"""A dependency-free declarative schema layer.

Why not ``pydantic``?
---------------------
The core of NemoTCAD is deliberately importable on a bare Python install, so
this module provides the subset of pydantic behaviour the project actually
needs:

* declarative fields with types, defaults, units, ranges and choices,
* **strict unknown-field rejection** (the primary safety gate for
  LLM-generated Decks and module parameters),
* nested schemas and lists of schemas,
* structured :class:`~nemo.errors.Issue` reporting with dotted paths,
* JSON Schema emission, which is what the LLM layer is fed with.

Behaviour is intentionally pydantic-compatible in spirit, so a later swap to
pydantic can be done behind the same call surface (``from_dict`` /
``to_dict`` / ``json_schema`` / ``check``).

Quick example
-------------
>>> class GateSpec(Schema):
...     length_um = Float(0.15, minimum=0.0, exclusive_minimum=True, doc="gate length")
...     kind = Enum(("schottky", "mis"), default="schottky")
>>> gate = GateSpec(length_um=0.25)
>>> gate.length_um
0.25
>>> GateSpec.from_dict({"lenght_um": 0.25})
Traceback (most recent call last):
    ...
nemo.errors.SchemaError: schema validation failed...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import (
    Any,
    ClassVar,
    Iterable,
    Mapping,
    Sequence,
)

from .errors import Issue, SchemaError

__all__ = [
    "MISSING",
    "coerce_value",
    "Field",
    "Float",
    "Int",
    "Str",
    "Bool",
    "Enum",
    "ListOf",
    "DictOf",
    "Nested",
    "AnyField",
    "Schema",
]


class _MissingType:
    """Sentinel distinguishing "no default declared" from ``None``."""

    _instance: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING = _MissingType()


# ---------------------------------------------------------------------------
# Field specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Field:
    """Declarative description of one schema field.

    Prefer the helper constructors (:func:`Float`, :func:`Str`, ...) over
    instantiating this class directly.
    """

    type: str = "any"
    default: Any = MISSING
    required: bool = False
    doc: str = ""
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: tuple[Any, ...] | None = None
    item: "Field | None" = None
    schema: Any = None
    nullable: bool = False
    pattern: str | None = None
    meta: Mapping[str, Any] = dc_field(default_factory=dict, compare=False, repr=False)

    # -- introspection --------------------------------------------------
    @property
    def has_default(self) -> bool:
        return self.default is not MISSING

    def _extras(self) -> dict[str, Any]:
        return {f"x-{key}": value for key, value in self.meta.items()}

    def json_schema(self) -> dict[str, Any]:
        return _field_json_schema(self)

    def __repr__(self) -> str:
        bits = [f"type={self.type!r}"]
        if self.has_default:
            bits.append(f"default={self.default!r}")
        if self.required:
            bits.append("required=True")
        if self.unit:
            bits.append(f"unit={self.unit!r}")
        return f"Field({', '.join(bits)})"


def AnyField(default: Any = MISSING, *, required: bool = False, doc: str = "", **kw: Any) -> Field:
    return Field(type="any", default=default, required=required, doc=doc, **kw)


def Float(
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
    nullable: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="float",
        default=default,
        required=required,
        doc=doc,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
        exclusive_maximum=exclusive_maximum,
        nullable=nullable,
        meta=dict(meta or {}),
    )


def Int(
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    unit: str | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
    nullable: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="int",
        default=default,
        required=required,
        doc=doc,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        nullable=nullable,
        meta=dict(meta or {}),
    )


def Str(
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    pattern: str | None = None,
    choices: Sequence[str] | None = None,
    nullable: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    if choices is not None:
        return Field(
            type="enum",
            default=default,
            required=required,
            doc=doc,
            choices=tuple(choices),
            nullable=nullable,
            meta=dict(meta or {}),
        )
    return Field(
        type="str",
        default=default,
        required=required,
        doc=doc,
        pattern=pattern,
        nullable=nullable,
        meta=dict(meta or {}),
    )


def Bool(
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    nullable: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="bool",
        default=default,
        required=required,
        doc=doc,
        nullable=nullable,
        meta=dict(meta or {}),
    )


def Enum(
    choices: Sequence[Any],
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    nullable: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="enum",
        default=default,
        required=required,
        doc=doc,
        choices=tuple(choices),
        nullable=nullable,
        meta=dict(meta or {}),
    )


def ListOf(
    item: Field,
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    unit: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="list",
        item=item,
        default=default,
        required=required,
        doc=doc,
        unit=unit,
        meta=dict(meta or {}),
    )


def DictOf(
    item: Field | None = None,
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    unit: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Field:
    return Field(
        type="dict",
        item=item,
        default=default,
        required=required,
        doc=doc,
        unit=unit,
        meta=dict(meta or {}),
    )


def Nested(
    schema_cls: type,
    default: Any = MISSING,
    *,
    required: bool = False,
    doc: str = "",
    nullable: bool = False,
) -> Field:
    return Field(
        type="schema",
        schema=schema_cls,
        default=default,
        required=required,
        doc=doc,
        nullable=nullable,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _typename(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "list"
    return type(value).__name__


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _prefix_issues(issues: Iterable[Issue], prefix: str) -> list[Issue]:
    if not prefix:
        return list(issues)
    out = []
    for issue in issues:
        path = _join(prefix, issue.path) if issue.path else prefix
        out.append(Issue(path=path, message=issue.message, code=issue.code, hint=issue.hint))
    return out


def _check_bounds(path: str, spec: Field, value: float, issues: list[Issue]) -> Any:
    failed = False
    if spec.minimum is not None:
        if spec.exclusive_minimum and value <= spec.minimum:
            issues.append(
                Issue(path, f"value {value} must be greater than {spec.minimum}", "range")
            )
            failed = True
        elif not spec.exclusive_minimum and value < spec.minimum:
            issues.append(
                Issue(path, f"value {value} must be greater than or equal to {spec.minimum}", "range")
            )
            failed = True
    if spec.maximum is not None:
        if spec.exclusive_maximum and value >= spec.maximum:
            issues.append(Issue(path, f"value {value} must be less than {spec.maximum}", "range"))
            failed = True
        elif not spec.exclusive_maximum and value > spec.maximum:
            issues.append(
                Issue(path, f"value {value} must be less than or equal to {spec.maximum}", "range")
            )
            failed = True
    return None if failed else value


def _coerce_schema(path: str, spec: Field, value: Any, issues: list[Issue]) -> Any:
    schema_cls = spec.schema
    if schema_cls is None:
        issues.append(Issue(path, "schema field declared without a target class", "schema-def"))
        return None
    if isinstance(value, schema_cls):
        return value
    if not isinstance(value, Mapping):
        issues.append(
            Issue(path, f"expected an object for {schema_cls.__name__}, got {_typename(value)}", "type")
        )
        return None
    values, sub_issues = schema_cls._build(value, path)
    if sub_issues:
        issues.extend(sub_issues)
        return None
    obj = schema_cls._make(values)
    semantic = _prefix_issues(obj.check(), path)
    if semantic:
        issues.extend(semantic)
        return None
    return obj


def _coerce(path: str, spec: Field, value: Any, issues: list[Issue]) -> Any:
    if value is None:
        if spec.nullable or spec.type == "any":
            return None
        issues.append(Issue(path, "value must not be null", "null"))
        return None

    kind = spec.type
    if kind == "any":
        return value

    if kind == "bool":
        if isinstance(value, bool):
            return value
        issues.append(Issue(path, f"expected bool, got {_typename(value)}", "type"))
        return None

    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(Issue(path, f"expected integer, got {_typename(value)}", "type"))
            return None
        return _check_bounds(path, spec, value, issues)

    if kind in ("float", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(Issue(path, f"expected number, got {_typename(value)}", "type"))
            return None
        return _check_bounds(path, spec, float(value), issues)

    if kind == "str":
        if not isinstance(value, str):
            issues.append(Issue(path, f"expected string, got {_typename(value)}", "type"))
            return None
        if spec.pattern is not None and re.fullmatch(spec.pattern, value) is None:
            issues.append(
                Issue(path, f"string {value!r} does not match pattern {spec.pattern!r}", "pattern")
            )
            return None
        return value

    if kind == "enum":
        choices = tuple(spec.choices or ())
        if value not in choices:
            issues.append(
                Issue(path, f"expected one of {list(choices)}, got {value!r}", "enum")
            )
            return None
        return value

    if kind == "list":
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            issues.append(Issue(path, f"expected a list, got {_typename(value)}", "type"))
            return None
        out: list[Any] = []
        for index, raw_item in enumerate(value):
            if spec.item is None:
                out.append(raw_item)
            else:
                out.append(_coerce(f"{path}[{index}]", spec.item, raw_item, issues))
        return out

    if kind == "dict":
        if not isinstance(value, Mapping):
            issues.append(Issue(path, f"expected an object, got {_typename(value)}", "type"))
            return None
        if spec.item is None:
            return dict(value)
        out_map: dict[str, Any] = {}
        for key, raw_item in value.items():
            if not isinstance(key, str):
                issues.append(
                    Issue(path, f"object keys must be strings, got {_typename(key)}", "type")
                )
                continue
            out_map[key] = _coerce(f"{path}.{key}", spec.item, raw_item, issues)
        return out_map

    if kind == "schema":
        return _coerce_schema(path, spec, value, issues)

    issues.append(Issue(path, f"unknown field type {kind!r} in schema definition", "schema-def"))
    return None


def coerce_value(path: str, spec: Field, value: Any) -> tuple[Any, list[Issue]]:
    """Validate a single value against ``spec``.

    Public, non-raising entry point used by module manifests to validate
    parameters outside of a :class:`Schema` document. Returns
    ``(coerced_value, issues)``.
    """
    issues: list[Issue] = []
    result = _coerce(path, spec, value, issues)
    return result, issues


def _to_plain(value: Any) -> Any:
    if isinstance(value, Schema):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _field_json_schema(spec: Field) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kind = spec.type
    if kind == "any":
        pass
    elif kind in ("float", "number"):
        out["type"] = "number"
    elif kind == "int":
        out["type"] = "integer"
    elif kind == "bool":
        out["type"] = "boolean"
    elif kind == "str":
        out["type"] = "string"
    elif kind == "enum":
        out["enum"] = list(spec.choices or ())
    elif kind == "list":
        out["type"] = "array"
        out["items"] = _field_json_schema(spec.item) if spec.item is not None else {}
    elif kind == "dict":
        out["type"] = "object"
        out["additionalProperties"] = (
            _field_json_schema(spec.item) if spec.item is not None else True
        )
    elif kind == "schema":
        if spec.schema is not None:
            nested = spec.schema.json_schema()
            nested.pop("$schema", None)
            out.update(nested)
        else:
            out["type"] = "object"
    else:
        out["type"] = "object"

    if spec.doc:
        out["description"] = spec.doc
    if spec.has_default:
        out["default"] = _to_plain(spec.default)
    if spec.minimum is not None:
        if spec.exclusive_minimum:
            out["exclusiveMinimum"] = spec.minimum
        else:
            out["minimum"] = spec.minimum
    if spec.maximum is not None:
        if spec.exclusive_maximum:
            out["exclusiveMaximum"] = spec.maximum
        else:
            out["maximum"] = spec.maximum
    if spec.pattern:
        out["pattern"] = spec.pattern
    if spec.unit:
        out["x-unit"] = spec.unit
    if spec.nullable:
        existing_type = out.get("type")
        out["type"] = [existing_type, "null"] if existing_type else ["object", "null"]
    out.update(spec._extras())
    return out


# ---------------------------------------------------------------------------
# Schema base class
# ---------------------------------------------------------------------------


class Schema:
    """Base class for declarative, strictly-validated documents.

    Subclasses declare fields as class attributes using the helper
    constructors. Field order follows declaration order, which is preserved in
    :meth:`json_schema` output.
    """

    __fields__: ClassVar[dict[str, Field]] = {}
    __strict__: ClassVar[bool] = True
    __title__: ClassVar[str | None] = None

    def __init_subclass__(cls, strict: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field] = {}
        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, "__fields__", None) or {})
        for name, value in list(vars(cls).items()):
            if isinstance(value, Field):
                fields[name] = value
                delattr(cls, name)
        cls.__fields__ = fields
        cls.__strict__ = strict

    # -- construction ---------------------------------------------------
    @classmethod
    def _make(cls, values: Mapping[str, Any]) -> "Schema":
        obj = cls.__new__(cls)
        obj.__dict__.update(values)
        return obj

    @classmethod
    def _build(cls, data: Any, path: str = "") -> tuple[dict[str, Any], list[Issue]]:
        values: dict[str, Any] = {}
        issues: list[Issue] = []
        if not isinstance(data, Mapping):
            issues.append(
                Issue(path or "<root>", f"expected an object, got {_typename(data)}", "type")
            )
            return values, issues

        fields = cls.__fields__
        if cls.__strict__:
            for key in data:
                if key not in fields:
                    hint = _closest_match(key, fields)
                    issues.append(
                        Issue(
                            _join(path, str(key)),
                            f"unknown field '{key}'",
                            "unknown-field",
                            hint=hint,
                        )
                    )

        for name, spec in fields.items():
            field_path = _join(path, name)
            if name in data:
                raw = data[name]
            elif spec.required:
                issues.append(Issue(field_path, "required field is missing", "required"))
                continue
            elif spec.has_default:
                raw = spec.default
            else:
                values[name] = None
                continue
            values[name] = _coerce(field_path, spec, raw, issues)
        return values, issues

    def __init__(self, **data: Any) -> None:
        cls = type(self)
        values, issues = cls._build(data)
        if not issues:
            probe = cls._make(values)
            issues = list(probe.check())
        if issues:
            raise SchemaError(issues, where=cls.__title__ or cls.__name__)
        self.__dict__.update(values)

    @classmethod
    def from_dict(cls, data: Any) -> "Schema":
        """Build an instance from a mapping, raising :class:`SchemaError`."""
        if not isinstance(data, Mapping):
            raise SchemaError(
                [Issue("<root>", f"expected an object, got {_typename(data)}", "type")],
                where=cls.__title__ or cls.__name__,
            )
        return cls(**data)

    @classmethod
    def validate_dict(cls, data: Any) -> tuple["Schema | None", list[Issue]]:
        """Non-raising validation; returns ``(instance_or_None, issues)``."""
        values, issues = cls._build(data)
        if issues:
            return None, issues
        obj = cls._make(values)
        issues = list(obj.check())
        if issues:
            return None, issues
        return obj, []

    @classmethod
    def fields(cls) -> Mapping[str, Field]:
        """Read-only view of the declared fields."""
        return dict(cls.__fields__)

    # -- semantic hook --------------------------------------------------
    def check(self) -> Iterable[Issue]:
        """Cross-field validation hook; override in subclasses."""
        return ()

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            name: _to_plain(getattr(self, name, None)) for name in type(self).__fields__
        }

    def update(self, **changes: Any) -> "Schema":
        """Return a new instance with ``changes`` applied on top of this one."""
        merged = self.to_dict()
        merged.update(changes)
        return type(self)(**merged)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """JSON Schema (2020-12) description of this document."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, spec in cls.__fields__.items():
            properties[name] = _field_json_schema(spec)
            if spec.required:
                required.append(name)
        out: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": cls.__title__ or cls.__name__,
            "type": "object",
            "properties": properties,
            "additionalProperties": not cls.__strict__,
        }
        if required:
            out["required"] = required
        return out

    # -- dunder ---------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Schema):
            return type(self) is type(other) and self.to_dict() == other.to_dict()
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        parts = []
        for name in list(type(self).__fields__)[:6]:
            parts.append(f"{name}={getattr(self, name, None)!r}")
        if len(type(self).__fields__) > 6:
            parts.append("...")
        return f"{type(self).__name__}({', '.join(parts)})"


def _closest_match(key: str, candidates: Iterable[str]) -> str | None:
    """Cheap typo hint: same lowercase or a close prefix."""
    lowered = key.lower()
    for candidate in candidates:
        if candidate.lower() == lowered:
            return f"did you mean '{candidate}'?"
    for candidate in candidates:
        if candidate.startswith(key) or key.startswith(candidate):
            return f"did you mean '{candidate}'?"
    return None
