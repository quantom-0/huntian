"""Module registry: the plug-and-play heart of Huntian.

Everything that can be replaced - physics models, solvers, compute backends,
EM adapters, AI helpers - is a *module* implementing :class:`IModule` and
describing itself with a :class:`ModuleManifest`. The registry is responsible
for:

* **discovery** - finding plugin packages on disk or by dotted import path;
* **contract checking** - refusing modules written against an incompatible
  plugin API version;
* **multi-version coexistence** - several versions of one module id may be
  registered at once, so old results stay reproducible;
* **dependency resolution** - version-range-aware topological ordering, with
  actionable errors for missing modules and cycles;
* **hot swapping** - registering a new version at runtime without disturbing
  already-running instances;
* **parameter validation** - rejecting unknown or out-of-range parameters,
  which is the safety gate for LLM-driven configuration.

Plugin entry-point convention
-----------------------------
A plugin is either a directory containing ``__init__.py`` (or ``module.py``)
or a dotted import path. It must expose two module-level names::

    MANIFEST: ModuleManifest
    MODULE: type[IModule]
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import time
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from .errors import (
    DependencyError,
    DuplicateModuleError,
    Issue,
    ModuleConfigError,
    ModuleLifecycleError,
    ModuleUnavailableError,
    HuntianError,
    RegistryError,
    RequirementError,
    VersionConflictError,
    VersionError,
    format_issues,
)
from .events import EventBus
from .schema import MISSING, Field, coerce_value
from .semver import Requirement, Version, VersionRange
from .version import API_VERSION

__all__ = [
    "ModuleCategory",
    "VERIFICATION_LEVELS",
    "ParamSpec",
    "ModuleManifest",
    "ModuleState",
    "IModule",
    "BaseModule",
    "ModuleRecord",
    "ResolutionPlan",
    "SwapReport",
    "DiscoveryReport",
    "ModuleRegistry",
    "discover_plugin_classes",
]

LOGGER_NAME = "huntian"

VERIFICATION_LEVELS = ("verified", "partially-verified", "unverified")

_MODULE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class ModuleCategory(str, Enum):
    """Broad role a module plays, used for catalogue grouping and selection."""

    MESH = "mesh"
    PHYSICS = "physics"
    TRANSPORT = "transport"
    SOLVER = "solver"
    BACKEND = "backend"
    EXTRACT = "extract"
    COMPACT = "compact"
    EM = "em"
    AI = "ai"
    OPTIMIZE = "optimize"
    IO = "io"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ParamSpec(Field):
    """A module parameter: a :class:`~huntian.schema.Field` plus provenance.

    ``calibratable`` marks parameters that may be freed by the automatic
    fitting loop. Everything else is treated as fixed structure, which keeps
    the LLM out of trouble.
    """

    name: str = ""
    source: str = ""
    calibratable: bool = False

    # -- constructors ---------------------------------------------------
    @classmethod
    def _make_spec(cls, kind: str, name: str, default: Any, **kw: Any) -> "ParamSpec":
        doc = kw.pop("doc", "")
        unit = kw.pop("unit", None)
        source = kw.pop("source", "")
        calibratable = kw.pop("calibratable", False)
        required = kw.pop("required", False)
        nullable = kw.pop("nullable", False)
        choices = kw.pop("choices", None)
        return cls(
            type=kind,
            name=name,
            default=default,
            required=required,
            doc=doc,
            unit=unit,
            choices=tuple(choices) if choices is not None else None,
            nullable=nullable,
            source=source,
            calibratable=calibratable,
            meta=dict(kw.pop("meta", {}) or {}),
            **kw,
        )

    @classmethod
    def float(cls, name: str, default: Any = MISSING, **kw: Any) -> "ParamSpec":
        return cls._make_spec("float", name, default, **kw)

    @classmethod
    def integer(cls, name: str, default: Any = MISSING, **kw: Any) -> "ParamSpec":
        return cls._make_spec("int", name, default, **kw)

    @classmethod
    def string(cls, name: str, default: Any = MISSING, **kw: Any) -> "ParamSpec":
        return cls._make_spec("str", name, default, **kw)

    @classmethod
    def boolean(cls, name: str, default: Any = MISSING, **kw: Any) -> "ParamSpec":
        return cls._make_spec("bool", name, default, **kw)

    @classmethod
    def choice(
        cls, name: str, choices: Sequence[Any], default: Any = MISSING, **kw: Any
    ) -> "ParamSpec":
        return cls._make_spec("enum", name, default, choices=tuple(choices), **kw)

    # -- presentation ---------------------------------------------------
    def _extras(self) -> dict[str, Any]:
        extras = super()._extras()
        if self.source:
            extras["x-source"] = self.source
        extras["x-calibratable"] = bool(self.calibratable)
        return extras

    def __repr__(self) -> str:
        bits = [f"{self.name}: {self.type}"]
        if self.has_default:
            bits.append(f"={self.default!r}")
        if self.unit:
            bits.append(f"[{self.unit}]")
        if self.calibratable:
            bits.append("calibratable")
        return f"ParamSpec({', '.join(bits)})"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleManifest:
    """Everything the registry, the pipeline and the LLM need to know."""

    module_id: str
    module_version: str = "0.1.0"
    api_version: str = API_VERSION
    category: ModuleCategory = ModuleCategory.PHYSICS
    description: str = ""
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    parameters: tuple[ParamSpec, ...] = ()
    author: str = ""
    tags: tuple[str, ...] = ()
    verification: str = "unverified"

    # -- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        def _set(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.module_id, str) or not _MODULE_ID_RE.match(self.module_id):
            raise RegistryError(
                f"invalid module_id {self.module_id!r}; expected dotted lowercase, "
                "e.g. 'physics.polarization.ambacher'"
            )
        try:
            Version.parse(self.module_version)
        except VersionError as exc:
            raise RegistryError(f"{self.module_id}: invalid module_version: {exc}") from exc
        try:
            Version.parse(self.api_version)
        except VersionError as exc:
            raise RegistryError(f"{self.module_id}: invalid api_version: {exc}") from exc

        category = self.category
        if not isinstance(category, ModuleCategory):
            try:
                category = ModuleCategory(str(category))
            except ValueError as exc:
                valid = ", ".join(c.value for c in ModuleCategory)
                raise RegistryError(
                    f"{self.module_id}: unknown category {category!r}; expected one of {valid}"
                ) from exc
            _set("category", category)

        if self.verification not in VERIFICATION_LEVELS:
            raise RegistryError(
                f"{self.module_id}: verification must be one of {list(VERIFICATION_LEVELS)}, "
                f"got {self.verification!r}"
            )

        for attr in ("capabilities", "dependencies", "requires", "produces", "tags"):
            value = getattr(self, attr)
            if not isinstance(value, tuple):
                _set(attr, tuple(value or ()))

        params = self.parameters
        if not isinstance(params, tuple):
            params = tuple(params or ())
            _set("parameters", params)
        seen: set[str] = set()
        for spec in params:
            if not isinstance(spec, ParamSpec):
                raise RegistryError(
                    f"{self.module_id}: parameters must be ParamSpec instances, got {type(spec).__name__}"
                )
            if not spec.name:
                raise RegistryError(f"{self.module_id}: a ParamSpec is missing its name")
            if spec.name in seen:
                raise RegistryError(f"{self.module_id}: duplicate parameter {spec.name!r}")
            seen.add(spec.name)

        for dependency in self.dependencies:
            Requirement.parse(dependency)  # raises RequirementError with detail

    # -- parameter access -----------------------------------------------
    def param(self, name: str) -> ParamSpec | None:
        for spec in self.parameters:
            if spec.name == name:
                return spec
        return None

    def param_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.parameters)

    def calibratable_params(self) -> tuple[ParamSpec, ...]:
        return tuple(spec for spec in self.parameters if spec.calibratable)

    def defaults(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.parameters if spec.has_default}

    def params_schema(self) -> dict[str, Any]:
        """JSON Schema of the parameter object (fed to the LLM layer)."""
        properties = {spec.name: spec.json_schema() for spec in self.parameters}
        required = [spec.name for spec in self.parameters if spec.required]
        out: dict[str, Any] = {
            "type": "object",
            "title": f"{self.module_id} parameters",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            out["required"] = required
        return out

    def validate_params(
        self, params: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], list[Issue]]:
        """Coerce and validate a parameter mapping; never raises."""
        supplied = dict(params or {})
        issues: list[Issue] = []
        values: dict[str, Any] = {}
        known = {spec.name: spec for spec in self.parameters}

        for key in supplied:
            if key not in known:
                message = f"unknown parameter {key!r} for {self.module_id}"
                hint = None
                if known:
                    hint = f"known parameters: {', '.join(sorted(known))}"
                issues.append(Issue(f"params.{key}", message, "unknown-param", hint=hint))

        for name, spec in known.items():
            path = f"params.{name}"
            if name in supplied:
                raw = supplied[name]
            elif spec.required:
                issues.append(Issue(path, "required parameter is missing", "required"))
                continue
            elif spec.has_default:
                raw = spec.default
            else:
                values[name] = None
                continue
            value, value_issues = coerce_value(path, spec, raw)
            if value_issues:
                issues.extend(value_issues)
            else:
                values[name] = value
        return values, issues

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "module_version": self.module_version,
            "api_version": self.api_version,
            "category": self.category.value,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "requires": list(self.requires),
            "produces": list(self.produces),
            "author": self.author,
            "tags": list(self.tags),
            "verification": self.verification,
            "parameters": [
                {
                    "name": spec.name,
                    "type": spec.type,
                    "default": None if spec.default is MISSING else spec.default,
                    "unit": spec.unit,
                    "minimum": spec.minimum,
                    "maximum": spec.maximum,
                    "choices": list(spec.choices) if spec.choices else None,
                    "doc": spec.doc,
                    "source": spec.source,
                    "calibratable": spec.calibratable,
                }
                for spec in self.parameters
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModuleManifest":
        params_raw = raw.get("parameters") or []
        specs: list[ParamSpec] = []
        for entry in params_raw:
            if not isinstance(entry, Mapping):
                raise RegistryError("manifest parameter entries must be objects")
            kind = str(entry.get("type", "any"))
            kwargs: dict[str, Any] = {
                "doc": entry.get("doc", ""),
                "unit": entry.get("unit"),
                "source": entry.get("source", ""),
                "calibratable": bool(entry.get("calibratable", False)),
            }
            if entry.get("minimum") is not None:
                kwargs["minimum"] = entry["minimum"]
            if entry.get("maximum") is not None:
                kwargs["maximum"] = entry["maximum"]
            if entry.get("choices"):
                kwargs["choices"] = tuple(entry["choices"])
            default = entry.get("default", MISSING)
            if default is None and entry.get("required"):
                default = MISSING
            specs.append(ParamSpec(type=kind, name=str(entry["name"]), default=default, **kwargs))
        return cls(
            module_id=str(raw["module_id"]),
            module_version=str(raw.get("module_version", "0.1.0")),
            api_version=str(raw.get("api_version", API_VERSION)),
            category=ModuleCategory(str(raw.get("category", "physics"))),
            description=str(raw.get("description", "")),
            capabilities=tuple(raw.get("capabilities", ()) or ()),
            dependencies=tuple(raw.get("dependencies", ()) or ()),
            requires=tuple(raw.get("requires", ()) or ()),
            produces=tuple(raw.get("produces", ()) or ()),
            parameters=tuple(specs),
            author=str(raw.get("author", "")),
            tags=tuple(raw.get("tags", ()) or ()),
            verification=str(raw.get("verification", "unverified")),
        )

    def __str__(self) -> str:
        return f"{self.module_id}@{self.module_version}"


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------


class ModuleState(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    TERMINATED = "terminated"
    FAILED = "failed"


class IModule(ABC):
    """The contract every pluggable module implements."""

    MANIFEST: ClassVar[ModuleManifest]

    # -- metadata --------------------------------------------------------
    @classmethod
    def describe(cls) -> ModuleManifest:
        manifest = getattr(cls, "MANIFEST", None)
        if manifest is None:
            raise RegistryError(f"{cls.__name__} does not declare a MANIFEST")
        return manifest

    # -- lifecycle -------------------------------------------------------
    @abstractmethod
    def initialize(self, params: Mapping[str, Any] | None = None) -> None:
        """Adopt validated parameters and acquire resources."""

    @abstractmethod
    def validate(self, ctx: Any) -> list[Issue]:
        """Pre-flight check against the current context; return problems."""

    @abstractmethod
    def run(self, ctx: Any) -> Any:
        """Execute against the context and return it (or a new one)."""

    @abstractmethod
    def terminate(self) -> None:
        """Release resources. Safe to call more than once."""


class BaseModule(IModule):
    """Lifecycle, parameter validation and state tracking, ready to subclass.

    Subclasses implement :meth:`execute` (and optionally :meth:`on_initialize`,
    :meth:`on_terminate`) and declare a ``MANIFEST`` class attribute.
    """

    MANIFEST: ClassVar[ModuleManifest]

    def __init__(self) -> None:
        self._state = ModuleState.CREATED
        self._params: dict[str, Any] = {}

    # -- state -----------------------------------------------------------
    @property
    def module_id(self) -> str:
        return type(self).MANIFEST.module_id

    @property
    def state(self) -> ModuleState:
        return self._state

    @property
    def params(self) -> Mapping[str, Any]:
        return MappingProxyType(self._params)

    def param(self, name: str, default: Any = MISSING) -> Any:
        """Resolved parameter value, falling back to the manifest default."""
        if name in self._params and self._params[name] is not None:
            return self._params[name]
        spec = type(self).MANIFEST.param(name)
        if spec is not None and spec.has_default and spec.default is not None:
            return spec.default
        if default is not MISSING:
            return default
        raise ModuleConfigError(f"{self.module_id}: parameter {name!r} has no value")

    # -- lifecycle -------------------------------------------------------
    def initialize(self, params: Mapping[str, Any] | None = None) -> None:
        if self._state is ModuleState.INITIALIZED:
            return
        if self._state in (ModuleState.TERMINATED, ModuleState.FAILED):
            raise ModuleLifecycleError(
                f"{self.module_id}: cannot initialize from state {self._state.value!r}"
            )
        manifest = type(self).MANIFEST
        values, issues = manifest.validate_params(params)
        if issues:
            raise ModuleConfigError(
                f"invalid parameters for {manifest.module_id}:\n{format_issues(issues)}"
            )
        self._params = values
        self._state = ModuleState.INITIALIZED
        self.on_initialize()

    def on_initialize(self) -> None:
        """Hook called once after parameters are adopted."""

    def validate(self, ctx: Any) -> list[Issue]:
        return []

    def run(self, ctx: Any) -> Any:
        if self._state is ModuleState.CREATED:
            self.initialize({})
        if self._state is not ModuleState.INITIALIZED:
            raise ModuleLifecycleError(
                f"{self.module_id}: cannot run from state {self._state.value!r}"
            )
        self._state = ModuleState.RUNNING
        try:
            result = self.execute(ctx)
        except BaseException:
            self._state = ModuleState.FAILED
            raise
        self._state = ModuleState.INITIALIZED
        return ctx if result is None else result

    @abstractmethod
    def execute(self, ctx: Any) -> Any:
        """Actual work. Return ``None`` to leave the context unchanged."""

    def terminate(self) -> None:
        if self._state is ModuleState.TERMINATED:
            return
        self.on_terminate()
        self._state = ModuleState.TERMINATED

    def on_terminate(self) -> None:
        """Hook called once when the module is released."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.module_id} state={self._state.value}>"


# ---------------------------------------------------------------------------
# Records and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleRecord:
    """A registered module class together with where it came from."""

    manifest: ModuleManifest
    module_cls: type[IModule]
    origin: str = "<runtime>"
    loaded_at: float = 0.0

    @property
    def module_id(self) -> str:
        return self.manifest.module_id

    @property
    def version(self) -> Version:
        return Version.parse(self.manifest.module_version)

    def __str__(self) -> str:
        return f"{self.manifest} ({self.origin})"


@dataclass(frozen=True)
class ResolutionPlan:
    """A resolved, topologically ordered set of modules."""

    records: tuple[ModuleRecord, ...] = ()
    requirements: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def ids(self) -> list[str]:
        return [record.module_id for record in self.records]

    def record(self, module_id: str) -> ModuleRecord:
        for record in self.records:
            if record.module_id == module_id:
                return record
        raise ModuleUnavailableError(f"{module_id!r} is not part of this plan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": [str(r.manifest) for r in self.records],
            "requirements": list(self.requirements),
            "edges": [list(e) for e in self.edges],
            "warnings": list(self.warnings),
        }

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __str__(self) -> str:
        return " -> ".join(self.ids())


@dataclass(frozen=True)
class SwapReport:
    """Outcome of a hot swap."""

    module_id: str
    active_version: str
    previous_versions: tuple[str, ...] = ()
    replaced: bool = False
    live_instances_unaffected: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "active_version": self.active_version,
            "previous_versions": list(self.previous_versions),
            "replaced": self.replaced,
            "live_instances_unaffected": self.live_instances_unaffected,
        }

    def __str__(self) -> str:
        return (
            f"hot-swap {self.module_id} -> {self.active_version} "
            f"(previous: {', '.join(self.previous_versions) or 'none'}, "
            f"live instances unaffected: {self.live_instances_unaffected})"
        )


@dataclass
class DiscoveryReport:
    """What a plugin discovery pass found and what it could not load."""

    loaded: list[ModuleRecord] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded": [str(record) for record in self.loaded],
            "errors": [{"path": path, "error": message} for path, message in self.errors],
        }

    def __str__(self) -> str:
        lines = [f"discovered {len(self.loaded)} module(s)"]
        for record in self.loaded:
            lines.append(f"  + {record.manifest}  <-  {record.origin}")
        for path, message in self.errors:
            lines.append(f"  ! {path}: {message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _ensure_namespace_package(dotted_name: str) -> None:
    """Register a synthetic namespace package so plugin-relative imports work.

    A plugin package is loaded under a synthetic name such as
    ``huntian_plugins.<dirname>_<index>``. For ``from .solver import x`` inside
    that package to resolve, its parent (``huntian_plugins``) must exist in
    ``sys.modules`` with a ``__path__``. A real importable package wins if one
    is already present.
    """
    if not dotted_name or dotted_name in sys.modules:
        return
    module = types.ModuleType(dotted_name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[dotted_name] = module


def _load_plugin_module(path: Path, import_name: str) -> types.ModuleType:
    """Load a plugin package/file without requiring it to be on sys.path."""
    _ensure_namespace_package(import_name.rpartition(".")[0])
    if path.is_dir():
        init_file = path / "__init__.py"
        standalone = path / "module.py"
        if init_file.is_file():
            source, search = init_file, [str(path)]
        elif standalone.is_file():
            source, search = standalone, None
        else:
            raise RegistryError(
                f"{path}: plugin directory needs an __init__.py or module.py entry point"
            )
    else:
        source, search = path, None

    spec = importlib.util.spec_from_file_location(
        import_name, source, submodule_search_locations=search
    )
    if spec is None or spec.loader is None:
        raise RegistryError(f"cannot build an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(import_name, None)
        raise
    return module


def _plugin_class_from_module(module: types.ModuleType, origin: str) -> type[IModule]:
    manifest = getattr(module, "MANIFEST", None)
    module_cls = getattr(module, "MODULE", None)
    if manifest is None or module_cls is None:
        raise RegistryError(
            f"{origin}: a plugin must define module-level MANIFEST and MODULE "
            f"(found MANIFEST={manifest is not None}, MODULE={module_cls is not None})"
        )
    if not isinstance(manifest, ModuleManifest):
        raise RegistryError(f"{origin}: MANIFEST must be a ModuleManifest instance")
    if not isinstance(module_cls, type) or not issubclass(module_cls, IModule):
        raise RegistryError(f"{origin}: MODULE must be an IModule subclass")
    declared = getattr(module_cls, "MANIFEST", None)
    if declared is None:
        module_cls.MANIFEST = manifest  # type: ignore[attr-defined]
    elif declared.module_id != manifest.module_id:
        raise RegistryError(
            f"{origin}: MANIFEST ({manifest.module_id}) disagrees with "
            f"MODULE.MANIFEST ({declared.module_id})"
        )
    return module_cls


def discover_plugin_classes(
    paths: Iterable[str | Path],
    *,
    import_prefix: str = "huntian_plugins",
) -> tuple[list[tuple[type[IModule], str]], list[tuple[str, str]]]:
    """Import every plugin under ``paths``.

    Returns ``(plugins, errors)`` where each plugin is ``(class, origin)`` and
    each error is ``(path, message)``. A broken plugin never aborts discovery.
    """
    found: list[tuple[type[IModule], str]] = []
    errors: list[tuple[str, str]] = []
    for raw_path in paths:
        root = Path(raw_path)
        if not root.exists():
            errors.append((str(root), "path does not exist"))
            continue
        candidates: list[Path] = []
        if root.is_dir():
            if (root / "__init__.py").is_file() or (root / "module.py").is_file():
                candidates.append(root)
            else:
                candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))
        else:
            candidates.append(root)

        for index, candidate in enumerate(candidates):
            if candidate.name.startswith((".", "_")):
                continue
            import_name = f"{import_prefix}.{candidate.name}_{index}"
            origin = str(candidate)
            try:
                module = _load_plugin_module(candidate, import_name)
                module_cls = _plugin_class_from_module(module, origin)
                found.append((module_cls, origin))
            except Exception as exc:  # noqa: BLE001 - discovery is best-effort
                errors.append((origin, f"{type(exc).__name__}: {exc}"))
    return found, errors


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModuleRegistry:
    """Indexes module classes by id and version and resolves requests."""

    def __init__(
        self,
        *,
        api_version: str = API_VERSION,
        bus: EventBus | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_version = api_version
        self._records: dict[str, dict[Version, ModuleRecord]] = {}
        self._active: dict[str, list[IModule]] = {}
        self._bus = bus if bus is not None else EventBus()
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._discovery_errors: list[tuple[str, str]] = []

    # -- registration ----------------------------------------------------
    @property
    def bus(self) -> EventBus:
        return self._bus

    def register(
        self,
        module_cls: type[IModule],
        *,
        origin: str = "<runtime>",
        replace: bool = False,
    ) -> ModuleRecord:
        """Add a module class to the registry."""
        if not isinstance(module_cls, type) or not issubclass(module_cls, IModule):
            raise RegistryError(f"{module_cls!r} is not an IModule subclass")
        manifest = module_cls.describe()
        if not self.api_compatible(manifest):
            raise RegistryError(
                f"{manifest.module_id}@{manifest.module_version} targets plugin API "
                f"{manifest.api_version}, but this registry implements {self.api_version}"
            )
        version = Version.parse(manifest.module_version)
        bucket = self._records.setdefault(manifest.module_id, {})
        if version in bucket and not replace:
            existing = bucket[version]
            if existing.module_cls is module_cls:
                return existing
            raise DuplicateModuleError(
                f"{manifest.module_id}@{version} is already registered from {existing.origin}; "
                "pass replace=True to override"
            )
        record = ModuleRecord(
            manifest=manifest,
            module_cls=module_cls,
            origin=origin,
            loaded_at=time.time(),
        )
        bucket[version] = record
        self._bus.publish(
            "registry.registered",
            module_id=manifest.module_id,
            version=str(version),
            origin=origin,
            category=manifest.category.value,
        )
        self._log.debug("registered %s from %s", manifest, origin)
        return record

    def unregister(self, module_id: str, version: str | None = None) -> int:
        """Remove one version, or every version, of a module id."""
        bucket = self._records.get(module_id)
        if not bucket:
            return 0
        if version is None:
            removed = len(bucket)
            del self._records[module_id]
        else:
            parsed = Version.parse(version)
            removed = 1 if bucket.pop(parsed, None) is not None else 0
            if not bucket:
                self._records.pop(module_id, None)
        if removed:
            self._bus.publish(
                "registry.unregistered", module_id=module_id, version=version or "*"
            )
        return removed

    def clear(self) -> None:
        self._records.clear()
        self._active.clear()
        self._discovery_errors.clear()

    def api_compatible(self, manifest: ModuleManifest) -> bool:
        return Version.parse(manifest.api_version).major == Version.parse(self.api_version).major

    # -- loading ---------------------------------------------------------
    def load_from_paths(self, paths: Iterable[str | Path]) -> DiscoveryReport:
        """Discover and register every plugin found under ``paths``."""
        report = DiscoveryReport()
        found, errors = discover_plugin_classes(paths)
        report.errors.extend(errors)
        self._discovery_errors.extend(errors)
        for module_cls, origin in found:
            try:
                report.loaded.append(self.register(module_cls, origin=origin))
            except HuntianError as exc:
                report.errors.append((origin, str(exc)))
                self._discovery_errors.append((origin, str(exc)))
        return report

    def load_from_module_path(self, dotted_path: str) -> ModuleRecord:
        """Register the plugin exposed by an importable dotted module path."""
        try:
            module = importlib.import_module(dotted_path)
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"cannot import {dotted_path!r}: {exc}") from exc
        module_cls = _plugin_class_from_module(module, dotted_path)
        return self.register(module_cls, origin=f"import:{dotted_path}")

    # -- inspection ------------------------------------------------------
    @property
    def discovery_errors(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._discovery_errors)

    def records(
        self,
        *,
        category: ModuleCategory | str | None = None,
        capability: str | None = None,
        verification: str | None = None,
        module_id: str | None = None,
    ) -> list[ModuleRecord]:
        want_category = ModuleCategory(str(category)) if category is not None else None
        out: list[ModuleRecord] = []
        for mid, bucket in self._records.items():
            if module_id is not None and mid != module_id:
                continue
            for record in bucket.values():
                manifest = record.manifest
                if want_category is not None and manifest.category is not want_category:
                    continue
                if capability is not None and capability not in manifest.capabilities:
                    continue
                if verification is not None and manifest.verification != verification:
                    continue
                out.append(record)
        out.sort(key=lambda r: (r.module_id, tuple(-p for p in r.version.release)))
        return out

    def module_ids(self) -> list[str]:
        return sorted(self._records)

    def versions(self, module_id: str) -> list[Version]:
        return sorted(self._records.get(module_id, {}), reverse=True)

    def has(self, module_id: str, spec: str = "*") -> bool:
        return any(VersionRange.parse(spec).matches(v) for v in self.versions(module_id))

    def select(self, module_id: str, spec: str | VersionRange = "*") -> ModuleRecord:
        """Pick the highest registered version satisfying ``spec``."""
        range_ = VersionRange.parse(spec)
        bucket = self._records.get(module_id)
        if not bucket:
            raise ModuleUnavailableError(
                f"no module registered as {module_id!r}; "
                f"known ids: {', '.join(self.module_ids()) or '(none)'}"
            )
        matching = [record for version, record in bucket.items() if range_.matches(version)]
        if not matching:
            available = ", ".join(str(v) for v in sorted(bucket, reverse=True))
            raise ModuleUnavailableError(
                f"module {module_id!r} has no version matching {range_}; available: {available}"
            )
        return max(matching, key=lambda r: r.version)

    def select_by_capability(
        self, capability: str, spec: str = "*"
    ) -> list[ModuleRecord]:
        """All records declaring ``capability``, best version per module id."""
        best: dict[str, ModuleRecord] = {}
        range_ = VersionRange.parse(spec)
        for record in self.records(capability=capability):
            if not range_.matches(record.version):
                continue
            current = best.get(record.module_id)
            if current is None or record.version > current.version:
                best[record.module_id] = record
        return [best[mid] for mid in sorted(best)]

    # -- resolution ------------------------------------------------------
    def resolve(self, requirements: Iterable[str | Requirement]) -> ResolutionPlan:
        """Resolve requirements plus their transitive dependencies."""
        parsed: list[Requirement] = [Requirement.parse(r) for r in requirements]
        selected: dict[str, ModuleRecord] = {}
        constraints: dict[str, list[str]] = {}
        missing: list[str] = []
        queue: list[Requirement] = list(parsed)

        while queue:
            requirement = queue.pop(0)
            constraints.setdefault(requirement.module_id, []).append(str(requirement.version))
            current = selected.get(requirement.module_id)
            if current is not None:
                if not requirement.matches(current.version):
                    raise VersionConflictError(
                        f"{requirement.module_id}: {current.manifest.module_version} was selected, "
                        f"but {requirement.version} is also required (from {constraints[requirement.module_id]})"
                    )
                continue
            try:
                record = self.select(requirement.module_id, requirement.version)
            except ModuleUnavailableError as exc:
                missing.append(str(exc))
                continue
            selected[record.module_id] = record
            for dependency in record.manifest.dependencies:
                queue.append(Requirement.parse(dependency))

        if missing:
            raise ModuleUnavailableError(
                "unresolved module requirements:\n" + format_issues(
                    [Issue("", m, "unavailable") for m in missing]
                )
            )

        edges: list[tuple[str, str]] = []
        for record in selected.values():
            for dependency in record.manifest.dependencies:
                dep_id = Requirement.parse(dependency).module_id
                if dep_id in selected:
                    edges.append((dep_id, record.module_id))

        order = self._topological_order(selected, edges, constraints)
        warnings: list[str] = []
        for module_id in order:
            record = selected[module_id]
            if record.manifest.verification == "unverified":
                warnings.append(
                    f"{record.manifest} is marked 'unverified' - results must not be "
                    "reported as validated"
                )
            elif record.manifest.verification == "partially-verified":
                warnings.append(f"{record.manifest} is only 'partially-verified'")
            if not record.manifest.produces:
                warnings.append(f"{record.manifest} declares no produced artifacts")

        return ResolutionPlan(
            records=tuple(selected[mid] for mid in order),
            requirements=tuple(str(r) for r in parsed),
            edges=tuple(edges),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _topological_order(
        selected: Mapping[str, ModuleRecord],
        edges: Sequence[tuple[str, str]],
        constraints: Mapping[str, list[str]],
    ) -> list[str]:
        incoming: dict[str, set[str]] = {mid: set() for mid in selected}
        outgoing: dict[str, set[str]] = {mid: set() for mid in selected}
        for dep_id, module_id in edges:
            incoming[module_id].add(dep_id)
            outgoing[dep_id].add(module_id)
        ready = sorted(mid for mid, deps in incoming.items() if not deps)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for downstream in sorted(outgoing[node]):
                incoming[downstream].discard(node)
                if not incoming[downstream]:
                    ready.append(downstream)
                    ready.sort()
        if len(order) != len(selected):
            cyclic = sorted(mid for mid, deps in incoming.items() if deps)
            raise DependencyError(
                "module dependency cycle detected among: " + ", ".join(cyclic)
            )
        return order

    # -- instantiation ---------------------------------------------------
    def instantiate(
        self,
        module_id: str,
        *,
        version: str | VersionRange = "*",
        params: Mapping[str, Any] | None = None,
        track: bool = True,
    ) -> IModule:
        """Create and initialize a module instance."""
        record = self.select(module_id, version)
        instance = record.module_cls()
        instance.initialize(params or {})
        if track:
            self._active.setdefault(module_id, []).append(instance)
        self._bus.publish(
            "registry.instantiated",
            module_id=module_id,
            version=record.manifest.module_version,
        )
        return instance

    def build(
        self,
        plan: ResolutionPlan,
        params_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[IModule]:
        """Instantiate every module of a plan, in dependency order."""
        params_by_id = params_by_id or {}
        instances: list[IModule] = []
        for record in plan:
            instance = record.module_cls()
            instance.initialize(params_by_id.get(record.module_id, {}))
            self._active.setdefault(record.module_id, []).append(instance)
            instances.append(instance)
        return instances

    def release(self, instance: IModule) -> None:
        """Terminate an instance and drop it from the active list."""
        module_id = type(instance).MANIFEST.module_id
        live = self._active.get(module_id, [])
        self._active[module_id] = [item for item in live if item is not instance]
        try:
            instance.terminate()
        finally:
            self._bus.publish("registry.released", module_id=module_id)

    @property
    def active_instances(self) -> Mapping[str, int]:
        return {mid: len(items) for mid, items in self._active.items() if items}

    # -- hot swap --------------------------------------------------------
    def hot_swap(
        self,
        module_cls: type[IModule],
        *,
        origin: str = "<hot-swap>",
        replace: bool = False,
    ) -> SwapReport:
        """Register a new module version at runtime.

        Instances created before the swap keep running against the version they
        were built from; only later resolutions see the new version.
        """
        manifest = module_cls.describe()
        previous = tuple(str(v) for v in self.versions(manifest.module_id))
        record = self.register(module_cls, origin=origin, replace=replace)
        report = SwapReport(
            module_id=manifest.module_id,
            active_version=manifest.module_version,
            previous_versions=previous,
            replaced=replace,
            live_instances_unaffected=len(self._active.get(manifest.module_id, [])),
        )
        self._bus.publish("registry.swapped", **report.as_dict())
        return report

    # -- reporting -------------------------------------------------------
    def health(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_verification: dict[str, int] = {}
        for bucket in self._records.values():
            for record in bucket.values():
                by_category[str(record.manifest.category)] = (
                    by_category.get(str(record.manifest.category), 0) + 1
                )
                level = record.manifest.verification
                by_verification[level] = by_verification.get(level, 0) + 1
        return {
            "api_version": self.api_version,
            "module_ids": len(self._records),
            "module_versions": sum(len(b) for b in self._records.values()),
            "by_category": by_category,
            "by_verification": by_verification,
            "active_instances": dict(self.active_instances),
            "discovery_errors": len(self._discovery_errors),
            "ok": not self._discovery_errors,
        }

    def catalog(self) -> list[dict[str, Any]]:
        """Manifest descriptions of every registered module (LLM-readable)."""
        return [record.manifest.to_dict() for record in self.records()]

    def render_catalog(self) -> str:
        lines: list[str] = []
        for module_id in self.module_ids():
            for record in sorted(
                self._records[module_id].values(), key=lambda r: r.version, reverse=True
            ):
                manifest = record.manifest
                lines.append(
                    f"{manifest.module_id}@{manifest.module_version} "
                    f"[{manifest.category}] ({manifest.verification})"
                )
                if manifest.description:
                    lines.append(f"    {manifest.description}")
                if manifest.capabilities:
                    lines.append(f"    capabilities: {', '.join(manifest.capabilities)}")
                if manifest.parameters:
                    names = ", ".join(
                        f"{p.name}{'*' if p.calibratable else ''}" for p in manifest.parameters
                    )
                    lines.append(f"    params: {names}   (* = calibratable)")
        return "\n".join(lines) if lines else "(registry is empty)"

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._records.values())

    def __contains__(self, module_id: object) -> bool:
        return module_id in self._records

    def __repr__(self) -> str:
        return (
            f"ModuleRegistry(api={self.api_version}, ids={len(self._records)}, "
            f"versions={len(self)})"
        )
