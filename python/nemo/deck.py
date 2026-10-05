"""Deck: the structured document describing one simulation campaign.

A Deck is the single source of truth for *what to simulate* and *which plugin
implements each physical role*. It is deliberately declarative so that:

* a human can hand-write it,
* the LLM layer can generate it and have it validated mechanically,
* and the registry can resolve every ``module`` reference to a concrete,
  version-pinned plugin.

Two-stage validation is used throughout:

1. **structural** - types, ranges, required fields, cross-field rules. Runs on
   construction and raises :class:`~nemo.errors.SchemaError`.
2. **contextual** - does the plugin exist, is the version satisfiable, are the
   parameters known to that plugin, do the material names resolve. Runs on
   demand and *returns* issues so a repair loop can consume them.

JSON is always supported. YAML is supported when PyYAML is importable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import DeckError, Issue
from .materials import MaterialDB, MaterialLookupError
from .registry import ModuleRegistry
from .schema import (
    MISSING,
    Bool,
    DictOf,
    Enum,
    Float,
    Int,
    ListOf,
    Nested,
    Schema,
    Str,
)
from .version import SCHEMA_VERSION

__all__ = [
    "LayerSpec",
    "GateSpec",
    "GeometrySpec",
    "DeviceSpec",
    "ModuleBinding",
    "PhysicsSpec",
    "ModulesSpec",
    "SweepSpec",
    "SimulationSpec",
    "RunSpec",
    "Deck",
    "SLOT_ORDER",
    "SLOT_CAPABILITY",
    "SERVICE_SLOTS",
    "load_deck",
    "loads_deck",
    "dump_deck",
    "deck_json_schema",
    "yaml_available",
]

#: Canonical execution order used as a tie-breaker when the dependency graph
#: does not fully constrain the order of stages.
SLOT_ORDER: tuple[str, ...] = (
    "materials",
    "mesh",
    "backend",
    "polarization",
    "quantum",
    "mobility",
    "traps",
    "thermal",
    "transport",
    "nonlinear",
    "linear",
)

#: Capability each well-known slot expects a module to declare. Used as a
#: contract check so that, say, a mobility model cannot be dropped into the
#: polarization slot.
SLOT_CAPABILITY: dict[str, str] = {
    "materials": "materials",
    "mesh": "mesh",
    "polarization": "polarization",
    "quantum": "quantum",
    "mobility": "mobility",
    "traps": "traps",
    "thermal": "thermal",
    "transport": "transport",
    "nonlinear": "nonlinear",
    "linear": "linear",
    "backend": "backend",
}

#: Slots whose modules are *collaborators* rather than pipeline stages. They
#: transform no context and produce no artifacts; physics modules resolve them
#: through the context's service registry. Keeping them in the Deck means the
#: iteration strategy is swapped by editing one binding, exactly like a physics
#: model.
SERVICE_SLOTS: tuple[str, ...] = ("backend", "nonlinear", "linear")

LAYER_ROLES = (
    "barrier",
    "channel",
    "buffer",
    "spacer",
    "nucleation",
    "cap",
    "substrate",
    "passivation",
)


# ---------------------------------------------------------------------------
# Device description
# ---------------------------------------------------------------------------


class LayerSpec(Schema):
    """One epitaxial layer, described top-down in the stack."""

    name = Str(required=True, doc="unique layer name, e.g. 'barrier'")
    material = Str(required=True, doc="material key in the material database")
    role = Enum(LAYER_ROLES, default="channel", doc="functional role in the stack")
    thickness_nm = Float(
        required=True, unit="nm", minimum=0.0, exclusive_minimum=True, doc="layer thickness"
    )
    x_al = Float(
        0.0, unit="1", minimum=0.0, maximum=1.0, doc="AlN mole fraction for AlGaN alloys"
    )
    doping_cm3 = Float(0.0, unit="cm^-3", minimum=0.0, doc="net doping concentration")
    doping_kind = Enum(
        ("unintentional", "donor", "acceptor"),
        default="unintentional",
        doc="doping polarity",
    )

    def check(self) -> Iterable[Issue]:
        issues: list[Issue] = []
        if self.role == "barrier" and self.x_al <= 0.0:
            issues.append(
                Issue(
                    "x_al",
                    "a barrier layer must contain aluminium, but x_al is 0",
                    "physics",
                    hint="set x_al > 0 (e.g. 0.25 for Al0.25Ga0.75N) or change the role",
                )
            )
        return issues


class GateSpec(Schema):
    """Gate electrode description."""

    length_um = Float(
        required=True, unit="um", minimum=0.0, exclusive_minimum=True, doc="physical gate length"
    )
    kind = Enum(
        ("schottky", "mis", "pn", "ohmic"),
        default="schottky",
        doc="gate technology",
    )
    barrier_eV = Float(
        0.90,
        unit="eV",
        minimum=0.0,
        maximum=3.0,
        doc="metal/semiconductor barrier height",
    )
    workfunction_eV = Float(
        None, nullable=True, unit="eV", doc="metal work function, when known"
    )

    def check(self) -> Iterable[Issue]:
        if self.kind == "schottky" and self.barrier_eV <= 0.0:
            return [
                Issue(
                    "barrier_eV",
                    "a Schottky gate needs a positive barrier height",
                    "physics",
                )
            ]
        return ()


class GeometrySpec(Schema):
    """Lateral layout parameters of a single-finger device."""

    gate_width_um = Float(
        100.0, unit="um", minimum=0.0, exclusive_minimum=True, doc="gate (unit) width"
    )
    source_gate_spacing_um = Float(
        0.7, unit="um", minimum=0.0, doc="source-to-gate access region length"
    )
    gate_drain_spacing_um = Float(
        2.0, unit="um", minimum=0.0, doc="gate-to-drain access region length"
    )
    fingers = Int(1, minimum=1, maximum=1000, doc="number of gate fingers")


class DeviceSpec(Schema):
    """The device under test: layer stack plus gate and layout."""

    name = Str("device", doc="device identifier")
    substrate = Str("SiC-4H", doc="substrate material key")
    layers = ListOf(
        Nested(LayerSpec),
        required=True,
        doc="epitaxial stack ordered from the surface downwards",
    )
    gate = Nested(GateSpec, required=True)
    geometry = Nested(GeometrySpec, default={})

    def check(self) -> Iterable[Issue]:
        issues: list[Issue] = []
        if not self.layers:
            issues.append(Issue("layers", "at least one layer is required", "required"))
            return issues

        barriers = [i for i, layer in enumerate(self.layers) if layer.role == "barrier"]
        channels = [i for i, layer in enumerate(self.layers) if layer.role == "channel"]

        if not barriers:
            issues.append(
                Issue(
                    "layers",
                    "no layer has role 'barrier'",
                    "physics",
                    hint="an AlGaN/GaN HEMT needs exactly one barrier layer above the channel",
                )
            )
        elif len(barriers) > 1:
            issues.append(
                Issue(
                    "layers",
                    f"found {len(barriers)} barrier layers; exactly one is supported",
                    "physics",
                )
            )
        if not channels:
            issues.append(
                Issue(
                    "layers",
                    "no layer has role 'channel'",
                    "physics",
                    hint="mark the GaN channel layer with role 'channel'",
                )
            )
        if barriers and channels and barriers[0] > channels[0]:
            issues.append(
                Issue(
                    "layers",
                    "the barrier layer must sit above the channel layer",
                    "physics",
                    hint="layers are ordered from the surface downwards",
                )
            )

        names = [layer.name for layer in self.layers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            issues.append(
                Issue(
                    "layers",
                    f"duplicate layer name(s): {', '.join(duplicates)}",
                    "duplicate",
                )
            )
        return issues

    # -- convenience ----------------------------------------------------
    def layer_by_role(self, role: str) -> LayerSpec | None:
        for layer in self.layers:
            if layer.role == role:
                return layer
        return None

    def layer_by_name(self, name: str) -> LayerSpec | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None


# ---------------------------------------------------------------------------
# Module bindings
# ---------------------------------------------------------------------------


class ModuleBinding(Schema):
    """A reference from a physical role to a concrete plugin."""

    module = Str(
        required=True,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$",
        doc="module id, e.g. 'physics.polarization.ambacher'",
    )
    version = Str("*", doc="version requirement, e.g. '>=0.1,<1.0' or '^0.1.0'")
    params = DictOf(None, default={}, doc="module parameters, validated by the module manifest")
    enabled = Bool(True, doc="set false to skip this stage without deleting the binding")


class PhysicsSpec(Schema):
    """Physics-role bindings - the primary plug-and-play surface."""

    polarization = Nested(
        ModuleBinding, default={"module": "physics.polarization.ambacher"}
    )
    quantum = Nested(ModuleBinding, default={"module": "physics.quantum.analytic_tw"})
    mobility = Nested(ModuleBinding, None, nullable=True)
    traps = Nested(ModuleBinding, None, nullable=True)
    thermal = Nested(ModuleBinding, None, nullable=True)
    transport = Nested(ModuleBinding, None, nullable=True)


class ModulesSpec(Schema):
    """Numerical and infrastructure bindings, plus the extractor chain."""

    materials = Nested(
        ModuleBinding,
        None,
        nullable=True,
        doc=(
            "optional drop-in replacement for the seeded material database; "
            "when bound, it runs first and overwrites the 'materials' artifact "
            "downstream modules consume (e.g. materials.dft)"
        ),
    )
    mesh = Nested(ModuleBinding, default={"module": "mesh.structured2d"})
    nonlinear = Nested(ModuleBinding, None, nullable=True)
    linear = Nested(ModuleBinding, None, nullable=True)
    backend = Nested(ModuleBinding, None, nullable=True)
    extractors = ListOf(
        Nested(ModuleBinding), default=[], doc="post-processing modules, run last"
    )


# ---------------------------------------------------------------------------
# Simulation description
# ---------------------------------------------------------------------------


class SweepSpec(Schema):
    """What to sweep and over which range."""

    kind = Enum(
        ("iv", "cv", "sparam", "transient", "none"),
        default="iv",
        doc="sweep family",
    )
    vgs = ListOf(Float(), default=[-6.0, 2.0], doc="gate bias range [start, stop] in V")
    vds = ListOf(Float(), default=[0.0, 20.0], doc="drain bias range [start, stop] in V")
    vgs_points = Int(21, minimum=2, maximum=100000, doc="number of gate bias points")
    vds_points = Int(11, minimum=2, maximum=100000, doc="number of drain bias points")
    frequencies_hz = ListOf(Float(), default=[1.0e9, 1.0e10, 4.0e10], unit="Hz")

    def check(self) -> Iterable[Issue]:
        issues: list[Issue] = []
        for name in ("vgs", "vds"):
            values = getattr(self, name)
            if len(values) != 2:
                issues.append(
                    Issue(name, f"{name} must be exactly [start, stop], got {values}", "range")
                )
            elif values[0] > values[1]:
                issues.append(
                    Issue(
                        name,
                        f"{name} start ({values[0]}) is greater than stop ({values[1]})",
                        "range",
                        hint="swap the two bounds",
                    )
                )
        if self.kind == "sparam" and not self.frequencies_hz:
            issues.append(
                Issue("frequencies_hz", "an sparam sweep needs at least one frequency", "required")
            )
        if any(f <= 0.0 for f in self.frequencies_hz):
            issues.append(
                Issue("frequencies_hz", "frequencies must be strictly positive", "range")
            )
        return issues


class SimulationSpec(Schema):
    """Temperature, sweep and requested outputs."""

    temperature_K = Float(
        300.0, unit="K", minimum=0.0, exclusive_minimum=True, doc="lattice temperature"
    )
    sweep = Nested(SweepSpec, default={})
    output = ListOf(Str(), default=["ns"], doc="artifact keys the run must deliver")


class RunSpec(Schema):
    """Free-form bookkeeping for the result library."""

    notes = Str("", doc="human notes recorded with the run")
    tags = ListOf(Str(), default=[], doc="tags for searching the result library")


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


class Deck(Schema):
    """The complete description of a simulation campaign."""

    __title__ = "NemoTCAD Deck"

    schema_version = Str(
        SCHEMA_VERSION,
        pattern=r"^\d+\.\d+$",
        doc="Deck document format version",
    )
    name = Str(required=True, doc="campaign name")
    device = Nested(DeviceSpec, required=True)
    physics = Nested(PhysicsSpec, default={})
    modules = Nested(ModulesSpec, default={})
    simulation = Nested(SimulationSpec, default={})
    run = Nested(RunSpec, default={})

    # -- bindings --------------------------------------------------------
    def physics_bindings(self) -> list[tuple[str, ModuleBinding]]:
        out: list[tuple[str, ModuleBinding]] = []
        for slot in SLOT_ORDER:
            binding = getattr(self.physics, slot, None) or getattr(self.modules, slot, None)
            if binding is not None:
                out.append((slot, binding))
        return out

    def extractor_bindings(self) -> list[tuple[str, ModuleBinding]]:
        return [
            (f"extract:{index}", binding)
            for index, binding in enumerate(self.modules.extractors)
        ]

    def bindings(self) -> list[tuple[str, ModuleBinding]]:
        """All (slot, binding) pairs in canonical execution order."""
        return self.physics_bindings() + self.extractor_bindings()

    def stage_bindings(self) -> list[tuple[str, ModuleBinding]]:
        """Bindings that become pipeline stages (data transforms)."""
        return [
            (slot, binding)
            for slot, binding in self.physics_bindings()
            if slot not in SERVICE_SLOTS
        ]

    def service_bindings(self) -> list[tuple[str, ModuleBinding]]:
        """Bindings that become collaborator services, not stages."""
        return [
            (slot, binding)
            for slot, binding in self.physics_bindings()
            if slot in SERVICE_SLOTS
        ]

    def enabled_bindings(self) -> list[tuple[str, ModuleBinding]]:
        return [(slot, b) for slot, b in self.bindings() if b.enabled]

    def requirements(self) -> list[str]:
        """Module requirement strings, ready for :meth:`ModuleRegistry.resolve`."""
        out: list[str] = []
        for _slot, binding in self.enabled_bindings():
            version = binding.version or "*"
            out.append(binding.module if version == "*" else f"{binding.module}{version}")
        return out

    # -- contextual validation -------------------------------------------
    def validate_against_registry(
        self, registry: ModuleRegistry
    ) -> tuple[list[Issue], list[str]]:
        """Check every binding against the registry.

        Returns ``(errors, warnings)``. Errors block a run; warnings do not.
        """
        errors: list[Issue] = []
        warnings: list[str] = []

        for slot, binding in self.enabled_bindings():
            path = f"binding[{slot}]"
            try:
                record = registry.select(binding.module, binding.version or "*")
            except Exception as exc:  # noqa: BLE001 - surfaced as an actionable issue
                errors.append(Issue(path, str(exc), "module-unavailable"))
                continue

            manifest = record.manifest
            _values, param_issues = manifest.validate_params(binding.params)
            for issue in param_issues:
                errors.append(
                    Issue(
                        f"{path}.{issue.path}",
                        issue.message,
                        issue.code,
                        hint=issue.hint,
                    )
                )

            expected = SLOT_CAPABILITY.get(slot)
            if expected and expected not in manifest.capabilities:
                warnings.append(
                    f"{path}: {manifest.module_id} does not advertise capability "
                    f"{expected!r} (declares: {', '.join(manifest.capabilities) or 'none'})"
                )
            if manifest.verification == "unverified":
                warnings.append(
                    f"{path}: {manifest.module_id} is marked 'unverified'"
                )

        # Artifact reachability: every required artifact must be produced.
        produced: set[str] = set()
        required: dict[str, str] = {}
        for slot, binding in self.enabled_bindings():
            try:
                manifest = registry.select(binding.module, binding.version or "*").manifest
            except Exception:  # noqa: BLE001 - already reported above
                continue
            produced.update(manifest.produces)
            for key in manifest.requires:
                required.setdefault(key, slot)
        builtin_inputs = {"structure", "materials", "simulation", "device", "deck", "run"}
        for key, slot in required.items():
            if key not in produced and key not in builtin_inputs:
                errors.append(
                    Issue(
                        f"binding[{slot}]",
                        f"requires artifact {key!r}, but no bound module produces it",
                        "artifact-unreachable",
                        hint="add a module that produces it, or disable this binding",
                    )
                )

        for key in self.simulation.output:
            if key not in produced and key not in builtin_inputs:
                warnings.append(
                    f"simulation.output declares {key!r}, which no bound module produces"
                )
        return errors, warnings

    def validate_against_materials(self, materials: MaterialDB) -> list[Issue]:
        """Check that every referenced material or alloy recipe exists."""
        errors: list[Issue] = []
        names = [("substrate", self.device.substrate)]
        names += [
            (f"layers[{index}].material", layer.material)
            for index, layer in enumerate(self.device.layers)
        ]
        available = ", ".join(materials.names() + sorted(materials.alloys))
        for path, name in names:
            if materials.has(name) or name in materials.alloys:
                continue
            errors.append(
                Issue(
                    path,
                    f"unknown material {name!r}",
                    "material-unavailable",
                    hint=f"available: {available}",
                )
            )
        return errors

    def validate_all(
        self, registry: ModuleRegistry, materials: MaterialDB | None = None
    ) -> tuple[list[Issue], list[str]]:
        """Run every contextual check in one call."""
        errors, warnings = self.validate_against_registry(registry)
        if materials is not None:
            errors.extend(self.validate_against_materials(materials))
        return errors, warnings

    def describe(self) -> str:
        lines = [f"Deck {self.name!r} (schema {self.schema_version})"]
        lines.append(
            f"  device: {self.device.name} on {self.device.substrate}, "
            f"{len(self.device.layers)} layer(s), Lg={self.device.gate.length_um} um"
        )
        for slot, binding in self.bindings():
            state = "" if binding.enabled else "  [disabled]"
            lines.append(f"  {slot:<14} -> {binding.module}{binding.version} {state}")
        lines.append(
            f"  sweep: {self.simulation.sweep.kind} "
            f"Vgs={self.simulation.sweep.vgs} Vds={self.simulation.sweep.vds} "
            f"@ {self.simulation.temperature_K} K"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing and serialisation
# ---------------------------------------------------------------------------

_YAML_HINT = (
    "YAML support requires PyYAML, which is not installed in this environment. "
    "Use JSON instead, or run: pip install PyYAML"
)


def yaml_available() -> bool:
    """Whether optional YAML support is importable."""
    try:
        import yaml  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _detect_format(path: Path, fmt: str | None) -> str:
    if fmt:
        return fmt.lower()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    raise DeckError(
        f"cannot infer the format of {path}; pass fmt='json' or fmt='yaml' explicitly"
    )


def loads_deck(text: str, *, fmt: str = "json", source: str = "<string>") -> Deck:
    """Parse a Deck from a string."""
    fmt = fmt.lower()
    if fmt in ("yaml", "yml"):
        try:
            import yaml
        except Exception as exc:  # noqa: BLE001
            raise DeckError(_YAML_HINT) from exc
        try:
            raw = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            raise DeckError(f"{source}: invalid YAML: {exc}") from exc
    elif fmt == "json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DeckError(
                f"{source}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
            ) from exc
    else:
        raise DeckError(f"unsupported deck format {fmt!r}; expected 'json' or 'yaml'")

    if not isinstance(raw, Mapping):
        raise DeckError(f"{source}: a Deck must be a mapping at the top level")

    instance, issues = Deck.validate_dict(raw)
    if instance is None:
        from .errors import SchemaError, format_issues

        raise DeckError(
            f"{source}: deck validation failed:\n{format_issues(issues)}"
        ) from SchemaError(issues, where=source)
    return instance  # type: ignore[return-value]


def load_deck(path: str | Path) -> Deck:
    """Load a Deck from a ``.json`` / ``.yaml`` file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DeckError(f"deck file not found: {file_path}")
    fmt = _detect_format(file_path, None)
    return loads_deck(file_path.read_text(encoding="utf-8"), fmt=fmt, source=str(file_path))


def dump_deck(deck: Deck, path: str | Path, *, fmt: str | None = None) -> Path:
    """Write a Deck to disk."""
    target = Path(path)
    fmt = _detect_format(target, fmt)
    payload = deck.to_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("yaml", "yml"):
        if not yaml_available():
            raise DeckError(_YAML_HINT)
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    target.write_text(text, encoding="utf-8")
    return target


def deck_json_schema() -> dict[str, Any]:
    """JSON Schema of the Deck document, for LLM-constrained generation."""
    return Deck.json_schema()
