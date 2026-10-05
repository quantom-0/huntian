"""NemoTCAD - a lightweight, modular TCAD core for GaN RF HEMTs.

The package is organised around three ideas:

* a **schema** layer that validates documents and emits JSON Schema, which is
  what keeps LLM-generated input mechanically checkable;
* a **registry** of pluggable modules, each describing itself with a manifest
  and able to be swapped at runtime;
* a **context** that carries artifacts between modules, recording provenance
  and a reproducibility fingerprint.

Typical use::

    from nemo import ModuleRegistry, Executor, Pipeline, load_deck

    registry = ModuleRegistry()
    registry.load_from_paths(["plugins"])
    deck = load_deck("decks/demo_hemt.json")
    result = Executor(registry).run(Pipeline.from_deck(deck), deck=deck)
"""

from __future__ import annotations

from .context import Artifact, ProvenanceRecord, SimContext, StageScope
from .deck import (
    Deck,
    DeviceSpec,
    GateSpec,
    GeometrySpec,
    LayerSpec,
    ModuleBinding,
    ModulesSpec,
    PhysicsSpec,
    RunSpec,
    SimulationSpec,
    SweepSpec,
    deck_json_schema,
    dump_deck,
    load_deck,
    loads_deck,
    yaml_available,
)
from .errors import (
    ArtifactMissingError,
    ContextError,
    DeckError,
    DependencyError,
    DuplicateModuleError,
    Issue,
    ModuleConfigError,
    ModuleLifecycleError,
    ModuleUnavailableError,
    NemoError,
    PipelineError,
    RegistryError,
    RequirementError,
    SchemaError,
    ServiceUnavailableError,
    VersionConflictError,
    VersionError,
    format_issues,
)
from .events import Event, EventBus
from .materials import Material, MaterialDB, MaterialLookupError
from .pipeline import Executor, Pipeline, RunResult, Stage, StageResult
from .registry import (
    BaseModule,
    DiscoveryReport,
    IModule,
    ModuleCategory,
    ModuleManifest,
    ModuleRecord,
    ModuleRegistry,
    ModuleState,
    ParamSpec,
    ResolutionPlan,
    SwapReport,
)
from .schema import Field, Schema
from .semver import Requirement, Version, VersionRange, parse_requirement, parse_version
from .services import ServiceRegistry
from .version import API_VERSION, SCHEMA_VERSION, __version__

__all__ = [
    "__version__",
    "API_VERSION",
    "SCHEMA_VERSION",
    # schema
    "Schema",
    "Field",
    # versions
    "Version",
    "VersionRange",
    "Requirement",
    "parse_version",
    "parse_requirement",
    # errors
    "NemoError",
    "Issue",
    "format_issues",
    "SchemaError",
    "DeckError",
    "VersionError",
    "RequirementError",
    "RegistryError",
    "ModuleUnavailableError",
    "VersionConflictError",
    "DependencyError",
    "DuplicateModuleError",
    "ModuleConfigError",
    "ServiceUnavailableError",
    "ModuleLifecycleError",
    "ContextError",
    "ArtifactMissingError",
    "PipelineError",
    # events
    "Event",
    "EventBus",
    # context
    "SimContext",
    "Artifact",
    "ProvenanceRecord",
    "StageScope",
    # materials
    "Material",
    "MaterialDB",
    "MaterialLookupError",
    # registry
    "IModule",
    "BaseModule",
    "ModuleState",
    "ModuleCategory",
    "ModuleManifest",
    "ParamSpec",
    "ModuleRecord",
    "ModuleRegistry",
    "ResolutionPlan",
    "SwapReport",
    "DiscoveryReport",
    # deck
    "Deck",
    "DeviceSpec",
    "LayerSpec",
    "GateSpec",
    "GeometrySpec",
    "ModuleBinding",
    "PhysicsSpec",
    "ModulesSpec",
    "SweepSpec",
    "SimulationSpec",
    "RunSpec",
    "load_deck",
    "loads_deck",
    "dump_deck",
    "deck_json_schema",
    "yaml_available",
    # pipeline
    "Pipeline",
    "Stage",
    "Executor",
    "RunResult",
    "StageResult",
    # services
    "ServiceRegistry",
]
