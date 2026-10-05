"""Pipeline assembly and execution.

A :class:`Pipeline` is an ordered list of :class:`Stage`, each naming a module
and its parameters. The :class:`Executor` turns a pipeline plus a Deck into a
run: it resolves modules through the registry, seeds the context with the
Deck-derived inputs, then executes stages one by one, recording provenance and
publishing events.

Stage ordering is derived from the manifests themselves: if module A
*produces* an artifact that module B *requires*, A runs before B. The
canonical slot order is only a tie-breaker. That is what lets a module be
swapped for one with different data dependencies without editing the pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .context import SimContext
from .deck import SLOT_ORDER, Deck
from .errors import (
    DependencyError,
    Issue,
    ModuleUnavailableError,
    NemoError,
    PipelineError,
    format_issues,
)
from .events import EventBus
from .materials import MaterialDB
from .registry import ModuleRecord, ModuleRegistry
from .services import ServiceRegistry

__all__ = ["Stage", "Pipeline", "StageResult", "RunResult", "Executor"]

LOGGER_NAME = "nemo"


@dataclass(frozen=True)
class Stage:
    """One executable step of a pipeline."""

    slot: str
    module_id: str
    version: str = "*"
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    optional: bool = False

    def requirement(self) -> str:
        return self.module_id if self.version == "*" else f"{self.module_id}{self.version}"

    def __str__(self) -> str:
        suffix = "" if self.enabled else " [disabled]"
        return f"{self.slot}: {self.module_id}{self.version}{suffix}"


class Pipeline:
    """An ordered collection of stages plus its dependency ordering logic."""

    def __init__(
        self,
        name: str,
        stages: Sequence[Stage],
        services: Sequence[Stage] | None = None,
    ) -> None:
        self.name = name
        self.stages: list[Stage] = list(stages)
        #: Library modules (solvers, backends) rather than data stages. They are
        #: configured from the Deck but never ordered or executed as a stage.
        self.services: list[Stage] = list(services or ())

    # -- construction ---------------------------------------------------
    @classmethod
    def from_deck(cls, deck: Deck, *, optional_extractors: bool = True) -> "Pipeline":
        """Build the canonical pipeline implied by a Deck."""
        stages: list[Stage] = []
        for slot, binding in deck.stage_bindings():
            stages.append(
                Stage(
                    slot=slot,
                    module_id=binding.module,
                    version=binding.version or "*",
                    params=dict(binding.params),
                    enabled=binding.enabled,
                )
            )
        for slot, binding in deck.extractor_bindings():
            stages.append(
                Stage(
                    slot=slot,
                    module_id=binding.module,
                    version=binding.version or "*",
                    params=dict(binding.params),
                    enabled=binding.enabled,
                    optional=optional_extractors,
                )
            )
        services = [
            Stage(
                slot=slot,
                module_id=binding.module,
                version=binding.version or "*",
                params=dict(binding.params),
                enabled=binding.enabled,
            )
            for slot, binding in deck.service_bindings()
        ]
        return cls(name=deck.name, stages=stages, services=services)

    def add(self, stage: Stage) -> "Pipeline":
        self.stages.append(stage)
        return self

    # -- introspection ---------------------------------------------------
    def enabled_stages(self) -> list[Stage]:
        return [stage for stage in self.stages if stage.enabled]

    def requirements(self, *, include_optional: bool = True) -> list[str]:
        out = [
            stage.requirement()
            for stage in self.enabled_stages()
            if include_optional or not stage.optional
        ]
        out.extend(service.requirement() for service in self.services if service.enabled)
        return out

    def describe(self) -> str:
        lines = [f"Pipeline {self.name!r} ({len(self.enabled_stages())} stage(s))"]
        for stage in self.stages:
            lines.append(f"  {stage}")
        if self.services:
            lines.append("  services:")
            for service in self.services:
                lines.append(f"    {service}")
        return "\n".join(lines)

    # -- ordering --------------------------------------------------------
    def ordered(self, registry: ModuleRegistry) -> list[Stage]:
        """Order stages so that producers run before consumers."""
        stages = self.enabled_stages()
        if not stages:
            return []

        requires: dict[str, list[str]] = {}
        produces: dict[str, list[str]] = {}
        for stage in stages:
            try:
                manifest = registry.select(stage.module_id, stage.version).manifest
            except ModuleUnavailableError:
                requires[stage.slot] = []
                produces[stage.slot] = []
                continue
            requires[stage.slot] = list(manifest.requires)
            produces[stage.slot] = list(manifest.produces)

        producer_of: dict[str, str] = {}
        for slot, keys in produces.items():
            for key in keys:
                producer_of.setdefault(key, slot)

        incoming: dict[str, set[str]] = {stage.slot: set() for stage in stages}
        outgoing: dict[str, set[str]] = {stage.slot: set() for stage in stages}
        for slot, keys in requires.items():
            for key in keys:
                source = producer_of.get(key)
                if source is not None and source != slot:
                    incoming[slot].add(source)
                    outgoing[source].add(slot)

        def sort_key(slot: str) -> tuple[int, str]:
            base = slot.split(":", 1)[0]
            index = SLOT_ORDER.index(base) if base in SLOT_ORDER else len(SLOT_ORDER)
            return (index, slot)

        by_slot = {stage.slot: stage for stage in stages}
        ready = sorted((s for s, deps in incoming.items() if not deps), key=sort_key)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for downstream in sorted(outgoing[node], key=sort_key):
                incoming[downstream].discard(node)
                if not incoming[downstream]:
                    ready.append(downstream)
                    ready.sort(key=sort_key)
        if len(order) != len(stages):
            cyclic = sorted(s for s, deps in incoming.items() if deps)
            raise DependencyError(
                "pipeline has an unresolvable artifact dependency cycle among: "
                + ", ".join(cyclic)
            )
        return [by_slot[slot] for slot in order]


@dataclass
class StageResult:
    """Outcome of one executed stage."""

    slot: str
    module_id: str
    module_version: str
    status: str  # ok | failed | skipped
    duration_s: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "module_id": self.module_id,
            "module_version": self.module_version,
            "status": self.status,
            "duration_s": round(self.duration_s, 6),
            "error": self.error,
            "warnings": list(self.warnings),
        }


@dataclass
class RunResult:
    """Outcome of a whole run."""

    run_id: str
    status: str  # success | partial | failed
    fingerprint: str
    duration_s: float
    stages: list[StageResult] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def stage(self, slot: str) -> StageResult | None:
        for result in self.stages:
            if result.slot == slot:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "fingerprint": self.fingerprint,
            "duration_s": round(self.duration_s, 6),
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": list(self.artifacts),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class Executor:
    """Runs pipelines against the registry and a simulation context.

    The executor also owns service wiring: it creates the context's
    :class:`~nemo.services.ServiceRegistry` and configures every library module
    the Deck binds (``nonlinear``, ``linear``, ``backend``). Service instances
    are cached on the context and are *not* torn down automatically - call
    ``ctx.services.release_all()`` when the context is done.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        *,
        bus: EventBus | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry
        self._bus = bus if bus is not None else registry.bus
        self._log = logger or logging.getLogger(LOGGER_NAME)

    # -- context seeding --------------------------------------------------
    def seed(
        self,
        ctx: SimContext,
        deck: Deck,
        *,
        materials: MaterialDB | None = None,
    ) -> SimContext:
        """Populate a fresh context with the Deck-derived inputs."""
        producer = "executor.seed"
        db = materials if materials is not None else ctx.materials
        if db is None:
            db = MaterialDB.builtin()
        ctx.materials = db

        ctx.put("deck", deck.to_dict(), producer=producer, tags={"input"})
        ctx.put("device", deck.device.to_dict(), producer=producer, tags={"input"})
        ctx.put("structure", deck.device.to_dict(), producer=producer, tags={"input"})
        ctx.put("gate", deck.device.gate.to_dict(), producer=producer, tags={"input"})
        ctx.put("geometry", deck.device.geometry.to_dict(), producer=producer, tags={"input"})
        ctx.put("simulation", deck.simulation.to_dict(), producer=producer, tags={"input"})
        ctx.put("materials", db, producer=producer, tags={"input"})
        ctx.put("temperature_K", deck.simulation.temperature_K, producer=producer, tags={"input"})
        ctx.mark_fingerprint_input("deck.device", deck.device.to_dict())
        ctx.mark_fingerprint_input("deck.simulation", deck.simulation.to_dict())
        return ctx

    # -- execution ---------------------------------------------------------
    def run(
        self,
        pipeline: Pipeline,
        *,
        ctx: SimContext | None = None,
        deck: Deck | None = None,
        materials: MaterialDB | None = None,
        strict: bool = True,
    ) -> RunResult:
        """Execute a pipeline, returning a :class:`RunResult`.

        ``strict=True`` aborts on the first failure of a non-optional stage.
        ``strict=False`` records the failure and keeps going.
        """
        started = time.time()
        ctx = ctx if ctx is not None else SimContext(name=pipeline.name, bus=self._bus)
        if ctx.services is None:
            ctx.services = ServiceRegistry(self.registry, logger=self._log)
        if deck is not None:
            self.seed(ctx, deck, materials=materials)
            for service in pipeline.services:
                if not service.enabled:
                    continue
                ctx.services.configure(
                    service.module_id,
                    version=service.version,
                    params=service.params,
                )
        elif materials is not None and ctx.materials is None:
            ctx.materials = materials

        warnings: list[str] = []
        errors: list[str] = []
        ordered = pipeline.ordered(self.registry)

        # Resolve the mandatory part of the pipeline up front so that version
        # conflicts and cycles surface before any work starts. Optional stages
        # are probed individually and may simply be missing.
        try:
            plan = self.registry.resolve(pipeline.requirements(include_optional=False))
        except NemoError as exc:
            warnings.append(f"module resolution incomplete: {exc}")
            plan = None
        if plan is not None:
            warnings.extend(plan.warnings)
            for record in plan:
                ctx.record_module_version(record.module_id, record.manifest.module_version)
        for stage in ordered:
            if not stage.optional:
                continue
            try:
                record = self.registry.select(stage.module_id, stage.version)
                ctx.record_module_version(record.module_id, record.manifest.module_version)
            except NemoError as exc:
                warnings.append(f"optional stage {stage.slot} is unavailable: {exc}")

        self._bus.publish(
            "run.start", run_id=ctx.run_id, pipeline=pipeline.name, stages=len(ordered)
        )

        results: list[StageResult] = []
        aborted = False

        for stage in ordered:
            try:
                record = self.registry.select(stage.module_id, stage.version)
            except ModuleUnavailableError as exc:
                results.append(
                    StageResult(
                        slot=stage.slot,
                        module_id=stage.module_id,
                        module_version="?",
                        status="failed",
                        error=str(exc),
                    )
                )
                errors.append(f"{stage.slot}: {exc}")
                if strict and not stage.optional:
                    aborted = True
                    break
                continue

            result = self._run_stage(stage, record, ctx)
            results.append(result)
            warnings.extend(f"{stage.slot}: {w}" for w in result.warnings)
            if result.status == "failed":
                errors.append(f"{stage.slot}: {result.error}")
                if strict and not stage.optional:
                    aborted = True
                    break

        if aborted:
            status = "failed"
        elif any(r.status == "failed" for r in results):
            status = "partial"
        else:
            status = "success"

        duration = time.time() - started
        run_result = RunResult(
            run_id=ctx.run_id,
            status=status,
            fingerprint=ctx.fingerprint(),
            duration_s=duration,
            stages=results,
            artifacts=ctx.keys(),
            errors=errors,
            warnings=warnings,
        )
        self._bus.publish(
            "run.end",
            run_id=ctx.run_id,
            status=status,
            duration_s=round(duration, 6),
            fingerprint=run_result.fingerprint,
        )
        return run_result

    def _run_stage(
        self,
        stage: Stage,
        record: ModuleRecord,
        ctx: SimContext,
    ) -> StageResult:
        manifest = record.manifest
        warnings: list[str] = []
        started = time.time()
        instance = None

        ctx.set_params(stage.slot, dict(stage.params))
        self._bus.publish(
            "run.stage.dispatch",
            run_id=ctx.run_id,
            slot=stage.slot,
            module_id=manifest.module_id,
            module_version=manifest.module_version,
        )

        try:
            instance = record.module_cls()
            instance.initialize(stage.params)

            missing = [key for key in manifest.requires if not ctx.has(key)]
            if missing:
                raise PipelineError(
                    f"{manifest.module_id} requires artifact(s) {missing}, which are not present"
                )

            pre_issues: list[Issue] = list(instance.validate(ctx) or [])
            if pre_issues:
                raise PipelineError(
                    f"{manifest.module_id} rejected the current context:\n"
                    f"{format_issues(pre_issues)}"
                )

            with ctx.stage(
                stage.slot,
                manifest.module_id,
                manifest.module_version,
                inputs=tuple(manifest.requires),
                outputs=tuple(manifest.produces),
            ) as scope:
                instance.run(ctx)
                for key in manifest.produces:
                    if ctx.has(key):
                        scope.declare_output(key)

            for key in manifest.produces:
                if not ctx.has(key):
                    warnings.append(
                        f"{manifest.module_id} declares it produces {key!r} but left it absent"
                    )
        except Exception as exc:  # noqa: BLE001 - reported through StageResult
            self._log.debug("stage %s failed: %s", stage.slot, exc)
            return StageResult(
                slot=stage.slot,
                module_id=manifest.module_id,
                module_version=manifest.module_version,
                status="failed",
                duration_s=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
                warnings=warnings,
            )
        finally:
            if instance is not None:
                self.registry.release(instance)

        return StageResult(
            slot=stage.slot,
            module_id=manifest.module_id,
            module_version=manifest.module_version,
            status="ok",
            duration_s=time.time() - started,
            warnings=warnings,
        )
