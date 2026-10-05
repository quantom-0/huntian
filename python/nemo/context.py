"""Simulation context: the shared data bus of a single run.

A :class:`SimContext` is the *only* thing modules exchange. Modules never
import each other; they read named artifacts from the context and write named
artifacts back. That indirection is what makes modules independently
replaceable.

Beyond data flow the context also carries:

* **provenance** - which module produced what, when, and how long it took;
* **a reproducibility fingerprint** - a stable hash over resolved parameters,
  module versions and the material database, used by golden-data regression;
* **run metadata snapshots** - serialisable records for the result library.

Artifact *values* are deliberately not serialised into snapshots: they are
frequently large arrays. Only metadata, parameters and provenance are.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .errors import ArtifactMissingError, ContextError
from .events import EventBus

__all__ = ["Artifact", "ProvenanceRecord", "StageScope", "SimContext"]

LOGGER_NAME = "nemo"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary values to JSON-safe structures."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:  # noqa: BLE001 - fingerprinting must never fail a run
            return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical(payload: Any) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Artifact:
    """A named value in the context, with producer provenance."""

    key: str
    value: Any
    producer: str
    module_version: str | None = None
    created_at: float = 0.0
    tags: frozenset[str] = frozenset()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Serialisable description (value excluded on purpose)."""
        return {
            "key": self.key,
            "producer": self.producer,
            "module_version": self.module_version,
            "created_at": self.created_at,
            "tags": sorted(self.tags),
            "value_type": type(self.value).__name__,
            "meta": _jsonable(self.meta),
        }


@dataclass(frozen=True)
class ProvenanceRecord:
    """One executed stage of a run."""

    stage: str
    module_id: str
    module_version: str
    status: str
    started_at: float
    ended_at: float
    duration_s: float
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "module_id": self.module_id,
            "module_version": self.module_version,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 6),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "notes": list(self.notes),
            "error": self.error,
        }


class StageScope:
    """Handle yielded by :meth:`SimContext.stage`."""

    def __init__(
        self,
        stage: str,
        module_id: str,
        module_version: str,
        inputs: tuple[str, ...],
        outputs: list[str],
    ) -> None:
        self.stage = stage
        self.module_id = module_id
        self.module_version = module_version
        self.inputs = inputs
        self._outputs = outputs
        self._notes: list[str] = []

    def note(self, text: str) -> None:
        """Attach a free-form note to the provenance record."""
        self._notes.append(str(text))

    def declare_output(self, key: str) -> None:
        """Record an artifact key produced by this stage."""
        if key not in self._outputs:
            self._outputs.append(key)

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(self._outputs)


class SimContext:
    """The shared state of one simulation run."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        name: str = "run",
        bus: EventBus | None = None,
        materials: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self.name = name
        self.created_at: float = time.time()
        self.bus: EventBus = bus if bus is not None else EventBus()
        self.materials = materials
        self.log: logging.Logger = logger or logging.getLogger(LOGGER_NAME)
        #: Collaborator modules (nonlinear/linear solvers, backends) resolved on
        #: demand. Populated by :class:`~nemo.pipeline.Executor`; a context built
        #: by hand has none, which is what makes the numerics testable in isolation.
        self.services: Any = None

        self._artifacts: dict[str, Artifact] = {}
        self._provenance: list[ProvenanceRecord] = []
        self._params: dict[str, Any] = {}
        self._module_versions: dict[str, str] = {}
        self._marked: dict[str, Any] = {}
        self._stage_stack: list[str] = []

    # -- artifacts ------------------------------------------------------
    def put(
        self,
        key: str,
        value: Any,
        *,
        producer: str = "unknown",
        module_version: str | None = None,
        tags: frozenset[str] | set[str] | tuple[str, ...] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Store an artifact, replacing any previous value with the same key."""
        if not isinstance(key, str) or not key:
            raise ContextError("artifact key must be a non-empty string")
        artifact = Artifact(
            key=key,
            value=value,
            producer=producer,
            module_version=module_version,
            created_at=time.time(),
            tags=frozenset(tags),
            meta=dict(meta or {}),
        )
        self._artifacts[key] = artifact
        self.bus.publish(
            "context.artifact.put", run_id=self.run_id, key=key, producer=producer
        )
        return artifact

    def get(self, key: str, default: Any = None) -> Any:
        artifact = self._artifacts.get(key)
        return default if artifact is None else artifact.value

    def require(self, key: str) -> Any:
        """Return an artifact value, raising a helpful error when absent."""
        artifact = self._artifacts.get(key)
        if artifact is None:
            raise ArtifactMissingError(
                f"context has no artifact {key!r}; available: {self.keys() or '(none)'}"
            )
        return artifact.value

    def artifact(self, key: str) -> Artifact:
        artifact = self._artifacts.get(key)
        if artifact is None:
            raise ArtifactMissingError(f"context has no artifact {key!r}")
        return artifact

    def has(self, key: str) -> bool:
        return key in self._artifacts

    def keys(self) -> list[str]:
        return sorted(self._artifacts)

    def artifacts_with_tag(self, tag: str) -> list[Artifact]:
        return [a for a in self._artifacts.values() if tag in a.tags]

    def remove(self, key: str) -> bool:
        return self._artifacts.pop(key, None) is not None

    # -- parameters -----------------------------------------------------
    def set_params(self, slot: str, params: Mapping[str, Any]) -> None:
        """Record the resolved parameters for one pipeline slot."""
        self._params[slot] = dict(params)

    @property
    def params(self) -> Mapping[str, Any]:
        return dict(self._params)

    def record_module_version(self, module_id: str, version: str) -> None:
        self._module_versions[module_id] = version

    @property
    def module_versions(self) -> Mapping[str, str]:
        return dict(self._module_versions)

    def mark_fingerprint_input(self, key: str, value: Any) -> None:
        """Include an extra value in the reproducibility fingerprint."""
        self._marked[key] = value

    # -- provenance -----------------------------------------------------
    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(self._provenance)

    @contextmanager
    def stage(
        self,
        stage: str,
        module_id: str,
        module_version: str = "0.0.0",
        *,
        inputs: tuple[str, ...] | list[str] = (),
        outputs: tuple[str, ...] | list[str] = (),
    ) -> Iterator[StageScope]:
        """Time-box a stage, recording provenance and emitting bus events."""
        scope = StageScope(
            stage=stage,
            module_id=module_id,
            module_version=module_version,
            inputs=tuple(inputs),
            outputs=list(outputs),
        )
        self._stage_stack.append(stage)
        started = time.time()
        self.bus.publish(
            "run.stage.start",
            run_id=self.run_id,
            stage=stage,
            module_id=module_id,
            module_version=module_version,
        )
        try:
            yield scope
        except BaseException as exc:
            ended = time.time()
            self._provenance.append(
                ProvenanceRecord(
                    stage=stage,
                    module_id=module_id,
                    module_version=module_version,
                    status="failed",
                    started_at=started,
                    ended_at=ended,
                    duration_s=ended - started,
                    inputs=scope.inputs,
                    outputs=scope.outputs,
                    notes=scope.notes,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            self.bus.publish(
                "run.stage.failed",
                run_id=self.run_id,
                stage=stage,
                module_id=module_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            ended = time.time()
            self._provenance.append(
                ProvenanceRecord(
                    stage=stage,
                    module_id=module_id,
                    module_version=module_version,
                    status="ok",
                    started_at=started,
                    ended_at=ended,
                    duration_s=ended - started,
                    inputs=scope.inputs,
                    outputs=scope.outputs,
                    notes=scope.notes,
                )
            )
            self.bus.publish(
                "run.stage.end",
                run_id=self.run_id,
                stage=stage,
                module_id=module_id,
                duration_s=round(ended - started, 6),
            )
        finally:
            self._stage_stack.pop()

    @property
    def current_stage(self) -> str | None:
        return self._stage_stack[-1] if self._stage_stack else None

    # -- services ----------------------------------------------------------
    def require_services(self) -> Any:
        """Return the service registry, or explain why there is none."""
        if self.services is None:
            raise ContextError(
                "this context has no service registry; build runs through "
                "nemo.pipeline.Executor (or assign ctx.services) so that library "
                "modules can be resolved"
            )
        return self.services

    # -- reproducibility ------------------------------------------------
    def fingerprint(self) -> str:
        """Stable hash identifying *what was computed* (not when)."""
        payload = {
            "params": self._params,
            "module_versions": self._module_versions,
            "materials": self.materials.fingerprint() if self.materials is not None else None,
            "marked": self._marked,
        }
        blob = _canonical(payload)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def snapshot(self) -> dict[str, Any]:
        """Serialisable run metadata for the result library."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint(),
            "nemo_version": _nemo_version(),
            "materials": (
                None
                if self.materials is None
                else {
                    "version": getattr(self.materials, "version", None),
                    "fingerprint": self.materials.fingerprint(),
                    "verification": getattr(self.materials, "verification", None),
                }
            ),
            "params": _jsonable(self._params),
            "module_versions": dict(self._module_versions),
            "artifacts": [a.metadata() for a in self._artifacts.values()],
            "provenance": [p.to_dict() for p in self._provenance],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        """Small human-facing summary of the run so far."""
        ok = sum(1 for p in self._provenance if p.status == "ok")
        failed = sum(1 for p in self._provenance if p.status == "failed")
        total = sum(p.duration_s for p in self._provenance)
        return {
            "artifacts": len(self._artifacts),
            "stages_ok": ok,
            "stages_failed": failed,
            "duration_s": round(total, 6),
        }

    def save_metadata(self, path: str | Path) -> Path:
        """Write a metadata snapshot (never artifact values) to JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=False, default=str), encoding="utf-8"
        )
        return target

    # -- dunder ---------------------------------------------------------
    def __contains__(self, key: object) -> bool:
        return key in self._artifacts

    @property
    def artifact_count(self) -> int:
        """Number of stored artifacts.

        Deliberately *not* ``__len__``: a context must stay truthy even when it
        is empty, otherwise ``ctx or SimContext(...)`` silently discards a
        caller-supplied context.
        """
        return len(self._artifacts)

    def __repr__(self) -> str:
        return (
            f"SimContext(run_id={self.run_id!r}, artifacts={len(self._artifacts)}, "
            f"stages={len(self._provenance)})"
        )


def _nemo_version() -> str:
    from .version import __version__

    return __version__
