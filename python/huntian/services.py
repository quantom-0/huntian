"""On-demand resolution of collaborator modules for the stages of a run.

Most modules are *data stages*: they read artifacts, write artifacts, and are
ordered by the pipeline. Some are *library modules* - a nonlinear solver, a
linear solver, a compute backend - which transform nothing and produce nothing,
but are needed **inside** another module's work.

Those are resolved through a :class:`ServiceRegistry` hanging off the
simulation context. The Deck still configures them, in its dedicated slots::

    "modules": { "nonlinear": { "module": "nonlinear.newton", "params": {...} } }

so swapping the iteration strategy is the same one-line change as swapping a
physics model. Instances are created lazily on first use and cached per
configuration.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from .errors import ModuleUnavailableError, ServiceUnavailableError
from .registry import IModule, ModuleRegistry

__all__ = ["ServiceRegistry"]

LOGGER_NAME = "huntian"


def _canonical(params: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(params or {}), sort_keys=True, default=str)


class ServiceRegistry:
    """Lazily instantiates library modules and hands them to stages."""

    def __init__(
        self,
        registry: ModuleRegistry,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._configured: dict[str, tuple[str, dict[str, Any]]] = {}
        self._instances: dict[str, IModule] = {}

    @property
    def module_registry(self) -> ModuleRegistry:
        return self._registry

    # -- configuration ----------------------------------------------------
    def configure(
        self,
        module_id: str,
        *,
        version: str = "*",
        params: Mapping[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> None:
        """Record how a service should be built, verifying it is registered."""
        if not allow_missing and not self._registry.has(module_id, version):
            raise ModuleUnavailableError(
                f"service {module_id}{version} is not registered; "
                f"known ids: {', '.join(self._registry.module_ids()) or '(none)'}"
            )
        self._configured[module_id] = (str(version), dict(params or {}))

    def configured(self) -> dict[str, str]:
        return {module_id: version for module_id, (version, _) in self._configured.items()}

    # -- lookup ------------------------------------------------------------
    def has(self, module_id: str, version: str = "*") -> bool:
        return self._registry.has(module_id, version)

    def get(
        self,
        module_id: str,
        *,
        version: str | None = None,
        params: Mapping[str, Any] | None = None,
        required: bool = True,
    ) -> IModule:
        """Return a configured instance of ``module_id``.

        Parameters come from the Deck's service slot, overridden by anything
        passed here (which is how a physics module injects problem-specific
        settings such as a tolerance in its own units).
        """
        configured_version, configured_params = self._configured.get(module_id, ("*", {}))
        effective_version = version if version is not None else configured_version
        merged = dict(configured_params)
        merged.update(params or {})

        record = None
        try:
            record = self._registry.select(module_id, effective_version)
        except ModuleUnavailableError:
            if required:
                raise
            return None  # type: ignore[return-value]

        key = f"{record.manifest.module_id}|{record.manifest.module_version}|{_canonical(merged)}"
        cached = self._instances.get(key)
        if cached is not None:
            return cached

        instance = record.module_cls()
        instance.initialize(merged)
        self._instances[key] = instance
        self._log.debug("service %s@%s instantiated", record.module_id, record.manifest.module_version)
        return instance

    def get_by_capability(
        self,
        capability: str,
        *,
        version: str = "*",
        params: Mapping[str, Any] | None = None,
        required: bool = True,
    ) -> IModule:
        """Resolve the best module declaring ``capability``."""
        candidates = self._registry.select_by_capability(capability, version)
        if not candidates:
            if not required:
                return None  # type: ignore[return-value]
            raise ServiceUnavailableError(
                f"no registered module declares capability {capability!r}"
            )
        return self.get(
            candidates[0].module_id, version=version, params=params, required=required
        )

    # -- lifecycle ---------------------------------------------------------
    def instantiated(self) -> list[str]:
        return sorted(self._instances)

    def release_all(self) -> int:
        """Terminate every cached instance and clear the cache."""
        count = 0
        for instance in list(self._instances.values()):
            try:
                instance.terminate()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                self._log.warning("service teardown failed: %s", exc)
            count += 1
        self._instances.clear()
        return count

    def __repr__(self) -> str:
        return (
            f"ServiceRegistry(configured={sorted(self._configured)}, "
            f"instantiated={len(self._instances)})"
        )
