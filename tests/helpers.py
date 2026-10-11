"""Shared helpers for the test suite."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from huntian import Deck, Executor, MaterialDB, ModuleRegistry, Pipeline, SimContext

from . import PLUGIN_DIR

__all__ = [
    "build_registry",
    "builtin_materials",
    "run_deck",
    "load_sp_solver",
    "load_newton_solver",
    "NewtonSolverStub",
    "SP_SOLVER_PATH",
    "NEWTON_SOLVER_PATH",
]

SP_SOLVER_PATH = PLUGIN_DIR / "quantum_sp1d" / "solver.py"
NEWTON_SOLVER_PATH = PLUGIN_DIR / "nonlinear_newton" / "solver.py"


def _load_by_path(name: str, path: Any) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_registry(*, plugin_dir: Any = PLUGIN_DIR) -> ModuleRegistry:
    """A registry loaded with the bundled plugins."""
    registry = ModuleRegistry()
    report = registry.load_from_paths([plugin_dir])
    if not report.ok:
        raise AssertionError(f"plugin discovery failed: {report.errors}")
    return registry


def builtin_materials() -> MaterialDB:
    return MaterialDB.builtin()


def run_deck(
    deck: Deck,
    *,
    registry: ModuleRegistry | None = None,
    strict: bool = True,
) -> tuple[Any, SimContext]:
    """Run a deck end to end, returning ``(result, context)``."""
    registry = registry if registry is not None else build_registry()
    ctx = SimContext(name=deck.name, bus=registry.bus)
    result = Executor(registry).run(
        Pipeline.from_deck(deck),
        ctx=ctx,
        deck=deck,
        materials=builtin_materials(),
        strict=strict,
    )
    return result, ctx


def load_sp_solver() -> Any:
    """Import the Schroedinger-Poisson numerics directly from their file.

    ``solver.py`` deliberately has no relative imports, so it can be exercised
    as standalone numerics as well as through the plugin package. Loading it
    this way also proves that property.
    """
    return _load_by_path("huntian_tests_sp1d_solver", SP_SOLVER_PATH)


def load_newton_solver() -> Any:
    """Import the generic Newton numerics directly from their file."""
    return _load_by_path("huntian_tests_newton_solver", NEWTON_SOLVER_PATH)


class NewtonSolverStub:
    """Adapter exposing the ``nonlinear.newton`` module's library surface.

    Used so the Schroedinger-Poisson unit tests can exercise ``algorithm=
    "newton"`` without standing up a whole registry.
    """

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)
        self.last_result: Any = None

    def solve(self, residual, jacobian, x0, *, options=None):  # noqa: ANN001
        newton = load_newton_solver()
        merged = dict(self.options)
        merged.update(options or {})
        opts = newton.NewtonOptions.from_mapping(merged)
        self.last_result = newton.solve(residual, jacobian, x0, options=opts)
        return self.last_result
