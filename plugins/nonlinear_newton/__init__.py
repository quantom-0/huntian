"""``nonlinear.newton`` - a pluggable dense Newton-Raphson solver module.

This is a **library module**, not a pipeline stage: it produces no artifacts and
transforms no context. Physics modules obtain it as a collaborator through the
context's service registry, configured from the Deck's ``nonlinear`` slot::

    "modules": {
      "nonlinear": {
        "module": "nonlinear.newton",
        "params": { "max_iterations": 40, "line_search": true }
      }
    }

and then call it with their own residual and Jacobian::

    solver = ctx.services.get("nonlinear.newton")
    result = solver.solve(system.residual, system.jacobian, x0,
                          options={"tolerance": 1e-6})

Keeping the iteration strategy in its own module means a Newton variant with
pseudo-transient continuation, a trust region, or a Krylov linear solver can be
dropped in by changing one Deck line - the same mechanism that swaps the
quantum physics.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from nemo.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

from .solver import (
    JacobianCheck,
    NewtonOptions,
    NewtonResult,
    check_jacobian,
    solve as _newton_solve,
)

MANIFEST = ModuleManifest(
    module_id="nonlinear.newton",
    module_version="1.0.0",
    category=ModuleCategory.SOLVER,
    description=(
        "Generic dense damped Newton-Raphson with a backtracking line search and "
        "optional diagonal (pseudo-transient) regularisation. Consumed as a service "
        "by physics modules; not run as a pipeline stage."
    ),
    capabilities=("nonlinear", "nonlinear.newton"),
    parameters=(
        ParamSpec.integer(
            "max_iterations",
            40,
            minimum=1,
            maximum=1000,
            doc="Newton iteration cap",
        ),
        ParamSpec.float(
            "tolerance",
            1.0e-8,
            minimum=1.0e-16,
            maximum=1.0e-1,
            doc=(
                "convergence tolerance on the update norm, in the units of the "
                "unknown; the calling physics module normally overrides this"
            ),
        ),
        ParamSpec.boolean(
            "line_search",
            True,
            doc="backtracking line search on the residual norm",
        ),
        ParamSpec.integer(
            "max_backtracks",
            12,
            minimum=0,
            maximum=60,
            doc="halvings attempted per Newton step when line searching",
        ),
        ParamSpec.float(
            "min_damping",
            1.0e-4,
            minimum=1.0e-12,
            maximum=1.0,
            doc="smallest damping factor the line search will try",
        ),
        ParamSpec.float(
            "armijo",
            1.0e-4,
            minimum=0.0,
            maximum=1.0,
            doc="Armijo sufficient-decrease constant",
        ),
        ParamSpec.float(
            "ptc_shift",
            0.0,
            minimum=0.0,
            doc=(
                "diagonal shift added to the Jacobian before each solve; a "
                "pseudo-transient regularisation for badly conditioned steps"
            ),
        ),
        ParamSpec.integer(
            "stall_limit",
            3,
            minimum=1,
            maximum=100,
            doc="abort after this many consecutive rejected line searches",
        ),
    ),
    verification="partially-verified",
    author="NemoTCAD",
    tags=("solver", "newton", "nonlinear", "service"),
)


class NewtonSolver(BaseModule):
    """Exposes :func:`~nonlinear_newton.solver.solve` with module-level options."""

    def __init__(self) -> None:
        super().__init__()
        #: Most recent solve, for diagnostics read back by the caller.
        self.last_result: NewtonResult | None = None

    # -- library API ------------------------------------------------------
    def option_defaults(self) -> dict[str, Any]:
        """Options taken from this module's configured parameters."""
        return {
            "max_iterations": int(self.param("max_iterations")),
            "tolerance": float(self.param("tolerance")),
            "line_search": bool(self.param("line_search")),
            "max_backtracks": int(self.param("max_backtracks")),
            "min_damping": float(self.param("min_damping")),
            "armijo": float(self.param("armijo")),
            "ptc_shift": float(self.param("ptc_shift")),
            "stall_limit": int(self.param("stall_limit")),
        }

    def solve(
        self,
        residual: Callable[[np.ndarray], "tuple[np.ndarray, Any]"],
        jacobian: Callable[[np.ndarray, Any], np.ndarray],
        x0: np.ndarray,
        *,
        options: Mapping[str, Any] | NewtonOptions | None = None,
    ) -> NewtonResult:
        """Run Newton on a caller-supplied residual and Jacobian.

        ``options`` override this module's configured parameters, which is how
        a physics module injects problem-specific settings such as a tolerance
        expressed in electronvolts.
        """
        if isinstance(options, NewtonOptions):
            merged = options
        else:
            combined = self.option_defaults()
            combined.update(options or {})
            merged = NewtonOptions.from_mapping(combined)
        outcome = _newton_solve(residual, jacobian, x0, options=merged)
        self.last_result = outcome
        return outcome

    def verify_jacobian(
        self,
        residual: Callable[[np.ndarray], "tuple[np.ndarray, Any]"],
        jacobian: Callable[[np.ndarray, Any], np.ndarray],
        x: np.ndarray,
        *,
        epsilon: float | None = None,
        columns: Any = None,
    ) -> JacobianCheck:
        """Diagnostic hook: compare the Jacobian with finite differences."""
        return check_jacobian(
            residual, jacobian, x, epsilon=epsilon, columns=columns
        )

    # -- stage lifecycle ---------------------------------------------------
    def validate(self, ctx: Any) -> list[Any]:
        return []

    def execute(self, ctx: Any) -> Any:
        raise RuntimeError(
            f"{MANIFEST.module_id} is a solver service, not a pipeline stage. "
            "Bind it to the Deck's 'nonlinear' slot and let a physics module "
            "resolve it through ctx.services."
        )


MODULE = NewtonSolver
