"""``linear.blocktridiag`` - a pluggable block-tridiagonal direct solver.

A **library module**, not a pipeline stage: it transforms no context and produces
no artifacts. Physics modules resolve it as a collaborator through the context's
service registry, configured from the Deck's ``linear`` slot::

    "modules": {
      "linear": { "module": "linear.blocktridiag", "params": {} }
    }

and then call it with the three block bands of their Jacobian::

    linear = ctx.require_services().get_by_capability("linear")
    step = linear.solve(lower, diagonal, upper, -residual)

This is the slot the drift-diffusion module fills: its Jacobian is block
tridiagonal by construction on a structured grid, and ~2e4 unknowns rule out a
dense factorisation. Swapping in a banded, GPU or iterative backend later is one
Deck line - the same mechanism that swaps the quantum or nonlinear module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nemo.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

from .solver import (
    BlockTridiagonalError,
    check_block_system,
    solve_block_tridiagonal,
    to_dense,
)

MANIFEST = ModuleManifest(
    module_id="linear.blocktridiag",
    module_version="0.1.0",
    category=ModuleCategory.SOLVER,
    description=(
        "Direct block-tridiagonal solver for structured-grid Jacobians: block "
        "Thomas elimination with LAPACK block pivots, O(nz*m^3) time and "
        "O(nz*m^2) memory. Pure NumPy - no SciPy, which the deployment target "
        "does not have. Consumed as a service by physics modules."
    ),
    capabilities=("linear", "linear.blocktridiag"),
    parameters=(
        ParamSpec.float(
            "regularization",
            0.0,
            minimum=0.0,
            doc=(
                "diagonal shift added to every block before factorising, i.e. "
                "solve (J + shift*I)x = b; a pseudo-transient step is exactly "
                "this, so a caller can request it without rebuilding J"
            ),
        ),
    ),
    verification="partially-verified",
    author="NemoTCAD",
    tags=("solver", "linear", "sparse", "block-tridiagonal", "service"),
)


class BlockTridiagonalSolver(BaseModule):
    """Exposes :func:`~linear_blocktridiag.solver.solve_block_tridiagonal`."""

    def __init__(self) -> None:
        super().__init__()
        #: Diagnostics read back by the caller after a solve.
        self.last_solve: dict[str, Any] | None = None

    # -- library API ------------------------------------------------------
    def solve(
        self,
        lower: Any,
        diagonal: Any,
        upper: Any,
        rhs: Any,
        *,
        regularization: float | None = None,
    ) -> np.ndarray:
        """Solve the block-tridiagonal system for ``rhs`` shaped ``(n, m)``."""
        shift = (
            float(self.param("regularization"))
            if regularization is None
            else float(regularization)
        )
        _lower, _diagonal, _upper, count, size = check_block_system(
            lower, diagonal, upper
        )
        self.last_solve = {
            "blocks": count,
            "block_size": size,
            "unknowns": count * size,
            "regularization": shift,
        }
        return solve_block_tridiagonal(
            lower, diagonal, upper, rhs, regularization=shift
        )

    def dense(self, lower: Any, diagonal: Any, upper: Any) -> np.ndarray:
        """Diagnostic hook: the full matrix, for tests and small systems."""
        return to_dense(lower, diagonal, upper)

    # -- stage lifecycle ---------------------------------------------------
    def validate(self, ctx: Any) -> list[Any]:
        return []

    def execute(self, ctx: Any) -> Any:
        raise RuntimeError(
            f"{MANIFEST.module_id} is a linear solver service, not a pipeline "
            "stage. Bind it to the Deck's 'linear' slot and let a physics module "
            "resolve it through ctx.services."
        )


MODULE = BlockTridiagonalSolver

__all__ = [
    "MANIFEST",
    "MODULE",
    "BlockTridiagonalSolver",
    "BlockTridiagonalError",
    "solve_block_tridiagonal",
    "to_dense",
]
