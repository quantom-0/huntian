"""Damped Newton with a backtracking line search and pseudo-transient continuation.

Three layers of insurance, in the order the design note prescribes:

1. **bias continuation** - the caller sweeps and reuses the previous solution
   (:mod:`transport_dd.plugin`);
2. **damped line search** - halve the step until the scaled residual actually
   falls (Armijo-style acceptance);
3. **pseudo-transient continuation** - when Newton cannot find a descent step,
   regularise the *iteration matrix* with the physical time term
   ``V/tau * dn/du`` on the continuity rows. The physical residual is untouched:
   the time term only shapes the step, which is what keeps the convergence test
   measuring the steady-state equations rather than the continuation path.

The linear step comes from a service resolved through the Deck's ``linear``
slot, so the factorisation is swappable without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = ["PseudoTransient", "NewtonOutcome", "solve"]


@dataclass
class PseudoTransient:
    """Pseudo-transient schedule: a time step that grows while it keeps working."""

    tau_s: float
    growth: float = 10.0
    shrink: float = 0.1
    tau_max_s: float = 1.0
    steps: int = 0

    def relax(self) -> None:
        self.tau_s = min(self.tau_s * self.growth, self.tau_max_s)
        self.steps += 1

    def stiffen(self) -> None:
        self.tau_s = max(self.tau_s * self.shrink, 1.0e-30)
        self.steps += 1


@dataclass
class NewtonOutcome:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: list[float] = field(default_factory=list)
    damping_history: list[float] = field(default_factory=list)
    linear_solves: int = 0
    ptc_steps: int = 0
    message: str = ""
    residual_evaluations: int = 0
    update_norm: float = float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "update_norm": self.update_norm,
            "residual_history": list(self.residual_history),
            "damping_history": list(self.damping_history),
            "linear_solves": self.linear_solves,
            "ptc_steps": self.ptc_steps,
            "residual_evaluations": self.residual_evaluations,
            "message": self.message,
        }


def _inf_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def solve(
    system: Any,
    x0: np.ndarray,
    *,
    linear: Any,
    max_iterations: int = 200,
    tolerance: float = 1.0e-9,
    update_tolerance: float = 1.0e-9,
    damping_max_halvings: int = 30,
    ptc: PseudoTransient | None = None,
    stall_limit: int = 4,
    options: Mapping[str, Any] | None = None,
) -> NewtonOutcome:
    """Solve ``system`` from ``x0``.

    ``system`` must provide ``residual(u, ptc=None) -> (R, aux)`` with the
    steady residual under ``aux['physical']``, ``jacobian(u, aux, ptc=None)``
    returning the ``(lower, diagonal, upper)`` block bands, and
    ``solve_linear(linear, lower, diagonal, upper, rhs_flat)`` so this loop can
    stay independent of the block layout.
    """
    if options:
        max_iterations = int(options.get("max_iterations", max_iterations))
        tolerance = float(options.get("tolerance", tolerance))
        update_tolerance = float(options.get("update_tolerance", update_tolerance))
        damping_max_halvings = int(
            options.get("damping_max_halvings", damping_max_halvings)
        )

    x = np.array(x0, dtype=float, copy=True)
    shape = x.shape
    residual, aux = system.residual(x)
    norm = _inf_norm(aux["physical"])
    history = [norm]
    damping: list[float] = []
    linear_solves = 0
    evaluations = 1
    stalled = 0
    use_ptc = False
    message = "iteration limit reached"
    converged = norm <= tolerance
    update_norm = float("inf")

    if converged:
        return NewtonOutcome(
            x=x,
            converged=True,
            iterations=0,
            residual_norm=norm,
            residual_history=history,
            message="initial residual already below tolerance",
            residual_evaluations=evaluations,
        )

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        # -- assemble the iteration matrix --------------------------------
        # Plain Newton while it works; the pseudo-transient term appears only
        # after a failed line search, and retires once full steps succeed again.
        ptc_state = (aux["density"], ptc.tau_s) if (use_ptc and ptc is not None) else None
        if ptc_state is None:
            iteration_residual = residual
        else:
            iteration_residual, aux = system.residual(x, ptc=ptc_state)
        lower, diagonal, upper = system.jacobian(x, aux, ptc=ptc_state)

        step = system.solve_linear(
            linear, lower, diagonal, upper, -iteration_residual.reshape(-1)
        )
        linear_solves += 1
        step = np.asarray(step, dtype=float).reshape(shape)
        if not np.all(np.isfinite(step)):
            message = f"non-finite Newton step at iteration {iteration}"
            break

        # -- backtracking line search on the *physical* residual ----------
        # A row-equilibrated system can sit on a floor set by cancellation in its
        # weakest rows, where no step can lower the residual norm any further.
        # The update norm is what decides convergence there, exactly as in the
        # dense Newton service: a tiny step means the iterate has stopped moving.
        factor = 1.0
        accepted = False
        trial = x
        trial_aux = aux
        trial_norm = norm
        for _ in range(max(1, int(damping_max_halvings)) + 1):
            trial = x + factor * step
            update_norm = _inf_norm(factor * step)
            _, trial_aux = system.residual(trial, ptc=ptc_state)
            evaluations += 1
            trial_norm = _inf_norm(trial_aux["physical"])
            if trial_norm < norm or trial_norm <= tolerance:
                accepted = True
                break
            factor *= 0.5
            if factor < 1.0e-8:
                break

        if not accepted:
            stalled += 1
            if ptc is not None and not use_ptc:
                use_ptc = True
                ptc.stiffen()
                stalled = 0
                message = (
                    f"line search failed at iteration {iteration}; switching to "
                    f"pseudo-transient continuation (tau = {ptc.tau_s:.3e} s)"
                )
                continue
            if ptc is not None and use_ptc and stalled <= max(1, int(stall_limit)):
                ptc.stiffen()
                message = (
                    f"line search failed at iteration {iteration}; stiffening "
                    f"pseudo-transient tau -> {ptc.tau_s:.3e} s"
                )
                continue
            message = (
                f"line search stalled at iteration {iteration} "
                f"(residual {norm:.3e}); the Jacobian may be inconsistent"
            )
            break

        x = trial
        residual, aux = system.residual(x, ptc=ptc_state)
        evaluations += 1
        damping.append(factor)
        norm = _inf_norm(aux["physical"])
        history.append(norm)
        stalled = 0

        if use_ptc and ptc is not None:
            if factor >= 1.0:
                ptc.relax()
                if ptc.tau_s >= ptc.tau_max_s:
                    use_ptc = False  # back to plain Newton
            else:
                ptc.stiffen()

        if norm <= tolerance:
            converged = True
            message = f"scaled residual {norm:.3e} below tolerance {tolerance:.3e}"
            break
        if update_norm <= update_tolerance:
            converged = True
            message = (
                f"update norm {update_norm:.3e} below tolerance "
                f"{update_tolerance:.3e} (residual {norm:.3e})"
            )
            break

    return NewtonOutcome(
        x=x,
        converged=converged,
        iterations=len(history) - 1,
        residual_norm=norm,
        residual_history=history,
        damping_history=damping,
        linear_solves=linear_solves,
        ptc_steps=0 if ptc is None else ptc.steps,
        message=message,
        residual_evaluations=evaluations,
        update_norm=update_norm,
    )
