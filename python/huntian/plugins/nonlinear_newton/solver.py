"""A generic dense damped Newton-Raphson solver.

Deliberately problem-agnostic: it is handed a residual and a Jacobian as
callables and knows nothing about Schroedinger-Poisson. The physics modules own
their own residual assembly; this module owns the nonlinear iteration strategy
(damping, backtracking line search, convergence criteria, pseudo-transient
regularisation).

Calling convention
------------------
>>> R, aux = residual(x)          # aux is opaque, carried to the Jacobian
>>> J = jacobian(x, aux)          # dense (n, n) array
>>> result = solve(residual, jacobian, x0, options=NewtonOptions())

The ``aux`` channel exists so the residual's expensive intermediate state (for
the SP system, the Schroedinger eigen-decomposition) does not have to be
recomputed when the Jacobian is assembled.

Why Newton matters here
-----------------------
The coupled SP fixed-point map has a linear gain of roughly -30: a 1 % change
in the sheet density deepens the well enough to change the density by ~30 %.
Plain damped mixing therefore needs a damping below 0.06 and hundreds of
iterations. Newton solves the same system in single-digit iterations, and the
backtracking line search keeps it safe when the starting guess is poor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "NewtonOptions",
    "NewtonResult",
    "JacobianCheck",
    "solve",
    "check_jacobian",
]

#: A residual returns ``(R, aux)``; a Jacobian consumes ``(x, aux)``.
ResidualFn = Callable[[np.ndarray], "tuple[np.ndarray, Any]"]
JacobianFn = Callable[[np.ndarray, Any], np.ndarray]


@dataclass(frozen=True)
class NewtonOptions:
    """Tuning knobs for the Newton loop."""

    max_iterations: int = 40
    tolerance: float = 1.0e-8
    residual_tolerance: float | None = None
    line_search: bool = True
    max_backtracks: int = 12
    min_damping: float = 1.0e-4
    armijo: float = 1.0e-4
    ptc_shift: float = 0.0
    stall_limit: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, **overrides: Any) -> "NewtonOptions":
        data: dict[str, Any] = dict(values or {})
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        if unknown:
            raise TypeError(
                f"unknown Newton option(s) {unknown}; expected {sorted(known)}"
            )
        return cls(**data)


@dataclass
class NewtonResult:
    """Outcome of a Newton solve."""

    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: list[float] = field(default_factory=list)
    damping_history: list[float] = field(default_factory=list)
    message: str = ""
    linear_solves: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "residual_history": list(self.residual_history),
            "damping_history": list(self.damping_history),
            "message": self.message,
            "linear_solves": self.linear_solves,
        }

    def __repr__(self) -> str:
        return (
            f"NewtonResult(converged={self.converged}, iterations={self.iterations}, "
            f"residual={self.residual_norm:.3e})"
        )


def _inf_norm(vector: np.ndarray) -> float:
    return float(np.max(np.abs(vector))) if vector.size else 0.0


def solve(
    residual: ResidualFn,
    jacobian: JacobianFn,
    x0: np.ndarray,
    *,
    options: NewtonOptions | Mapping[str, Any] | None = None,
) -> NewtonResult:
    """Damped Newton-Raphson with a backtracking line search."""
    if isinstance(options, NewtonOptions):
        opts = options
    else:
        opts = NewtonOptions.from_mapping(options)

    x = np.array(x0, dtype=float, copy=True)
    if x.ndim != 1:
        raise ValueError("x0 must be a one-dimensional array")

    R, aux = residual(x)
    norm = _inf_norm(R)
    history: list[float] = [norm]
    damping: list[float] = []
    linear_solves = 0
    stalled = 0
    message = "iteration limit reached"

    if opts.residual_tolerance is not None and norm <= opts.residual_tolerance:
        return NewtonResult(
            x=x,
            converged=True,
            iterations=0,
            residual_norm=norm,
            residual_history=history,
            damping_history=damping,
            message="initial residual already below tolerance",
        )

    matrix = None
    converged = False

    for iteration in range(1, max(1, int(opts.max_iterations)) + 1):
        matrix = np.asarray(jacobian(x, aux), dtype=float)
        if matrix.shape != (x.size, x.size):
            raise ValueError(
                f"jacobian returned shape {matrix.shape}, expected {(x.size, x.size)}"
            )
        if opts.ptc_shift > 0.0:
            matrix = matrix + float(opts.ptc_shift) * np.eye(x.size)

        try:
            step = np.linalg.solve(matrix, -R)
        except np.linalg.LinAlgError as exc:
            message = f"linear solve failed at iteration {iteration}: {exc}"
            break
        linear_solves += 1
        if not np.all(np.isfinite(step)):
            message = f"non-finite Newton step at iteration {iteration}"
            break

        damping_factor = 1.0
        if opts.line_search:
            accepted = False
            trial_x = x
            trial_R = R
            trial_aux = aux
            for _ in range(max(1, int(opts.max_backtracks))):
                trial_x = x + damping_factor * step
                trial_R, trial_aux = residual(trial_x)
                trial_norm = _inf_norm(trial_R)
                if trial_norm <= (1.0 - opts.armijo * damping_factor) * norm:
                    accepted = True
                    break
                if opts.residual_tolerance is not None and trial_norm <= opts.residual_tolerance:
                    accepted = True
                    break
                damping_factor *= 0.5
                if damping_factor < opts.min_damping:
                    break
            if not accepted:
                stalled += 1
            else:
                stalled = 0
            x, R, aux = trial_x, trial_R, trial_aux
        else:
            x = x + damping_factor * step
            R, aux = residual(x)

        norm = _inf_norm(R)
        history.append(norm)
        damping.append(damping_factor)
        update = _inf_norm(damping_factor * step)

        if update <= opts.tolerance:
            converged = True
            message = f"update norm {update:.3e} below tolerance {opts.tolerance:.3e}"
            break
        if opts.residual_tolerance is not None and norm <= opts.residual_tolerance:
            converged = True
            message = f"residual {norm:.3e} below tolerance"
            break
        if stalled >= max(1, int(opts.stall_limit)):
            message = (
                f"line search stalled for {stalled} iteration(s) at residual {norm:.3e}; "
                "the Jacobian may be inconsistent with the residual"
            )
            break

    return NewtonResult(
        x=x,
        converged=converged,
        iterations=len(history) - 1,
        residual_norm=norm,
        residual_history=history,
        damping_history=damping,
        message=message,
        linear_solves=linear_solves,
    )


@dataclass(frozen=True)
class JacobianCheck:
    """Result of comparing an analytic Jacobian with central differences."""

    max_relative_error: float
    worst_column: int
    per_column: tuple[float, ...]
    step: float

    @property
    def ok(self) -> bool:
        return self.max_relative_error < 1.0e-4

    def __str__(self) -> str:
        return (
            f"JacobianCheck(max_relative_error={self.max_relative_error:.3e}, "
            f"worst_column={self.worst_column}, step={self.step:.3e})"
        )


def check_jacobian(
    residual: ResidualFn,
    jacobian: JacobianFn,
    x: np.ndarray,
    *,
    epsilon: float | None = None,
    columns: Sequence[int] | None = None,
) -> JacobianCheck:
    """Compare an analytic Jacobian against central finite differences.

    Returns the worst relative error over the sampled columns. A correct
    Jacobian typically agrees to 1e-6..1e-10 with a well-scaled step, so a
    threshold of 1e-4 catches sign errors and missing terms while tolerating
    the truncation error of the difference formula.
    """
    x = np.asarray(x, dtype=float)
    R0, aux = residual(x)
    del R0
    matrix = np.asarray(jacobian(x, aux), dtype=float)

    scale = max(1.0, float(np.max(np.abs(x))))
    h = float(epsilon) if epsilon is not None else 1.0e-6 * scale
    indices = list(range(x.size)) if columns is None else list(columns)

    per_column: list[float] = []
    worst = 0.0
    worst_column = -1
    for index in indices:
        perturbation = np.zeros_like(x)
        perturbation[index] = h
        plus, _ = residual(x + perturbation)
        minus, _ = residual(x - perturbation)
        numeric = (plus - minus) / (2.0 * h)
        analytic = matrix[:, index]
        denominator = max(
            float(np.max(np.abs(numeric))), float(np.max(np.abs(analytic))), 1.0e-300
        )
        error = float(np.max(np.abs(numeric - analytic)) / denominator)
        per_column.append(error)
        if error > worst:
            worst = error
            worst_column = index

    return JacobianCheck(
        max_relative_error=worst,
        worst_column=worst_column,
        per_column=tuple(per_column),
        step=h,
    )
