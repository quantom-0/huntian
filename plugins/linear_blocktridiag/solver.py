"""Block-tridiagonal direct solves, in pure NumPy.

Why this exists
---------------
Steady-state drift-diffusion on a structured ``nx x nz`` tensor grid, with the
unknowns grouped per node and ordered so that ``nx`` varies fastest, produces a
Jacobian that is **block tridiagonal**: node row ``j`` couples only to rows
``j-1`` and ``j+1`` through dense ``m x m`` blocks, with ``m`` the number of
unknowns carried per lateral row (``2 * nx`` for the ``(psi, phi_n)`` pair).
Factorising that structure directly costs ``O(nz * m^3)`` instead of the
``O((nx*nz)^3)`` a dense solve would need, and it needs no third-party sparse
library - which matters here, because the deployment target has NumPy and not
SciPy.

The elimination is the block analogue of the Thomas algorithm. Each block pivot
is factorised by LAPACK (``numpy.linalg.solve``), so partial pivoting is always
in play: the solver does not rely on the Jacobian being an M-matrix, which
keeps it honest for a coupled Newton system where that property is easy to lose
once the flux depends on the potential.

Complexity and honest limits
----------------------------
* time: ``O(nz * m^3)``; memory: ``O(nz * m^2)`` for the three block bands.
* the blocks are stored dense, so a lateral row of 200 unknowns costs ~1.6 MB
  per band per 1000 vertical rows. Refining ``nx`` is therefore the expensive
  direction - a banded or nested-dissection factorisation is the obvious next
  step if a lateral refinement ever matters.
* an *iterative* fallback (BiCGSTAB + ILU(0)) is deliberately absent: it would
  add a second convergence story to debug. If the direct factorisation ever
  becomes the bottleneck, that is the place to reach for, and the Deck's
  ``linear`` slot is exactly where the replacement binds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "BlockTridiagonalError",
    "check_block_system",
    "solve_block_tridiagonal",
    "to_dense",
]


class BlockTridiagonalError(ValueError):
    """Raised when the block system is malformed or a block pivot is singular."""


def _as_blocks(array: Any, name: str, count: int, size: int) -> np.ndarray:
    blocks = np.asarray(array, dtype=float)
    if blocks.shape != (count, size, size):
        raise BlockTridiagonalError(
            f"{name} has shape {blocks.shape}, expected {(count, size, size)}"
        )
    return blocks


def check_block_system(
    lower: Any, diagonal: Any, upper: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate a block-tridiagonal triple and return it with ``(n, m)``.

    ``lower[k]`` is the block coupling row ``k+1`` to row ``k`` (so there are
    ``n-1`` of them), ``upper[k]`` couples row ``k`` to row ``k+1``, and
    ``diagonal`` holds the ``n`` diagonal blocks.
    """
    diagonal = np.asarray(diagonal, dtype=float)
    if diagonal.ndim != 3 or diagonal.shape[1] != diagonal.shape[2]:
        raise BlockTridiagonalError(
            f"diagonal has shape {diagonal.shape}, expected (n, m, m)"
        )
    count, size = diagonal.shape[0], diagonal.shape[1]
    if count < 1 or size < 1:
        raise BlockTridiagonalError("the block system is empty")
    lower = _as_blocks(lower, "lower", max(count - 1, 0), size)
    upper = _as_blocks(upper, "upper", max(count - 1, 0), size)
    return lower, diagonal, upper, count, size


def solve_block_tridiagonal(
    lower: Any,
    diagonal: Any,
    upper: Any,
    rhs: Any,
    *,
    regularization: float = 0.0,
) -> np.ndarray:
    """Solve a block-tridiagonal system by block elimination.

    Parameters
    ----------
    lower, diagonal, upper
        The three block bands, shaped ``(n-1, m, m)``, ``(n, m, m)`` and
        ``(n-1, m, m)``.
    rhs
        Right-hand sides shaped ``(n, m)`` - one vector per block row.
    regularization
        Optional scalar added to every diagonal block, i.e. ``(J + shift*I)``.
        A pseudo-transient step is exactly that, so the caller can ask for it
        without rebuilding the Jacobian.

    Returns
    -------
    numpy.ndarray
        The solution, shaped ``(n, m)``.
    """
    lower, diagonal, upper, count, size = check_block_system(lower, diagonal, upper)
    values = np.asarray(rhs, dtype=float)
    if values.shape != (count, size):
        raise BlockTridiagonalError(
            f"rhs has shape {values.shape}, expected {(count, size)}"
        )
    if regularization:
        diagonal = diagonal + float(regularization) * np.eye(size)

    # Forward elimination. ``pivot`` holds the eliminated diagonal blocks, so
    # each row is reduced once and reused for the full block of right-hand
    # sides (a block Thomas algorithm).
    pivot_diagonal = np.empty_like(diagonal)
    pivot_rhs = np.empty_like(values)
    pivot_diagonal[0] = diagonal[0]
    pivot_rhs[0] = values[0]

    for row in range(1, count):
        try:
            reduced = np.linalg.solve(
                pivot_diagonal[row - 1],
                np.concatenate([upper[row - 1], pivot_rhs[row - 1][:, None]], axis=1),
            )
        except np.linalg.LinAlgError as exc:
            raise BlockTridiagonalError(
                f"singular block pivot at row {row - 1}: {exc}; the Jacobian is "
                "not factorisable - check the residual/Jacobian consistency"
            ) from exc
        coupling = reduced[:, :size]
        pivot_diagonal[row] = diagonal[row] - lower[row - 1] @ coupling
        pivot_rhs[row] = values[row] - lower[row - 1] @ reduced[:, size]

    solution = np.empty_like(values)
    try:
        solution[-1] = np.linalg.solve(pivot_diagonal[-1], pivot_rhs[-1])
    except np.linalg.LinAlgError as exc:
        raise BlockTridiagonalError(
            f"singular block pivot at row {count - 1}: {exc}"
        ) from exc
    for row in range(count - 2, -1, -1):
        solution[row] = np.linalg.solve(
            pivot_diagonal[row], pivot_rhs[row] - upper[row] @ solution[row + 1]
        )
    return solution


def to_dense(lower: Any, diagonal: Any, upper: Any) -> np.ndarray:
    """Materialise the full matrix - for tests and small diagnostics only."""
    lower, diagonal, upper, count, size = check_block_system(lower, diagonal, upper)
    matrix = np.zeros((count * size, count * size), dtype=float)
    for row in range(count):
        start = row * size
        matrix[start : start + size, start : start + size] = diagonal[row]
        if row > 0:
            matrix[start : start + size, start - size : start] = lower[row - 1]
        if row < count - 1:
            matrix[start : start + size, start + size : start + 2 * size] = upper[row]
    return matrix
