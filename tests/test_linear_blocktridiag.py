"""Numerics and service-contract tests for ``linear.blocktridiag``.

Verification strategy: the block solver is *direct*, so it can be checked
exactly - every case is compared against a dense ``numpy.linalg.solve`` of the
materialised matrix. The service half is checked through the real registry, so
the capability/slot wiring is exercised rather than asserted.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest

import numpy as np

from nemo import ModuleRegistry
from nemo.services import ServiceRegistry

from . import PLUGIN_DIR
from .helpers import build_registry

LINEAR_SOLVER_PATH = PLUGIN_DIR / "linear_blocktridiag" / "solver.py"


def load_linear_solver():
    """Import the numerics straight from the file, as the other kernels do."""
    name = "nemo_tests_linear_blocktridiag"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LINEAR_SOLVER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"cannot load {LINEAR_SOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load_linear_solver()


def random_system(count: int, size: int, *, seed: int = 0, shift: float = 4.0):
    """A diagonally dominant block system - well conditioned by construction."""
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-1.0, -0.1, (count - 1, size, size))
    upper = rng.uniform(-1.0, -0.1, (count - 1, size, size))
    diagonal = np.empty((count, size, size))
    for row in range(count):
        block = rng.uniform(-0.5, 0.5, (size, size))
        diagonal[row] = block
        # Row-sum dominance, the classic M-matrix construction.
        off = np.zeros(size)
        if row > 0:
            off += lower[row - 1].sum(axis=1)
        if row < count - 1:
            off += upper[row].sum(axis=1)
        diagonal[row] += np.diag(shift - off)
    return lower, diagonal, upper


class BlockSolveTests(unittest.TestCase):
    def test_matches_a_dense_solve(self) -> None:
        for count, size, seed in ((2, 2, 1), (5, 3, 2), (9, 4, 3), (17, 6, 4)):
            lower, diagonal, upper = random_system(count, size, seed=seed)
            rng = np.random.default_rng(seed + 100)
            rhs = rng.uniform(-1.0, 1.0, (count, size))
            dense = L.to_dense(lower, diagonal, upper)
            expected = np.linalg.solve(dense, rhs.reshape(-1)).reshape(count, size)
            solution = L.solve_block_tridiagonal(lower, diagonal, upper, rhs)
            residual = float(np.max(np.abs(solution - expected)))
            self.assertLess(residual, 1e-10, msg=f"n={count} m={size}: {residual:.2e}")

    def test_single_block_is_a_dense_solve(self) -> None:
        rng = np.random.default_rng(7)
        lower = np.zeros((0, 3, 3))
        upper = np.zeros((0, 3, 3))
        diagonal = rng.uniform(-1.0, 1.0, (1, 3, 3)) + 4.0 * np.eye(3)
        rhs = rng.uniform(-1.0, 1.0, (1, 3))
        solution = L.solve_block_tridiagonal(lower, diagonal, upper, rhs)
        self.assertLess(
            float(np.max(np.abs(solution - np.linalg.solve(diagonal[0], rhs[0])))),
            1e-12,
        )

    def test_regularization_shifts_the_diagonal_blocks(self) -> None:
        lower, diagonal, upper = random_system(6, 3, seed=11)
        rng = np.random.default_rng(12)
        rhs = rng.uniform(-1.0, 1.0, (6, 3))
        shift = 0.75
        solution = L.solve_block_tridiagonal(
            lower, diagonal, upper, rhs, regularization=shift
        )
        shifted = diagonal + shift * np.eye(3)
        dense = L.to_dense(lower, shifted, upper)
        expected = np.linalg.solve(dense, rhs.reshape(-1)).reshape(6, 3)
        self.assertLess(float(np.max(np.abs(solution - expected))), 1e-10)

    def test_a_singular_pivot_is_reported_with_its_row(self) -> None:
        size = 2
        lower = np.zeros((2, size, size))
        upper = np.zeros((2, size, size))
        diagonal = np.zeros((3, size, size))
        diagonal[0] = np.eye(size)
        diagonal[1] = 0.0  # singular block, and nothing couples it away
        diagonal[2] = np.eye(size)
        rhs = np.zeros((3, size))
        with self.assertRaises(L.BlockTridiagonalError) as caught:
            L.solve_block_tridiagonal(lower, diagonal, upper, rhs)
        self.assertIn("row 1", str(caught.exception))

    def test_malformed_shapes_are_rejected(self) -> None:
        good = np.eye(3)
        with self.assertRaises(L.BlockTridiagonalError):
            L.solve_block_tridiagonal(
                np.zeros((1, 3, 3)), np.stack([good, good]), np.zeros((1, 3, 3)),
                np.zeros((2, 4)),
            )
        with self.assertRaises(L.BlockTridiagonalError):
            L.solve_block_tridiagonal(
                np.zeros((1, 3, 3)), np.stack([good, good]), np.zeros((1, 3, 3)),
                np.zeros((3, 3)),
            )
        with self.assertRaises(L.BlockTridiagonalError):
            L.check_block_system(np.zeros((1, 3, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3, 3)))

    def test_to_dense_lays_the_blocks_out_in_order(self) -> None:
        lower, diagonal, upper = random_system(4, 2, seed=21)
        dense = L.to_dense(lower, diagonal, upper)
        self.assertEqual(dense.shape, (8, 8))
        self.assertTrue(np.allclose(dense[:2, 2:4], upper[0]))
        self.assertTrue(np.allclose(dense[2:4, :2], lower[0]))
        self.assertTrue(np.allclose(dense[6:8, 6:8], diagonal[3]))
        self.assertTrue(np.allclose(dense[:2, 4:6], 0.0))


class ServiceContractTests(unittest.TestCase):
    """The plugin must be reachable as the ``linear`` capability."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_the_module_declares_the_linear_capability(self) -> None:
        record = self.registry.select("linear.blocktridiag")
        self.assertIn("linear", record.manifest.capabilities)
        self.assertEqual(record.manifest.module_version, "0.1.0")

    def test_resolving_by_capability_returns_a_working_solver(self) -> None:
        services = ServiceRegistry(self.registry)
        solver = services.get_by_capability("linear")
        lower, diagonal, upper = random_system(5, 3, seed=31)
        rng = np.random.default_rng(32)
        rhs = rng.uniform(-1.0, 1.0, (5, 3))
        solution = solver.solve(lower, diagonal, upper, rhs)
        dense = solver.dense(lower, diagonal, upper)
        expected = np.linalg.solve(dense, rhs.reshape(-1)).reshape(5, 3)
        self.assertLess(float(np.max(np.abs(solution - expected))), 1e-10)
        self.assertEqual(solver.last_solve["unknowns"], 15)

    def test_the_service_cannot_be_run_as_a_pipeline_stage(self) -> None:
        services = ServiceRegistry(self.registry)
        solver = services.get_by_capability("linear")
        with self.assertRaises(RuntimeError):
            solver.execute(None)

    def test_an_empty_registry_reports_a_missing_capability(self) -> None:
        services = ServiceRegistry(ModuleRegistry())
        with self.assertRaises(Exception):
            services.get_by_capability("linear")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
