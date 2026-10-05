"""Tests for the generic Newton-Raphson solver module.

The numerics are exercised directly (file import) and through the plugin
manifest (registry instantiation), so both the algorithm and the module
contract are covered.
"""

from __future__ import annotations

import unittest

import numpy as np

from nemo.errors import ModuleConfigError
from nemo.registry import ModuleState

from .helpers import build_registry, load_newton_solver

N = load_newton_solver()


def linear_system(size: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-2.0, 2.0, (size, size)) + 3.0 * np.eye(size)
    target = rng.uniform(-1.0, 1.0, size)

    def residual(x):
        return matrix @ x - target, None

    def jacobian(x, aux):
        del aux
        return matrix

    return residual, jacobian, np.linalg.solve(matrix, target)


class OptionsTests(unittest.TestCase):
    def test_from_mapping_applies_overrides(self) -> None:
        options = N.NewtonOptions.from_mapping({"max_iterations": 7}, tolerance=1e-3)
        self.assertEqual(options.max_iterations, 7)
        self.assertAlmostEqual(options.tolerance, 1e-3)

    def test_unknown_option_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            N.NewtonOptions.from_mapping({"nope": 1})

    def test_none_overrides_are_ignored(self) -> None:
        options = N.NewtonOptions.from_mapping({}, tolerance=None)
        self.assertEqual(options.tolerance, N.NewtonOptions().tolerance)


class NewtonConvergenceTests(unittest.TestCase):
    def test_linear_system_is_solved_in_two_iterations(self) -> None:
        residual, jacobian, exact = linear_system()
        result = N.solve(residual, jacobian, np.zeros(3))
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 2)
        np.testing.assert_allclose(result.x, exact, atol=1e-10)

    def test_quadratic_convergence_on_a_scalar_root(self) -> None:
        def residual(x):
            return np.array([x[0] ** 2 - 2.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[2.0 * x[0]]])

        result = N.solve(residual, jacobian, np.array([1.0]))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.x[0]), np.sqrt(2.0), places=12)

        history = [v for v in result.residual_history if v > 0.0]
        self.assertGreaterEqual(len(history), 4)
        orders = []
        for i in range(1, len(history) - 1):
            a, b, c = history[i - 1], history[i], history[i + 1]
            if a > b > c:
                orders.append(np.log(c / b) / np.log(b / a))
        self.assertTrue(orders, "no measurable convergence order")
        self.assertGreater(max(orders), 1.7, msg=f"orders {orders}")

    def test_line_search_recovers_from_a_far_start(self) -> None:
        # From x = 0.1 the undamped Newton step jumps to x ~ 33, where the
        # residual is far larger: without backtracking the iteration diverges.
        def residual(x):
            return np.array([x[0] ** 3 - 1.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[3.0 * x[0] ** 2]])

        result = N.solve(residual, jacobian, np.array([0.1]))
        self.assertTrue(result.converged, msg=result.message)
        self.assertAlmostEqual(float(result.x[0]), 1.0, places=9)
        self.assertTrue(
            any(d < 1.0 for d in result.damping_history),
            msg=f"no backtracking happened: {result.damping_history}",
        )

    def test_line_search_keeps_the_residual_monotone(self) -> None:
        def residual(x):
            return np.array([x[0] ** 3 - 1.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[3.0 * x[0] ** 2]])

        result = N.solve(residual, jacobian, np.array([0.1]))
        history = result.residual_history
        for previous, current in zip(history, history[1:]):
            self.assertLessEqual(current, previous * (1.0 + 1e-12))

    def test_a_well_behaved_start_needs_no_backtracking(self) -> None:
        def residual(x):
            return np.array([x[0] ** 2 - 2.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[2.0 * x[0]]])

        result = N.solve(residual, jacobian, np.array([1.0]))
        self.assertTrue(result.converged)
        self.assertTrue(all(d == 1.0 for d in result.damping_history))

    def test_without_line_search_a_far_start_can_fail(self) -> None:
        def residual(x):
            return np.array([x[0] ** 3 - 1.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[3.0 * x[0] ** 2]])

        result = N.solve(
            residual,
            jacobian,
            np.array([40.0]),
            options=N.NewtonOptions(line_search=False, max_iterations=4),
        )
        self.assertFalse(result.converged)

    def test_iteration_limit_is_reported(self) -> None:
        def residual(x):
            return np.array([np.exp(x[0]) - 1e6]), None

        def jacobian(x, aux):
            del aux
            return np.array([[np.exp(x[0])]])

        result = N.solve(
            residual, jacobian, np.array([0.0]),
            options=N.NewtonOptions(max_iterations=1, line_search=True, tolerance=1e-14),
        )
        self.assertFalse(result.converged)
        self.assertIn("iteration limit", result.message)

    def test_ptc_shift_changes_the_step(self) -> None:
        def residual(x):
            return np.array([x[0] ** 2 - 4.0]), None

        def jacobian(x, aux):
            del aux
            return np.array([[2.0 * x[0]]])

        plain = N.solve(residual, jacobian, np.array([1.0]),
                        options=N.NewtonOptions(line_search=False, max_iterations=1))
        shifted = N.solve(residual, jacobian, np.array([1.0]),
                          options=N.NewtonOptions(line_search=False, max_iterations=1,
                                                  ptc_shift=5.0))
        self.assertNotAlmostEqual(float(plain.x[0]), float(shifted.x[0]), places=6)

    def test_singular_jacobian_is_reported_not_raised(self) -> None:
        def residual(x):
            return np.array([1.0, 1.0]), None

        def jacobian(x, aux):
            del aux
            return np.zeros((2, 2))

        result = N.solve(residual, jacobian, np.zeros(2))
        self.assertFalse(result.converged)
        self.assertIn("linear solve failed", result.message)

    def test_non_square_jacobian_is_rejected(self) -> None:
        def residual(x):
            return np.array([1.0, 1.0]), None

        def jacobian(x, aux):
            del aux
            return np.zeros((2, 3))

        with self.assertRaises(ValueError):
            N.solve(residual, jacobian, np.zeros(2))

    def test_result_serialises(self) -> None:
        residual, jacobian, _ = linear_system()
        payload = N.solve(residual, jacobian, np.zeros(3)).as_dict()
        for key in ("converged", "iterations", "residual_history", "linear_solves"):
            self.assertIn(key, payload)


class JacobianCheckTests(unittest.TestCase):
    @staticmethod
    def _system():
        def residual(x):
            return np.array([x[0] ** 2 + 3.0 * x[1], np.sin(x[0]) + x[1] ** 3]), None

        def jacobian(x, aux):
            del aux
            return np.array([[2.0 * x[0], 3.0], [np.cos(x[0]), 3.0 * x[1] ** 2]])

        return residual, jacobian

    def test_correct_jacobian_is_confirmed(self) -> None:
        residual, jacobian = self._system()
        check = N.check_jacobian(residual, jacobian, np.array([0.7, -0.4]))
        self.assertLess(check.max_relative_error, 1e-8)
        self.assertTrue(check.ok)

    def test_wrong_jacobian_is_detected(self) -> None:
        residual, jacobian = self._system()

        def wrong(x, aux):
            matrix = jacobian(x, aux).copy()
            matrix[0, 1] = 4.0  # should be 3.0
            return matrix

        check = N.check_jacobian(residual, wrong, np.array([0.7, -0.4]))
        self.assertGreater(check.max_relative_error, 1e-2)
        self.assertFalse(check.ok)
        self.assertEqual(check.worst_column, 1)

    def test_column_subset_is_honoured(self) -> None:
        residual, jacobian = self._system()
        check = N.check_jacobian(
            residual, jacobian, np.array([0.7, -0.4]), columns=[0]
        )
        self.assertEqual(len(check.per_column), 1)


class NewtonModuleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.record = cls.registry.select("nonlinear.newton")

    def test_manifest_declares_the_solver_capability(self) -> None:
        self.assertIn("nonlinear", self.record.manifest.capabilities)
        self.assertEqual(self.record.manifest.category.value, "solver")
        self.assertEqual(self.record.manifest.produces, ())

    def test_parameters_are_validated(self) -> None:
        instance = self.record.module_cls()
        with self.assertRaises(ModuleConfigError):
            instance.initialize({"not_a_parameter": 1})

    def test_solve_uses_the_modules_configured_options(self) -> None:
        instance = self.record.module_cls()
        instance.initialize({"max_iterations": 3, "tolerance": 1e-3})
        self.assertEqual(instance.option_defaults()["max_iterations"], 3)

        residual, jacobian, _ = linear_system()
        result = instance.solve(residual, jacobian, np.zeros(3))
        self.assertTrue(result.converged)
        self.assertIs(instance.last_result, result)

    def test_caller_options_override_the_module_configuration(self) -> None:
        instance = self.record.module_cls()
        instance.initialize({"tolerance": 1e-12})

        def residual(x):
            return np.array([np.exp(x[0]) - 1e6]), None

        def jacobian(x, aux):
            del aux
            return np.array([[np.exp(x[0])]])

        tight = instance.solve(
            residual, jacobian, np.zeros(1), options={"max_iterations": 1,
                                                      "tolerance": 1e-14}
        )
        self.assertFalse(tight.converged)

    def test_verify_jacobian_hook(self) -> None:
        instance = self.record.module_cls()
        instance.initialize({})

        def residual(x):
            return np.array([x[0] ** 2]), None

        def jacobian(x, aux):
            del aux
            return np.array([[2.0 * x[0]]])

        check = instance.verify_jacobian(residual, jacobian, np.array([1.3]))
        self.assertLess(check.max_relative_error, 1e-8)

    def test_running_it_as_a_pipeline_stage_is_rejected(self) -> None:
        instance = self.record.module_cls()
        instance.initialize({})
        self.assertEqual(instance.state, ModuleState.INITIALIZED)
        with self.assertRaises(RuntimeError) as caught:
            instance.execute(None)
        self.assertIn("service", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
