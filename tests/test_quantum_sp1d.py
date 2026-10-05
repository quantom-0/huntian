"""Numerics tests for the self-consistent Schroedinger-Poisson solver.

Verification strategy, matching the accuracy programme:

* each **sub-solver** is checked against a closed-form solution and its
  convergence order is asserted (Poisson, Schroedinger, Anderson);
* the **E_c Poisson operator** is checked against the independently verified
  phi formulation of the same discretisation;
* the **coupled Jacobian** is checked against central finite differences;
* the **coupled solve** is checked for its convergence order, physical bounds,
  seed independence, and insensitivity to spacing and domain size.

Verified against closed form:
* two-dielectric parallel plate, no sheet charge -> exact to machine precision
* same with an interface sheet charge -> second-order convergent
* infinite square well, harmonic oscillator, triangular well (Airy zeros)
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from .helpers import NewtonSolverStub, load_newton_solver, load_sp_solver

S = load_sp_solver()
N = load_newton_solver()

ELEMENTARY_CHARGE = S.ELEMENTARY_CHARGE
VACUUM_PERMITTIVITY = S.VACUUM_PERMITTIVITY
ELECTRON_MASS = S.ELECTRON_MASS

# -- demo device parameters (Al0.25Ga0.75N / GaN, 20 nm barrier) -------------
D_NM = 20.0
EXTENT_NM = 60.0
SIGMA_EFF = 0.01540689  # C/m^2, screening 0.8 of the Ambacher sheet charge
PHI_B = 0.94
DELTA_EC = 0.6055
EPS_BARRIER = 9.25
EPS_CHANNEL = 9.5
M_BARRIER = 0.225 * ELECTRON_MASS
M_CHANNEL = 0.20 * ELECTRON_MASS
TEMPERATURE = 300.0

#: The analytic Ambacher result this solver improves upon.
ANALYTIC_NS_CM2 = 7.5829e12

AIRY_ZEROS = (2.33810741, 4.08794944, 5.52055983)


def uniform_grid(
    z_min_nm: float, z_max_nm: float, nodes: int, *, m_rel: float = 0.2, eps_r: float = 9.5
) -> "S.Grid":
    """A plain uniform grid for pure-Schroedinger tests (d_m is unused there)."""
    z = np.linspace(z_min_nm * 1e-9, z_max_nm * 1e-9, nodes)
    m_kg = m_rel * ELECTRON_MASS
    return S.Grid(
        z_m=z,
        barrier_index=0,
        eps_r=np.full(nodes, eps_r),
        m_eff_kg=np.full(nodes, m_kg),
        eps_barrier=eps_r,
        eps_channel=eps_r,
        m_barrier_kg=m_kg,
        m_channel_kg=m_kg,
        d_m=0.0,
    )


def uniform_face_grid(target_h_nm: float, *, extent_nm: float = EXTENT_NM) -> "S.Grid":
    """Fully uniform grid with the interface exactly on a face and spacing set."""
    d = D_NM * 1e-9
    n_barrier = max(2, int(round(d / (target_h_nm * 1e-9) - 0.5)))
    h = 2.0 * d / (2.0 * n_barrier + 1.0)
    z_barrier = np.linspace(0.0, d - 0.5 * h, n_barrier + 1)
    n_channel = max(2, int(round(extent_nm * 1e-9 / h)))
    z_channel = np.linspace(d + 0.5 * h, d + 0.5 * h + n_channel * h, n_channel + 1)
    z = np.concatenate([z_barrier, z_channel])
    return S.Grid(
        z_m=z,
        barrier_index=n_barrier,
        eps_r=np.where(z < d, EPS_BARRIER, EPS_CHANNEL),
        m_eff_kg=np.where(z < d, M_BARRIER, M_CHANNEL),
        eps_barrier=EPS_BARRIER,
        eps_channel=EPS_CHANNEL,
        m_barrier_kg=M_BARRIER,
        m_channel_kg=M_CHANNEL,
        d_m=d,
    )


def sp_grid(
    *,
    h_fine_nm: float = 0.2,
    fine_width_nm: float = 10.0,
    extent_nm: float = EXTENT_NM,
    barrier_fine_nm: float = 3.0,
) -> "S.Grid":
    return S.build_grid(
        barrier_thickness_m=D_NM * 1e-9,
        channel_extent_m=extent_nm * 1e-9,
        h_fine_m=h_fine_nm * 1e-9,
        fine_width_m=min(fine_width_nm, extent_nm) * 1e-9,
        barrier_fine_width_m=barrier_fine_nm * 1e-9,
        h_barrier_m=1.0e-9,
        h_max_m=2.0e-9,
        growth=1.15,
        eps_r_barrier=EPS_BARRIER,
        eps_r_channel=EPS_CHANNEL,
        m_eff_barrier_kg=M_BARRIER,
        m_eff_channel_kg=M_CHANNEL,
    )


def split_sheet(grid, sigma: float) -> np.ndarray:
    """The exact flux-jump weights used by the operator (see PoissonOperator)."""
    eps_b = float(grid.eps_barrier)
    eps_c = float(grid.eps_channel)
    sheet = np.zeros(grid.size)
    sheet[grid.barrier_index] += sigma * eps_b / (eps_b + eps_c)
    sheet[grid.channel_index] += sigma * eps_c / (eps_b + eps_c)
    return sheet


def sp_operator(grid=None, *, sigma: float = SIGMA_EFF, band_offset: float = DELTA_EC):
    return S.PoissonOperator.build(
        grid if grid is not None else sp_grid(),
        sheet_charge_C_m2=sigma,
        gate_ec_eV=PHI_B,
        band_offset_eV=band_offset,
    )


def sp_solve(grid=None, *, algorithm: str = "newton", **overrides):
    kwargs = dict(
        grid=grid if grid is not None else sp_grid(),
        sigma_eff_C_m2=SIGMA_EFF,
        phi_b_V=PHI_B,
        delta_ec_eV=DELTA_EC,
        temperature_K=TEMPERATURE,
        valley_degeneracy=1,
        n_subbands=12,
        max_iterations=30,
        tolerance_eV=1e-8,
        adaptive_mixing=True,
        algorithm=algorithm,
    )
    if algorithm == "newton":
        kwargs["nonlinear_solver"] = NewtonSolverStub(tolerance=1e-8, max_iterations=60)
    else:
        kwargs["anderson_depth"] = 8
        kwargs["mixing"] = 0.5
    kwargs.update(overrides)
    return S.solve_schrodinger_poisson(**kwargs)


# ---------------------------------------------------------------------------
# Linear algebra and grid
# ---------------------------------------------------------------------------


class TridiagonalTests(unittest.TestCase):
    def test_solves_a_known_system(self) -> None:
        rng = np.random.default_rng(7)
        n = 12
        matrix = (
            np.diag(rng.uniform(2.0, 4.0, n))
            + np.diag(rng.uniform(-1.0, -0.2, n - 1), 1)
            + np.diag(rng.uniform(-1.0, -0.2, n - 1), -1)
        )
        rhs = rng.uniform(-1.0, 1.0, n)
        solution = S.solve_tridiagonal(
            np.concatenate([[0.0], np.diag(matrix, -1)]),
            np.diag(matrix).copy(),
            np.concatenate([np.diag(matrix, 1), [0.0]]),
            rhs,
        )
        self.assertLess(float(np.max(np.abs(matrix @ solution - rhs))), 1e-12)

    def test_singular_system_raises(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            S.solve_tridiagonal(
                np.zeros(3), np.array([1.0, 0.0, 1.0]), np.zeros(3), np.zeros(3)
            )

    def test_a_block_of_right_hand_sides_matches_one_at_a_time(self) -> None:
        # deflated_resolvent relies on this: one factorisation, n columns.
        rng = np.random.default_rng(11)
        n = 9
        matrix = (
            np.diag(rng.uniform(2.0, 4.0, n))
            + np.diag(rng.uniform(-1.0, -0.2, n - 1), 1)
            + np.diag(rng.uniform(-1.0, -0.2, n - 1), -1)
        )
        lower = np.concatenate([[0.0], np.diag(matrix, -1)])
        diag = np.diag(matrix).copy()
        upper = np.concatenate([np.diag(matrix, 1), [0.0]])
        block = rng.uniform(-1.0, 1.0, (n, 5))
        together = S.solve_tridiagonal(lower, diag, upper, block)
        for column in range(block.shape[1]):
            alone = S.solve_tridiagonal(lower, diag, upper, block[:, column])
            self.assertLess(float(np.max(np.abs(together[:, column] - alone))), 1e-14)
        self.assertLess(
            float(np.max(np.abs(matrix @ together - block))), 1e-12
        )


class GridTests(unittest.TestCase):
    def test_interface_lands_exactly_on_a_face(self) -> None:
        grid = sp_grid()
        half = 0.5 * float(grid.spacing_m[grid.barrier_index])
        self.assertGreater(half, 0.0)
        self.assertNotAlmostEqual(
            float(grid.z_m[grid.barrier_index]), grid.d_m, delta=1e-12
        )
        self.assertNotAlmostEqual(
            float(grid.z_m[grid.channel_index]), grid.d_m, delta=1e-12
        )
        self.assertAlmostEqual(grid.interface_position_m(), grid.d_m, places=15)
        self.assertEqual(grid.interface_face, grid.barrier_index)

    def test_axis_is_strictly_increasing(self) -> None:
        grid = sp_grid()
        self.assertTrue(np.all(np.diff(grid.z_m) > 0.0))

    def test_channels_and_barrier_are_separated(self) -> None:
        grid = sp_grid()
        self.assertTrue(np.all(grid.z_m[: grid.channel_index] < grid.d_m))
        self.assertTrue(np.all(grid.z_m[grid.channel_index :] > grid.d_m))

    def test_rejects_a_barrier_thinner_than_half_the_fine_spacing(self) -> None:
        with self.assertRaises(ValueError):
            S.build_grid(
                barrier_thickness_m=0.05e-9,
                channel_extent_m=60e-9,
                h_fine_m=0.2e-9,
                fine_width_m=10e-9,
                barrier_fine_width_m=1e-9,
                h_barrier_m=1e-9,
                h_max_m=2e-9,
                growth=1.15,
                eps_r_barrier=EPS_BARRIER,
                eps_r_channel=EPS_CHANNEL,
                m_eff_barrier_kg=M_BARRIER,
                m_eff_channel_kg=M_CHANNEL,
            )


# ---------------------------------------------------------------------------
# Poisson operator
# ---------------------------------------------------------------------------


class PoissonOperatorTests(unittest.TestCase):
    """The E_c operator must reproduce the independently verified phi solve."""

    def test_matches_the_phi_formulation(self) -> None:
        grid = sp_grid()
        operator = sp_operator(grid)
        ns = S.depletion_seed_ns(
            sigma_eff_C_m2=SIGMA_EFF,
            barrier_thickness_m=grid.d_m,
            eps_r_barrier=EPS_BARRIER,
            phi_b_V=PHI_B,
            delta_ec_eV=DELTA_EC,
            m_eff_channel_kg=M_CHANNEL,
            valley_degeneracy=1,
            gamma0_eV_m43=2.123e-12,
        )
        density = S.fang_howard_profile(grid, ns)

        from_operator = operator.potential(operator.solve(density))
        phi = S.solve_poisson(
            grid, density, split_sheet(grid, SIGMA_EFF), gate_value_V=0.0
        )
        from_phi = S.conduction_band_edge(
            phi, grid, phi_b_V=PHI_B, gate_bias_V=0.0, delta_ec_eV=DELTA_EC
        )
        self.assertLess(float(np.max(np.abs(from_operator - from_phi))), 1e-10)

    def test_residual_vanishes_on_the_operators_own_solution(self) -> None:
        grid = sp_grid()
        operator = sp_operator(grid)
        density = S.fang_howard_profile(grid, 7.5e16)
        ec = operator.solve(density)
        residual = operator.residual(ec, density)
        scale = float(np.max(np.abs(operator.dense() @ ec))) + 1e-30
        self.assertLess(float(np.max(np.abs(residual))) / scale, 1e-12)

    def test_gate_boundary_condition_is_applied(self) -> None:
        operator = sp_operator()
        ec = operator.solve(np.zeros(operator.grid.size))
        self.assertAlmostEqual(float(ec[0]), PHI_B, places=12)

    def test_band_offset_lowers_the_channel_side(self) -> None:
        grid = sp_grid()
        operator = sp_operator(grid, band_offset=DELTA_EC)
        offset = operator.offset_eV
        self.assertAlmostEqual(float(offset[grid.barrier_index]), 0.0, places=15)
        self.assertAlmostEqual(float(offset[grid.channel_index]), -DELTA_EC, places=15)
        # potential() must add exactly that step to the electrostatic part.
        probe = np.linspace(0.0, 1.0, grid.size)
        np.testing.assert_allclose(operator.potential(probe), probe + offset)

    def test_zero_band_offset_leaves_the_profile_continuous(self) -> None:
        grid = sp_grid()
        operator = sp_operator(grid, band_offset=0.0)
        np.testing.assert_allclose(operator.offset_eV, np.zeros(grid.size))
        np.testing.assert_allclose(operator.potential(np.arange(grid.size, dtype=float)),
                                   np.arange(grid.size, dtype=float))

    def test_sheet_charge_weights_are_the_exact_flux_jump_weights(self) -> None:
        grid = sp_grid()
        operator = sp_operator(grid)
        self.assertAlmostEqual(float(operator.sheet.sum()), SIGMA_EFF, places=15)
        eps_b, eps_c = EPS_BARRIER, EPS_CHANNEL
        self.assertAlmostEqual(
            float(operator.sheet[grid.barrier_index]),
            SIGMA_EFF * eps_b / (eps_b + eps_c),
            places=16,
        )


class PoissonAnalyticTests(unittest.TestCase):
    """A two-dielectric parallel plate, with and without an interface sheet."""

    L_NM = 100.0
    V0, VL = 0.0, -0.5

    def _plate_grid(self, h_nm: float) -> "S.Grid":
        # The channel segment starts half a spacing past the interface, so trim
        # the extent to put the last node exactly on the far Dirichlet face.
        d = D_NM * 1e-9
        extent = (self.L_NM - D_NM) * 1e-9 - 0.5 * h_nm * 1e-9
        grid = S.build_grid(
            barrier_thickness_m=d,
            channel_extent_m=extent,
            h_fine_m=h_nm * 1e-9,
            fine_width_m=extent,
            barrier_fine_width_m=d,
            h_barrier_m=h_nm * 1e-9,
            h_max_m=h_nm * 1e-9,
            growth=1.0,
            eps_r_barrier=EPS_BARRIER,
            eps_r_channel=EPS_CHANNEL,
            m_eff_barrier_kg=M_BARRIER,
            m_eff_channel_kg=M_CHANNEL,
        )
        self.assertAlmostEqual(float(grid.z_m[-1]), self.L_NM * 1e-9, places=15)
        return grid

    def _exact(self, grid, sigma: float) -> np.ndarray:
        d = grid.d_m
        length = self.L_NM * 1e-9
        e1 = (self.V0 - self.VL - sigma * (length - d) / (EPS_CHANNEL * VACUUM_PERMITTIVITY)) / (
            d + EPS_BARRIER * (length - d) / EPS_CHANNEL
        )
        e2 = (sigma + EPS_BARRIER * VACUUM_PERMITTIVITY * e1) / (
            EPS_CHANNEL * VACUUM_PERMITTIVITY
        )
        phi_d = self.V0 - e1 * d
        return np.where(grid.z_m < d, self.V0 - e1 * grid.z_m, phi_d - e2 * (grid.z_m - d))

    def test_without_a_sheet_charge_the_solution_is_exact(self) -> None:
        # With no sheet the displacement is continuous, and the length-weighted
        # series face permittivity makes the finite-volume scheme reproduce the
        # piecewise-linear solution to round-off.
        for h_nm in (2.0, 1.0, 0.5):
            grid = self._plate_grid(h_nm)
            phi = S.solve_poisson(
                grid,
                np.zeros(grid.size),
                np.zeros(grid.size),
                gate_value_V=self.V0,
                back_bc="dirichlet",
                back_value_V=self.VL,
            )
            error = float(np.max(np.abs(phi - self._exact(grid, 0.0))))
            self.assertLess(error, 1e-10, msg=f"h={h_nm} nm, error={error}")

    def test_with_a_sheet_charge_the_solution_is_exact(self) -> None:
        # Eliminating the interface potential between the two one-sided flux
        # relations moves the sheet into the constant vector with the exact
        # permittivity weights, and the scheme reproduces the piecewise-linear
        # solution to round-off - with or without a sheet.
        sigma = 0.01
        for h_nm in (2.0, 1.0, 0.5):
            grid = self._plate_grid(h_nm)
            phi = S.solve_poisson(
                grid,
                np.zeros(grid.size),
                split_sheet(grid, sigma),
                gate_value_V=self.V0,
                back_bc="dirichlet",
                back_value_V=self.VL,
            )
            error = float(np.max(np.abs(phi - self._exact(grid, sigma))))
            self.assertLess(error, 1e-10, msg=f"h={h_nm} nm, error={error}")

    def test_neumann_back_face_has_zero_field(self) -> None:
        grid = sp_grid()
        phi = S.solve_poisson(
            grid, np.zeros(grid.size), split_sheet(grid, SIGMA_EFF), gate_value_V=0.0
        )
        field = (phi[-1] - phi[-2]) / (grid.z_m[-1] - grid.z_m[-2])
        self.assertLess(abs(field), 1e-6)

    def test_sheet_charge_raises_the_interface_potential(self) -> None:
        grid = sp_grid()
        phi = S.solve_poisson(
            grid, np.zeros(grid.size), split_sheet(grid, SIGMA_EFF), gate_value_V=0.0
        )
        self.assertGreater(float(phi[grid.barrier_index]), 0.0)
        self.assertGreater(float(phi[grid.barrier_index]), float(phi[1]))


# ---------------------------------------------------------------------------
# Schroedinger vs closed form
# ---------------------------------------------------------------------------


class SchrodingerAnalyticTests(unittest.TestCase):
    def test_infinite_square_well(self) -> None:
        length = 10e-9
        grid = uniform_grid(-length / 2 * 1e9, length / 2 * 1e9, 401, m_rel=1.0)
        energies, _psi = S.solve_schrodinger(grid, np.zeros(grid.size), n_states=3)
        base = np.pi**2 * S.HBAR2_OVER_2M0_EV_M2 / length**2
        self.assertAlmostEqual(float(energies[0] / base), 1.0, places=3)
        self.assertAlmostEqual(float(energies[1] / base), 4.0, places=3)
        self.assertAlmostEqual(float(energies[2] / base), 9.0, places=2)

    def test_infinite_square_well_converges_second_order(self) -> None:
        length = 10e-9
        base = np.pi**2 * S.HBAR2_OVER_2M0_EV_M2 / length**2
        errors = []
        for nodes in (101, 201, 401):
            grid = uniform_grid(-length / 2 * 1e9, length / 2 * 1e9, nodes, m_rel=1.0)
            energies, _ = S.solve_schrodinger(grid, np.zeros(grid.size), n_states=1)
            errors.append(abs(float(energies[0]) - base))
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            self.assertGreater(ratio, 3.0, msg=f"observed order too low: {errors}")
            self.assertLess(ratio, 5.0, msg=f"observed order too high: {errors}")

    def test_harmonic_oscillator(self) -> None:
        hbar_omega = 0.1
        omega = hbar_omega * ELEMENTARY_CHARGE / S.REDUCED_PLANCK
        mass = 0.2 * ELECTRON_MASS
        grid = uniform_grid(-10.0, 10.0, 401, m_rel=0.2)
        potential = 0.5 * mass * omega**2 * grid.z_m**2 / ELEMENTARY_CHARGE
        energies, _psi = S.solve_schrodinger(grid, potential, n_states=3)
        for index, level in enumerate((0.05, 0.15, 0.25)):
            self.assertAlmostEqual(
                float(energies[index]), level, delta=1e-3 * max(level, 0.05)
            )

    def test_triangular_well_matches_the_airy_zeros(self) -> None:
        field_eV_per_m = 1.0e7
        scale = (S.HBAR2_OVER_2M0_EV_M2 * (1.0 / 0.2) * field_eV_per_m**2) ** (1.0 / 3.0)
        grid = uniform_grid(0.0, 40.0, 401, m_rel=0.2)
        energies, _psi = S.solve_schrodinger(grid, field_eV_per_m * grid.z_m, n_states=3)
        for index, zeta in enumerate(AIRY_ZEROS):
            expected = zeta * scale
            self.assertLess(
                abs(float(energies[index]) - expected) / expected,
                5e-3,
                msg=f"E{index}: {energies[index]} vs {expected}",
            )

    def test_triangular_well_converges_second_order(self) -> None:
        field_eV_per_m = 1.0e7
        scale = (S.HBAR2_OVER_2M0_EV_M2 * (1.0 / 0.2) * field_eV_per_m**2) ** (1.0 / 3.0)
        expected = AIRY_ZEROS[0] * scale
        errors = []
        for nodes in (401, 801, 1601):
            grid = uniform_grid(0.0, 40.0, nodes, m_rel=0.2)
            energies, _ = S.solve_schrodinger(grid, field_eV_per_m * grid.z_m, n_states=1)
            errors.append(abs(float(energies[0]) - expected))
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            self.assertGreater(ratio, 3.0, msg=f"observed order too low: {errors}")
            self.assertLess(ratio, 5.0, msg=f"observed order too high: {errors}")

    def test_wavefunctions_are_normalised(self) -> None:
        grid = sp_grid()
        _energies, psi = S.solve_schrodinger(grid, np.zeros(grid.size), n_states=4)
        widths = S.cell_widths(grid)
        for index in range(psi.shape[1]):
            self.assertAlmostEqual(
                float((psi[:, index] ** 2 * widths).sum()), 1.0, places=10
            )

    def test_wavefunctions_are_orthogonal(self) -> None:
        grid = sp_grid()
        _energies, psi = S.solve_schrodinger(grid, np.zeros(grid.size), n_states=4)
        widths = S.cell_widths(grid)
        gram = (psi * widths[:, None]).T @ psi
        off_diagonal = gram - np.diag(np.diag(gram))
        self.assertLess(float(np.max(np.abs(off_diagonal))), 1e-10)

    def test_mass_step_keeps_the_solver_well_behaved(self) -> None:
        nodes = 401
        z = np.linspace(0.0, 20e-9, nodes)
        m1, m2 = 0.2 * ELECTRON_MASS, 0.4 * ELECTRON_MASS
        step = nodes // 2
        grid = S.Grid(
            z_m=z,
            barrier_index=step,
            eps_r=np.full(nodes, 9.5),
            m_eff_kg=np.where(np.arange(nodes) < step, m1, m2),
            eps_barrier=9.5,
            eps_channel=9.5,
            m_barrier_kg=m1,
            m_channel_kg=m2,
            d_m=z[step],
        )
        potential = np.where(np.abs(z - z[step]) < 3e-9, -0.4, 0.5)
        energies, psi = S.solve_schrodinger(grid, potential, n_states=2)
        widths = S.cell_widths(grid)
        self.assertTrue(np.all(np.diff(energies) > 0.0))
        for index in range(2):
            self.assertAlmostEqual(
                float((psi[:, index] ** 2 * widths).sum()), 1.0, places=10
            )
        masses = S.subband_masses(grid, psi)
        for mass in masses:
            self.assertGreaterEqual(float(mass), min(m1, m2) - 1e-30)
            self.assertLessEqual(float(mass), max(m1, m2) + 1e-30)

    def test_subband_mass_of_a_confined_state_is_the_well_mass(self) -> None:
        nodes = 401
        z = np.linspace(0.0, 20e-9, nodes)
        heavy = 0.4 * ELECTRON_MASS
        grid = uniform_grid(0.0, 20.0, nodes, m_rel=0.4)
        potential = np.where(np.abs(z - 2e-9) < 2e-9, -0.5, 1.0)
        _energies, psi = S.solve_schrodinger(grid, potential, n_states=1)
        mass = float(S.subband_masses(grid, psi)[0])
        self.assertLess(abs(mass - heavy) / heavy, 1e-6)

    def test_energies_are_sorted(self) -> None:
        grid = sp_grid()
        energies, _ = S.solve_schrodinger(grid, np.zeros(grid.size), n_states=6)
        self.assertTrue(np.all(np.diff(energies) > 0.0))

    def test_rejects_a_grid_that_is_too_small(self) -> None:
        grid = uniform_grid(0.0, 1.0, 4)
        with self.assertRaises(ValueError):
            S.solve_schrodinger(grid, np.zeros(4), n_states=2)


# ---------------------------------------------------------------------------
# Anderson acceleration
# ---------------------------------------------------------------------------


class AndersonMixerTests(unittest.TestCase):
    @staticmethod
    def _linear_map(coefficients):
        matrix = np.diag(coefficients)
        offset = np.arange(1.0, len(coefficients) + 1)
        exact = np.linalg.solve(np.eye(len(coefficients)) - matrix, offset)
        return matrix, offset, exact

    def test_solves_an_expansive_linear_map_quickly(self) -> None:
        matrix, offset, exact = self._linear_map([-5.0, -3.0, -8.0])
        mixer = S.AndersonMixer(depth=4, beta=1.0)
        x = np.zeros(3)
        for _ in range(40):
            x = mixer.step(x, matrix @ x + offset)
            if float(np.max(np.abs(x - exact))) < 1e-12:
                break
        self.assertLess(float(np.max(np.abs(x - exact))), 1e-10)

    def test_plain_mixing_diverges_where_anderson_converges(self) -> None:
        matrix, offset, exact = self._linear_map([-5.0, -3.0, -8.0])
        plain = np.zeros(3)
        for _ in range(60):
            plain = plain + 0.3 * ((matrix @ plain + offset) - plain)
        self.assertGreater(float(np.max(np.abs(plain - exact))), 1.0)

    def test_depth_zero_is_plain_damped_mixing(self) -> None:
        mixer = S.AndersonMixer(depth=0, beta=0.5)
        x = np.array([1.0, 2.0])
        gx = np.array([3.0, 4.0])
        np.testing.assert_allclose(mixer.step(x, gx), x + 0.5 * (gx - x))

    def test_singular_history_falls_back_without_nan(self) -> None:
        mixer = S.AndersonMixer(depth=4, beta=0.5)
        x = np.zeros(3)
        for _ in range(6):
            x = mixer.step(x, np.array([1.0, 1.0, 1.0]))
        self.assertTrue(np.all(np.isfinite(x)))


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


class DepletionSeedTests(unittest.TestCase):
    def _seed(self, **overrides):
        kwargs = dict(
            sigma_eff_C_m2=SIGMA_EFF,
            barrier_thickness_m=D_NM * 1e-9,
            eps_r_barrier=EPS_BARRIER,
            phi_b_V=PHI_B,
            delta_ec_eV=DELTA_EC,
            m_eff_channel_kg=M_CHANNEL,
            valley_degeneracy=1,
            gamma0_eV_m43=2.123e-12,
        )
        kwargs.update(overrides)
        return S.depletion_seed_ns(**kwargs)

    def test_lands_near_the_analytic_answer(self) -> None:
        seed_cm2 = self._seed() * 1e-4
        self.assertLess(abs(seed_cm2 - ANALYTIC_NS_CM2) / ANALYTIC_NS_CM2, 0.05)

    def test_is_below_the_polarization_upper_bound(self) -> None:
        upper = SIGMA_EFF / ELEMENTARY_CHARGE
        self.assertGreater(self._seed(), 0.0)
        self.assertLess(self._seed(), upper)

    def test_no_polarization_gives_zero(self) -> None:
        self.assertEqual(self._seed(sigma_eff_C_m2=0.0), 0.0)

    def test_more_screening_gives_more_charge(self) -> None:
        low = self._seed(sigma_eff_C_m2=SIGMA_EFF * 0.5)
        high = self._seed(sigma_eff_C_m2=SIGMA_EFF * 1.0)
        self.assertLess(low, high)


# ---------------------------------------------------------------------------
# Coupled self-consistent solve
# ---------------------------------------------------------------------------


class SelfConsistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = sp_solve()

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)
        self.assertLess(self.result.iterations, 30)

    def test_sheet_density_is_physically_plausible(self) -> None:
        ns = self.result.ns_cm2
        self.assertGreater(ns, 4.0e12)
        self.assertLess(ns, 1.2e13)

    def test_sheet_density_is_below_the_polarization_upper_bound(self) -> None:
        self.assertLess(self.result.ns_m2, self.result.ns_upper_m2)
        self.assertGreater(self.result.ns_m2 / self.result.ns_upper_m2, 0.5)

    def test_agrees_with_the_analytic_estimate_within_the_quantum_correction(self) -> None:
        deviation = abs(self.result.ns_cm2 - ANALYTIC_NS_CM2) / ANALYTIC_NS_CM2
        self.assertLess(deviation, 0.20)
        self.assertGreater(deviation, 0.001, msg="not merely the analytic model")

    def test_subbands_are_ordered_and_physically_placed(self) -> None:
        offsets = self.result.subband_offsets_eV
        self.assertTrue(np.all(np.diff(offsets) > 0.0))
        self.assertGreater(float(offsets[0]), 0.0)
        self.assertLess(float(self.result.energies_eV[0]), 0.0)

    def test_fermi_level_is_above_the_ground_subband(self) -> None:
        self.assertGreater(self.result.ef_eV, float(self.result.subband_offsets_eV[0]))

    def test_occupation_is_monotonic_across_subbands(self) -> None:
        densities = self.result.ns_per_subband_m2
        self.assertTrue(np.all(densities >= 0.0))
        self.assertTrue(np.all(np.diff(densities) < 0.0))

    def test_sheet_density_equals_the_sum_of_subband_occupations(self) -> None:
        self.assertAlmostEqual(
            self.result.ns_m2, float(self.result.ns_per_subband_m2.sum()), places=6
        )

    def test_volume_density_integrates_to_the_sheet_density(self) -> None:
        widths = S.cell_widths(self.result.grid)
        integrated = float((self.result.n_profile_m3 * widths).sum())
        self.assertAlmostEqual(integrated / self.result.ns_m2, 1.0, places=8)

    def test_charge_balance_reproduces_the_interface_band_edge(self) -> None:
        # The reported well bottom is the first channel node at d + h/2, so it
        # sits one half-spacing of band bending below the interface value that
        # the charge-control relation predicts.
        q_ns = self.result.ns_m2 * ELEMENTARY_CHARGE
        barrier_drop = (SIGMA_EFF - q_ns) * self.result.grid.d_m / (
            EPS_BARRIER * VACUUM_PERMITTIVITY
        )
        predicted = -barrier_drop + PHI_B - DELTA_EC
        self.assertAlmostEqual(self.result.ec_well_bottom_eV, predicted, delta=0.05)

    def test_wavefunction_is_confined_near_the_interface(self) -> None:
        self.assertGreater(self.result.centroid_nm, 0.0)
        self.assertLess(self.result.centroid_nm, 5.0)
        self.assertLess(self.result.rms_width_nm, 3.0)

    def test_zero_iterations_reports_non_convergence(self) -> None:
        result = sp_solve(grid=sp_grid(h_fine_nm=0.4, fine_width_nm=6.0), max_iterations=1)
        self.assertFalse(result.converged)


class ConvergenceOrderTests(unittest.TestCase):
    """The headline result of the face-based interface: second order in h."""

    def test_coupled_solve_is_second_order_in_the_vertical_spacing(self) -> None:
        values = []
        spacings = []
        for target in (0.4, 0.2, 0.1):
            grid = uniform_face_grid(target)
            result = sp_solve(grid)
            self.assertTrue(result.converged)
            values.append(result.ns_cm2)
            spacings.append(float(np.mean(np.diff(grid.z_m))))

        # Richardson: with three levels, p = log(|u1-u2| / |u2-u3|) / log(h1/h2)
        finer = abs(values[0] - values[1])
        finest = abs(values[1] - values[2])
        order = np.log(finer / finest) / np.log(spacings[0] / spacings[1])
        self.assertGreater(
            order, 1.7, msg=f"observed order {order:.3f} (values {values}) - expected ~2"
        )

    def test_domain_size_does_not_change_the_answer(self) -> None:
        # Proves the box-continuum states are a legitimate quadrature of the 3D
        # continuum, not a domain artefact.
        short = sp_solve(
            sp_grid(h_fine_nm=0.2, fine_width_nm=10.0, extent_nm=30.0),
            tolerance_eV=1e-6,
            max_iterations=150,
        )
        long = sp_solve(
            sp_grid(h_fine_nm=0.2, fine_width_nm=10.0, extent_nm=120.0),
            tolerance_eV=1e-6,
            max_iterations=150,
        )
        self.assertTrue(short.converged, msg=short.solver_message)
        self.assertTrue(long.converged, msg=long.solver_message)
        self.assertLess(abs(short.ns_cm2 - long.ns_cm2) / long.ns_cm2, 1e-3)


def sp_start_profile(grid) -> np.ndarray:
    """The band-edge profile the Jacobian tests linearise about (the warm start)."""
    operator = sp_operator(grid)
    ns = S.depletion_seed_ns(
        sigma_eff_C_m2=SIGMA_EFF,
        barrier_thickness_m=grid.d_m,
        eps_r_barrier=EPS_BARRIER,
        phi_b_V=PHI_B,
        delta_ec_eV=DELTA_EC,
        m_eff_channel_kg=M_CHANNEL,
        valley_degeneracy=1,
        gamma0_eV_m43=2.123e-12,
    )
    return operator.solve(S.fang_howard_profile(grid, ns))


def sp_start_state(grid, *, n_subbands: int = 12) -> "S.QuantumState":
    """A populated state at that profile - a realistic linearisation point."""
    operator = sp_operator(grid)
    return S.quantum_state(
        grid,
        operator.potential(sp_start_profile(grid)),
        n_subbands=n_subbands,
        temperature_K=TEMPERATURE,
        valley_degeneracy=1,
        ns_upper_m2=None,
        density_clamp=False,
    )


def local_convergence_orders(history, *, floor: float = 1e-15) -> list[float]:
    """Local orders ``p`` from consecutive residual triples: ``e3 ~ e2**p``.

    Only triples with a *falling* ratio - the superlinear signature, since for a
    quadratic method the ratio itself shrinks in proportion to the residual -
    and with the smallest member still above the round-off floor (``1e-15``, the
    double-precision limit for residuals of order one) are used. Below the floor
    the ratios are noise, which is why the *order* is asserted here rather than a
    ratio threshold.
    """
    values = [v for v in history if v > 0.0]
    orders: list[float] = []
    for first, second, third in zip(values, values[1:], values[2:]):
        if not (second < first and third < second) or third <= floor:
            continue
        first_ratio = second / first
        second_ratio = third / second
        if first_ratio >= 0.5 or second_ratio >= first_ratio:
            continue
        orders.append(float(np.log(second / third) / np.log(first / second)))
    return orders


class ResolventTests(unittest.TestCase):
    """The rank-1 deflation must reproduce the eigenvector sum exactly."""

    def _bands(self):
        grid = sp_grid()
        operator = sp_operator(grid)
        state = sp_start_state(grid)
        interior, widths, diagonal, off = S.schrodinger_bands(
            grid, state.potential_eV
        )
        vectors = state.psi[interior, :] * np.sqrt(widths)[:, None]
        return diagonal, off, vectors, state.energies_eV

    def test_deflated_resolvent_matches_the_eigenvector_sum(self) -> None:
        diagonal, off, vectors, energies = self._bands()
        values, full = np.linalg.eigh(S._tridiagonal_matrix(diagonal, off))
        for index in range(energies.size):
            weights = np.zeros(values.size)
            other = np.arange(values.size) != index
            weights[other] = 1.0 / (values[other] - energies[index])
            reference = full @ (weights[:, None] * full.T)
            deflated = S.deflated_resolvent(
                diagonal, off, energies[index], vectors[:, index]
            )
            relative = float(np.max(np.abs(deflated - reference))) / float(
                np.max(np.abs(reference))
            )
            self.assertLess(relative, 1e-9, msg=f"state {index}: {relative:.3e}")

    def test_deflated_resolvent_is_the_pseudo_inverse(self) -> None:
        diagonal, off, vectors, energies = self._bands()
        for index in range(energies.size):
            vector = vectors[:, index]
            resolvent = S.deflated_resolvent(diagonal, off, energies[index], vector)
            n = diagonal.size
            residual = (diagonal - energies[index])[:, None] * resolvent
            if off.size:
                residual[:-1, :] += off[:, None] * resolvent[1:, :]
                residual[1:, :] += off[:, None] * resolvent[:-1, :]
            # (A - E) G = I - v v^T and v^T G = 0 are the definition.
            expected = np.eye(n) - np.outer(vector, vector)
            self.assertLess(float(np.max(np.abs(residual - expected))), 1e-9)
            self.assertLess(float(np.max(np.abs(vector @ resolvent))), 1e-9)


class JacobianTests(unittest.TestCase):
    """The Jacobian must be the exact derivative of the residual."""

    def _system(self, grid, *, response: bool = True, n_subbands: int = 12):
        return S.SchrodingerPoissonSystem(
            sp_operator(grid),
            n_subbands=n_subbands,
            temperature_K=TEMPERATURE,
            valley_degeneracy=1,
            density_clamp=False,
            ns_upper_m2=SIGMA_EFF / ELEMENTARY_CHARGE,
            include_wavefunction_response=response,
        )

    def _start(self, grid):
        return sp_start_profile(grid)

    def test_jacobian_matches_central_differences(self) -> None:
        grid = uniform_face_grid(0.4, extent_nm=20.0)
        system = self._system(grid)
        x = self._start(grid)
        columns = [grid.channel_index + 2, grid.channel_index + 6, grid.channel_index + 12]
        check = N.check_jacobian(system.residual, system.jacobian, x, columns=columns)
        # Exact to the noise of the difference formula. This is a real assertion,
        # not a loose one: dropping the d m*/dE_c term leaves a floor of ~1.3e-5
        # that no choice of step removes.
        self.assertLess(check.max_relative_error, 1e-8, msg=str(check))

    def test_response_matches_central_differences_of_the_density(self) -> None:
        # Checked at the level of n[E_c] itself, so a mistake in the node-space
        # measure or in the charge weight cannot hide behind the operator.
        grid = uniform_face_grid(0.4, extent_nm=20.0)
        operator = sp_operator(grid)
        x = self._start(grid)
        step = 1e-6
        for node in (grid.channel_index + 2, grid.channel_index + 8):
            state = sp_start_state(grid)
            analytic = S.quantum_response_jacobian(
                grid, state, temperature_K=TEMPERATURE, valley_degeneracy=1
            )[:, node]
            perturbation = np.zeros_like(x)
            perturbation[node] = step

            def density(profile):
                return S.quantum_state(
                    grid,
                    operator.potential(profile),
                    n_subbands=12,
                    temperature_K=TEMPERATURE,
                    valley_degeneracy=1,
                    ns_upper_m2=None,
                    density_clamp=False,
                ).n_volume_m3

            numeric = (density(x + perturbation) - density(x - perturbation)) / (2 * step)
            scale = max(float(np.max(np.abs(numeric))), 1e-300)
            self.assertLess(
                float(np.max(np.abs(numeric - analytic))) / scale,
                1e-7,
                msg=f"node {node}",
            )

    def test_the_response_no_longer_depends_on_the_subband_count(self) -> None:
        # Truncation used to be the reason more subbands helped. With the exact
        # resolvent the response is a property of the operator, not of the
        # carried set, so four and sixteen subbands must be equally exact.
        grid = uniform_face_grid(0.4, extent_nm=20.0)
        x = self._start(grid)
        errors = []
        for n_subbands in (4, 16):
            system = self._system(grid, n_subbands=n_subbands)
            check = N.check_jacobian(
                system.residual,
                system.jacobian,
                x,
                columns=[grid.channel_index + 4],
            )
            errors.append(check.max_relative_error)
        for error in errors:
            self.assertLess(error, 1e-8, msg=f"subband-dependent errors {errors}")

    def test_production_and_reference_responses_agree(self) -> None:
        # Two independent ways of obtaining (A - E_i)^dagger: the rank-1
        # deflation and the full eigenvector sum. Same assembly, different
        # resolvent, so a disagreement isolates the deflation.
        grid = sp_grid()
        state = sp_start_state(grid)
        production = S.quantum_response_jacobian(
            grid, state, temperature_K=TEMPERATURE, valley_degeneracy=1
        )
        reference = S.quantum_response_jacobian_reference(
            grid, state, temperature_K=TEMPERATURE, valley_degeneracy=1
        )
        scale = float(np.max(np.abs(reference)))
        self.assertLess(
            float(np.max(np.abs(production - reference))) / scale, 1e-9
        )

    def test_a_state_from_another_operator_is_rejected(self) -> None:
        # The deflation assumes the null vector really annihilates A - E. A
        # foreign state must fail loudly instead of returning a plausible but
        # meaningless Jacobian.
        grid = sp_grid()
        state = sp_start_state(grid)
        foreign = dataclasses.replace(
            state, potential_eV=state.potential_eV + 0.05
        )
        with self.assertRaises(ValueError):
            S.quantum_response_jacobian(
                grid, foreign, temperature_K=TEMPERATURE, valley_degeneracy=1
            )

    def test_a_state_without_its_potential_is_rejected(self) -> None:
        grid = sp_grid()
        state = dataclasses.replace(sp_start_state(grid), potential_eV=None)
        with self.assertRaises(ValueError):
            S.quantum_response_jacobian(
                grid, state, temperature_K=TEMPERATURE, valley_degeneracy=1
            )

    def test_rank_only_jacobian_is_more_approximate(self) -> None:
        grid = uniform_face_grid(0.4, extent_nm=20.0)
        x = self._start(grid)
        columns = [grid.channel_index + 2, grid.channel_index + 6]
        full = self._system(grid)
        rank = self._system(grid, response=False)
        exact_err = N.check_jacobian(full.residual, full.jacobian, x, columns=columns)
        rank_err = N.check_jacobian(rank.residual, rank.jacobian, x, columns=columns)
        self.assertLessEqual(
            exact_err.max_relative_error, rank_err.max_relative_error
        )


class NewtonAlgorithmTests(unittest.TestCase):
    """Newton must beat the fixed point and land on the same answer."""

    @classmethod
    def setUpClass(cls) -> None:
        grid = sp_grid(h_fine_nm=0.2, fine_width_nm=10.0)
        cls.newton = sp_solve(grid, algorithm="newton")
        # The Anderson fallback is slower and less robust; give it both a large
        # iteration budget and the density guard, which is what that guard is for.
        cls.anderson = sp_solve(
            grid, algorithm="anderson", max_iterations=800, density_clamp=True
        )

    def test_newton_converges(self) -> None:
        self.assertTrue(self.newton.converged, msg=self.newton.solver_message)

    def test_anderson_converges_with_its_guard_enabled(self) -> None:
        self.assertTrue(self.anderson.converged, msg=self.anderson.solver_message)

    def test_they_agree_closely(self) -> None:
        # The guard makes Anderson land slightly inside the true answer, so the
        # agreement is at the sub-percent level rather than at the tolerance.
        relative = abs(self.newton.ns_cm2 - self.anderson.ns_cm2) / self.newton.ns_cm2
        self.assertLess(relative, 5e-3)

    def test_newton_needs_far_fewer_iterations(self) -> None:
        self.assertLess(self.newton.iterations * 5, self.anderson.iterations)

    def test_newton_converges_superlinearly(self) -> None:
        history = [v for v in self.newton.residual_history if v > 0.0]
        self.assertGreater(len(history), 4)
        # The final decade of the residual must fall faster than linearly.
        tail = history[-4:]
        ratios = [tail[i + 1] / tail[i] for i in range(len(tail) - 1)]
        self.assertLess(min(ratios), 0.5, msg=f"residual ratios {ratios}")

    def test_newton_converges_quadratically(self) -> None:
        # The headline payoff of the exact Jacobian: the local order estimated
        # from three consecutive residuals is 2, where the truncated
        # wavefunction response could only reach about 1.5.
        orders = local_convergence_orders(self.newton.residual_history)
        self.assertTrue(orders, msg="no asymptotic window in the residual history")
        self.assertGreater(
            max(orders), 1.9, msg=f"local orders {[round(o, 3) for o in orders]}"
        )

    def test_the_exact_jacobian_reaches_round_off(self) -> None:
        # An inconsistent Jacobian leaves a linear tail: the truncated response
        # stalled at ~5e-13, the exact one reaches the representation floor.
        smallest = min(v for v in self.newton.residual_history if v > 0.0)
        self.assertLess(smallest, 1e-13, msg=f"smallest residual {smallest:.3e}")

    def test_newton_records_the_algorithm(self) -> None:
        self.assertEqual(self.newton.algorithm, "newton")
        self.assertEqual(self.anderson.algorithm, "anderson")
        self.assertTrue(self.newton.solver_message)

    def test_newton_without_a_solver_module_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            sp_solve(grid=sp_grid(h_fine_nm=0.4, fine_width_nm=6.0), nonlinear_solver=None)

    def test_unknown_algorithm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sp_solve(algorithm="magic")


class InvarianceTests(unittest.TestCase):
    """The answer must not depend on the warm start or the grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.coarse = sp_grid(h_fine_nm=0.2, fine_width_nm=10.0)

    def test_warm_start_does_not_change_the_answer(self) -> None:
        upper = SIGMA_EFF / ELEMENTARY_CHARGE
        low = sp_solve(self.coarse, seed_ns_m2=0.35 * upper)
        high = sp_solve(self.coarse, seed_ns_m2=0.95 * upper)
        self.assertTrue(low.converged and high.converged)
        self.assertLess(abs(low.ns_cm2 - high.ns_cm2) / low.ns_cm2, 5e-3)

    def test_vertical_spacing_refinement_stays_within_tolerance(self) -> None:
        coarse = sp_solve(sp_grid(h_fine_nm=0.2, fine_width_nm=10.0))
        fine = sp_solve(sp_grid(h_fine_nm=0.1, fine_width_nm=12.0))
        self.assertTrue(coarse.converged and fine.converged)
        self.assertLess(abs(coarse.ns_cm2 - fine.ns_cm2) / fine.ns_cm2, 1.0e-2)

    def test_temperature_changes_the_occupation_as_expected(self) -> None:
        cold = sp_solve(sp_grid(h_fine_nm=0.3, fine_width_nm=8.0), temperature_K=77.0)
        warm = sp_solve(sp_grid(h_fine_nm=0.3, fine_width_nm=8.0), temperature_K=300.0)
        self.assertTrue(cold.converged and warm.converged)
        self.assertLess(cold.ns_cm2, warm.ns_cm2 * 1.001)
        self.assertGreater(cold.ns_cm2, 0.9 * warm.ns_cm2)


class GateBiasTests(unittest.TestCase):
    def test_negative_gate_bias_depletes_the_channel(self) -> None:
        grid = sp_grid(h_fine_nm=0.3, fine_width_nm=8.0)
        flat = sp_solve(grid, gate_bias_V=0.0)
        depleted = sp_solve(grid, gate_bias_V=-4.0)
        self.assertTrue(flat.converged and depleted.converged)
        self.assertLess(depleted.ns_cm2, flat.ns_cm2)
        self.assertGreaterEqual(depleted.ns_cm2, 0.0)

    def test_forward_bias_does_not_reduce_the_channel_charge(self) -> None:
        grid = sp_grid(h_fine_nm=0.3, fine_width_nm=8.0)
        flat = sp_solve(grid, gate_bias_V=0.0)
        forward = sp_solve(grid, gate_bias_V=0.5)
        self.assertTrue(forward.converged)
        self.assertGreaterEqual(forward.ns_cm2, flat.ns_cm2 * 0.999)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
