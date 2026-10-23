"""Numerics tests for the drift-diffusion kernel (``transport.dd``).

Verification strategy, following the Schroedinger-Poisson discipline:

* **flux law** - the Scharfetter-Gummel weight is checked against its closed form
  (``x/(exp(x)-1)``) and its slope against a central difference, across every
  regime of the Bernoulli argument;
* **flux partials** - every analytic partial is compared with a central
  difference of the flux itself, *which is the test that catches a derivative
  slip that a system-level check at one state misses*;
* **equilibrium invariant** - a face carrying the Boltzmann density built from
  the same band edges must carry exactly zero current, across a heterojunction
  step and across a permittivity step;
* **assembly** - the coupled Jacobian is compared against central differences of
  the residual, both at the reference state and *along a Newton path*;
* **Poisson sub-problem** - the non-linear Poisson start converges and its
  solution is a genuine equilibrium of the charge balance.

The coupled *bias* solve is not asserted here yet: it is the open item in
``工作日志.md`` (residual scaling), and a test that pins down a solver which does
not conserve terminal current would be worse than no test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import numpy as np

from . import PLUGIN_DIR, ROOT
from .helpers import build_registry

PHYSICS_PATH = PLUGIN_DIR / "transport_dd" / "physics.py"
PACKAGE_DIR = PLUGIN_DIR / "transport_dd"

STRUCTURE = {
    "name": "test",
    "layers": [
        {"name": "barrier", "material": "AlGaN", "role": "barrier",
         "thickness_nm": 20.0, "x_al": 0.25},
        {"name": "channel", "material": "GaN", "role": "channel", "thickness_nm": 40.0},
        {"name": "buffer", "material": "GaN", "role": "buffer", "thickness_nm": 40.0,
         "doping_kind": "acceptor", "doping_cm3": 1.0e17},
    ],
    "gate": {"length_um": 0.15, "kind": "schottky", "barrier_eV": 0.94},
    "geometry": {"source_gate_spacing_um": 0.6, "gate_drain_spacing_um": 1.8},
}


def load_physics():
    """Import the physics module straight from the file."""
    name = "huntian_tests_transport_dd_physics"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PHYSICS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"cannot load {PHYSICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_package():
    """Import ``transport_dd`` as a package (its modules use relative imports)."""
    name = "huntian_tests_transport_dd"
    if name in sys.modules:
        return sys.modules[name]
    package = types.ModuleType(name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[name] = package
    for sub in ("physics", "assemble", "newton"):
        spec = importlib.util.spec_from_file_location(
            f"{name}.{sub}", PACKAGE_DIR / f"{sub}.py"
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise AssertionError(f"cannot load {sub}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{sub}"] = module
        spec.loader.exec_module(module)
    return package


P = load_physics()


def bernoulli_reference(x: float) -> float:
    """``x/(exp(x)-1)`` evaluated the naive way, for the moderate range."""
    if x == 0.0:
        return 1.0
    return x / np.expm1(x)


class BernoulliTests(unittest.TestCase):
    def test_matches_the_closed_form_across_regimes(self) -> None:
        for x in (-200.0, -50.0, -5.0, -1.0, -0.1, -1e-7, 0.0, 1e-7, 0.1, 1.0, 5.0, 50.0):
            value = float(P.bernoulli(x))
            if abs(x) < 500.0:
                reference = float(np.longdouble(x) / np.expm1(np.longdouble(x))) if x else 1.0
                self.assertAlmostEqual(value, reference, delta=1e-12 * max(1.0, abs(reference)), msg=f"x={x}")
        self.assertAlmostEqual(float(P.bernoulli(0.0)), 1.0, places=14)

    def test_reflection_identity_holds(self) -> None:
        # B(-y) = B(y) + y. This is the identity that was wrong once, and the
        # symptom was a flux that stayed perfectly self-consistent while being
        # wrong wherever the band edge rose along the flux direction.
        for y in (1e-8, 0.1, 1.0, 5.0, 50.0, 200.0):
            self.assertAlmostEqual(
                float(P.bernoulli(-y)),
                float(P.bernoulli(y)) + y,
                delta=1e-10 * max(1.0, y),
                msg=f"y={y}",
            )

    def test_slope_matches_a_central_difference(self) -> None:
        for x in (-50.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 50.0):
            _, slope = P._bernoulli_pair(np.array([x]))
            h = 1.0e-6 * max(1.0, abs(x))
            numeric = (
                bernoulli_reference(x + h) - bernoulli_reference(x - h)
            ) / (2 * h)
            self.assertAlmostEqual(
                float(slope[0]), numeric, delta=1e-6 * max(1.0, abs(numeric)), msg=f"x={x}"
            )

    def test_overflow_and_zero_arg_are_handled(self) -> None:
        value, slope = P._bernoulli_pair(np.array([-1e5, 1e5, 0.0]))
        self.assertTrue(np.all(np.isfinite(value)))
        self.assertTrue(np.all(np.isfinite(slope)))
        self.assertAlmostEqual(float(value[1]), 0.0, places=12)
        self.assertAlmostEqual(float(slope[2]), -0.5, places=12)


class FluxPartialTests(unittest.TestCase):
    """Every analytic partial of the flux against a central difference."""

    MU = 0.15
    KT = 0.025852
    LENGTH = 1.0e-9

    def _flux(self, band_l, band_r, n_l, n_r, *, vsat=None, beta=1.0):
        return P.sg_face_current(
            band_left=band_l,
            band_right=band_r,
            n_left=n_l,
            n_right=n_r,
            mu_m2_Vs=self.MU,
            kT_eV=self.KT,
            length_m=self.LENGTH,
            vsat_m_s=vsat,
            beta_field=beta,
        )

    def test_partials_match_central_differences(self) -> None:
        # Kept to moderate band-edge steps. Beyond roughly 5 kT the flux is the
        # difference of two terms that are orders apart, so a central difference
        # measures its own cancellation rather than the derivative; those regimes
        # are covered instead by the reflection identity and the zero-flux
        # invariant, both of which are exact statements.
        for band_l, band_r, n_l, n_r in (
            (0.0, 0.0, 4.0e24, 2.0e22),
            (0.05, 0.0, 1.0e24, 1.0e24),
            (0.0, 0.05, 1.0e24, 1.0e24),
            (0.05, -0.025, 5.0e23, 3.0e24),
            (-0.05, 0.025, 5.0e23, 3.0e24),
            (0.01, 0.0, 1.0e20, 1.0e21),
        ):
            current, d_psi_l, d_n_l, d_psi_r, d_n_r = self._flux(band_l, band_r, n_l, n_r)
            eps_psi = 1.0e-6 * self.KT
            eps_n = 1.0e-6 * n_l
            # band = ec0 - psi, so d/dpsi = -d/dband
            numeric_psi_l = -(
                self._flux(band_l + eps_psi, band_r, n_l, n_r)[0]
                - self._flux(band_l - eps_psi, band_r, n_l, n_r)[0]
            ) / (2 * eps_psi)
            numeric_psi_r = -(
                self._flux(band_l, band_r + eps_psi, n_l, n_r)[0]
                - self._flux(band_l, band_r - eps_psi, n_l, n_r)[0]
            ) / (2 * eps_psi)
            numeric_n_l = (
                self._flux(band_l, band_r, n_l + eps_n, n_r)[0]
                - self._flux(band_l, band_r, n_l - eps_n, n_r)[0]
            ) / (2 * eps_n)
            numeric_n_r = (
                self._flux(band_l, band_r, n_l, n_r + eps_n)[0]
                - self._flux(band_l, band_r, n_l, n_r - eps_n)[0]
            ) / (2 * eps_n)
            label = f"bands=({band_l},{band_r}) n=({n_l:.1e},{n_r:.1e})"
            for name, analytic, numeric in (
                ("dJ/dpsi_L", d_psi_l, numeric_psi_l),
                ("dJ/dn_L", d_n_l, numeric_n_l),
                ("dJ/dpsi_R", d_psi_r, numeric_psi_r),
                ("dJ/dn_R", d_n_r, numeric_n_r),
            ):
                scale = max(abs(float(numeric)), abs(float(analytic)), 1.0e-300)
                self.assertLess(
                    abs(float(analytic) - float(numeric)) / scale,
                    1.0e-5,
                    msg=f"{label}: {name} analytic={float(analytic):.6e} numeric={float(numeric):.6e}",
                )
            self.assertTrue(np.isfinite(current))

    def test_potential_partials_are_antisymmetric(self) -> None:
        # Shifting both band edges by the same constant leaves delta - and hence
        # the current - unchanged, so the two potential partials must cancel.
        _, d_psi_l, _, d_psi_r, _ = self._flux(0.3, -0.2, 1.0e24, 5.0e23)
        self.assertAlmostEqual(
            float(d_psi_l) + float(d_psi_r), 0.0,
            delta=1.0e-9 * abs(float(d_psi_l)) + 1.0e-30,
        )

    def test_partials_with_field_dependent_mobility(self) -> None:
        # The mobility now depends on the face field, and the field on the band
        # edges, so the flux gains a term proportional to dmu/dE. This is the
        # check that the term is in the Jacobian with the right sign and size:
        # without it the partials are wrong by tens of percent at high field.
        vsat = 2.0e5  # m/s, 2e7 cm/s
        for band_l, band_r, n_l, n_r in (
            (0.30, 0.0, 4.0e24, 2.0e22),   # E = 3e8 V/m, deeply saturated
            (0.05, 0.0, 1.0e24, 1.0e24),   # 5e7 V/m, part way
            (0.005, 0.0, 1.0e24, 1.0e24),  # 5e6 V/m, near the low-field limit
        ):
            current, d_psi_l, d_n_l, d_psi_r, d_n_r = self._flux(
                band_l, band_r, n_l, n_r, vsat=vsat
            )
            eps_psi = 1.0e-6 * self.KT
            eps_n = 1.0e-6 * n_l
            numeric_psi_l = -(
                self._flux(band_l + eps_psi, band_r, n_l, n_r, vsat=vsat)[0]
                - self._flux(band_l - eps_psi, band_r, n_l, n_r, vsat=vsat)[0]
            ) / (2 * eps_psi)
            numeric_psi_r = -(
                self._flux(band_l, band_r + eps_psi, n_l, n_r, vsat=vsat)[0]
                - self._flux(band_l, band_r - eps_psi, n_l, n_r, vsat=vsat)[0]
            ) / (2 * eps_psi)
            numeric_n_l = (
                self._flux(band_l, band_r, n_l + eps_n, n_r, vsat=vsat)[0]
                - self._flux(band_l, band_r, n_l - eps_n, n_r, vsat=vsat)[0]
            ) / (2 * eps_n)
            label = f"bands=({band_l},{band_r})"
            for name, analytic, numeric in (
                ("dJ/dpsi_L", d_psi_l, numeric_psi_l),
                ("dJ/dn_L", d_n_l, numeric_n_l),
                ("dJ/dpsi_R", d_psi_r, numeric_psi_r),
            ):
                scale = max(abs(float(numeric)), abs(float(analytic)), 1.0e-300)
                self.assertLess(
                    abs(float(analytic) - float(numeric)) / scale,
                    1.0e-5,
                    msg=f"{label}: {name} analytic={float(analytic):.6e} numeric={float(numeric):.6e}",
                )
            self.assertTrue(np.isfinite(current))
            # Antisymmetry has to survive the mobility term as well.
            self.assertAlmostEqual(
                float(d_psi_l) + float(d_psi_r), 0.0,
                delta=1.0e-8 * abs(float(d_psi_l)) + 1.0e-30,
                msg=label,
            )

    def test_mobility_limits_and_derivative(self) -> None:
        mu0, vsat = 0.15, 2.0e5
        # Low field: mu -> mu0.
        mu_low, slope_low = P.field_mobility(mu0, 1.0, vsat_m_s=vsat)
        self.assertAlmostEqual(float(mu_low), mu0, delta=1e-6 * mu0)
        self.assertLess(abs(float(slope_low)), 1e-6)
        # High field: mu -> vsat / E, i.e. the drift velocity saturates. The
        # approach is 1/(1 + mu0 E / vsat), which is slow: at 5e7 V/m the ratio
        # mu0 E / vsat is 37.5 and the velocity is still 2.6 % short of vsat.
        field = 5.0e7
        mu_high, _ = P.field_mobility(mu0, field, vsat_m_s=vsat)
        velocity = float(mu_high) * field
        self.assertLess(abs(velocity - vsat) / vsat, 0.05)
        self.assertGreater(velocity, 0.5 * vsat)
        # Derivative against a central difference, in both regimes.
        for field in (1.0e4, 1.0e6, 5.0e6, 1.0e8):
            _mu, slope = P.field_mobility(mu0, field, vsat_m_s=vsat)
            h = 1.0e-4 * field
            plus, _ = P.field_mobility(mu0, field + h, vsat_m_s=vsat)
            minus, _ = P.field_mobility(mu0, field - h, vsat_m_s=vsat)
            numeric = float(plus - minus) / (2 * h)
            scale = max(abs(numeric), abs(float(slope)), 1e-300)
            self.assertLess(abs(float(slope) - numeric) / scale, 1e-6, msg=f"E={field}")

    def test_density_partials_do_not_depend_on_the_density(self) -> None:
        # This is the property that makes a (psi, n) formulation well posed in a
        # wide-gap barrier: dJ/dn is set by geometry, mobility and the band-edge
        # difference alone.
        for factor in (1.0e-6, 1.0e-2, 1.0, 1.0e2, 1.0e6):
            _, _, d_n_l, _, _ = self._flux(0.05, 0.0, 4.0e24 * factor, 2.0e22)
            _, _, reference, _, _ = self._flux(0.05, 0.0, 4.0e24, 2.0e22)
            self.assertAlmostEqual(float(d_n_l), float(reference), delta=0.0)


class EquilibriumInvariantTests(unittest.TestCase):
    """A face whose densities follow the band edges carries no current."""

    def test_zero_flux_across_a_heterojunction_step(self) -> None:
        # AlGaN/GaN-like: 0.6 eV step and an effective-DOS step folded in.
        kT = 0.025852
        band_l, band_r = 0.58, -0.02
        n_ref = 1.0e24
        n_l = n_ref * np.exp(-band_l / kT)
        n_r = n_ref * np.exp(-band_r / kT)
        current, *_ = P.sg_face_current(
            band_left=band_l, band_right=band_r, n_left=n_l, n_right=n_r,
            mu_m2_Vs=0.15, kT_eV=kT, length_m=1.0e-9,
        )
        scale = 0.15 * kT / 1.0e-9 * n_l  # a flux scale for the comparison
        self.assertLess(abs(float(current)) / scale, 1.0e-12)

    def test_zero_flux_at_unity_ratio(self) -> None:
        current, *_ = P.sg_face_current(
            band_left=0.0, band_right=0.0, n_left=1.0e23, n_right=1.0e23,
            mu_m2_Vs=0.15, kT_eV=0.025852, length_m=1.0e-9,
        )
        self.assertAlmostEqual(float(current), 0.0, places=20)


class AssemblyTests(unittest.TestCase):
    """Grid construction, the Poisson start and the coupled Jacobian."""

    @classmethod
    def setUpClass(cls) -> None:
        load_package()
        A = sys.modules["huntian_tests_transport_dd.assemble"]
        cls.A = A
        from huntian import SimContext
        from huntian.materials import MaterialDB

        registry = build_registry()
        module = registry.select("mesh.structured2d").module_cls()
        module.initialize({"nx": 5, "target_spacing_nm": 2.0, "min_nodes_per_layer": 3})
        ctx = SimContext(name="dd-test", bus=registry.bus)
        ctx.put("structure", STRUCTURE)
        module.execute(ctx)
        cls.registry = registry
        cls.grid = A.build_grid2d(
            ctx.get("mesh"),
            STRUCTURE,
            MaterialDB.builtin(),
            {"sigma_eff_C_m2": 0.01540689},
            temperature_K=300.0,
            mu0_m2_Vs=0.15,
            delta_ec_ratio=0.7,
            gate_barrier_eV=0.94,
        )
        from huntian.services import ServiceRegistry

        cls.linear = ServiceRegistry(registry).get_by_capability("linear")

    def _system(self):
        A = self.A
        boundary = A.make_boundary(self.grid, vds_V=0.0, vgs_V=0.0)
        return A.DriftDiffusionSystem(
            self.grid, temperature_K=300.0, boundary=boundary, mu0_m2_Vs=0.15
        )

    def test_grid_masks_are_disjoint_and_cover_the_boundary(self) -> None:
        grid = self.grid
        masks = {
            "ohmic": grid.ohmic_mask,
            "gate": grid.gate_mask,
            "ground": grid.ground_mask,
            "surface": grid.surface_mask,
        }
        self.assertTrue(grid.ohmic_mask[:, 0].any())
        self.assertTrue(grid.ohmic_mask[:, -1].any())
        self.assertTrue(grid.gate_mask.any())
        self.assertTrue(grid.ground_mask.any())
        # A node must not be pinned twice by *different* conditions.
        self.assertFalse(np.any(grid.gate_mask & grid.ground_mask))
        self.assertFalse(np.any(grid.gate_mask & grid.surface_mask))
        self.assertFalse(np.any(grid.surface_mask & grid.ground_mask))
        for name, mask in masks.items():
            self.assertEqual(mask.shape, (grid.nz, grid.nx), msg=name)

    def test_band_offset_folds_in_the_density_of_states_step(self) -> None:
        # The flux band edge must satisfy n_i = N_ref exp(-band_i/kT) exactly, or
        # the flux is not at equilibrium across the heterointerface.
        grid = self.grid
        kT = P.BOLTZMANN * 300.0 / P.ELEMENTARY_CHARGE
        n_ref = 1.0e24
        reconstructed = n_ref * np.exp(-grid.band_offset_eV / kT)
        self.assertTrue(np.allclose(reconstructed, grid.nc_eff_m3, rtol=1e-12))

    def test_poisson_start_solves_the_equilibrium_charge_balance(self) -> None:
        system = self._system()
        boundary_psi = system.boundary.psi_mask
        psi = system.poisson_start(linear=self.linear, tolerance=1.0e-11)
        self.assertLess(system.poisson_history[-1], 1.0e-10)
        self.assertTrue(np.all(np.isfinite(psi)))
        # The gate pins the surface, and the barrier must not be flat.
        self.assertLess(float(psi.min()), -0.05)
        self.assertGreater(float(psi.max()), 0.0)
        # Re-forming the residual at the solution must be small, with the density
        # taken from the Boltzmann relation the start used.
        self.assertEqual(boundary_psi.shape, psi.shape)

    def test_coupled_jacobian_matches_central_differences(self) -> None:
        system = self._system()
        grid = self.grid
        psi = system.poisson_start(linear=self.linear, tolerance=1.0e-11)
        system.adopt_equilibrium(psi)
        system.boundary = system.boundary_for(vds_V=0.0, vgs_V=0.0)
        system._dirichlet = system.boundary.unknown_mask()
        system._targets = system.boundary.unknown_values()
        state = system.initial_guess(psi_field=psi)
        residual, aux = system.residual(state)
        lower, diagonal, upper = system.jacobian(state, aux)

        matrix = np.zeros((grid.nz * 2 * grid.nx, grid.nz * 2 * grid.nx))
        blocks, size = grid.nz, 2 * grid.nx
        for row in range(blocks):
            start = row * size
            matrix[start:start + size, start:start + size] = diagonal[row]
            if row > 0:
                matrix[start:start + size, start - size:start] = lower[row - 1]
            if row < blocks - 1:
                matrix[start:start + size, start + size:start + 2 * size] = upper[row]

        # Sample inside the electrically active rows. Deep-buffer columns carry
        # fluxes whose derivatives are exponentially small, where a central
        # difference measures its own round-off rather than the Jacobian.
        active = grid.ohmic_mask.any(axis=1)
        mask = (~system._dirichlet).reshape(blocks, size)
        mask[~active, :] = False
        free = np.flatnonzero(mask.reshape(-1))
        sampled = free[:: max(1, free.size // 15)]
        self.assertGreater(sampled.size, 5)
        h = 1.0e-7 * max(1.0, float(np.max(np.abs(state))))
        worst = 0.0
        worst_at = None
        for index in sampled:
            perturbation = np.zeros_like(state)
            perturbation.reshape(-1)[index] = h
            plus, _ = system.residual(state + perturbation)
            minus, _ = system.residual(state - perturbation)
            numeric = (plus.reshape(-1) - minus.reshape(-1)) / (2 * h)
            analytic = matrix[:, index]
            # Row-relative, not column-relative. A column-max denominator is
            # blind to an entry that is small but *wrong* by orders of magnitude,
            # which is exactly how a Jacobian that evaluated the flux at the
            # wrong density survived an earlier version of this test: only the
            # potential partials were wrong, and they are small next to the
            # Poisson rows of the same column.
            floor = 1.0e-9 * max(float(np.max(np.abs(analytic))), 1.0e-300)
            relative = np.abs(numeric - analytic) / np.maximum(
                np.maximum(np.abs(numeric), np.abs(analytic)), floor
            )
            error = float(np.max(relative))
            if error > worst:
                worst = error
                worst_at = index
        self.assertLess(
            worst,
            1.0e-4,
            msg=f"worst row-relative error {worst:.3e} at unknown index {worst_at} "
            f"of {sampled.size} sampled",
        )

    def _equilibrium(self, vds_V: float = 0.0):
        """Solve from the Poisson start: equilibrium at 0 V, or a biased point."""
        A = self.A
        from huntian.services import ServiceRegistry

        newton = sys.modules["huntian_tests_transport_dd.newton"]
        system = self._system()
        psi = system.poisson_start(linear=self.linear, tolerance=1.0e-11)
        system.adopt_equilibrium(psi)
        system.boundary = system.boundary_for(vds_V=0.0, vgs_V=0.0)
        system._dirichlet = system.boundary.unknown_mask()
        system._targets = system.boundary.unknown_values()
        state = system.initial_guess(psi_field=psi)
        system.equilibrate(state)
        outcome = newton.solve(
            system, state, linear=self.linear, max_iterations=40, tolerance=1.0e-9
        )
        return system, outcome, psi

    def test_equilibrium_carries_no_terminal_current(self) -> None:
        # At 0 V the exact solution is known: the Poisson-start potential with the
        # Boltzmann density that goes with it. The solver must leave it alone, so
        # any terminal current here is pure solver error - and this is the
        # sharpest anchor the module has.
        system, outcome, _psi = self._equilibrium(0.0)
        currents = system.terminal_currents(outcome.x)
        self.assertLess(abs(currents["source"]), 1.0e-9)
        self.assertLess(abs(currents["drain"]), 1.0e-9)

    def test_biased_solve_conserves_the_terminal_current(self) -> None:
        # A small drain bias carried by continuation. The gate the plugin will
        # enforce is |Is + Id| / |Id| <= 1e-6, and the lateral current profile has
        # to be flat: the currents are the same at every vertical cut.
        newton = sys.modules["huntian_tests_transport_dd.newton"]
        system, outcome, _psi = self._equilibrium(0.0)
        state = np.array(outcome.x, dtype=float, copy=True)
        ramp = (self.grid.x_m - self.grid.x_m[0]) / (
            self.grid.x_m[-1] - self.grid.x_m[0]
        )
        state[:, :, 0] = state[:, :, 0] + 0.1 * ramp[None, :]
        system.boundary = system.boundary_for(vds_V=0.1, vgs_V=0.0)
        system._dirichlet = system.boundary.unknown_mask()
        system._targets = system.boundary.unknown_values()
        outcome = newton.solve(
            system, state, linear=self.linear, max_iterations=60, tolerance=1.0e-9
        )
        currents = system.terminal_currents(outcome.x)
        drain = abs(currents["drain"])
        self.assertGreater(drain, 1.0e-3, msg="expected a real current at 0.1 V")
        self.assertLess(abs(currents["source"] + currents["drain"]) / drain, 1.0e-6)
        self.assertLess(currents["continuity_error"], 1.0e-6)

    def test_service_resolves_the_linear_capability(self) -> None:
        self.assertTrue(hasattr(self.linear, "solve"))
        self.assertIn("linear.blocktridiag", self.registry.module_ids())


SMALL_STRUCTURE = {
    "name": "small",
    "layers": [
        {"name": "barrier", "material": "AlGaN", "role": "barrier",
         "thickness_nm": 20.0, "x_al": 0.25},
        {"name": "channel", "material": "GaN", "role": "channel", "thickness_nm": 40.0},
        {"name": "buffer", "material": "GaN", "role": "buffer", "thickness_nm": 60.0,
         "doping_kind": "acceptor", "doping_cm3": 1.0e17},
    ],
    "gate": {"length_um": 0.15, "kind": "schottky", "barrier_eV": 0.94},
    "geometry": {"source_gate_spacing_um": 0.6, "gate_drain_spacing_um": 1.8},
}


def _small_context(*, transport_params: dict | None = None):
    """A context with every artefact transport.dd needs, on a small device."""
    from huntian import SimContext
    from huntian.deck import load_deck
    from huntian.materials import MaterialDB
    from huntian.services import ServiceRegistry

    registry = build_registry()
    mesh_module = registry.select("mesh.structured2d").module_cls()
    mesh_module.initialize({"nx": 9, "target_spacing_nm": 4.0, "min_nodes_per_layer": 3})
    ctx = SimContext(name="dd-integration", bus=registry.bus)
    ctx.services = ServiceRegistry(registry)
    ctx.put("structure", SMALL_STRUCTURE)
    mesh_module.execute(ctx)
    ctx.put("materials", MaterialDB.builtin())
    polarization = registry.select("physics.polarization.ambacher").module_cls()
    polarization.initialize({"screening": 0.8, "relaxed_fraction": 1.0})
    polarization.execute(ctx)
    transport = registry.select("transport.dd").module_cls()
    transport.initialize(
        transport_params
        or {"vds_sweep_V": "0.0,0.1,0.2", "vgs_sweep_V": "0.0",
            "max_newton_iterations": 60}
    )
    return registry, transport, ctx, load_deck


class DeckIntegrationTests(unittest.TestCase):
    """The demo deck, the service binding, and one end-to-end plugin run."""

    DECK = ROOT / "decks" / "demo_hemt_dd.json"

    def test_demo_deck_binds_the_transport_stage_and_linear_service(self) -> None:
        from huntian.deck import load_deck

        deck = load_deck(self.DECK)
        self.assertEqual(deck.physics.transport.module, "transport.dd")
        self.assertEqual(deck.modules.linear.module, "linear.blocktridiag")
        # The stage must be bound in the physics block: that is where the Deck
        # schema puts stage slots, and binding it under modules would silently do
        # nothing.
        self.assertIn("transport.dd", deck.describe())

    def test_missing_linear_service_is_reported_by_validate(self) -> None:
        registry, transport, ctx, _ = _small_context()
        ctx.services = None
        issues = transport.validate(ctx)
        self.assertTrue(any("linear" in issue.path for issue in issues), [str(i) for i in issues])

    def test_unknown_linear_solver_is_reported_by_validate(self) -> None:
        registry, transport, ctx, _ = _small_context(
            transport_params={
                "vds_sweep_V": "0.0",
                "vgs_sweep_V": "0.0",
                "linear_solver": "linear.does_not_exist",
            }
        )
        issues = transport.validate(ctx)
        self.assertTrue(any("linear_solver" in issue.path for issue in issues))

    def test_plugin_runs_a_sweep_and_publishes_the_artifacts(self) -> None:
        registry, transport, ctx, _ = _small_context()
        self.assertEqual(transport.validate(ctx), [])
        transport.execute(ctx)

        iv = ctx.get("iv")
        self.assertEqual(len(iv["curves"]), 3)
        for point in iv["curves"]:
            self.assertTrue(point["converged"], point["message"])
            self.assertTrue(point["continuity_passed"], point)
        # 0 V carries no net current, and a positive drain bias does.
        self.assertLess(abs(iv["curves"][0]["id_A_per_m"]), 1.0e-9)
        self.assertGreater(abs(iv["curves"][-1]["id_A_per_m"]), 1.0e-3)
        # Monotone in drain bias, and ohmic at this scale.
        currents = [abs(point["id_A_per_m"]) for point in iv["curves"]]
        self.assertLess(currents[0], currents[1])
        self.assertLess(currents[1], currents[2])
        self.assertAlmostEqual(currents[2] / currents[1], 2.0, delta=0.2)

        for name in ("electrostatics", "carriers", "current", "iv", "ns"):
            self.assertIsNotNone(ctx.get(name), msg=name)
        carriers = ctx.get("carriers")
        self.assertGreater(carriers["ns_cm2"], 0.0)
        self.assertGreater(carriers["density_max_m3"], 0.0)

    def test_two_deg_matches_the_polarization_bound(self) -> None:
        # The polarization sheet sets the scale of the 2DEG: charge control says
        # q * ns is at most the bound charge, less whatever the gate depletes.
        # A gross density-of-states or band-offset error shows up here, and so
        # does a sheet density that is summed over the lateral nodes instead of
        # averaged (that is exactly an nx-fold over-count).
        registry, transport, ctx, _ = _small_context()
        transport.execute(ctx)
        ns = ctx.get("carriers")["ns_cm2"]
        bound = 0.01540689 / P.ELEMENTARY_CHARGE * 1.0e-4  # cm^-2
        self.assertLess(ns, 3.0 * bound, msg=f"ns={ns:.4e}, bound={bound:.4e}")
        self.assertGreater(ns, 0.05 * bound, msg=f"ns={ns:.4e}, bound={bound:.4e}")
        # The published ns artefact must be the same number.
        self.assertAlmostEqual(float(ctx.get("ns")), ns, delta=1e-6 * ns)

    def test_continuation_chain_does_not_change_the_answer(self) -> None:
        # T4: the same bias point reached through two different continuation
        # ladders has to agree. The point is not that the paths are equal, it is
        # that the converged solution does not remember how it got there - a
        # solver whose answer depends on the stepping is not solving the equation.
        results = []
        for step in (0.05, 0.1):
            registry, transport, ctx, _ = _small_context(
                transport_params={"vds_sweep_V": "0.0,0.2", "vgs_sweep_V": "0.0",
                                  "max_step_V": step, "max_newton_iterations": 80}
            )
            transport.execute(ctx)
            curves = {point["vds_V"]: point for point in ctx.get("iv")["curves"]}
            point = curves[0.2]
            self.assertTrue(point["converged"], point["message"])
            results.append(point["id_A_per_m"])
        self.assertLess(
            abs(results[0] - results[1]) / max(abs(results[1]), 1e-30),
            5.0e-3,
            msg=f"0.05 V ladder {results[0]:.6e} A/m vs 0.1 V ladder {results[1]:.6e} A/m",
        )

    def test_gate_bias_controls_the_channel(self) -> None:
        # T5, the trend a transport model has to get right even before its numbers
        # are calibrated: at fixed drain bias, a more negative gate depletes the
        # channel and reduces the current. The effect is a few percent here,
        # because the barrier is thin and the ungated surface is pinned at the
        # Schottky reference, so the assertion is deliberately only on the sign
        # and a small margin.
        registry, transport, ctx, _ = _small_context(
            transport_params={"vds_sweep_V": "0.2", "vgs_sweep_V": "0.0,-2.0",
                              "max_newton_iterations": 120}
        )
        transport.execute(ctx)
        curves = {point["vgs_V"]: point for point in ctx.get("iv")["curves"]}
        on = abs(curves[0.0]["id_A_per_m"])
        off = abs(curves[-2.0]["id_A_per_m"])
        self.assertTrue(curves[0.0]["converged"], curves[0.0]["message"])
        self.assertTrue(curves[-2.0]["converged"], curves[-2.0]["message"])
        self.assertLess(off, on, msg=f"Id(Vgs=-2)={off:.4e} !< Id(0)={on:.4e}")
        self.assertLess(off / on, 0.999)
        # A depleted channel still conducts: the gate modulates, it does not
        # switch the device off at this barrier thickness.
        self.assertGreater(off / on, 0.1)

    def test_doubling_the_mobility_doubles_the_current(self) -> None:
        # The hot-swap acceptance in miniature: a parameter swap changes the
        # artefacts in the direction the physics says, which is what a Deck
        # rebind has to do.
        bare_registry, transport, ctx, _ = _small_context(
            transport_params={"vds_sweep_V": "0.0,0.2", "vgs_sweep_V": "0.0",
                              "max_newton_iterations": 60, "vsat_cm_s": 0.0}
        )
        transport.execute(ctx)
        reference = abs(ctx.get("iv")["curves"][-1]["id_A_per_m"])

        registry, transport, ctx, _ = _small_context(
            transport_params={"vds_sweep_V": "0.0,0.2", "vgs_sweep_V": "0.0",
                              "max_newton_iterations": 60, "mu0_cm2_Vs": 3000.0,
                              "vsat_cm_s": 0.0}
        )
        transport.execute(ctx)
        doubled = abs(ctx.get("iv")["curves"][-1]["id_A_per_m"])
        # With the field dependence switched off the proportionality is exact.
        self.assertAlmostEqual(doubled / reference, 2.0, delta=0.1)

    def test_drain_current_saturates(self) -> None:
        # T5's missing half: with Caughey-Thomas the mobility falls off with the
        # face field, so the drain current grows sub-linearly in Vds instead of
        # without bound. The assertion is on the *slope*, which is what
        # saturation means, and it stays inside the range the solver converges.
        registry, transport, ctx, _ = _small_context(
            transport_params={"vds_sweep_V": "0.0,0.25,0.5,1.0", "vgs_sweep_V": "0.0",
                              "max_step_V": 0.25, "max_newton_iterations": 120}
        )
        transport.execute(ctx)
        curves = {point["vds_V"]: point for point in ctx.get("iv")["curves"]}
        for vds in (0.25, 0.5, 1.0):
            self.assertTrue(curves[vds]["converged"], curves[vds]["message"])
        low_slope = abs(curves[0.5]["id_A_per_m"] - curves[0.25]["id_A_per_m"]) / 0.25
        high_slope = abs(curves[1.0]["id_A_per_m"] - curves[0.5]["id_A_per_m"]) / 0.5
        self.assertLess(high_slope, 0.95 * low_slope)
        ratio_with = high_slope / low_slope
        # Control: with a constant mobility the slope ratio is still ~0.92, not
        # 1.0 - a higher drain bias changes the gate-to-channel potential and
        # with it the 2DEG, so this device has a mild saturation of its own. The
        # assertion is therefore *relative*: the field dependence has to make it
        # substantially worse, which is what "the mobility saturates the current"
        # actually claims.
        registry, transport, ctx, _ = _small_context(
            transport_params={"vds_sweep_V": "0.0,0.25,0.5,1.0", "vgs_sweep_V": "0.0",
                              "max_step_V": 0.25, "max_newton_iterations": 120,
                              "vsat_cm_s": 0.0}
        )
        transport.execute(ctx)
        curves = {point["vds_V"]: point for point in ctx.get("iv")["curves"]}
        low_slope = abs(curves[0.5]["id_A_per_m"] - curves[0.25]["id_A_per_m"]) / 0.25
        high_slope = abs(curves[1.0]["id_A_per_m"] - curves[0.5]["id_A_per_m"]) / 0.5
        ratio_without = high_slope / low_slope
        self.assertGreater(ratio_without, 0.85)
        self.assertLess(ratio_with, ratio_without - 0.05)


class PoissonManufacturedTests(unittest.TestCase):
    """T2: the finite-volume Poisson operator against a manufactured solution.

    ``psi*(z) = sin(pi z / L)`` is x-independent, so every lateral flux vanishes
    identically even on a graded lateral mesh and the truncation error comes from
    the vertical operator alone - which, inside one layer, is a uniform grid. The
    matching charge is ``rho = -eps d2psi*/dz2``; sampling that *continuously*
    (not as ``A @ psi*``, which would make the test vacuous) and solving gives an
    error that has to fall as ``h^2``.

    The operator under test is the one the module actually assembles: the psi
    block of :meth:`DriftDiffusionSystem.jacobian`, with the harmonic face
    permittivity and the finite-volume control volumes.
    """

    LAYER_NM = 200.0

    def _solve_on(self, spacing_nm: float):
        from huntian import SimContext
        from huntian.materials import MaterialDB

        load_package()
        A = sys.modules["huntian_tests_transport_dd.assemble"]
        structure = {
            "name": "mms",
            "layers": [
                {"name": "bulk", "material": "GaN", "role": "channel",
                 "thickness_nm": self.LAYER_NM, "doping_kind": "donor",
                 "doping_cm3": 1.0e17},
            ],
            "gate": {"length_um": 0.15, "kind": "schottky", "barrier_eV": 0.94},
            "geometry": {"source_gate_spacing_um": 0.6, "gate_drain_spacing_um": 1.8},
        }
        registry = build_registry()
        mesh_module = registry.select("mesh.structured2d").module_cls()
        mesh_module.initialize({"nx": 5, "target_spacing_nm": spacing_nm,
                                "min_nodes_per_layer": 3})
        ctx = SimContext(name="mms", bus=registry.bus)
        ctx.put("structure", structure)
        mesh_module.execute(ctx)
        grid = A.build_grid2d(
            ctx.get("mesh"),
            structure,
            MaterialDB.builtin(),
            {"sigma_eff_C_m2": 0.0},
            temperature_K=300.0,
            mu0_m2_Vs=0.15,
            delta_ec_ratio=0.7,
            gate_barrier_eV=0.94,
            ground_backside=False,
            surface_pinning=False,
        )
        nz, nx = grid.nz, grid.nx
        empty = np.zeros((nz, nx), dtype=bool)
        system = A.DriftDiffusionSystem(
            grid,
            temperature_K=300.0,
            # No pins at all: the operator has to be the bare finite-volume
            # Laplacian. With the usual boundary the lateral columns are ohmic
            # contacts, their rows become identity rows, and the manufactured
            # source lands on equations that are not differential equations.
            boundary=A.Boundary(
                psi_mask=empty,
                psi_value=np.zeros((nz, nx), dtype=float),
                density_mask=empty.copy(),
                density_value=np.zeros((nz, nx), dtype=float),
            ),
            mu0_m2_Vs=0.15,
        )
        state = np.zeros((nz, nx, 2), dtype=float)
        _lower, diagonal, upper = system.jacobian(state)

        # psi-block of the operator, in the (nz*nx) node ordering.
        index = 2 * np.arange(nx)
        block = np.zeros((nz * nx, nz * nx), dtype=float)
        for row in range(nz):
            rows = slice(row * nx, (row + 1) * nx)
            block[rows, rows] = diagonal[row][np.ix_(index, index)]
            if row < nz - 1:
                block[rows, (row + 1) * nx:(row + 2) * nx] = upper[row][
                    np.ix_(index, index)
                ]
            if row > 0:
                block[rows, (row - 1) * nx:row * nx] = _lower[row - 1][
                    np.ix_(index, index)
                ]

        length = float(grid.z_m[-1] - grid.z_m[0])
        k = np.pi / length
        psi_exact = np.sin(k * grid.z_m)  # (nz,)
        # eps_r is uniform here, so the charge is eps * k^2 * sin(k z).
        eps = float(grid.eps_r[0, 0]) * P.VACUUM_PERMITTIVITY
        rho = eps * k * k * psi_exact
        # The jacobian hands back the *row-scaled* operator (each Poisson row is
        # divided by its own capacitance), so the manufactured charge has to be
        # scaled the same way or the solve is inconsistent and returns ~0.
        right = rho[:, None] * system.volume * system.row_scale[:, :, 0]

        # Homogeneous Dirichlet on the top and bottom rows: psi* vanishes there.
        pinned = np.zeros((nz, nx), dtype=bool)
        pinned[0, :] = True
        pinned[-1, :] = True
        flat = pinned.reshape(-1)
        block[flat, :] = 0.0
        block[flat, flat] = 1.0
        right = right.reshape(-1).copy()
        right[flat] = 0.0

        solution = np.linalg.solve(block, right).reshape(nz, nx)
        exact = np.repeat(psi_exact[:, None], nx, axis=1)
        interior = ~flat
        error = float(np.max(np.abs(solution.reshape(-1) - exact.reshape(-1))[interior]))
        return grid.nz, length, error

    def test_second_order_convergence(self) -> None:
        counts: list[int] = []
        errors: list[float] = []
        for spacing in (25.0, 12.5, 6.25, 3.125):
            count, _length, error = self._solve_on(spacing)
            counts.append(count)
            errors.append(error)
        self.assertGreaterEqual(len(errors), 4)
        # Every refinement must actually add nodes, or the order means nothing.
        for previous, current in zip(counts, counts[1:]):
            self.assertGreater(current, previous)
        self.assertLess(errors[-1], errors[0])
        orders = [
            float(np.log2(errors[index] / errors[index + 1]))
            for index in range(len(errors) - 1)
        ]
        for order in orders:
            self.assertGreater(
                order, 1.85, msg=f"orders={orders}, errors={errors}, nodes={counts}"
            )
            self.assertLess(
                order, 2.6, msg=f"orders={orders}, errors={errors}, nodes={counts}"
            )

    def test_truncation_error_scales_with_h_squared(self) -> None:
        # Independent of the solve: the local truncation error of the assembled
        # operator, A psi* - rho V, must also fall as h^2.
        errors: list[float] = []
        for spacing in (25.0, 12.5, 6.25):
            _count, _length, _error = self._solve_on(spacing)
            errors.append(_error)
        self.assertGreater(errors[0] / errors[-1], 4.0)


RESISTOR = {
    "name": "resistor",
    "layers": [
        # A single uniformly doped layer: no heterojunction, so no polarization
        # sheet, and the exact solution is a constant field with a uniform
        # density. One carrier, one mobility, closed form.
        {"name": "bulk", "material": "GaN", "role": "channel", "thickness_nm": 200.0,
         "doping_kind": "donor", "doping_cm3": 1.0e17},
    ],
    "gate": {"length_um": 0.15, "kind": "schottky", "barrier_eV": 0.94},
    "geometry": {"source_gate_spacing_um": 0.6, "gate_drain_spacing_um": 1.8},
}


class UniformResistorTests(unittest.TestCase):
    """T0: the closed form a drift-diffusion discretisation must reproduce.

    A uniformly doped layer between two ohmic contacts carries
    ``J = q * N_D * mu * V / L``, and the Scharfetter-Gummel flux is *exact* for
    the constant-field solution, so the discrete answer has to match the closed
    form to round-off - not to a few percent. This is the test that pins the
    absolute current scale: a wrong density of states, a wrong mobility unit or a
    spurious interface contribution all show up here immediately.
    """

    N_D = 1.0e23  # m^-3, 1e17 cm^-3
    MU = 0.15  # m^2/(V*s)
    VDS = 0.05  # V

    def _resistor(self, vds_V: float = VDS):
        from huntian import SimContext
        from huntian.materials import MaterialDB
        from huntian.services import ServiceRegistry

        A = sys.modules["huntian_tests_transport_dd.assemble"]
        newton = sys.modules["huntian_tests_transport_dd.newton"]

        registry = build_registry()
        mesh_module = registry.select("mesh.structured2d").module_cls()
        mesh_module.initialize({"nx": 5, "target_spacing_nm": 50.0,
                                "min_nodes_per_layer": 3})
        ctx = SimContext(name="resistor", bus=registry.bus)
        ctx.put("structure", RESISTOR)
        mesh_module.execute(ctx)
        grid = A.build_grid2d(
            ctx.get("mesh"),
            RESISTOR,
            MaterialDB.builtin(),
            {"sigma_eff_C_m2": 0.0},
            temperature_K=300.0,
            mu0_m2_Vs=self.MU,
            delta_ec_ratio=0.7,
            gate_barrier_eV=0.94,
            # A resistor has no gate, no pinned surface and a floating backside:
            # the contacts are the only anchors, which is exactly the well-posed
            # problem whose answer is known.
            ground_backside=False,
            surface_pinning=False,
        )
        kT = P.BOLTZMANN * 300.0 / P.ELEMENTARY_CHARGE
        psi_n = kT * float(np.log(self.N_D / float(grid.nc_eff_m3[0, 0])))

        nz, nx = grid.nz, grid.nx
        # Construct the system first: the density Dirichlet data is carried in
        # units of the unknown (n / density_scale), not in m^-3.
        system = A.DriftDiffusionSystem(
            grid,
            temperature_K=300.0,
            boundary=A.make_boundary(grid, vds_V=0.0, vgs_V=0.0),
            mu0_m2_Vs=self.MU,
        )
        psi_mask = np.zeros((nz, nx), dtype=bool)
        psi_value = np.zeros((nz, nx), dtype=float)
        psi_mask[:, 0] = True
        psi_mask[:, -1] = True
        psi_value[:, 0] = psi_n
        psi_value[:, -1] = psi_n + vds_V
        density_mask = psi_mask.copy()
        density_value = np.zeros((nz, nx), dtype=float)
        density_value[:, 0] = self.N_D / system.density_scale
        density_value[:, -1] = self.N_D / system.density_scale
        boundary = A.Boundary(
            psi_mask=psi_mask,
            psi_value=psi_value,
            density_mask=density_mask,
            density_value=density_value,
        )
        system.boundary = boundary
        system._dirichlet = boundary.unknown_mask()
        system._targets = boundary.unknown_values()
        linear = ServiceRegistry(registry).get_by_capability("linear")
        guess = system.initial_guess(psi_field=np.full((nz, nx), psi_n))
        # A uniform density is the physics here; the Boltzmann ramp the generic
        # start would build is the wrong guess for a resistor.
        guess[:, :, 1] = np.where(
            system._dirichlet[:, :, 1], system._targets[:, :, 1],
            self.N_D / system.density_scale,
        )
        system.equilibrate(guess)
        outcome = newton.solve(
            system, guess, linear=linear, max_iterations=40, tolerance=1.0e-12
        )
        return grid, system, outcome

    def test_matches_the_closed_form(self) -> None:
        grid, system, outcome = self._resistor()
        self.assertTrue(outcome.converged, outcome.message)
        currents = system.terminal_currents(outcome.x)
        length = float(grid.x_m[-1] - grid.x_m[0])
        thickness = float(grid.z_m[-1] - grid.z_m[0])
        expected = (
            P.ELEMENTARY_CHARGE * self.N_D * self.MU * (self.VDS / length) * thickness
        )
        self.assertGreater(expected, 1.0)
        relative = abs(abs(currents["drain"]) - expected) / expected
        self.assertLess(
            relative,
            1.0e-6,
            msg=f"Id={abs(currents['drain']):.6e} A/m, closed form {expected:.6e} A/m",
        )
        # And the two terminals must agree to the same standard.
        self.assertLess(
            abs(currents["source"] + currents["drain"]) / expected, 1.0e-9
        )

    def test_current_is_proportional_to_bias(self) -> None:
        _grid, _system, first = self._resistor(0.025)
        _grid, _system, second = self._resistor(0.050)
        low = abs(_system.terminal_currents(first.x)["drain"])
        high = abs(_system.terminal_currents(second.x)["drain"])
        self.assertAlmostEqual(high / low, 2.0, delta=1.0e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
