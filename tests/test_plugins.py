"""Physics and data-contract tests for the bundled plugins.

These are *plausibility* checks, not validation against experiment: they assert
the invariants a correct implementation must satisfy (the analytic sheet
charge, monotonic trends, the upper bound on n_s) rather than magic numbers.
Absolute accuracy still has to come from the L2 benchmark and Hall data.
"""

from __future__ import annotations

import json
import unittest

from huntian.deck import Deck, load_deck

from . import DECK_PATH
from .helpers import build_registry, run_deck

RAW = json.loads(DECK_PATH.read_text(encoding="utf-8"))


def _deck(**mutations) -> Deck:
    """Copy the demo deck and apply ``a__b__0__c=value`` style mutations."""
    clone = json.loads(json.dumps(RAW))
    for path, value in mutations.items():
        cursor = clone
        parts = path.split("__")
        for part in parts[:-1]:
            cursor = cursor[int(part)] if part.isdigit() else cursor[part]
        cursor[parts[-1]] = value
    return Deck(**clone)


class DemoRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.result, cls.ctx = run_deck(load_deck(DECK_PATH), registry=cls.registry)
        cls.pol = cls.ctx.get("polarization")
        cls.quantum = cls.ctx.get("quantum")
        cls.fom = cls.ctx.get("results.fom")
        cls.mesh = cls.ctx.get("mesh")

    # -- run ------------------------------------------------------------
    def test_run_is_successful(self) -> None:
        self.assertEqual(self.result.status, "success")
        self.assertEqual(self.result.errors, [])

    # -- polarization ----------------------------------------------------
    def test_sheet_charge_matches_the_ambacher_result(self) -> None:
        # Analytic value for x=0.25: sigma_pol/e ~ 1.20e13 cm^-2.
        self.assertGreater(self.pol["sigma_pol_cm2"], 1.10e13)
        self.assertLess(self.pol["sigma_pol_cm2"], 1.30e13)

    def test_strain_is_tensile_and_small(self) -> None:
        self.assertGreater(self.pol["strain"], 0.0)
        self.assertLess(self.pol["strain"], 0.02)

    def test_screening_scales_the_effective_charge(self) -> None:
        self.assertAlmostEqual(
            self.pol["sigma_eff_cm2"], 0.8 * self.pol["sigma_pol_cm2"], places=6
        )

    def test_piezoelectric_and_spontaneous_contributions_are_both_negative(self) -> None:
        self.assertLess(self.pol["p_pz_barrier_C_m2"], 0.0)
        self.assertLess(self.pol["p_sp_barrier_C_m2"], self.pol["p_sp_channel_C_m2"])

    # -- quantum ----------------------------------------------------------
    def test_sheet_density_is_physically_plausible(self) -> None:
        ns = self.quantum["ns_cm2"]
        self.assertGreater(ns, 4.0e12)
        self.assertLess(ns, 1.2e13)

    def test_sheet_density_stays_below_the_upper_bound(self) -> None:
        self.assertLess(self.quantum["ns_cm2"], self.quantum["ns_upper_bound_cm2"])
        self.assertGreater(self.quantum["occupancy_fraction"], 0.0)

    def test_fixed_point_converged_quickly(self) -> None:
        self.assertTrue(self.quantum["converged"])
        self.assertLess(self.quantum["iterations"], 500)

    def test_fermi_level_is_above_the_ground_subband(self) -> None:
        self.assertGreaterEqual(self.quantum["ef_eV"], self.quantum["e0_eV"])

    def test_barrier_height_is_taken_from_the_deck(self) -> None:
        self.assertEqual(self.quantum["phi_b_source"], "deck")
        self.assertAlmostEqual(self.quantum["phi_b_eV"], 0.94, places=9)

    # -- figure of merit ---------------------------------------------------
    def test_threshold_is_negative_and_plausible(self) -> None:
        vth = self.fom["vth_est_V"]
        self.assertLess(vth, -1.0)
        self.assertGreater(vth, -5.0)

    def test_barrier_drop_is_consistent_with_charge_control(self) -> None:
        # sigma_eff * d / (eps0 * eps_r) must reproduce the reported drop.
        sigma_eff = self.pol["sigma_eff_C_m2"]
        thickness_m = self.pol["barrier_thickness_nm"] * 1.0e-9
        eps0 = 8.8541878128e-12
        expected = sigma_eff * thickness_m / (eps0 * self.pol["eps_r_barrier"])
        self.assertAlmostEqual(self.fom["barrier_drop_V"], expected, places=6)

    def test_oxide_capacitance_is_plausible(self) -> None:
        self.assertGreater(self.fom["cox_uF_cm2"], 0.2)
        self.assertLess(self.fom["cox_uF_cm2"], 0.8)

    # -- mesh ---------------------------------------------------------------
    def test_mesh_shape(self) -> None:
        self.assertEqual(self.mesh["nx"], 41)
        self.assertEqual(len(self.mesh["x_nm"]), 41)
        self.assertEqual(self.mesh["n_nodes"], self.mesh["nx"] * self.mesh["nz"])

    def test_vertical_axis_spans_the_whole_stack(self) -> None:
        z = self.mesh["z_nm"]
        self.assertEqual(z[0], 0.0)
        self.assertAlmostEqual(z[-1], 20.0 + 100.0 + 1500.0, places=9)
        for lower, upper in zip(z, z[1:]):
            self.assertLess(lower, upper)

    def test_layers_and_interfaces(self) -> None:
        self.assertEqual(len(self.mesh["layers"]), 3)
        self.assertEqual(len(self.mesh["interfaces_nm"]), 2)
        self.assertAlmostEqual(self.mesh["interfaces_nm"][0], 20.0, places=9)
        self.assertAlmostEqual(self.mesh["interfaces_nm"][1], 120.0, places=9)

    def test_node_slices_agree_with_the_axis(self) -> None:
        z = self.mesh["z_nm"]
        for layer in self.mesh["layers"]:
            start, end = layer["node_slice"]
            self.assertAlmostEqual(z[start], layer["top_nm"], places=9)
            self.assertAlmostEqual(z[end], layer["bottom_nm"], places=9)

    def test_contact_ordering(self) -> None:
        contacts = self.mesh["contacts_nm"]
        self.assertLess(contacts["source"], contacts["gate_left"])
        self.assertLess(contacts["gate_left"], contacts["gate_right"])
        self.assertLess(contacts["gate_right"], contacts["drain"])


class ParameterTrendTests(unittest.TestCase):
    """The signs of the physical trends must be right."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def _ctx(self, **mutations):  # noqa: ANN003
        _result, ctx = run_deck(_deck(**mutations), registry=self.registry)
        return ctx

    def test_higher_al_fraction_increases_polarization(self) -> None:
        low = self._ctx(device__layers__0__x_al=0.20).get("polarization")
        high = self._ctx(device__layers__0__x_al=0.35).get("polarization")
        self.assertLess(low["sigma_pol_cm2"], high["sigma_pol_cm2"])

    def test_more_screening_increases_sheet_density(self) -> None:
        low = self._ctx(physics__polarization__params={"screening": 0.4}).get("quantum")
        high = self._ctx(physics__polarization__params={"screening": 0.95}).get("quantum")
        self.assertLess(low["ns_cm2"], high["ns_cm2"])

    def test_thicker_barrier_increases_sheet_density(self) -> None:
        thin = self._ctx(device__layers__0__thickness_nm=15.0).get("quantum")
        thick = self._ctx(device__layers__0__thickness_nm=30.0).get("quantum")
        self.assertGreater(thick["ns_cm2"], thin["ns_cm2"])

    def test_relaxing_the_barrier_reduces_polarization(self) -> None:
        strained = self._ctx(physics__polarization__params={"relaxed_fraction": 1.0}).get(
            "polarization"
        )
        relaxed = self._ctx(physics__polarization__params={"relaxed_fraction": 0.5}).get(
            "polarization"
        )
        self.assertGreater(strained["sigma_pol_cm2"], relaxed["sigma_pol_cm2"])

    def test_higher_screening_deepens_the_threshold(self) -> None:
        low = self._ctx(physics__polarization__params={"screening": 0.6}).get("results.fom")
        high = self._ctx(physics__polarization__params={"screening": 0.9}).get("results.fom")
        self.assertLess(high["vth_est_V"], low["vth_est_V"])

    def test_more_gate_spacing_extends_the_mesh_laterally(self) -> None:
        narrow = self._ctx(device__geometry__gate_drain_spacing_um=1.0).get("mesh")
        wide = self._ctx(device__geometry__gate_drain_spacing_um=3.0).get("mesh")
        self.assertLess(
            narrow["lateral_extent_nm"][1],
            wide["lateral_extent_nm"][1],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
