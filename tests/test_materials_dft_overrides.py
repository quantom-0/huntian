"""Override/merge behaviour tests for ``materials.dft``.

Covers the ``materials`` + ``dft_provenance`` artifact contract: DFT values
become a top overlay on the builtin database, ``dft_*`` keys are attached
but never consumed downstream, and the override report records every
decision (adopted or not, converged or not, with sources).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nemo import MaterialDB, SimContext

from . import ROOT
from .dft_table_fixtures import MINIMAL_TABLE
from .helpers import build_registry

TABLE_PATH = ROOT / "data" / "dft_lookup" / "algan_gan.json"


def make_context() -> SimContext:
    ctx = SimContext(name="dft-test")
    ctx.put("structure", {"layers": []}, producer="test")
    ctx.put("materials", MaterialDB.builtin(), producer="test")
    return ctx


class ModuleFixture:
    """Instantiate materials.dft against a table (shipped or temp)."""

    def __init__(self, registry, table_path: Path, **params) -> None:
        module_cls = registry.select("materials.dft").module_cls
        self.module = module_cls()
        self.module.initialize({"table_path": str(table_path), **params})

    def run(self, ctx: SimContext | None = None):
        ctx = ctx or make_context()
        self.module.run(ctx)
        return ctx

    def close(self) -> None:
        self.module.terminate()


class ShippedTableOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_dft_values_become_the_top_overlay(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        merged = ctx.get("materials")
        gan = merged.get("GaN")
        self.assertAlmostEqual(gan.numeric("a_lattice_A"), 3.178, places=6)
        self.assertAlmostEqual(gan.numeric("Eg_eV"), 3.45, places=6)
        self.assertAlmostEqual(gan.numeric("p_sp_C_m2"), -0.034, places=6)
        # Provenance of the replaced entry is visible to downstream audits.
        self.assertIn("DFT", gan.source)
        self.assertEqual(gan.verification, "partially-verified")
        fx.close()

    def test_dft_prefixed_keys_are_attached_not_consumed(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        gan = ctx.get("materials").get("GaN")
        self.assertAlmostEqual(gan.numeric("dft_eg_uncorrected_eV"), 1.72, places=6)
        uncertainty = gan.get("dft_uncertainty")
        self.assertIsInstance(uncertainty, dict)
        for key in ("p_sp_C_m2", "Eg_eV", "dft_eg_uncorrected_eV"):
            self.assertIn(key, uncertainty)
            self.assertIn("kind", uncertainty[key])
        fx.close()

    def test_builtin_only_keys_survive_the_merge(self) -> None:
        # 0 K honesty: thermal keys stay from the builtin library.
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        gan = ctx.get("materials").get("GaN")
        self.assertAlmostEqual(gan.numeric("thermal_conductivity_W_mK"), 130.0)
        self.assertAlmostEqual(gan.numeric("thermal_exponent"), 1.4)
        self.assertAlmostEqual(gan.numeric("eps_r"), 9.5)
        fx.close()

    def test_alloy_interpolation_uses_dft_endpoints(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        merged = ctx.get("materials")
        # AlGaN x=0.25 = endpoints + builtin bowing, endpoints now DFT.
        p_sp = merged.alloy_from_spec("AlGaN", 0.25, "p_sp_C_m2")
        expected = 0.75 * (-0.034) + 0.25 * (-0.090) + 0.021 * 0.25 * 0.75
        self.assertAlmostEqual(p_sp, expected, places=9)
        fx.close()

    def test_override_report_is_complete_and_traceable(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        report = ctx.get("dft_provenance")["override_report"]
        self.assertTrue(report)
        for row in report:
            for key in ("material", "property", "builtin", "dft", "adopted", "converged", "source"):
                self.assertIn(key, row, msg=f"row missing {key}: {row}")
            self.assertIn(row["adopted"], ("dft", "reference"))
            self.assertTrue(row["source"])
        adopted = [r for r in report if r["adopted"] == "dft"]
        self.assertTrue(adopted)
        fx.close()

    def test_provenance_records_backend_numerics_and_assumptions(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH)
        ctx = fx.run()
        prov = ctx.get("dft_provenance")
        self.assertEqual(prov["backend"], "table")
        self.assertEqual(prov["functional"], "hse06")
        self.assertEqual(prov["cache_misses"], 0)
        self.assertEqual(prov["cache_hits"], 2)
        self.assertIn("table_content_hash", prov["numerics"])
        assumptions = " ".join(prov["assumptions"])
        self.assertIn("0 K", assumptions)
        self.assertIn("PBE", assumptions)
        self.assertIn("converged=false", assumptions)
        fx.close()


class OverridePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_reference_wins_keeps_builtin_values_but_records_dft(self) -> None:
        fx = ModuleFixture(self.registry, TABLE_PATH, override_policy="reference_wins")
        ctx = fx.run()
        merged = ctx.get("materials")
        self.assertAlmostEqual(merged.get("GaN").numeric("Eg_eV"), 3.42, places=6)
        self.assertAlmostEqual(
            merged.get("GaN").numeric("a_lattice_A"), 3.189, places=6
        )
        # The DFT candidates are still recorded, not silently dropped.
        self.assertAlmostEqual(
            merged.get("GaN").numeric("dft_eg_uncorrected_eV"), 1.72, places=6
        )
        report = ctx.get("dft_provenance")["override_report"]
        self.assertTrue(all(row["adopted"] == "reference" for row in report))
        fx.close()

    def test_dft_only_gapless_adopts_everything_but_the_gap(self) -> None:
        fx = ModuleFixture(
            self.registry, TABLE_PATH, override_policy="dft_only_gapless"
        )
        ctx = fx.run()
        merged = ctx.get("materials")
        self.assertAlmostEqual(merged.get("GaN").numeric("a_lattice_A"), 3.178, places=6)
        self.assertAlmostEqual(merged.get("GaN").numeric("Eg_eV"), 3.42, places=6)
        report = {
            (row["material"], row["property"]): row
            for row in ctx.get("dft_provenance")["override_report"]
        }
        self.assertEqual(report[("GaN", "Eg_eV")]["adopted"], "reference")
        self.assertEqual(report[("GaN", "a_lattice_A")]["adopted"], "dft")
        fx.close()


class SyntheticTableTests(unittest.TestCase):
    """Edge paths that need a hand-built table."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _table(self, payload) -> Path:
        path = self.dir / "table.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_unknown_material_is_skipped_with_a_note(self) -> None:
        payload = json.loads(json.dumps(MINIMAL_TABLE))
        payload["materials"]["InN"] = {
            "source": "fixture",
            "verification": "unverified",
            "convergence": None,
            "values": {"a_lattice_A": {"value": 3.54}},
        }
        fx = ModuleFixture(self.registry, self._table(payload))
        ctx = fx.run()
        self.assertFalse(ctx.get("materials").has("InN"))
        entry = ctx.get("dft_provenance")["entries"]["InN"]
        self.assertIn("skipped", entry)
        fx.close()

    def test_gap_functional_mismatch_keeps_builtin_with_a_note(self) -> None:
        # functional=hse06 requested, but the table's gap is tagged pbe and
        # eg_policy=reject_pbe: never adopted, always recorded.
        payload = json.loads(json.dumps(MINIMAL_TABLE))
        payload["materials"]["GaN"]["values"]["Eg_eV"] = {
            "value": 1.72,
            "functional": "pbe",
        }
        fx = ModuleFixture(self.registry, self._table(payload))
        ctx = fx.run()
        gan = ctx.get("materials").get("GaN")
        self.assertAlmostEqual(gan.numeric("Eg_eV"), 3.42, places=6)
        report = ctx.get("dft_provenance")["override_report"]
        gap_row = [r for r in report if r["property"] == "Eg_eV"][0]
        self.assertEqual(gap_row["adopted"], "reference")
        fx.close()

    def test_converged_entry_is_flagged_converged(self) -> None:
        payload = json.loads(json.dumps(MINIMAL_TABLE))
        payload["materials"]["GaN"]["convergence"] = {
            "ecut_wfc_ry": {"values": [50.0, 60.0], "change_pct": 0.4},
            "kspacing_1_A": {"values": [0.24, 0.20], "change_pct": 0.2},
            "extrapolated": True,
        }
        fx = ModuleFixture(self.registry, self._table(payload))
        ctx = fx.run()
        report = ctx.get("dft_provenance")["override_report"]
        self.assertTrue(all(row["converged"] for row in report))
        self.assertNotIn("converged=false", " ".join(ctx.get("dft_provenance")["assumptions"]))
        fx.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
