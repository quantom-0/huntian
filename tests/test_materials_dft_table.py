"""Lookup-table IO tests for ``materials.dft`` (backend="table").

The table is a versioned data asset feeding golden CI; its reader must be
strict about schema violations and explicit about staleness.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import ROOT
from .dft_table_fixtures import MINIMAL_TABLE, table_io_module


TABLE_PATH = ROOT / "data" / "dft_lookup" / "algan_gan.json"


class ShippedTableTests(unittest.TestCase):
    def test_shipped_table_loads_and_matches_default_numerics(self) -> None:
        table_io = table_io_module()
        table = table_io.LookupTable.load(TABLE_PATH)
        self.assertEqual(table.version, "1.0.0")
        self.assertEqual(set(table.materials), {"GaN", "AlN"})
        self.assertEqual(table.numerics["ecut_wfc_ry"], 60.0)
        self.assertEqual(table.numerics["kspacing_1_A"], 0.20)
        self.assertEqual(
            table.numerics["pseudopotential_family"], "SSSP/PBEsol/efficiency@1.3.0"
        )

    def test_shipped_table_has_no_convergence_records(self) -> None:
        # Honesty gate: literature-transcribed entries must NOT claim
        # convergence evidence they do not have.
        table_io = table_io_module()
        table = table_io.LookupTable.load(TABLE_PATH)
        for name, entry in table.materials.items():
            self.assertFalse(entry.converged, msg=name)

    def test_shipped_table_numerics_match_manifest_defaults(self) -> None:
        table_io = table_io_module()
        table = table_io.LookupTable.load(TABLE_PATH)
        params = {
            "pseudopotential_family": "SSSP/PBEsol/efficiency@1.3.0",
            "ecut_wfc_ry": 60.0,
            "kspacing_1_A": 0.20,
            "sqs_seed": 1,
            "spin_orbit": False,
        }
        self.assertEqual(table_io.numerics_mismatch(table, params), {})

    def test_content_hash_is_stable(self) -> None:
        table_io = table_io_module()
        table = table_io.LookupTable.load(TABLE_PATH)
        self.assertEqual(table.content_hash(), table.content_hash())
        self.assertEqual(len(table.content_hash()), 16)


class TableErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self, payload):
        table_io = table_io_module()
        path = self.dir / "table.json"
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )
        return table_io.LookupTable.load(path)

    def test_missing_file_mentions_regeneration_path(self) -> None:
        table_io = table_io_module()
        with self.assertRaises(table_io.TableError) as caught:
            table_io.LookupTable.load(self.dir / "nope.json")
        self.assertIn("not found", str(caught.exception))
        self.assertIn("qe_local", str(caught.exception))

    def test_invalid_json_is_rejected(self) -> None:
        with self.assertRaises(Exception) as caught:
            self._load("{not json")
        self.assertIn("TableError", type(caught.exception).__name__)

    def test_empty_materials_section_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self._load({"version": "1.0.0", "materials": {}})

    def test_unknown_value_key_is_rejected_with_known_keys(self) -> None:
        bad = json.loads(json.dumps(MINIMAL_TABLE))
        bad["materials"]["GaN"]["values"]["bogus_key"] = {"value": 1.0}
        with self.assertRaises(Exception) as caught:
            self._load(bad)
        self.assertIn("bogus_key", str(caught.exception))

    def test_value_without_numeric_value_is_rejected(self) -> None:
        bad = json.loads(json.dumps(MINIMAL_TABLE))
        bad["materials"]["GaN"]["values"]["a_lattice_A"] = {"lo": 1.0}
        with self.assertRaises(Exception):
            self._load(bad)

    def test_minimal_valid_table_loads(self) -> None:
        table = self._load(MINIMAL_TABLE)
        self.assertIn("GaN", table.materials)
        self.assertAlmostEqual(table.materials["GaN"].values["a_lattice_A"].value, 3.19)

    def test_convergence_record_is_recognised(self) -> None:
        good = json.loads(json.dumps(MINIMAL_TABLE))
        good["materials"]["GaN"]["convergence"] = {
            "ecut_wfc_ry": {"values": [50.0, 60.0], "change_pct": 0.4},
            "kspacing_1_A": {"values": [0.24, 0.20], "change_pct": 0.2},
            "extrapolated": True,
        }
        table = self._load(good)
        self.assertTrue(table.materials["GaN"].converged)

    def test_incomplete_convergence_record_does_not_count(self) -> None:
        partial = json.loads(json.dumps(MINIMAL_TABLE))
        partial["materials"]["GaN"]["convergence"] = {
            "ecut_wfc_ry": {"values": [50.0, 60.0], "change_pct": 0.4},
            "extrapolated": True,  # missing kspacing record
        }
        table = self._load(partial)
        self.assertFalse(table.materials["GaN"].converged)

    def test_stale_mismatch_reports_both_sides(self) -> None:
        table_io = table_io_module()
        table = self._load(MINIMAL_TABLE)
        mismatch = table_io.numerics_mismatch(table, {"ecut_wfc_ry": 90.0})
        self.assertEqual(mismatch["ecut_wfc_ry"], (90.0, 60.0))
        clean = table_io.numerics_mismatch(table, {"ecut_wfc_ry": 60.0})
        self.assertEqual(clean, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
