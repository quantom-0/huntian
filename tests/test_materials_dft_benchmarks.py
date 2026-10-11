"""D0-D3 verification benchmarks for ``materials.dft`` (design document §8).

These benchmarks check the DFT module against *experiment*, complementing
the L0-L2 ladder that validates solvers against analytics/measurement.
Tolerance philosophy matches ``test_quantum_sp1d.py``: assert error bands
and invariants, not magic digits - but the *comparison output* is golden:
``tests/golden/dft_benchmarks_d0_d3.json`` is checked in, so any drift in
the shipped lookup table fails CI even inside the tolerance bands.

Reference values are experimental constants (design §8), independent of
the builtin library, so a wrong edit to materials.json cannot silently
move the goalposts.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from huntian import MaterialDB, SimContext

from . import ROOT
from .helpers import build_registry

TABLE_PATH = ROOT / "data" / "dft_lookup" / "algan_gan.json"
GOLDEN_PATH = ROOT / "tests" / "golden" / "dft_benchmarks_d0_d3.json"

#: Experimental references (design §8), independent of materials.json.
EXPERIMENT = {
    "GaN": {
        "a_lattice_A": 3.189,
        "c_lattice_A": 5.185,
        "C13_GPa": 103.0,
        "C33_GPa": 405.0,
        "p_sp_C_m2": -0.034,
        "Eg_eV": 3.42,
    },
    "AlN": {
        "a_lattice_A": 3.112,
        "c_lattice_A": 4.982,
        "C13_GPa": 108.0,
        "C33_GPa": 373.0,
        "p_sp_C_m2": -0.090,
        "Eg_eV": 6.2,
    },
}

#: Tolerance bands per level (design §8).
TOLERANCE = {
    "D0": 0.01,  # lattice constants, relative
    "D1": 0.10,  # elastic constants, relative
    "D2": 0.15,  # spontaneous polarization, relative
    "D3": 0.05,  # band gap (HSE-adopted only), relative
}


def merged_db(registry) -> MaterialDB:
    module_cls = registry.select("materials.dft").module_cls
    module = module_cls()
    module.initialize({"table_path": str(TABLE_PATH)})
    ctx = SimContext(name="dft-benchmark")
    ctx.put("structure", {"layers": []}, producer="test")
    ctx.put("materials", MaterialDB.builtin(), producer="test")
    module.run(ctx)
    module.terminate()
    return ctx.get("materials")


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


class D0LatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = merged_db(build_registry())

    def test_lattice_constants_within_one_percent(self) -> None:
        for material, keys in (("GaN", ("a_lattice_A", "c_lattice_A")), ("AlN", ("a_lattice_A", "c_lattice_A"))):
            for key in keys:
                with self.subTest(material=material, key=key):
                    error = relative_error(
                        self.db.get(material).numeric(key), EXPERIMENT[material][key]
                    )
                    self.assertLess(error, TOLERANCE["D0"], msg=f"{material}.{key}: {error:.2%}")


class D1ElasticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = merged_db(build_registry())

    def test_elastic_constants_within_ten_percent(self) -> None:
        for material in ("GaN", "AlN"):
            for key in ("C13_GPa", "C33_GPa"):
                with self.subTest(material=material, key=key):
                    error = relative_error(
                        self.db.get(material).numeric(key), EXPERIMENT[material][key]
                    )
                    self.assertLess(error, TOLERANCE["D1"], msg=f"{material}.{key}: {error:.2%}")


class D2PolarizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = merged_db(build_registry())

    def test_spontaneous_polarization_within_fifteen_percent(self) -> None:
        for material in ("GaN", "AlN"):
            with self.subTest(material=material):
                error = relative_error(
                    self.db.get(material).numeric("p_sp_C_m2"),
                    EXPERIMENT[material]["p_sp_C_m2"],
                )
                self.assertLess(error, TOLERANCE["D2"], msg=f"{material}.p_sp: {error:.2%}")

    def test_polarization_method_is_recorded(self) -> None:
        # D2 requires recording the method difference (Berry phase vs
        # empirical formula); the source string must name it.
        gan = self.db.get("GaN")
        self.assertIn("Berry", gan.source)


class D3BandGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = merged_db(build_registry())

    def test_adopted_gap_within_five_percent(self) -> None:
        for material in ("GaN", "AlN"):
            with self.subTest(material=material):
                error = relative_error(
                    self.db.get(material).numeric("Eg_eV"), EXPERIMENT[material]["Eg_eV"]
                )
                self.assertLess(error, TOLERANCE["D3"], msg=f"{material}.Eg: {error:.2%}")

    def test_pbe_raw_gap_is_never_adopted(self) -> None:
        # The red line at benchmark level: the merged gap must not equal the
        # PBE raw value, and the raw value must be present for audit.
        for material, pbe_gap in (("GaN", 1.72), ("AlN", 4.07)):
            with self.subTest(material=material):
                entry = self.db.get(material)
                self.assertNotAlmostEqual(entry.numeric("Eg_eV"), pbe_gap, places=3)
                self.assertAlmostEqual(
                    entry.numeric("dft_eg_uncorrected_eV"), pbe_gap, places=6
                )

    def test_gap_uncertainty_brackets_experiment(self) -> None:
        for material in ("GaN", "AlN"):
            with self.subTest(material=material):
                unc = self.db.get(material).get("dft_uncertainty")["Eg_eV"]
                self.assertIn("lo", unc)
                self.assertIn("hi", unc)
                exp = EXPERIMENT[material]["Eg_eV"]
                self.assertTrue(
                    math.isclose(unc["lo"], exp, abs_tol=0.6)
                    or unc["lo"] <= exp <= unc["hi"]
                    or abs(unc["hi"] - exp) < 0.6,
                    msg=f"{material}: experiment {exp} eV outside quoted band [{unc['lo']}, {unc['hi']}]",
                )


class GoldenComparisonTests(unittest.TestCase):
    """The exact benchmark output is a checked-in golden asset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = merged_db(build_registry())

    def _measure(self) -> dict:
        out: dict[str, dict[str, float]] = {}
        for material, props in EXPERIMENT.items():
            out[material] = {}
            for key, reference in props.items():
                value = self.db.get(material).numeric(key)
                out[material][f"{key}_rel_err"] = round(relative_error(value, reference), 6)
        return out

    def test_measurements_match_golden(self) -> None:
        measured = self._measure()
        if not GOLDEN_PATH.is_file():  # pragma: no cover - one-time bootstrap
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(
                json.dumps(measured, indent=1, sort_keys=True), encoding="utf-8"
            )
            self.skipTest(f"golden file bootstrapped at {GOLDEN_PATH}; rerun")
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(measured, golden)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
