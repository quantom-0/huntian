"""Manifest and precision-red-line tests for ``materials.dft``.

Red lines (design document §5) are asserted here as behaviour, not prose:
PBE band gaps must be rejected by default, scissor correction must record
its shift, and no parameter of this module may be calibratable - DFT
settings are numerical-method choices, not device fit knobs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nemo import MaterialDB, SimContext
from nemo.errors import ModuleConfigError

from . import ROOT
from .dft_table_fixtures import MINIMAL_TABLE
from .helpers import build_registry

TABLE_PATH = ROOT / "data" / "dft_lookup" / "algan_gan.json"


def make_context() -> SimContext:
    ctx = SimContext(name="dft-test")
    ctx.put("structure", {"layers": []}, producer="test")
    ctx.put("materials", MaterialDB.builtin(), producer="test")
    return ctx


def make_module(registry, **params):
    module_cls = registry.select("materials.dft").module_cls
    module = module_cls()
    module.initialize({"table_path": str(TABLE_PATH), **params})
    return module


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.manifest = cls.registry.select("materials.dft").manifest

    def test_identity_and_contract(self) -> None:
        self.assertEqual(self.manifest.module_id, "materials.dft")
        self.assertEqual(self.manifest.module_version, "0.1.0")
        self.assertIn("materials", self.manifest.capabilities)
        self.assertIn("materials.validation", self.manifest.capabilities)
        self.assertEqual(self.manifest.requires, ("structure",))
        self.assertEqual(self.manifest.produces, ("materials", "dft_provenance"))
        self.assertEqual(self.manifest.verification, "partially-verified")

    def test_no_parameter_is_calibratable(self) -> None:
        # DFT settings are numerical-method choices; letting a fit loop move
        # ecut would be tuning the grid to match data (design §3).
        self.assertEqual(self.manifest.calibratable_params(), ())

    def test_default_mode_is_dependency_free_table(self) -> None:
        self.assertEqual(self.manifest.param("backend").default, "table")
        self.assertEqual(self.manifest.param("eg_policy").default, "reject_pbe")
        self.assertEqual(self.manifest.param("functional").default, "hse06")

    def test_qe_drivers_are_not_imported_in_table_mode(self) -> None:
        # Lazy imports are a hard requirement: discovery loads every plugin,
        # and loading this one must not pull ase/qe_driver/slurm in. Run in a
        # subprocess so other tests that *do* touch qe_driver cannot leak.
        code = (
            "import sys; sys.path.insert(0, 'python')"
            "; from nemo.registry import discover_plugin_classes"
            "; discover_plugin_classes(['plugins/materials_dft'])"
            "; leaked=[m for m in sys.modules if 'qe_driver' in m"
            " or 'slurm' in m or m == 'ase']"
            "; print('LEAKED', leaked)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("LEAKED []", proc.stdout, msg=proc.stdout + proc.stderr)


class RedLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, payload) -> Path:
        path = self.dir / "table.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_pbe_gap_is_rejected_by_default(self) -> None:
        module = make_module(self.registry, functional="pbe")
        issues = module.validate(make_context())
        self.assertTrue(
            any(issue.code == "eg-policy" for issue in issues),
            f"expected an eg-policy issue, got: {issues}",
        )
        module.terminate()

    def test_hse06_gap_passes_validation(self) -> None:
        module = make_module(self.registry)
        self.assertEqual(module.validate(make_context()), [])
        module.terminate()

    def test_scissor_mode_allows_pbe_and_records_the_shift(self) -> None:
        # The shipped table carries HSE gaps; scissor mode matters when the
        # tabulated gap is PBE, so build that table explicitly.
        payload = json.loads(json.dumps(MINIMAL_TABLE))
        payload["materials"]["GaN"]["values"]["Eg_eV"] = {
            "value": 1.72,
            "functional": "pbe",
        }
        table_path = self._write(payload)
        module = make_module(
            self.registry,
            functional="pbe",
            eg_policy="scissor_to_reference",
            table_path=str(table_path),
        )
        self.assertEqual(module.validate(make_context()), [])
        ctx = make_context()
        module.run(ctx)
        gan = ctx.get("materials").get("GaN")
        # The adopted gap equals the reference gap; the PBE raw value and the
        # shift are recorded instead of silently discarded.
        self.assertAlmostEqual(gan.numeric("Eg_eV"), 3.42, places=6)
        self.assertAlmostEqual(
            gan.numeric("dft_eg_uncorrected_eV"), 1.72, places=6
        )
        self.assertAlmostEqual(gan.numeric("dft_eg_shift_eV"), 1.72 - 3.42, places=6)
        assumptions = " ".join(ctx.get("dft_provenance")["assumptions"])
        self.assertIn("scissor", assumptions)
        module.terminate()

    def test_scissor_caps_verification_at_partially_verified(self) -> None:
        payload = json.loads(json.dumps(MINIMAL_TABLE))
        payload["materials"]["GaN"]["values"]["Eg_eV"] = {
            "value": 1.72,
            "functional": "pbe",
        }
        table_path = self._write(payload)
        module = make_module(
            self.registry,
            functional="pbe",
            eg_policy="scissor_to_reference",
            table_path=str(table_path),
        )
        ctx = make_context()
        module.run(ctx)
        self.assertEqual(
            ctx.get("materials").get("GaN").verification, "partially-verified"
        )
        module.terminate()


class BackendAvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_qe_local_reports_why_it_cannot_run(self) -> None:
        module_cls = self.registry.select("materials.dft").module_cls
        module = module_cls()
        with self.assertRaises(ModuleConfigError) as caught:
            module.initialize({"backend": "qe_local"})
        message = str(caught.exception)
        self.assertIn("qe_local", message)
        # Either the dependency is missing or the driver is not implemented
        # yet; both must be actionable, pointing back to backend='table'.
        self.assertTrue("ase" in message or "not implemented" in message)
        self.assertIn("table", message)

    def test_qe_slurm_reports_why_it_cannot_run(self) -> None:
        module_cls = self.registry.select("materials.dft").module_cls
        module = module_cls()
        with self.assertRaises(ModuleConfigError) as caught:
            module.initialize({"backend": "qe_slurm"})
        self.assertIn("qe_slurm", str(caught.exception))
        self.assertIn("table", str(caught.exception))


class StaleCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()

    def test_mismatched_numerics_are_refused_by_default(self) -> None:
        module = make_module(self.registry, ecut_wfc_ry=90.0)
        issues = module.validate(make_context())
        self.assertTrue(
            any(issue.code == "stale-cache" for issue in issues),
            f"expected a stale-cache issue, got: {issues}",
        )
        module.terminate()

    def test_stale_numerics_require_explicit_opt_in(self) -> None:
        module = make_module(
            self.registry, ecut_wfc_ry=90.0, allow_stale_cache=True
        )
        self.assertEqual(module.validate(make_context()), [])
        ctx = make_context()
        module.run(ctx)
        assumptions = " ".join(ctx.get("dft_provenance")["assumptions"])
        self.assertIn("stale", assumptions)
        self.assertIn("ecut_wfc_ry", assumptions)
        module.terminate()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
