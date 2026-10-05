"""Integration test: swapping the material database implementation at runtime.

Acceptance test for the ``materials.dft`` plugin, mirroring
``test_hotswap_quantum.py``: the same device, the same downstream modules
(``physics.polarization.ambacher``, ``physics.quantum.sp1d``,
``extract.fom``) and the same pipeline shape; the only difference is the
``materials`` slot, bound to ``materials.dft`` instead of the builtin seed.

Design-document §7 acceptance assertions:

* ``polarization.ambacher`` and ``physics.quantum.sp1d`` run unchanged;
* the pipeline gains exactly one stage (``materials``), scheduled first by
  manifest data dependency, nothing else reordered;
* ``dft_provenance.override_report`` is non-empty and every override is
  traceable to a source;
* two runs of the same Deck produce the same SimContext fingerprint.
"""

from __future__ import annotations

import json
import unittest

from nemo import Executor, Pipeline, SimContext, load_deck
from nemo.deck import Deck

from . import DECK_SP1D_PATH, DECKS_DIR
from .helpers import build_registry, builtin_materials

DECK_DFT_PATH = DECKS_DIR / "demo_hemt_dft.json"


class DeckSwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.materials = builtin_materials()
        cls.sp_deck = load_deck(DECK_SP1D_PATH)
        cls.dft_deck = load_deck(DECK_DFT_PATH)
        cls.sp_result, cls.sp_ctx = cls._run(cls.sp_deck)
        cls.dft_result, cls.dft_ctx = cls._run(cls.dft_deck)

    @classmethod
    def _run(cls, deck):
        ctx = SimContext(name=deck.name, bus=cls.registry.bus)
        result = Executor(cls.registry).run(
            Pipeline.from_deck(deck), ctx=ctx, deck=deck, materials=cls.materials
        )
        return result, ctx

    # -- validation ---------------------------------------------------------
    def test_both_decks_validate(self) -> None:
        for deck in (self.sp_deck, self.dft_deck):
            errors, _warnings = deck.validate_all(self.registry, self.materials)
            self.assertEqual([str(e) for e in errors], [], msg=deck.name)

    # -- pipeline shape -----------------------------------------------------
    def test_the_materials_stage_runs_first(self) -> None:
        stages = Pipeline.from_deck(self.dft_deck).stages
        self.assertEqual(stages[0].slot, "materials")
        self.assertEqual(stages[0].module_id, "materials.dft")

    def test_only_one_stage_is_added_and_nothing_else_moves(self) -> None:
        sp_slots = [stage.slot for stage in Pipeline.from_deck(self.sp_deck).stages]
        dft_slots = [stage.slot for stage in Pipeline.from_deck(self.dft_deck).stages]
        self.assertEqual(dft_slots[1:], sp_slots)

    def test_downstream_stage_ordering_comes_from_manifest_dependencies(self) -> None:
        # materials.dft requires only 'structure' (a seeded builtin) and
        # produces 'materials'; the edge that schedules it before polarization
        # is derived from the manifests, not from the slot table.
        manifest = self.registry.select("materials.dft").manifest
        self.assertEqual(manifest.requires, ("structure",))
        self.assertIn("materials", manifest.produces)

    # -- both runs succeed, downstream untouched ----------------------------
    def test_both_runs_succeed(self) -> None:
        self.assertEqual(self.sp_result.status, "success")
        self.assertEqual(self.dft_result.status, "success")

    def test_downstream_modules_are_identical(self) -> None:
        for slot in ("polarization", "quantum"):
            self.assertEqual(
                self.sp_result.stage(slot).module_id,
                self.dft_result.stage(slot).module_id,
            )
            self.assertEqual(
                self.sp_result.stage(slot).module_version,
                self.dft_result.stage(slot).module_version,
            )
        self.assertEqual(
            self.dft_result.stage("extract:0").module_id, "extract.fom"
        )
        self.assertTrue(self.dft_ctx.has("results.fom"))

    # -- the DFT overlay actually changed the physics ------------------------
    def test_downstream_consumes_the_merged_database(self) -> None:
        merged = self.dft_ctx.get("materials")
        self.assertAlmostEqual(merged.get("GaN").numeric("a_lattice_A"), 3.178, places=6)
        self.assertAlmostEqual(merged.get("GaN").numeric("Eg_eV"), 3.45, places=6)
        # The seeded builtin database is untouched for other contexts.
        self.assertAlmostEqual(
            builtin_materials().get("GaN").numeric("a_lattice_A"), 3.189, places=6
        )

    def test_the_swap_moves_the_sheet_density_physically(self) -> None:
        sp_ns = float(self.sp_ctx.get("ns"))
        dft_ns = float(self.dft_ctx.get("ns"))
        relative = abs(dft_ns - sp_ns) / sp_ns
        self.assertGreater(relative, 0.005, msg="the swap changed nothing measurable")
        self.assertLess(relative, 0.20, msg="the two parameter sets disagree unphysically")

    def test_the_two_runs_have_different_fingerprints(self) -> None:
        self.assertNotEqual(self.sp_result.fingerprint, self.dft_result.fingerprint)

    # -- provenance contract --------------------------------------------------
    def test_dft_provenance_is_complete_and_traceable(self) -> None:
        prov = self.dft_ctx.get("dft_provenance")
        self.assertEqual(prov["backend"], "table")
        self.assertTrue(prov["override_report"])
        for row in prov["override_report"]:
            self.assertTrue(row["source"], msg=f"override without source: {row}")
            self.assertIn(row["adopted"], ("dft", "reference"))
        for key in ("functional", "numerics", "entries", "assumptions", "cache_hits"):
            self.assertIn(key, prov, msg=key)
        for material, entry in prov["entries"].items():
            self.assertIn("source", entry, msg=material)
            self.assertIn("verification", entry, msg=material)

    def test_the_same_deck_twice_has_the_same_fingerprint(self) -> None:
        result, _ctx = self._run(self.dft_deck)
        self.assertEqual(result.fingerprint, self.dft_result.fingerprint)


class VersionPinningTests(unittest.TestCase):
    """The new module must not disturb the already-validated sp1d Deck."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.dft_deck = load_deck(DECK_DFT_PATH)

    def test_the_dft_deck_pins_the_module(self) -> None:
        binding = self.dft_deck.modules.materials
        self.assertEqual(binding.module, "materials.dft")
        self.assertEqual(binding.version, "^0.1.0")
        selected = self.registry.select(binding.module, binding.version)
        self.assertEqual(selected.manifest.module_version, "0.1.0")

    def test_binding_the_materials_slot_is_a_few_lines(self) -> None:
        raw = json.loads(DECK_SP1D_PATH.read_text(encoding="utf-8"))
        raw["name"] = "demo_algan_gan_hemt_dft_inline"
        raw["modules"]["materials"] = json.loads(
            DECK_DFT_PATH.read_text(encoding="utf-8")
        )["modules"]["materials"]
        swapped = Deck(**raw)
        errors, _warnings = swapped.validate_all(self.registry, builtin_materials())
        self.assertEqual([str(e) for e in errors], [])
        stages = Pipeline.from_deck(swapped).stages
        self.assertEqual(stages[0].slot, "materials")
        self.assertEqual(stages[0].module_id, "materials.dft")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
