"""Integration test: swapping the quantum implementation at runtime.

This is the acceptance test for the whole plug-and-play story. The same device,
the same downstream extractor and the same pipeline are used twice; only the
``quantum`` binding in the Deck changes.

    physics.quantum.analytic_tw   (analytic triangular-well estimate)
    physics.quantum.sp1d          (self-consistent Schroedinger-Poisson)

Both must satisfy the same contract, so ``extract.fom`` - which knows nothing
about either - keeps working untouched. And because each Deck pins a version,
adding the new module cannot silently change the previously validated result.
"""

from __future__ import annotations

import json
import unittest

from nemo import Executor, Pipeline, SimContext, load_deck
from nemo.deck import SLOT_CAPABILITY, Deck

from . import DECK_PATH, DECK_SP1D_PATH
from .helpers import build_registry, builtin_materials

#: Artefacts ``extract.fom`` consumes; the quantum slot must supply these.
DOWNSTREAM_CONTRACT = ("quantum", "ns")


class QuantumContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.sp = cls.registry.select("physics.quantum.sp1d").manifest
        cls.analytic = cls.registry.select("physics.quantum.analytic_tw").manifest

    def test_both_declare_the_quantum_capability(self) -> None:
        required = SLOT_CAPABILITY["quantum"]
        self.assertIn(required, self.sp.capabilities)
        self.assertIn(required, self.analytic.capabilities)

    def test_both_produce_the_downstream_contract_artifacts(self) -> None:
        for key in DOWNSTREAM_CONTRACT:
            self.assertIn(key, self.sp.produces)
            self.assertIn(key, self.analytic.produces)

    def test_the_replacement_only_asks_for_available_inputs(self) -> None:
        # The contract is *not* "identical inputs" - the self-consistent solver
        # legitimately needs the material database as well, to read the channel
        # permittivity and effective mass. It is that every input is either
        # produced by a bound module or seeded by the executor.
        builtin = {
            "structure",
            "materials",
            "simulation",
            "device",
            "deck",
            "run",
            "gate",
            "geometry",
            "temperature_K",
        }
        produced: set[str] = set()
        for record in self.registry.records():
            produced.update(record.manifest.produces)
        for key in self.sp.requires:
            self.assertTrue(key in produced or key in builtin, msg=f"unobtainable: {key}")
        # Everything the analytic module needed is still needed.
        self.assertTrue(set(self.analytic.requires).issubset(set(self.sp.requires)))

    def test_both_are_discoverable_by_capability(self) -> None:
        found = {record.module_id for record in self.registry.select_by_capability("quantum")}
        self.assertEqual(
            found, {"physics.quantum.analytic_tw", "physics.quantum.sp1d"}
        )

    def test_the_extractor_knows_nothing_about_either_implementation(self) -> None:
        extractor = self.registry.select("extract.fom").manifest
        for key in DOWNSTREAM_CONTRACT:
            self.assertNotIn(key, extractor.produces)
        # The extractor's dependency is expressed purely through artifacts.
        self.assertNotIn("physics.quantum", " ".join(extractor.dependencies))

    def test_sp_module_is_honest_about_its_verification_state(self) -> None:
        self.assertEqual(self.sp.verification, "partially-verified")
        self.assertIn("Schroedinger-Poisson", self.sp.description)
        self.assertIn("drop-in", self.sp.description.lower())

    def test_sp_manifest_declares_calibratable_physics_parameters(self) -> None:
        names = {spec.name for spec in self.sp.calibratable_params()}
        self.assertIn("delta_ec_ratio", names)
        self.assertIn("phi_b_0_eV", names)
        # The warm-start prefactor must NOT be offered as a fit parameter.
        self.assertNotIn("gamma0_eV_m43", names)


class DeckSwapTests(unittest.TestCase):
    """Run both Decks and compare what comes out."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.materials = builtin_materials()
        cls.analytic_deck = load_deck(DECK_PATH)
        cls.sp_deck = load_deck(DECK_SP1D_PATH)
        cls.analytic_result, cls.analytic_ctx = cls._run(cls.analytic_deck)
        cls.sp_result, cls.sp_ctx = cls._run(cls.sp_deck)

    @classmethod
    def _run(cls, deck):
        ctx = SimContext(name=deck.name, bus=cls.registry.bus)
        result = Executor(cls.registry).run(
            Pipeline.from_deck(deck), ctx=ctx, deck=deck, materials=cls.materials
        )
        return result, ctx

    # -- both decks are valid against the same registry -------------------
    def test_analytic_deck_still_validates(self) -> None:
        errors, _warnings = self.analytic_deck.validate_all(self.registry, self.materials)
        self.assertEqual([str(e) for e in errors], [])

    def test_sp_deck_validates(self) -> None:
        errors, _warnings = self.sp_deck.validate_all(self.registry, self.materials)
        self.assertEqual([str(e) for e in errors], [])

    def test_both_decks_produce_an_artifact_reachable_pipeline(self) -> None:
        for deck in (self.analytic_deck, self.sp_deck):
            errors, _ = deck.validate_against_registry(self.registry)
            self.assertFalse(
                any(e.code == "artifact-unreachable" for e in errors), msg=deck.name
            )

    # -- identical pipeline shape ------------------------------------------
    def test_stage_slots_are_identical(self) -> None:
        analytic_slots = [stage.slot for stage in Pipeline.from_deck(self.analytic_deck).stages]
        sp_slots = [stage.slot for stage in Pipeline.from_deck(self.sp_deck).stages]
        self.assertEqual(analytic_slots, sp_slots)

    def test_only_the_quantum_binding_and_the_solver_slot_differ(self) -> None:
        # The self-consistent Deck additionally binds the nonlinear solver it
        # uses as a service, and that is the whole difference.
        analytic = self.analytic_deck.to_dict()
        sp = self.sp_deck.to_dict()
        self.assertNotEqual(
            analytic["physics"]["quantum"]["module"], sp["physics"]["quantum"]["module"]
        )
        analytic["physics"]["quantum"] = sp["physics"]["quantum"]
        analytic["modules"]["nonlinear"] = sp["modules"]["nonlinear"]
        analytic["name"] = sp["name"]
        analytic["run"] = sp["run"]
        self.assertEqual(analytic, sp)

    # -- both runs succeed --------------------------------------------------
    def test_both_runs_succeed(self) -> None:
        self.assertEqual(self.analytic_result.status, "success")
        self.assertEqual(self.sp_result.status, "success")

    def test_downstream_extractor_runs_unchanged(self) -> None:
        for result, ctx in (
            (self.analytic_result, self.analytic_ctx),
            (self.sp_result, self.sp_ctx),
        ):
            self.assertEqual(result.stage("extract:0").module_id, "extract.fom")
            self.assertEqual(result.stage("extract:0").status, "ok")
            self.assertTrue(ctx.has("results.fom"))
            self.assertIn("vth_est_V", ctx.get("results.fom"))

    def test_quantum_stage_reports_the_expected_module(self) -> None:
        self.assertEqual(
            self.analytic_result.stage("quantum").module_id,
            "physics.quantum.analytic_tw",
        )
        self.assertEqual(
            self.sp_result.stage("quantum").module_id, "physics.quantum.sp1d"
        )

    # -- the physics actually changed --------------------------------------
    def test_the_self_consistent_solution_moves_the_sheet_density(self) -> None:
        analytic_ns = float(self.analytic_ctx.get("ns"))
        sp_ns = float(self.sp_ctx.get("ns"))
        relative = abs(sp_ns - analytic_ns) / analytic_ns
        self.assertGreater(relative, 0.005, msg="the swap changed nothing measurable")
        self.assertLess(relative, 0.20, msg="the two models disagree unphysically")

    def test_both_results_respect_the_polarization_upper_bound(self) -> None:
        for ctx in (self.analytic_ctx, self.sp_ctx):
            quantum = ctx.get("quantum")
            self.assertLess(quantum["ns_cm2"], quantum["ns_upper_bound_cm2"])

    def test_sp_result_carries_self_consistent_detail(self) -> None:
        quantum = self.sp_ctx.get("quantum")
        self.assertEqual(quantum["model"], "self_consistent_sp1d")
        self.assertTrue(quantum["converged"])
        self.assertGreater(quantum["iterations"], 0)
        for key in (
            "subband_energies_eV",
            "subband_offsets_eV",
            "subband_densities_cm2",
            "profiles",
            "assumptions",
            "notes",
        ):
            self.assertIn(key, quantum, msg=key)
        self.assertGreater(len(quantum["subband_energies_eV"]), 1)
        self.assertEqual(
            len(quantum["profiles"]["z_nm"]), quantum["grid_nodes"]
        )

    def test_sp_result_documents_the_jacobian_accuracy(self) -> None:
        quantum = self.sp_ctx.get("quantum")
        assumptions = " ".join(quantum["assumptions"])
        self.assertIn("exact derivative", assumptions)
        self.assertIn("quadratically convergent", assumptions)
        self.assertEqual(quantum["interface_discretisation"], "face")
        self.assertIn("wavefunction response", quantum["jacobian"])

    def test_sp_result_reports_which_algorithm_ran(self) -> None:
        quantum = self.sp_ctx.get("quantum")
        self.assertEqual(quantum["algorithm"], "newton")
        self.assertGreater(quantum["function_evaluations"], 0)
        self.assertIsNotNone(quantum["linear_solves"])
        self.assertGreater(quantum["linear_solves"], 0)
        self.assertTrue(quantum["damping_history"])

    def test_the_two_runs_have_different_fingerprints(self) -> None:
        self.assertNotEqual(
            self.analytic_result.fingerprint, self.sp_result.fingerprint
        )


class VersionPinningTests(unittest.TestCase):
    """The new module must not disturb an already-validated Deck."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.analytic_deck = load_deck(DECK_PATH)
        cls.sp_deck = load_deck(DECK_SP1D_PATH)

    def test_the_analytic_deck_pins_its_implementation(self) -> None:
        binding = self.analytic_deck.physics.quantum
        self.assertEqual(binding.module, "physics.quantum.analytic_tw")
        self.assertEqual(binding.version, "^0.1.0")
        selected = self.registry.select(binding.module, binding.version)
        self.assertEqual(selected.manifest.module_version, "0.1.0")

    def test_the_pin_cannot_be_satisfied_by_the_new_module(self) -> None:
        # sp1d is 1.0.0, which is outside ^0.1.0, so the pinned Deck can never
        # be silently upgraded to it.
        self.assertFalse(
            self.registry.select(
                "physics.quantum.analytic_tw", "^0.1.0"
            ).manifest.module_version.startswith("1.")
        )
        with self.assertRaises(Exception):
            self.registry.select("physics.quantum.sp1d", "^0.1.0")

    def test_the_sp_deck_pins_the_sp_module(self) -> None:
        binding = self.sp_deck.physics.quantum
        self.assertEqual(binding.module, "physics.quantum.sp1d")
        self.assertEqual(binding.version, "^1.0.0")
        selected = self.registry.select(binding.module, binding.version)
        self.assertEqual(selected.manifest.module_version, "1.1.1")

    def test_swapping_the_binding_is_a_one_line_change(self) -> None:
        raw = json.loads(DECK_SP1D_PATH.read_text(encoding="utf-8"))
        raw["physics"]["quantum"] = json.loads(
            DECK_PATH.read_text(encoding="utf-8")
        )["physics"]["quantum"]
        swapped = Deck(**raw)
        errors, _warnings = swapped.validate_all(self.registry, builtin_materials())
        self.assertEqual([str(e) for e in errors], [])
        self.assertEqual(
            Pipeline.from_deck(swapped).stages[2].module_id,
            "physics.quantum.analytic_tw",
        )

    def test_hot_swapping_a_third_implementation_is_accepted(self) -> None:
        # A future module only has to declare the capability and produce the
        # artifacts; the registry accepts it and live instances are untouched.
        # A private registry keeps this mutation out of the other tests.
        registry = build_registry()
        base = registry.select("physics.quantum.sp1d").module_cls
        manifest = registry.select("physics.quantum.sp1d").manifest
        import dataclasses

        class ThirdImplementation(base):  # type: ignore[misc, valid-type]
            MANIFEST = dataclasses.replace(
                manifest, module_version="1.2.0", description="test double"
            )

        live = registry.instantiate("physics.quantum.sp1d", version="^1.0.0")
        report = registry.hot_swap(ThirdImplementation, origin="<test>")
        self.assertEqual(report.active_version, "1.2.0")
        self.assertIn("1.1.1", report.previous_versions)
        self.assertEqual(report.live_instances_unaffected, 1)
        self.assertEqual(
            registry.select("physics.quantum.sp1d").manifest.module_version, "1.2.0"
        )
        # Both versions coexist, and an exact pin still gets the old one.
        self.assertEqual(
            registry.select("physics.quantum.sp1d", "1.1.1").manifest.module_version,
            "1.1.1",
        )
        # The already-built instance still runs the version it was built from.
        self.assertEqual(live.state.value, "initialized")
        registry.release(live)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
