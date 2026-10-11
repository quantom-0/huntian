"""Tests for pipeline assembly, ordering and execution."""

from __future__ import annotations

import json
import unittest

from huntian import Executor, MaterialDB, ModuleRegistry, Pipeline, SimContext, Stage
from huntian.deck import Deck, load_deck
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest

from . import DECK_PATH
from .helpers import build_registry, builtin_materials

RAW = json.loads(DECK_PATH.read_text(encoding="utf-8"))


class Boom(BaseModule):
    """A stage that always fails, used to exercise the failure paths."""

    MANIFEST = ModuleManifest(
        module_id="test.boom",
        module_version="1.0.0",
        category=ModuleCategory.EXTRACT,
        description="always raises",
    )

    def execute(self, ctx):  # noqa: ANN001
        raise RuntimeError("deliberate failure")


class PipelineAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.deck = load_deck(DECK_PATH)
        cls.pipeline = Pipeline.from_deck(cls.deck)

    def test_stage_order_from_deck(self) -> None:
        slots = [stage.slot for stage in self.pipeline.stages]
        self.assertEqual(slots, ["mesh", "polarization", "quantum", "extract:0"])

    def test_extractors_are_optional(self) -> None:
        self.assertTrue(self.pipeline.stages[-1].optional)
        self.assertFalse(self.pipeline.stages[0].optional)

    def test_ordering_is_topological(self) -> None:
        order = [stage.slot for stage in self.pipeline.ordered(self.registry)]
        self.assertEqual(order, ["mesh", "polarization", "quantum", "extract:0"])

    def test_describe(self) -> None:
        text = self.pipeline.describe()
        self.assertIn("mesh.structured2d", text)

    def test_requirements_split(self) -> None:
        self.assertEqual(len(self.pipeline.requirements()), 4)
        self.assertEqual(len(self.pipeline.requirements(include_optional=False)), 3)


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.deck = load_deck(DECK_PATH)
        cls.materials = builtin_materials()
        cls.ctx = SimContext(name="test-run", bus=cls.registry.bus)
        cls.result = Executor(cls.registry).run(
            Pipeline.from_deck(cls.deck), ctx=cls.ctx, deck=cls.deck, materials=cls.materials
        )

    def test_run_succeeds(self) -> None:
        self.assertEqual(self.result.status, "success")
        self.assertTrue(self.result.ok)
        self.assertEqual(self.result.errors, [])

    def test_every_stage_succeeded(self) -> None:
        self.assertEqual(len(self.result.stages), 4)
        for stage in self.result.stages:
            self.assertEqual(stage.status, "ok", msg=f"{stage.slot}: {stage.error}")

    def test_expected_artifacts_are_present(self) -> None:
        for key in ("mesh", "polarization", "quantum", "ns", "results.fom"):
            self.assertIn(key, self.result.artifacts)
            self.assertTrue(self.ctx.has(key))

    def test_seeded_inputs_are_available(self) -> None:
        for key in ("deck", "device", "structure", "gate", "simulation", "materials"):
            self.assertTrue(self.ctx.has(key), msg=key)

    def test_provenance_records_every_stage(self) -> None:
        stages = [record.stage for record in self.ctx.provenance]
        self.assertEqual(stages, ["mesh", "polarization", "quantum", "extract:0"])
        for record in self.ctx.provenance:
            self.assertEqual(record.status, "ok")

    def test_module_versions_are_recorded(self) -> None:
        versions = self.ctx.module_versions
        self.assertEqual(versions["mesh.structured2d"], "0.1.0")
        self.assertEqual(versions["extract.fom"], "0.1.0")

    def test_params_are_recorded_per_slot(self) -> None:
        self.assertEqual(self.ctx.params["mesh"]["nx"], 41)

    def test_fingerprint_is_reproducible(self) -> None:
        again_ctx = SimContext(name="test-run-2", bus=self.registry.bus)
        again = Executor(self.registry).run(
            Pipeline.from_deck(self.deck), ctx=again_ctx, deck=self.deck, materials=self.materials
        )
        self.assertEqual(self.result.fingerprint, again.fingerprint)

    def test_fingerprint_changes_with_a_parameter(self) -> None:
        raw = json.loads(json.dumps(RAW))
        raw["physics"]["polarization"]["params"]["screening"] = 0.6
        other = Deck(**raw)
        other_ctx = SimContext(name="other", bus=self.registry.bus)
        result = Executor(self.registry).run(
            Pipeline.from_deck(other), ctx=other_ctx, deck=other, materials=self.materials
        )
        self.assertEqual(result.status, "success")
        self.assertNotEqual(self.result.fingerprint, result.fingerprint)

    def test_metadata_snapshot_has_no_artifact_values(self) -> None:
        snapshot = self.ctx.snapshot()
        self.assertTrue(snapshot["artifacts"])
        for entry in snapshot["artifacts"]:
            self.assertNotIn("value", entry)
        self.assertIn("fingerprint", snapshot)

    def test_result_serialises(self) -> None:
        payload = self.result.to_dict()
        text = json.dumps(payload)
        self.assertIn("success", text)
        self.assertEqual(len(payload["stages"]), 4)


class FailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()
        self.registry.register(Boom, origin="test")
        self.deck = load_deck(DECK_PATH)
        self.materials = builtin_materials()

    def _run(self, *, optional: bool, strict: bool, registry: ModuleRegistry | None = None):
        registry = registry if registry is not None else self.registry
        pipeline = Pipeline(
            "failure-test",
            [
                Stage(slot="mesh", module_id="mesh.structured2d"),
                Stage(slot="polarization", module_id="physics.polarization.ambacher"),
                Stage(slot="boom", module_id="test.boom", optional=optional),
            ],
        )
        ctx = SimContext(name="failure-test", bus=registry.bus)
        return Executor(registry).run(
            pipeline, ctx=ctx, deck=self.deck, materials=self.materials, strict=strict
        )

    def test_strict_failure_aborts_the_run(self) -> None:
        result = self._run(optional=False, strict=True)
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("deliberate failure" in error for error in result.errors))

    def test_optional_failure_yields_partial(self) -> None:
        result = self._run(optional=True, strict=True)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.stage("boom").status, "failed")
        self.assertEqual(result.stage("mesh").status, "ok")

    def test_keep_going_records_failure_and_continues(self) -> None:
        result = self._run(optional=False, strict=False)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.stage("boom").status, "failed")

    def test_missing_module_fails_the_stage(self) -> None:
        pipeline = Pipeline(
            "missing",
            [
                Stage(slot="mesh", module_id="mesh.structured2d"),
                Stage(slot="ghost", module_id="test.does_not_exist"),
            ],
        )
        ctx = SimContext(bus=self.registry.bus)
        result = Executor(self.registry).run(
            pipeline, ctx=ctx, deck=self.deck, materials=self.materials, strict=True
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("test.does_not_exist", " ".join(result.errors))

    def test_missing_required_artifact_fails_the_stage(self) -> None:
        # `extract.fom` requires 'quantum', which is absent when polarization is skipped.
        pipeline = Pipeline(
            "missing-artifact",
            [Stage(slot="extract:0", module_id="extract.fom", optional=False)],
        )
        ctx = SimContext(bus=self.registry.bus)
        result = Executor(self.registry).run(
            pipeline, ctx=ctx, deck=self.deck, materials=self.materials, strict=True
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("requires artifact", " ".join(result.errors))

    def test_module_rejecting_parameters_fails_the_stage(self) -> None:
        pipeline = Pipeline(
            "bad-params",
            [
                Stage(slot="mesh", module_id="mesh.structured2d"),
                Stage(
                    slot="polarization",
                    module_id="physics.polarization.ambacher",
                    params={"screening": 42.0},
                ),
            ],
        )
        ctx = SimContext(bus=self.registry.bus)
        result = Executor(self.registry).run(
            pipeline, ctx=ctx, deck=self.deck, materials=self.materials, strict=True
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("invalid parameters", " ".join(result.errors))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
