"""Tests for the simulation context."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huntian.context import SimContext
from huntian.errors import ArtifactMissingError, ContextError
from huntian.events import EventBus
from huntian.materials import MaterialDB


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = SimContext(name="t")

    def test_put_get_require(self) -> None:
        self.ctx.put("ns", 9.2e12, producer="p", module_version="1.0.0")
        self.assertEqual(self.ctx.get("ns"), 9.2e12)
        self.assertEqual(self.ctx.require("ns"), 9.2e12)
        self.assertTrue(self.ctx.has("ns"))
        self.assertIn("ns", self.ctx)

    def test_get_default(self) -> None:
        self.assertIsNone(self.ctx.get("missing"))
        self.assertEqual(self.ctx.get("missing", 7), 7)

    def test_require_missing_lists_available(self) -> None:
        self.ctx.put("a", 1, producer="p")
        with self.assertRaises(ArtifactMissingError) as caught:
            self.ctx.require("b")
        message = str(caught.exception)
        self.assertIn("'b'", message)
        self.assertIn("a", message)

    def test_empty_key_is_rejected(self) -> None:
        with self.assertRaises(ContextError):
            self.ctx.put("", 1)

    def test_artifact_metadata(self) -> None:
        self.ctx.put("x", 1.0, producer="m", module_version="2.0.0", tags={"result"})
        artifact = self.ctx.artifact("x")
        self.assertEqual(artifact.producer, "m")
        self.assertEqual(artifact.module_version, "2.0.0")
        self.assertIn("result", artifact.tags)
        self.assertEqual(len(self.ctx.artifacts_with_tag("result")), 1)

    def test_keys_sorted_and_remove(self) -> None:
        self.ctx.put("b", 1, producer="p")
        self.ctx.put("a", 1, producer="p")
        self.assertEqual(self.ctx.keys(), ["a", "b"])
        self.assertTrue(self.ctx.remove("a"))
        self.assertFalse(self.ctx.remove("a"))


class ProvenanceTests(unittest.TestCase):
    def test_successful_stage_is_recorded(self) -> None:
        ctx = SimContext()
        with ctx.stage("mesh", "mesh.structured2d", "0.1.0", inputs=("structure",)) as scope:
            ctx.put("mesh", {}, producer="mesh.structured2d")
            scope.declare_output("mesh")
            scope.note("built 41x1621 nodes")
        record = ctx.provenance[0]
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.stage, "mesh")
        self.assertIn("mesh", record.outputs)
        self.assertEqual(record.notes, ("built 41x1621 nodes",))
        self.assertGreaterEqual(record.duration_s, 0.0)

    def test_failed_stage_records_and_reraises(self) -> None:
        ctx = SimContext()
        with self.assertRaises(ValueError):
            with ctx.stage("boom", "test.boom", "1.0.0"):
                raise ValueError("kaboom")
        record = ctx.provenance[0]
        self.assertEqual(record.status, "failed")
        self.assertIn("ValueError", record.error or "")
        self.assertIn("kaboom", record.error or "")

    def test_current_stage_tracking(self) -> None:
        ctx = SimContext()
        self.assertIsNone(ctx.current_stage)
        with ctx.stage("a", "m", "1.0.0"):
            self.assertEqual(ctx.current_stage, "a")
        self.assertIsNone(ctx.current_stage)

    def test_events_are_published(self) -> None:
        bus = EventBus()
        ctx = SimContext(bus=bus)
        with ctx.stage("s", "m", "1.0.0"):
            pass
        topics = [event.topic for event in bus.history()]
        self.assertIn("run.stage.start", topics)
        self.assertIn("run.stage.end", topics)

    def test_handler_errors_do_not_break_publish(self) -> None:
        bus = EventBus()

        def bad_handler(event):  # noqa: ANN001
            raise RuntimeError("subscriber bug")

        bus.subscribe("*", bad_handler)
        bus.publish("anything")
        self.assertEqual(len(bus.handler_errors), 1)


class FingerprintTests(unittest.TestCase):
    def test_identical_inputs_give_identical_fingerprint(self) -> None:
        materials = MaterialDB.builtin()
        first = SimContext(materials=materials)
        second = SimContext(materials=materials)
        for ctx in (first, second):
            ctx.set_params("mesh", {"nx": 41})
            ctx.record_module_version("mesh.structured2d", "0.1.0")
            ctx.mark_fingerprint_input("layers", [{"thickness_nm": 20.0}])
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_parameters_change_the_fingerprint(self) -> None:
        materials = MaterialDB.builtin()
        first = SimContext(materials=materials)
        first.set_params("mesh", {"nx": 41})
        second = SimContext(materials=materials)
        second.set_params("mesh", {"nx": 81})
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_module_version_changes_the_fingerprint(self) -> None:
        materials = MaterialDB.builtin()
        first = SimContext(materials=materials)
        first.record_module_version("m", "1.0.0")
        second = SimContext(materials=materials)
        second.record_module_version("m", "1.0.1")
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_material_database_changes_the_fingerprint(self) -> None:
        first = SimContext(materials=MaterialDB.builtin())
        second = SimContext(materials=MaterialDB(version="9.9.9"))
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_fingerprint_is_short_and_stable(self) -> None:
        ctx = SimContext()
        value = ctx.fingerprint()
        self.assertEqual(len(value), 16)
        self.assertEqual(value, ctx.fingerprint())


class SnapshotTests(unittest.TestCase):
    def test_snapshot_excludes_artifact_values(self) -> None:
        ctx = SimContext()
        ctx.put("big", list(range(1000)), producer="m")
        snapshot = ctx.snapshot()
        entry = snapshot["artifacts"][0]
        self.assertNotIn("value", entry)
        self.assertEqual(entry["value_type"], "list")
        self.assertEqual(entry["key"], "big")

    def test_snapshot_is_json_serialisable(self) -> None:
        ctx = SimContext(materials=MaterialDB.builtin())
        ctx.set_params("mesh", {"nx": 41})
        ctx.put("x", 1.0, producer="m")
        text = json.dumps(ctx.snapshot(), default=str)
        self.assertIn("fingerprint", text)

    def test_save_metadata(self) -> None:
        ctx = SimContext()
        with tempfile.TemporaryDirectory() as tmp:
            path = ctx.save_metadata(Path(tmp) / "nested" / "run.json")
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], ctx.run_id)

    def test_summary_counts(self) -> None:
        ctx = SimContext()
        with ctx.stage("ok", "m", "1.0.0"):
            pass
        with self.assertRaises(RuntimeError):
            with ctx.stage("bad", "m", "1.0.0"):
                raise RuntimeError("x")
        summary = ctx.summary()
        self.assertEqual(summary["stages_ok"], 1)
        self.assertEqual(summary["stages_failed"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
