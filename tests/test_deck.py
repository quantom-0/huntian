"""Tests for Deck parsing and two-stage validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from huntian.deck import (
    SLOT_ORDER,
    Deck,
    deck_json_schema,
    dump_deck,
    load_deck,
    loads_deck,
    yaml_available,
)
from huntian.errors import DeckError, SchemaError
from huntian.materials import MaterialDB

from . import DECK_PATH
from .helpers import build_registry

RAW = json.loads(DECK_PATH.read_text(encoding="utf-8"))


def _raw_with(**mutations) -> dict:
    """Deep-ish copy of the demo deck with dotted-path mutations applied."""
    clone = json.loads(json.dumps(RAW))
    for path, value in mutations.items():
        cursor = clone
        parts = path.split("__")
        for part in parts[:-1]:
            cursor = cursor[int(part)] if part.isdigit() else cursor[part]
        cursor[parts[-1]] = value
    return clone


class DeckLoadTests(unittest.TestCase):
    def test_load_demo_deck(self) -> None:
        deck = load_deck(DECK_PATH)
        self.assertEqual(deck.name, "demo_algan_gan_hemt")
        self.assertEqual(deck.schema_version, "1.0")
        self.assertEqual(len(deck.device.layers), 3)
        self.assertEqual(deck.device.gate.length_um, 0.15)

    def test_describe_mentions_bindings(self) -> None:
        text = load_deck(DECK_PATH).describe()
        self.assertIn("mesh.structured2d", text)
        self.assertIn("physics.polarization.ambacher", text)

    def test_missing_file(self) -> None:
        with self.assertRaises(DeckError):
            load_deck("does/not/exist.json")

    def test_invalid_json_reports_position(self) -> None:
        with self.assertRaises(DeckError) as caught:
            loads_deck("{not json", fmt="json")
        self.assertIn("invalid JSON", str(caught.exception))

    def test_non_mapping_root(self) -> None:
        with self.assertRaises(DeckError):
            loads_deck("[1, 2, 3]", fmt="json")

    def test_dump_and_reload_roundtrip(self) -> None:
        deck = load_deck(DECK_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out" / "deck.json"
            dump_deck(deck, target)
            self.assertTrue(target.is_file())
            self.assertEqual(load_deck(target), deck)

    def test_json_schema_shape(self) -> None:
        schema = deck_json_schema()
        self.assertEqual(schema["title"], "Huntian Deck")
        for key in ("name", "device", "physics", "modules", "simulation"):
            self.assertIn(key, schema["properties"])
        self.assertIn("name", schema["required"])

    def test_layers_helper_lookup(self) -> None:
        device = load_deck(DECK_PATH).device
        self.assertIsNotNone(device.layer_by_role("barrier"))
        self.assertIsNotNone(device.layer_by_name("channel"))
        self.assertIsNone(device.layer_by_role("spacer"))


class YamlTests(unittest.TestCase):
    def test_yaml_is_optional(self) -> None:
        yaml_path = DECK_PATH.with_suffix(".yaml")
        if yaml_available():
            deck = load_deck(yaml_path)
            self.assertEqual(deck.name, load_deck(DECK_PATH).name)
        else:
            with self.assertRaises(DeckError) as caught:
                load_deck(yaml_path)
            self.assertIn("PyYAML", str(caught.exception))


class StructuralRejectionTests(unittest.TestCase):
    def test_unknown_top_level_field(self) -> None:
        raw = _raw_with(nmae="typo")
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("nmae", str(caught.exception))

    def test_unknown_nested_field_reports_path(self) -> None:
        raw = _raw_with(device__gate__lenght_um=0.2)
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        paths = [issue.path for issue in caught.exception.issues]
        self.assertIn("device.gate.lenght_um", paths)

    def test_missing_required_layer(self) -> None:
        raw = _raw_with(device__layers=[])
        with self.assertRaises(SchemaError):
            Deck(**raw)

    def test_barrier_without_aluminium_is_rejected(self) -> None:
        raw = _raw_with(device__layers__0__x_al=0.0)
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("aluminium", str(caught.exception))

    def test_missing_barrier_role_is_rejected(self) -> None:
        raw = _raw_with(device__layers__0__role="spacer")
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("barrier", str(caught.exception))

    def test_barrier_below_channel_is_rejected(self) -> None:
        raw = json.loads(json.dumps(RAW))
        raw["device"]["layers"] = [raw["device"]["layers"][1], raw["device"]["layers"][0]]
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("above the channel", str(caught.exception))

    def test_duplicate_layer_names_are_rejected(self) -> None:
        raw = _raw_with(device__layers__1__name="barrier")
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("duplicate", str(caught.exception))

    def test_inverted_sweep_range_is_rejected(self) -> None:
        raw = _raw_with(simulation__sweep__vgs=[2.0, -6.0])
        with self.assertRaises(SchemaError) as caught:
            Deck(**raw)
        self.assertIn("greater than", str(caught.exception))

    def test_non_positive_temperature_is_rejected(self) -> None:
        raw = _raw_with(simulation__temperature_K=0.0)
        with self.assertRaises(SchemaError):
            Deck(**raw)


class BindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck = load_deck(DECK_PATH)

    def test_binding_order_follows_slot_order(self) -> None:
        slots = [slot for slot, _binding in self.deck.physics_bindings()]
        self.assertEqual(slots, [s for s in SLOT_ORDER if s in slots])
        self.assertEqual(self.deck.enabled_bindings()[-1][0], "extract:0")

    def test_requirements_include_version_specs(self) -> None:
        requirements = self.deck.requirements()
        self.assertIn("physics.polarization.ambacher>=0.1,<1.0", requirements)
        self.assertIn("physics.quantum.analytic_tw^0.1.0", requirements)
        self.assertIn("mesh.structured2d", requirements)

    def test_disabled_binding_is_skipped(self) -> None:
        raw = _raw_with(physics__quantum__enabled=False)
        deck = Deck(**raw)
        self.assertNotIn("physics.quantum.analytic_tw", " ".join(deck.requirements()))


class ContextualValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.materials = MaterialDB.builtin()
        cls.deck = load_deck(DECK_PATH)

    def test_demo_deck_is_valid(self) -> None:
        errors, _warnings = self.deck.validate_all(self.registry, self.materials)
        self.assertEqual([str(e) for e in errors], [])

    def test_warnings_mention_unverified_modules(self) -> None:
        _errors, warnings = self.deck.validate_all(self.registry, self.materials)
        self.assertTrue(any("unverified" in warning for warning in warnings))

    def test_unknown_parameter_is_reported(self) -> None:
        deck = Deck(**_raw_with(physics__polarization__params={"screning": 0.5}))
        errors, _warnings = deck.validate_against_registry(self.registry)
        self.assertTrue(any("screning" in str(error) for error in errors))

    def test_out_of_range_parameter_is_reported(self) -> None:
        deck = Deck(**_raw_with(physics__polarization__params={"screening": 1.5}))
        errors, _warnings = deck.validate_against_registry(self.registry)
        self.assertTrue(any(error.code == "range" for error in errors))

    def test_unknown_module_is_reported(self) -> None:
        deck = Deck(**_raw_with(physics__polarization__module="physics.nope"))
        errors, _warnings = deck.validate_against_registry(self.registry)
        self.assertTrue(any(error.code == "module-unavailable" for error in errors))

    def test_unsatisfiable_version_is_reported(self) -> None:
        deck = Deck(**_raw_with(physics__polarization__version="^9.0.0"))
        errors, _warnings = deck.validate_against_registry(self.registry)
        self.assertTrue(any(error.code == "module-unavailable" for error in errors))

    def test_unknown_material_is_reported(self) -> None:
        deck = Deck(**_raw_with(device__layers__1__material="Unobtainium"))
        errors = deck.validate_against_materials(self.materials)
        self.assertTrue(any("Unobtainium" in str(error) for error in errors))

    def test_artifact_reachability_passes_for_demo(self) -> None:
        errors, _warnings = self.deck.validate_against_registry(self.registry)
        self.assertFalse(any(error.code == "artifact-unreachable" for error in errors))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
