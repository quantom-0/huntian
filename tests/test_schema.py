"""Tests for the declarative schema layer."""

from __future__ import annotations

import unittest

from nemo.errors import SchemaError
from nemo.schema import (
    Bool,
    DictOf,
    Enum,
    Float,
    Int,
    ListOf,
    MISSING,
    Nested,
    Schema,
    Str,
    coerce_value,
)


class GateSchema(Schema):
    length_um = Float(0.15, minimum=0.0, exclusive_minimum=True, doc="gate length")
    kind = Enum(("schottky", "mis"), default="schottky")
    barrier_eV = Float(0.9, minimum=0.0, maximum=3.0, exclusive_maximum=True)


class DeviceSchema(Schema):
    name = Str("device")
    gate = Nested(GateSchema, required=True)
    fingers = Int(1, minimum=1, maximum=64)
    tags = ListOf(Str(), default=[])
    extras = DictOf(None, default={})


class CheckedSchema(Schema):
    low = Float(1.0)
    high = Float(2.0)

    def check(self):
        from nemo.errors import Issue

        if self.high <= self.low:
            return [Issue("high", "high must exceed low", "semantic")]
        return []


class SchemaBasicsTests(unittest.TestCase):
    def test_defaults_are_applied(self) -> None:
        gate = GateSchema()
        self.assertEqual(gate.length_um, 0.15)
        self.assertEqual(gate.kind, "schottky")
        self.assertEqual(gate.barrier_eV, 0.9)

    def test_values_are_coerced(self) -> None:
        gate = GateSchema(length_um=1)
        self.assertEqual(gate.length_um, 1.0)
        self.assertIsInstance(gate.length_um, float)

    def test_bool_is_not_accepted_as_number(self) -> None:
        with self.assertRaises(SchemaError):
            GateSchema(length_um=True)

    def test_fields_are_not_left_as_class_attributes(self) -> None:
        self.assertFalse(hasattr(GateSchema, "length_um"))
        self.assertIn("length_um", GateSchema.fields())

    def test_to_dict_roundtrip(self) -> None:
        device = DeviceSchema(name="dut", gate={"length_um": 0.25})
        again = DeviceSchema.from_dict(device.to_dict())
        self.assertEqual(device, again)
        self.assertEqual(device.to_dict(), again.to_dict())

    def test_update_returns_new_instance(self) -> None:
        device = DeviceSchema(name="dut", gate={"length_um": 0.25})
        updated = device.update(fingers=4)
        self.assertEqual(device.fingers, 1)
        self.assertEqual(updated.fingers, 4)
        self.assertEqual(updated.gate.length_um, 0.25)

    def test_equality_with_mapping(self) -> None:
        gate = GateSchema(length_um=0.2)
        self.assertEqual(gate, {"length_um": 0.2, "kind": "schottky", "barrier_eV": 0.9})


class SchemaValidationTests(unittest.TestCase):
    def test_unknown_field_is_rejected_with_hint(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            DeviceSchema.from_dict({"name": "dut", "gate": {"length_um": 0.1}, "finger": 2})
        codes = [issue.code for issue in caught.exception.issues]
        self.assertIn("unknown-field", codes)
        hint = caught.exception.issues[0].hint or ""
        self.assertIn("fingers", hint)

    def test_required_field_missing(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            DeviceSchema.from_dict({"name": "dut"})
        self.assertIn("required", [i.code for i in caught.exception.issues])

    def test_range_violation(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            DeviceSchema.from_dict({"gate": {"length_um": 0.0}})
        issue = caught.exception.issues[0]
        self.assertEqual(issue.code, "range")
        self.assertEqual(issue.path, "gate.length_um")

    def test_exclusive_minimum(self) -> None:
        with self.assertRaises(SchemaError):
            GateSchema(length_um=0.0)
        self.assertEqual(GateSchema(length_um=0.001).length_um, 0.001)

    def test_exclusive_maximum(self) -> None:
        with self.assertRaises(SchemaError):
            GateSchema(barrier_eV=3.0)
        self.assertEqual(GateSchema(barrier_eV=2.999).barrier_eV, 2.999)

    def test_enum_violation(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            GateSchema(kind="pn")
        self.assertEqual(caught.exception.issues[0].code, "enum")

    def test_type_violation(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            GateSchema(length_um="long")
        self.assertEqual(caught.exception.issues[0].code, "type")

    def test_nested_path_is_reported(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            DeviceSchema.from_dict({"gate": {"length_um": -1.0}})
        self.assertEqual(caught.exception.issues[0].path, "gate.length_um")

    def test_null_not_allowed(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            GateSchema(length_um=None)
        self.assertEqual(caught.exception.issues[0].code, "null")

    def test_list_item_paths(self) -> None:
        with self.assertRaises(SchemaError) as caught:
            DeviceSchema.from_dict({"gate": {"length_um": 0.1}, "fingers": 1, "tags": ["ok", 5]})
        self.assertEqual(caught.exception.issues[0].path, "tags[1]")

    def test_validate_dict_is_non_raising(self) -> None:
        instance, issues = DeviceSchema.validate_dict({"gate": {"length_um": -1}})
        self.assertIsNone(instance)
        self.assertTrue(issues)
        instance, issues = DeviceSchema.validate_dict({"gate": {"length_um": 1}})
        self.assertIsNotNone(instance)
        self.assertEqual(issues, [])

    def test_semantic_check_hook(self) -> None:
        self.assertIsNotNone(CheckedSchema(low=1.0, high=2.0))
        with self.assertRaises(SchemaError) as caught:
            CheckedSchema(low=3.0, high=1.0)
        self.assertEqual(caught.exception.issues[0].code, "semantic")

    def test_non_mapping_root_is_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            GateSchema.from_dict([1, 2, 3])  # type: ignore[arg-type]


class JsonSchemaTests(unittest.TestCase):
    def test_structure(self) -> None:
        schema = DeviceSchema.json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("gate", schema["properties"])
        self.assertIn("gate", schema["required"])
        self.assertFalse(schema["additionalProperties"])

    def test_field_keywords(self) -> None:
        properties = GateSchema.json_schema()["properties"]
        self.assertEqual(properties["length_um"]["type"], "number")
        self.assertEqual(properties["length_um"]["exclusiveMinimum"], 0.0)
        self.assertEqual(properties["length_um"]["default"], 0.15)
        self.assertEqual(properties["kind"]["enum"], ["schottky", "mis"])
        self.assertEqual(properties["barrier_eV"]["exclusiveMaximum"], 3.0)

    def test_nested_schema_is_inlined(self) -> None:
        nested = DeviceSchema.json_schema()["properties"]["gate"]
        self.assertEqual(nested["type"], "object")
        self.assertIn("length_um", nested["properties"])
        self.assertNotIn("$schema", nested)

    def test_unit_and_meta_extras(self) -> None:
        class WithUnit(Schema):
            thickness = Float(1.0, unit="nm", meta={"lab": "x"})

        properties = WithUnit.json_schema()["properties"]
        self.assertEqual(properties["thickness"]["x-unit"], "nm")
        self.assertEqual(properties["thickness"]["x-lab"], "x")


class CoerceValueTests(unittest.TestCase):
    def test_returns_value_and_no_issues(self) -> None:
        value, issues = coerce_value("p", Float(minimum=0.0), 3)
        self.assertEqual(value, 3.0)
        self.assertEqual(issues, [])

    def test_returns_issues(self) -> None:
        value, issues = coerce_value("p", Float(minimum=10.0), 3)
        self.assertIsNone(value)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].path, "p")

    def test_missing_sentinel(self) -> None:
        self.assertFalse(bool(MISSING))
        self.assertEqual(repr(MISSING), "<MISSING>")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
