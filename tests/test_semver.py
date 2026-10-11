"""Tests for version parsing and requirement matching."""

from __future__ import annotations

import unittest

from huntian.errors import RequirementError, VersionError
from huntian.semver import Requirement, Version, VersionRange, parse_requirement


class VersionTests(unittest.TestCase):
    def test_parse_full(self) -> None:
        version = Version.parse("1.2.3")
        self.assertEqual((version.major, version.minor, version.patch), (1, 2, 3))
        self.assertEqual(str(version), "1.2.3")

    def test_parse_partial_fills_zeros(self) -> None:
        self.assertEqual(str(Version.parse("0.1")), "0.1.0")
        self.assertEqual(str(Version.parse("2")), "2.0.0")

    def test_parse_prerelease(self) -> None:
        version = Version.parse("1.2.3-rc1")
        self.assertEqual(version.prerelease, ("rc1",))
        self.assertEqual(str(version), "1.2.3-rc1")

    def test_ordering(self) -> None:
        self.assertLess(Version.parse("1.2.3"), Version.parse("1.2.4"))
        self.assertLess(Version.parse("1.9.0"), Version.parse("1.10.0"))
        self.assertLess(Version.parse("1.2.3-rc1"), Version.parse("1.2.3"))
        self.assertGreater(Version.parse("2.0.0"), Version.parse("1.99.99"))

    def test_equality_and_hash(self) -> None:
        self.assertEqual(Version.parse("1.2"), Version.parse("1.2.0"))
        self.assertEqual(len({Version.parse("1.2.0"), Version.parse("1.2")}), 1)

    def test_invalid_versions(self) -> None:
        for text in ("v1.2", "1.2.3.4", "", "1.x", "abc"):
            with self.assertRaises(VersionError, msg=text):
                Version.parse(text)


class VersionRangeTests(unittest.TestCase):
    def test_any(self) -> None:
        for spec in ("*", "", None):
            self.assertTrue(VersionRange.parse(spec).matches("0.0.1"))
            self.assertTrue(VersionRange.parse(spec).matches("9.9.9"))

    def test_exact(self) -> None:
        self.assertTrue(VersionRange.parse("==1.2.3").matches("1.2.3"))
        self.assertFalse(VersionRange.parse("==1.2.3").matches("1.2.4"))
        self.assertTrue(VersionRange.parse("=1.2.3").matches("1.2.3"))

    def test_prefix_semantics(self) -> None:
        self.assertTrue(VersionRange.parse("1.2").matches("1.2.5"))
        self.assertFalse(VersionRange.parse("1.2").matches("1.3.0"))
        self.assertTrue(VersionRange.parse("1").matches("1.9.9"))
        self.assertFalse(VersionRange.parse("1").matches("2.0.0"))

    def test_comparison_operators(self) -> None:
        spec = VersionRange.parse(">=1.0,<2.0")
        self.assertTrue(spec.matches("1.0.0"))
        self.assertTrue(spec.matches("1.9.9"))
        self.assertFalse(spec.matches("0.9.9"))
        self.assertFalse(spec.matches("2.0.0"))

    def test_caret(self) -> None:
        self.assertTrue(VersionRange.parse("^1.2.3").matches("1.9.0"))
        self.assertFalse(VersionRange.parse("^1.2.3").matches("2.0.0"))
        self.assertTrue(VersionRange.parse("^0.2.3").matches("0.2.9"))
        self.assertFalse(VersionRange.parse("^0.2.3").matches("0.3.0"))
        self.assertTrue(VersionRange.parse("^0.0.3").matches("0.0.3"))
        self.assertFalse(VersionRange.parse("^0.0.3").matches("0.0.4"))

    def test_tilde(self) -> None:
        self.assertTrue(VersionRange.parse("~1.2.3").matches("1.2.9"))
        self.assertFalse(VersionRange.parse("~1.2.3").matches("1.3.0"))
        self.assertTrue(VersionRange.parse("~1").matches("1.9.9"))
        self.assertFalse(VersionRange.parse("~1").matches("2.0.0"))

    def test_not_equal(self) -> None:
        self.assertFalse(VersionRange.parse("!=1.2.3").matches("1.2.3"))
        self.assertTrue(VersionRange.parse("!=1.2.3").matches("1.2.4"))

    def test_or_clauses(self) -> None:
        spec = VersionRange.parse("^1.0.0|^2.0.0")
        self.assertTrue(spec.matches("1.5.0"))
        self.assertTrue(spec.matches("2.5.0"))
        self.assertFalse(spec.matches("3.0.0"))

    def test_wildcard_predicate_is_rejected(self) -> None:
        with self.assertRaises(VersionError):
            VersionRange.parse(">=*")

    def test_contains_protocol(self) -> None:
        self.assertIn("1.2.3", VersionRange.parse("^1.0.0"))


class RequirementTests(unittest.TestCase):
    def test_parse_with_spec(self) -> None:
        requirement = parse_requirement("physics.polarization.ambacher>=0.1,<1.0")
        self.assertEqual(requirement.module_id, "physics.polarization.ambacher")
        self.assertTrue(requirement.matches("0.1.0"))
        self.assertFalse(requirement.matches("1.0.0"))

    def test_parse_without_spec(self) -> None:
        requirement = parse_requirement("mesh.structured2d")
        self.assertEqual(requirement.module_id, "mesh.structured2d")
        self.assertTrue(requirement.matches("0.0.1"))

    def test_parse_with_space_separator(self) -> None:
        requirement = parse_requirement("mesh.structured2d >=0.1.0")
        self.assertEqual(requirement.module_id, "mesh.structured2d")
        self.assertTrue(requirement.matches("0.1.0"))

    def test_invalid_requirements(self) -> None:
        for text in ("", "   ", ">=1.0"):
            with self.assertRaises(RequirementError, msg=text):
                parse_requirement(text)

    def test_str_roundtrip(self) -> None:
        requirement = Requirement.parse("a.b^1.2.3")
        self.assertEqual(str(requirement), "a.b^1.2.3")
        self.assertEqual(Requirement.parse(str(requirement)), requirement)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
