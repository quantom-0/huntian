"""Dependency-free semantic versioning and requirement matching.

Supported version strings
-------------------------
``1.2.3``, ``1.2.3-rc1``, ``1.2``, ``1``

Supported requirement specs
---------------------------
===================  =========================================================
Spec                 Meaning
===================  =========================================================
``*`` / ``""``       any version
``1.2.3`` / ``=1.2.3``/ ``==1.2.3``   exact match
``1.2``              prefix match: ``>=1.2.0, <1.3.0``
``1``                prefix match: ``>=1.0.0, <2.0.0``
``!=1.2.3``          not equal
``>=``, ``<=``, ``>``, ``<``  zero-filled comparison
``^1.2.3``           compatible range: ``>=1.2.3, <2.0.0``
``^0.2.3``           ``>=0.2.3, <0.3.0``
``^0.0.3``           ``>=0.0.3, <0.0.4``
``~1.2.3`` / ``~=1.2.3``  ``>=1.2.3, <1.3.0``
``~1.2``             ``>=1.2.0, <1.3.0``
``,``                logical AND within a clause
``|``                logical OR between clauses
===================  =========================================================

This module intentionally avoids a dependency on ``packaging`` so the core
stays importable on a bare Python installation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import total_ordering

from .errors import RequirementError, VersionError

__all__ = ["Version", "VersionRange", "Requirement", "parse_requirement"]

__all__ += ["parse_version"]

_VERSION_RE = re.compile(
    r"^(?P<major>\d+)"
    r"(?:\.(?P<minor>\d+))?"
    r"(?:\.(?P<patch>\d+))?"
    r"(?:[-+](?P<pre>[0-9A-Za-z.\-]+))?$"
)

_PREDICATE_RE = re.compile(r"^(==|!=|>=|<=|>|<|\^|~=|~|=)?\s*(.+)$")

_CLAUSE_SPLIT_RE = re.compile(r"[<>=!~^]")


@total_ordering
@dataclass(frozen=True, eq=False)
class Version:
    """A parsed ``MAJOR.MINOR.PATCH[-prerelease]`` version."""

    major: int
    minor: int = 0
    patch: int = 0
    prerelease: tuple[str, ...] = ()

    # -- construction ---------------------------------------------------
    @classmethod
    def parse(cls, text: str | "Version") -> "Version":
        if isinstance(text, Version):
            return text
        if not isinstance(text, str):
            raise VersionError(f"version must be a string, got {type(text).__name__}")
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise VersionError(
                f"invalid version {text!r}; expected MAJOR[.MINOR[.PATCH]][-pre], e.g. '1.2.3'"
            )
        pre_raw = match.group("pre")
        prerelease = tuple(p for p in re.split(r"[.\-]", pre_raw) if p) if pre_raw else ()
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor") or 0),
            patch=int(match.group("patch") or 0),
            prerelease=prerelease,
        )

    # -- ordering -------------------------------------------------------
    def _sort_key(self) -> tuple:
        if not self.prerelease:
            return (self.major, self.minor, self.patch, 1, ())
        pre_key = tuple(
            (0, int(part)) if part.isdigit() else (1, part) for part in self.prerelease
        )
        return (self.major, self.minor, self.patch, 0, pre_key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._sort_key() == other._sort_key()

    def __hash__(self) -> int:
        return hash(self._sort_key())

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{'.'.join(self.prerelease)}" if self.prerelease else base

    def __repr__(self) -> str:
        return f"Version({str(self)!r})"

    # -- helpers --------------------------------------------------------
    @property
    def release(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def bump_major(self) -> "Version":
        return Version(self.major + 1, 0, 0)

    def bump_minor(self) -> "Version":
        return Version(self.major, self.minor + 1, 0)


def parse_version(text: str | Version) -> Version:
    """Convenience wrapper around :meth:`Version.parse`."""
    return Version.parse(text)


def _zero_filled(parts: list[int]) -> tuple[int, int, int]:
    parts = list(parts) + [0] * (3 - len(parts))
    return (parts[0], parts[1], parts[2])


def _prefix_upper(parts: list[int]) -> tuple[int, int, int]:
    """Smallest version strictly above the prefix ``parts``."""
    if len(parts) == 1:
        return (parts[0] + 1, 0, 0)
    if len(parts) == 2:
        return (parts[0], parts[1] + 1, 0)
    return (parts[0], parts[1], parts[2] + 1)


def _caret_upper(version: Version) -> tuple[int, int, int]:
    if version.major > 0:
        return version.bump_major().release
    if version.minor > 0:
        return Version(0, version.minor + 1, 0).release
    return Version(0, 0, version.patch + 1).release


def _tilde_upper(version: Version, given: int) -> tuple[int, int, int]:
    if given <= 1:
        return Version(version.major + 1, 0, 0).release
    return Version(version.major, version.minor + 1, 0).release


@dataclass(frozen=True)
class _Predicate:
    """A single ``op version`` constraint."""

    op: str
    version: Version
    given: int  # how many release components the user actually wrote

    def matches(self, candidate: Version) -> bool:
        op = self.op
        if op in ("==", "="):
            if self.given < 3:
                parts = [self.version.major]
                if self.given >= 2:
                    parts.append(self.version.minor)
                lower = _zero_filled(parts)
                upper = _prefix_upper(parts)
                return lower <= candidate.release < upper
            return candidate == self.version
        if op == "!=":
            if self.given < 3:
                parts = [self.version.major]
                if self.given >= 2:
                    parts.append(self.version.minor)
                lower = _zero_filled(parts)
                upper = _prefix_upper(parts)
                return not (lower <= candidate.release < upper)
            return candidate != self.version
        if op == ">=":
            return candidate >= self.version
        if op == "<=":
            return candidate <= self.version
        if op == ">":
            return candidate > self.version
        if op == "<":
            return candidate < self.version
        if op == "^":
            return self.version <= candidate < Version(*_caret_upper(self.version))
        if op in ("~", "~="):
            return self.version <= candidate < Version(*_tilde_upper(self.version, self.given))
        # Unreachable: the parser rejects unknown operators.
        raise VersionError(f"unsupported operator {op!r}")


@dataclass(frozen=True)
class VersionRange:
    """A boolean combination of version predicates.

    ``groups`` is a tuple of AND-clauses; the range matches when *any* clause
    matches (OR of ANDs). An empty ``groups`` tuple means "any version".
    """

    text: str = "*"
    groups: tuple[tuple[_Predicate, ...], ...] = field(default_factory=tuple)

    @classmethod
    def any(cls) -> "VersionRange":
        return cls("*", ())

    @classmethod
    def parse(cls, text: str | "VersionRange" | None) -> "VersionRange":
        if isinstance(text, VersionRange):
            return text
        raw = (text or "*").strip()
        if raw in ("", "*"):
            return cls(raw or "*", ())
        groups: list[tuple[_Predicate, ...]] = []
        for raw_group in raw.split("|"):
            predicates: list[_Predicate] = []
            for raw_predicate in raw_group.split(","):
                raw_predicate = raw_predicate.strip()
                if not raw_predicate:
                    continue
                predicates.append(_parse_predicate(raw_predicate))
            if predicates:
                groups.append(tuple(predicates))
        if not groups:
            return cls(raw, ())
        return cls(raw, tuple(groups))

    def matches(self, version: str | Version) -> bool:
        candidate = Version.parse(version) if isinstance(version, str) else version
        if not self.groups:
            return True
        return any(all(p.matches(candidate) for p in group) for group in self.groups)

    def __contains__(self, version: object) -> bool:
        return self.matches(version)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"VersionRange({self.text!r})"


def _parse_predicate(raw: str) -> _Predicate:
    match = _PREDICATE_RE.match(raw)
    if match is None:
        raise VersionError(f"invalid version predicate {raw!r}")
    op = match.group(1) or "=="
    version_text = match.group(2).strip()
    if version_text in ("*", "x", "X"):
        raise VersionError(
            f"wildcard {raw!r} is not a valid predicate; use '*' as the whole spec instead"
        )
    version = Version.parse(version_text)
    given = len(re.findall(r"\d+", version_text.split("-")[0]))
    given = max(1, min(3, given))
    return _Predicate(op=op, version=version, given=given)


@dataclass(frozen=True)
class Requirement:
    """A parsed ``<module-id><version-spec>`` requirement."""

    module_id: str
    version: VersionRange = field(default_factory=VersionRange.any)

    @classmethod
    def parse(cls, text: str | "Requirement") -> "Requirement":
        if isinstance(text, Requirement):
            return text
        if not isinstance(text, str):
            raise RequirementError(f"requirement must be a string, got {type(text).__name__}")
        raw = text.strip()
        if not raw:
            raise RequirementError("requirement string is empty")
        match = _CLAUSE_SPLIT_RE.search(raw)
        if match is None:
            parts = raw.split()
            if len(parts) == 2:
                module_id, spec = parts
            elif len(parts) == 1:
                module_id, spec = parts[0], "*"
            else:
                raise RequirementError(f"invalid requirement {text!r}")
        else:
            module_id, spec = raw[: match.start()].strip(), raw[match.start() :].strip()
            if not module_id:
                # e.g. ">=1.0" with no module id, or a typo like "huntian>="
                raise RequirementError(f"requirement {text!r} is missing a module id")
        if not module_id:
            raise RequirementError(f"requirement {text!r} is missing a module id")
        return cls(module_id=module_id, version=VersionRange.parse(spec))

    def matches(self, version: str | Version) -> bool:
        return self.version.matches(version)

    def __str__(self) -> str:
        return f"{self.module_id}{self.version}"

    def __repr__(self) -> str:
        return f"Requirement({self.module_id!r}, {self.version!r})"


def parse_requirement(text: str | Requirement) -> Requirement:
    """Convenience wrapper around :meth:`Requirement.parse`."""
    return Requirement.parse(text)
