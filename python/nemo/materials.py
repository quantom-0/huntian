"""Material parameter database.

Material constants live in versioned, cited data files rather than being
hard-coded inside physics modules. That separation matters for two reasons:

1. a physics module stays pure *form* (equations), while the numbers are
   swappable and auditable;
2. every constant carries a source and a verification status, which is what
   the accuracy programme in the design document requires.

The bundled database is loaded from package data. Callers may layer extra
material files on top with :meth:`MaterialDB.load` and :meth:`MaterialDB.merge`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import NemoError

__all__ = ["Material", "MaterialDB", "MaterialLookupError", "BUILTIN_MATERIALS_PATH"]

BUILTIN_MATERIALS_PATH = Path(__file__).resolve().parent / "data" / "materials.json"

#: Properties that participate in linear + bowing alloy interpolation.
DEFAULT_BOWING: dict[str, float] = {}


class MaterialLookupError(NemoError):
    """A material or material property is not available."""


@dataclass(frozen=True)
class Material:
    """A single material entry with provenance."""

    name: str
    props: Mapping[str, Any]
    source: str = ""
    verification: str = "unverified"
    notes: str = ""
    tags: tuple[str, ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        return self.props.get(key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.props[key]
        except KeyError as exc:
            raise MaterialLookupError(
                f"material {self.name!r} has no property {key!r}; "
                f"available: {sorted(self.props)}"
            ) from exc

    def require(self, *keys: str) -> dict[str, Any]:
        """Return the named properties, raising if any is missing."""
        return {key: self[key] for key in keys}

    def numeric(self, key: str) -> float:
        value = self[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MaterialLookupError(
                f"material {self.name!r} property {key!r} is not numeric (got {value!r})"
            )
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "verification": self.verification,
            "notes": self.notes,
            "tags": list(self.tags),
            "props": dict(self.props),
        }


@dataclass
class MaterialDB:
    """A collection of materials plus optional alloy interpolation metadata."""

    version: str = "0.0.0"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    alloys: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    # -- loading --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "MaterialDB":
        """Load a material database from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise MaterialLookupError(f"material database not found: {file_path}")
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MaterialLookupError(f"invalid material database {file_path}: {exc}") from exc
        return cls.from_dict(raw, source_path=file_path)

    @classmethod
    def builtin(cls) -> "MaterialDB":
        """Load the database shipped as package data."""
        return cls.load(BUILTIN_MATERIALS_PATH)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source_path: Path | None = None) -> "MaterialDB":
        materials: dict[str, Material] = {}
        for name, entry in (raw.get("materials") or {}).items():
            if not isinstance(entry, Mapping):
                raise MaterialLookupError(f"material {name!r} entry must be an object")
            props = entry.get("props")
            if not isinstance(props, Mapping):
                props = {k: v for k, v in entry.items() if k not in _MATERIAL_META_KEYS}
            materials[name] = Material(
                name=name,
                props=dict(props),
                source=str(entry.get("source", raw.get("provenance", {}).get("primary_source", ""))),
                verification=str(entry.get("verification", "unverified")),
                notes=str(entry.get("notes", "")),
                tags=tuple(entry.get("tags", ()) or ()),
            )
        return cls(
            version=str(raw.get("version", "0.0.0")),
            provenance=dict(raw.get("provenance", {})),
            materials=materials,
            alloys=dict(raw.get("alloys", {})),
        )

    def merge(self, other: "MaterialDB", *, override: bool = False) -> "MaterialDB":
        """Return a new database with ``other`` layered on top of this one."""
        merged = dict(self.materials)
        for name, material in other.materials.items():
            if name in merged and not override:
                continue
            merged[name] = material
        alloys = {**self.alloys, **other.alloys}
        return MaterialDB(
            version=other.version if other.version != "0.0.0" else self.version,
            provenance={**self.provenance, **other.provenance},
            materials=merged,
            alloys=alloys,
        )

    # -- lookup ---------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.materials

    def get(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError as exc:
            raise MaterialLookupError(
                f"unknown material {name!r}; available: {sorted(self.materials)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self.materials)

    @property
    def verification(self) -> str:
        return str(self.provenance.get("verification_status", "unverified"))

    @property
    def source(self) -> str:
        return str(self.provenance.get("primary_source", ""))

    # -- alloy interpolation --------------------------------------------
    def alloy(
        self,
        base: str,
        solute: str,
        x: float,
        key: str,
        *,
        bowing: float | None = None,
    ) -> float:
        """Linear (Vegard) interpolation with an optional bowing term.

        ``value(x) = (1 - x) * base + x * solute + bowing * x * (1 - x)``
        """
        if not 0.0 <= x <= 1.0:
            raise MaterialLookupError(f"alloy fraction x must be within [0, 1], got {x}")
        a = self.get(base).numeric(key)
        b = self.get(solute).numeric(key)
        bow = 0.0 if bowing is None else float(bowing)
        return (1.0 - x) * a + x * b + bow * x * (1.0 - x)

    def alloy_from_spec(self, alloy_name: str, x: float, key: str) -> float:
        """Interpolate using a named alloy recipe from the database."""
        spec = self.alloys.get(alloy_name)
        if spec is None:
            raise MaterialLookupError(
                f"unknown alloy {alloy_name!r}; available: {sorted(self.alloys)}"
            )
        endpoints = spec.get("endpoints") or []
        if len(endpoints) != 2:
            raise MaterialLookupError(f"alloy {alloy_name!r} must declare exactly two endpoints")
        bowing = (spec.get("bowing") or {}).get(key)
        return self.alloy(endpoints[0], endpoints[1], x, key, bowing=bowing)

    # -- reproducibility ------------------------------------------------
    def fingerprint(self) -> str:
        """Content hash of the database, for run reproducibility records."""
        payload = {
            "version": self.version,
            "materials": {name: dict(mat.props) for name, mat in sorted(self.materials.items())},
            "alloys": {k: dict(v) for k, v in sorted(self.alloys.items())},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provenance": dict(self.provenance),
            "alloys": {k: dict(v) for k, v in self.alloys.items()},
            "materials": {name: mat.to_dict() for name, mat in sorted(self.materials.items())},
        }

    def __len__(self) -> int:
        return len(self.materials)

    def __contains__(self, name: object) -> bool:
        return name in self.materials

    def __repr__(self) -> str:
        return f"MaterialDB(version={self.version!r}, materials={len(self.materials)})"


_MATERIAL_META_KEYS = {"source", "verification", "notes", "tags", "props"}


def available_properties(db: MaterialDB) -> Iterable[str]:
    """Union of all property names defined across a database (handy for docs)."""
    keys: set[str] = set()
    for material in db.materials.values():
        keys.update(material.props)
    return sorted(keys)
