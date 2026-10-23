"""Lookup-table IO for the ``materials.dft`` plugin (``backend="table"``).

The lookup table is a *versioned data asset*: a JSON file of precomputed
first-principles material constants with per-value sources, uncertainties and
optional convergence records. In this phase every value is transcribed from
published DFT literature (see per-entry ``source``); nothing here was computed
in-house. That honesty is load-bearing: entries without a two-point
convergence record are flagged ``converged: false`` and are only allowed into
the merged database when the user explicitly opts in with
``override_policy="dft_wins"``.

Table schema (version 1)::

    {
      "version": "1.0.0",
      "provenance": {...},
      "numerics": {                  # nominal production settings the table
        "functional": "pbe",         # would be regenerated with; params that
        "pseudopotential_family": ..,# disagree make the table "stale" unless
        "ecut_wfc_ry": 60.0,         # allow_stale_cache=True
        "kspacing_1_A": 0.20,
        "sqs_seed": 1,
        "spin_orbit": false
      },
      "materials": {
        "GaN": {
          "source": "...", "verification": "partially-verified",
          "convergence": null | {"ecut_wfc_ry": {"values": [coarse, fine],
                                   "change_pct": ...}, ...,
                               "extrapolated": true},
          "values": {
            "a_lattice_A": {"value": 3.19, "lo": ..., "hi": ...,
                            "kind": "convergence|systematic|literature",
                            "source": "...", "functional": "pbe"?},
            "Eg_eV": {"value": 3.45, "functional": "hse06", ...},
            "dft_eg_uncorrected_eV": {"value": 1.72, "functional": "pbe", ...}
          }
        }
      }
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Value keys that may override builtin material properties.
OVERRIDABLE_KEYS: tuple[str, ...] = (
    "a_lattice_A",
    "c_lattice_A",
    "C13_GPa",
    "C33_GPa",
    "e31_C_m2",
    "e33_C_m2",
    "p_sp_C_m2",
    "m_eff",
    "Eg_eV",
)

#: Keys that are recorded into the merged database but never consumed by
#: builtin physics modules (isolated by their ``dft_`` prefix).
ATTACH_ONLY_KEYS: tuple[str, ...] = (
    "dft_eg_uncorrected_eV",
    "dft_eg_shift_eV",
)

#: Numerics parameters that make a cached/table result "stale" when they
#: differ between the table record and the module parameters. ``functional``
#: is deliberately excluded: it is governed by the eg_policy red line, not by
#: staleness (a table may legitimately hold PBE structure + HSE gaps).
STALE_NUMERIC_KEYS: tuple[str, ...] = (
    "pseudopotential_family",
    "ecut_wfc_ry",
    "kspacing_1_A",
    "sqs_seed",
    "spin_orbit",
)


class TableError(Exception):
    """A lookup table is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ValueEntry:
    """One tabulated constant with uncertainty and provenance."""

    value: float
    lo: float | None = None
    hi: float | None = None
    kind: str = "literature"
    source: str = ""
    functional: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, where: str) -> "ValueEntry":
        if not isinstance(raw, Mapping) or "value" not in raw:
            raise TableError(f"{where}: expected an object with a 'value' field")
        try:
            value = float(raw["value"])
        except (TypeError, ValueError) as exc:
            raise TableError(f"{where}: 'value' must be numeric, got {raw['value']!r}") from exc
        lo = raw.get("lo")
        hi = raw.get("hi")
        return cls(
            value=value,
            lo=None if lo is None else float(lo),
            hi=None if hi is None else float(hi),
            kind=str(raw.get("kind", "literature")),
            source=str(raw.get("source", "")),
            functional=(None if raw.get("functional") is None else str(raw["functional"])),
            note=str(raw.get("note", "")),
        )

    def uncertainty(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.lo is not None:
            out["lo"] = self.lo
        if self.hi is not None:
            out["hi"] = self.hi
        if self.source:
            out["source"] = self.source
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class MaterialEntry:
    """All tabulated DFT values for one material, plus convergence evidence."""

    name: str
    source: str
    verification: str
    values: Mapping[str, ValueEntry]
    convergence: Mapping[str, Any] | None = None
    notes: str = ""

    @property
    def converged(self) -> bool:
        """True only with an explicit two-point convergence record."""
        return bool(
            self.convergence
            and self.convergence.get("extrapolated")
            and isinstance(self.convergence.get("ecut_wfc_ry"), Mapping)
            and isinstance(self.convergence.get("kspacing_1_A"), Mapping)
        )


@dataclass(frozen=True)
class LookupTable:
    """A parsed, validated lookup table."""

    path: Path
    version: str
    provenance: Mapping[str, Any]
    numerics: Mapping[str, Any]
    materials: Mapping[str, MaterialEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "LookupTable":
        file_path = Path(path)
        if not file_path.is_file():
            raise TableError(
                f"DFT lookup table not found: {file_path} "
                f"(backend='table' needs a precomputed table; regenerate it with "
                f"backend='qe_local'/'qe_slurm' or point table_path at one)"
            )
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TableError(f"invalid lookup table {file_path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise TableError(f"{file_path}: table must be a JSON object")
        materials_raw = raw.get("materials")
        if not isinstance(materials_raw, Mapping) or not materials_raw:
            raise TableError(f"{file_path}: table has no 'materials' entries")

        materials: dict[str, MaterialEntry] = {}
        for name, entry_raw in materials_raw.items():
            where = f"{file_path}: materials.{name}"
            if not isinstance(entry_raw, Mapping):
                raise TableError(f"{where}: entry must be an object")
            values_raw = entry_raw.get("values")
            if not isinstance(values_raw, Mapping) or not values_raw:
                raise TableError(f"{where}: entry must carry a non-empty 'values' object")
            values: dict[str, ValueEntry] = {}
            for key, value_raw in values_raw.items():
                if key not in OVERRIDABLE_KEYS + ATTACH_ONLY_KEYS:
                    raise TableError(
                        f"{where}.values: unknown key {key!r}; "
                        f"known: {', '.join(OVERRIDABLE_KEYS + ATTACH_ONLY_KEYS)}"
                    )
                values[key] = ValueEntry.from_dict(value_raw, where=f"{where}.values.{key}")
            convergence = entry_raw.get("convergence")
            if convergence is not None and not isinstance(convergence, Mapping):
                raise TableError(f"{where}: 'convergence' must be an object or null")
            materials[str(name)] = MaterialEntry(
                name=str(name),
                source=str(entry_raw.get("source", "")),
                verification=str(entry_raw.get("verification", "unverified")),
                values=values,
                convergence=convergence,
                notes=str(entry_raw.get("notes", "")),
            )
        numerics = raw.get("numerics")
        return cls(
            path=file_path,
            version=str(raw.get("version", "0.0.0")),
            provenance=dict(raw.get("provenance", {})),
            numerics=dict(numerics) if isinstance(numerics, Mapping) else {},
            materials=materials,
        )

    def content_hash(self) -> str:
        """Hash of the raw table file: part of the reproducibility record."""
        blob = self.path.read_bytes()
        return hashlib.sha256(blob).hexdigest()[:16]


def numerics_mismatch(
    table: LookupTable, params: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Numerics params that disagree with the table's recorded settings.

    Returns ``{key: (param_value, table_value)}``; empty when consistent.
    """
    mismatch: dict[str, tuple[Any, Any]] = {}
    for key in STALE_NUMERIC_KEYS:
        recorded = table.numerics.get(key)
        wanted = params.get(key)
        if recorded is None or wanted is None:
            continue
        if isinstance(recorded, float) or isinstance(wanted, float):
            same = recorded is not None and wanted is not None and abs(float(recorded) - float(wanted)) < 1e-12
        else:
            same = recorded == wanted
        if not same:
            mismatch[key] = (wanted, recorded)
    return mismatch
