"""``materials.dft`` module implementation: lifecycle, red lines, merge.

The module loads the builtin material database from the context, layers the
DFT values (lookup table in this phase; Quantum ESPRESSO drivers later) on
top as a *replacement* database, and re-publishes the ``materials`` artifact
so downstream modules - ``physics.polarization.ambacher``,
``physics.quantum.sp1d`` - keep working untouched.

Precision red lines (design document §5) live here, in code:

1. ``eg_policy="reject_pbe"`` (default): a PBE band gap must never feed the
   device model. ``validate()`` returns a hard issue, the stage refuses to
   run.
2. ``eg_policy="scissor_to_reference"``: PBE gaps are rigidly shifted to the
   reference gap, the shift is recorded in ``dft_eg_shift_eV`` and the
   entry verification is capped at ``partially-verified``.
3. Convergence evidence: entries without a two-point convergence record are
   only adopted when the user explicitly opts in (``override_policy``), are
   flagged ``converged: false`` in the override report, and are listed in
   the provenance assumptions - never used silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from huntian.errors import Issue, ModuleConfigError
from huntian.materials import Material, MaterialDB
from huntian.registry import BaseModule

from .table_io import (
    ATTACH_ONLY_KEYS,
    OVERRIDABLE_KEYS,
    LookupTable,
    MaterialEntry,
    TableError,
    ValueEntry,
    numerics_mismatch,
)

#: Lookup table bundled with this plugin. Resolved relative to this file
#: (``__file__`` is always absolute under the plugin loader, unlike the
#: synthetic package's ``submodule_search_locations``), so the default table
#: works from any working directory - including after ``pip install``.
_BUNDLED_TABLE = Path(__file__).resolve().parent / "data" / "algan_gan.json"

#: Properties the DFT layer is allowed to override, excluding the gap (which
#: is governed by the eg_policy red line).
_NON_GAP_KEYS: tuple[str, ...] = tuple(k for k in OVERRIDABLE_KEYS if k != "Eg_eV")

#: Honest boundaries, sp1d-style; recorded in every dft_provenance artifact.
ASSUMPTIONS: tuple[str, ...] = (
    "DFT values are 0 K, ideal-crystal, defect-free estimates; real MOCVD "
    "material spread (dislocations, impurities, composition fluctuations) is "
    "absorbed in the quoted uncertainties, not modelled.",
    "Band gaps carry a functional-dependent systematic error: PBE "
    "underestimates the GaN gap by ~1.7 eV versus experiment - that is "
    "expected behaviour, not a bug; PBE raw gaps are recorded but never "
    "adopted unless eg_policy='scissor_to_reference'.",
    "Alloy values come from endpoint (Vegard + builtin bowing) interpolation "
    "of DFT end-member values; SQS supercell statistics are a later phase.",
    "No temperature corrections: thermal_conductivity, thermal_exponent and "
    "other thermal keys stay from the builtin library.",
    "No defect or disorder corrections to polarization.",
)


class DftMaterials(BaseModule):
    """First-principles material parameter provider (materials slot)."""

    def on_initialize(self) -> None:
        backend = str(self.param("backend"))
        if backend == "table":
            # Resolve now so a missing table fails at initialize time with a
            # clear message instead of mid-run.
            self._table_path = self._resolve_table_path(str(self.param("table_path")))
            if not self._table_path.is_file():
                raise ModuleConfigError(
                    f"{self.module_id}: lookup table not found: {self._table_path} "
                    f"(backend='table' serves precomputed JSON; produce one with "
                    f"backend='qe_local'/'qe_slurm' or fix table_path)"
                )
        elif backend in ("qe_local", "qe_slurm"):
            from .qe_driver import backend_unavailable

            reason = backend_unavailable(backend)
            if reason is not None:
                raise ModuleConfigError(f"{self.module_id}: {reason}")
            raise ModuleConfigError(
                f"{self.module_id}: backend={backend!r} is declared but its driver "
                f"is not implemented yet (DFT roadmap phase 2); use backend='table'"
            )
        else:  # pragma: no cover - manifest choices make this unreachable
            raise ModuleConfigError(f"{self.module_id}: unknown backend {backend!r}")
        self._cache_dir = self._resolve_path(str(self.param("cache_dir")))

    # -- validation: the red lines -----------------------------------------
    def validate(self, ctx: Any) -> list[Issue]:
        issues: list[Issue] = []
        functional = str(self.param("functional"))
        eg_policy = str(self.param("eg_policy"))

        # Red line 1: PBE gaps must not reach gap-consuming modules.
        if functional == "pbe" and eg_policy == "reject_pbe":
            issues.append(
                Issue(
                    "params.functional",
                    "functional='pbe' with eg_policy='reject_pbe': the band gap "
                    "feeds physics.quantum.sp1d (delta_ec_ratio x dEg) and the "
                    "polarization band offsets, and PBE gaps are systematically "
                    "underestimated; refusing to run. Use functional='hse06' or "
                    "eg_policy='scissor_to_reference'.",
                    "eg-policy",
                )
            )

        # Red line 3 (stale numerics): numerical settings that disagree with
        # the table's recorded production settings are the most dangerous
        # reuse - refuse unless the user explicitly accepts them.
        table = self._load_table(issues)
        if table is not None:
            mismatch = numerics_mismatch(table, self.params)
            if mismatch and not bool(self.param("allow_stale_cache")):
                details = "; ".join(
                    f"{key}: param={want!r} vs table={got!r}"
                    for key, (want, got) in sorted(mismatch.items())
                )
                issues.append(
                    Issue(
                        "params.numerics",
                        f"numerical settings differ from the lookup table record ({details}); "
                        f"serving them anyway would be an unlabelled accuracy change. "
                        f"Regenerate the table or set allow_stale_cache=true to accept.",
                        "stale-cache",
                    )
                )
        return issues

    # -- execution ----------------------------------------------------------
    def execute(self, ctx: Any) -> Any:
        issues: list[Issue] = []
        table = self._load_table(issues)
        if table is None:  # pragma: no cover - validate() gates this
            raise TableError("lookup table unavailable")

        base: MaterialDB = ctx.require("materials")
        functional = str(self.param("functional"))
        eg_policy = str(self.param("eg_policy"))
        override_policy = str(self.param("override_policy"))
        stale = numerics_mismatch(table, self.params)

        override_materials: dict[str, Material] = {}
        override_report: list[dict[str, Any]] = []
        provenance_entries: dict[str, dict[str, Any]] = {}
        cache_hits = 0

        for name, entry in sorted(table.materials.items()):
            if not base.has(name):
                provenance_entries[name] = {
                    "source": entry.source,
                    "verification": "unverified",
                    "skipped": "material not present in the builtin database",
                }
                continue
            cache_hits += 1
            builtin = base.get(name)
            props = dict(builtin.props)
            uncertainties: dict[str, Any] = {}
            adopted_notes: list[str] = []

            for key in _NON_GAP_KEYS:
                value_entry = entry.values.get(key)
                if value_entry is None:
                    continue
                adopted = self._adopt_non_gap(override_policy, entry)
                if adopted:
                    props[key] = value_entry.value
                    uncertainties[key] = value_entry.uncertainty()
                override_report.append(
                    self._report_row(name, key, builtin, value_entry, adopted, entry)
                )

            # ---- band gap: governed by the eg_policy red line --------------
            gap_entry = entry.values.get("Eg_eV")
            if gap_entry is not None:
                adopted, shift, note = self._adopt_gap(
                    builtin, entry, gap_entry, functional, eg_policy, override_policy
                )
                if adopted:
                    if shift is not None:
                        # Scissor mode: the PBE gap is rigidly shifted ONTO the
                        # reference gap, so the adopted value IS the reference
                        # value; the raw PBE number stays in
                        # dft_eg_uncorrected_eV and the shift in dft_eg_shift_eV.
                        props["Eg_eV"] = builtin.numeric("Eg_eV")
                    else:
                        props["Eg_eV"] = gap_entry.value
                    uncertainties["Eg_eV"] = gap_entry.uncertainty()
                if shift is not None:
                    props["dft_eg_shift_eV"] = shift
                if note:
                    adopted_notes.append(note)
                override_report.append(
                    self._report_row(name, "Eg_eV", builtin, gap_entry, adopted, entry)
                )
            raw_gap = entry.values.get("dft_eg_uncorrected_eV")
            if raw_gap is not None:
                # Recorded no matter what; consumed by nothing downstream.
                props["dft_eg_uncorrected_eV"] = raw_gap.value
                uncertainties["dft_eg_uncorrected_eV"] = raw_gap.uncertainty()

            if uncertainties:
                props["dft_uncertainty"] = uncertainties

            verification = entry.verification
            if any("scissor" in n for n in adopted_notes) and verification == "verified":
                verification = "partially-verified"

            override_materials[name] = Material(
                name=name,
                props=props,
                source=self._source_string(table, entry),
                verification=verification,
                notes=builtin.notes,
                tags=builtin.tags,
            )
            provenance_entries[name] = {
                "source": self._source_string(table, entry),
                "verification": verification,
                "converged": entry.converged,
                "uncertainty": uncertainties,
                "assumptions": list(ASSUMPTIONS) + adopted_notes + [entry.notes]
                if entry.notes
                else list(ASSUMPTIONS) + adopted_notes,
            }

        override_db = MaterialDB(
            version=f"dft-table-{table.version}",
            provenance={
                "primary_source": f"materials.dft lookup table {table.path.name} "
                f"(content {table.content_hash()})",
                "verification_status": "partially-verified",
            },
            materials=override_materials,
        )
        merged = base.merge(override_db, override=True)

        assumptions = list(ASSUMPTIONS)
        if stale:
            assumptions.append(
                "Served despite stale numerics (allow_stale_cache=true): "
                + "; ".join(
                    f"{key} param={want!r} vs table={got!r}"
                    for key, (want, got) in sorted(stale.items())
                )
            )
        if any(not row["converged"] for row in override_report):
            assumptions.append(
                "Entries without two-point convergence records are adopted "
                "(override_policy opt-in) but flagged converged=false in the "
                "override report; treat them as downweighted."
            )

        provenance = {
            "backend": "table",
            "functional": functional,
            "pseudopotential_family": str(self.param("pseudopotential_family")),
            "numerics": {
                "ecut_wfc_ry": float(self.param("ecut_wfc_ry")),
                "kspacing_1_A": float(self.param("kspacing_1_A")),
                "sqs_seed": int(self.param("sqs_seed")),
                "spin_orbit": bool(self.param("spin_orbit")),
                "table_version": table.version,
                "table_content_hash": table.content_hash(),
            },
            "cache_hits": cache_hits,
            "cache_misses": 0,
            "entries": provenance_entries,
            "override_report": override_report,
            "assumptions": assumptions,
        }

        # The artifact every downstream module consumes, plus the audit trail.
        ctx.materials = merged
        ctx.put(
            "materials",
            merged,
            producer=type(self).MANIFEST.module_id,
            module_version=type(self).MANIFEST.module_version,
        )
        ctx.put(
            "dft_provenance",
            provenance,
            producer=type(self).MANIFEST.module_id,
            module_version=type(self).MANIFEST.module_version,
        )
        return ctx

    # -- adoption rules -----------------------------------------------------
    def _adopt_non_gap(self, override_policy: str, entry: MaterialEntry) -> bool:
        if override_policy == "reference_wins":
            return False
        # "dft_wins" and "dft_only_gapless" both adopt non-gap properties;
        # non-converged entries are adopted but flagged in the report.
        return True

    def _adopt_gap(
        self,
        builtin: Material,
        entry: MaterialEntry,
        gap: ValueEntry,
        functional: str,
        eg_policy: str,
        override_policy: str,
    ) -> tuple[bool, float | None, str | None]:
        """Decide the band-gap adoption; returns (adopted, scissor_shift, note)."""
        if override_policy in ("reference_wins", "dft_only_gapless"):
            return False, None, f"gap not adopted under override_policy={override_policy!r}"
        gap_functional = gap.functional or str(
            entry.convergence.get("functional", "pbe") if entry.convergence else "pbe"
        )
        if gap_functional == "pbe":
            if functional == "pbe" and eg_policy == "reject_pbe":
                # validate() already refuses; never reached through a stage.
                return False, None, "PBE gap rejected by eg_policy=reject_pbe"
            if eg_policy == "scissor_to_reference":
                reference = builtin.numeric("Eg_eV")
                shift = gap.value - reference
                return True, shift, (
                    f"PBE gap scissor-shifted by {shift:+.3f} eV to the reference "
                    f"value {reference:.3f} eV; verification capped at partially-verified"
                )
            return False, None, "PBE gap recorded but not adopted"
        if gap_functional != functional:
            return False, None, (
                f"gap functional {gap_functional!r} does not match the requested "
                f"functional {functional!r}; builtin value kept"
            )
        return True, None, None

    # -- helpers -------------------------------------------------------------
    def _report_row(
        self,
        name: str,
        key: str,
        builtin: Material,
        value_entry: ValueEntry,
        adopted: bool,
        entry: MaterialEntry,
    ) -> dict[str, Any]:
        builtin_value = builtin.get(key)
        delta_pct = None
        if isinstance(builtin_value, (int, float)) and not isinstance(builtin_value, bool):
            if builtin_value:
                delta_pct = round(
                    (value_entry.value - float(builtin_value)) / abs(float(builtin_value)) * 100.0,
                    3,
                )
        return {
            "material": name,
            "property": key,
            "builtin": builtin_value,
            "dft": value_entry.value,
            "adopted": "dft" if adopted else "reference",
            "delta_pct": delta_pct,
            "converged": entry.converged,
            "source": value_entry.source or entry.source,
        }

    def _source_string(self, table: LookupTable, entry: MaterialEntry) -> str:
        return (
            f"DFT (table v{table.version}, content {table.content_hash()}; "
            f"{entry.source})"
        )

    def _load_table(self, issues: list[Issue]) -> LookupTable | None:
        try:
            return LookupTable.load(self._table_path)
        except TableError as exc:
            issues.append(Issue("params.table_path", str(exc), "table-unavailable"))
            return None

    @staticmethod
    def _resolve_table_path(raw: str) -> Path:
        """Resolve the ``table_path`` parameter.

        An empty string selects the table bundled with this plugin (resolved
        relative to this file, independent of the working directory); any
        other value is treated as a filesystem path (absolute, or relative to
        the current working directory).
        """
        if raw == "":
            return _BUNDLED_TABLE
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path

    @staticmethod
    def _resolve_path(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else Path.cwd() / path
