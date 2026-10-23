"""First-principles (DFT) material parameter provider.

``materials.dft`` is a drop-in replacement for the builtin ``materials``
slot: it loads the builtin material database, layers first-principles
constants (with uncertainty and provenance) on top, and republishes the
``materials`` artifact so downstream modules - ``physics.polarization.ambacher``,
``physics.quantum.sp1d`` - work unchanged.

Three backends, one artifact contract (design document ``DFT材料计算模块_插件设计.md``):

* ``table`` (default, zero third-party dependency): serve a precomputed,
  versioned lookup table of DFT values;
* ``qe_local`` / ``qe_slurm``: drive Quantum ESPRESSO through ASE (lazy
  imports; unavailable in this phase).

Precision red lines are enforced in :mod:`materials_dft.plugin`, not in
documentation: PBE band gaps never reach the device model unless explicitly
scissor-corrected, and entries without convergence evidence are flagged,
never silently used.
"""

from __future__ import annotations

from huntian.registry import ModuleCategory, ModuleManifest, ParamSpec

from .plugin import DftMaterials

MANIFEST = ModuleManifest(
    module_id="materials.dft",
    module_version="0.1.0",
    category=ModuleCategory.PHYSICS,
    description=(
        "First-principles (DFT) material parameter provider: computes or looks up "
        "AlGaN/GaN material constants (polarization, elasticity, band edges, effective "
        "mass) from plane-wave DFT via an ASE/Quantum ESPRESSO driver, or serves them "
        "from a precomputed table. Drop-in replacement for the builtin materials slot."
    ),
    capabilities=("materials", "materials.dft", "materials.validation"),
    requires=("structure",),
    produces=("materials", "dft_provenance"),
    verification="partially-verified",
    author="Huntian",
    tags=("dft", "first-principles", "materials", "validation"),
    parameters=(
        # ---- run mode ---------------------------------------------------
        ParamSpec.choice(
            "backend",
            ("table", "qe_local", "qe_slurm"),
            default="table",
            doc="table: serve precomputed JSON; qe_*: drive Quantum ESPRESSO",
        ),
        ParamSpec.string(
            "table_path",
            "",
            doc="lookup-table path; empty string selects the table bundled with "
            "this plugin (data/algan_gan.json), any other value is resolved as a "
            "filesystem path (absolute, or relative to the current working "
            "directory)",
        ),
        # ---- functional & accuracy tiers (red lines in plugin.py) --------
        ParamSpec.choice(
            "functional",
            ("pbe", "hse06"),
            default="hse06",
            doc="exchange-correlation functional; PBE gaps are systematically "
            "underestimated and must not feed Eg directly (see eg_policy)",
        ),
        ParamSpec.choice(
            "eg_policy",
            ("reject_pbe", "scissor_to_reference"),
            default="reject_pbe",
            doc="reject_pbe: fail validation if functional=pbe and band gaps are "
            "requested; scissor_to_reference: shift PBE gap to reference Eg "
            "and record the shift in dft_provenance",
        ),
        ParamSpec.string(
            "reference_eg_source",
            "experiment",
            doc="where scissor reference gaps come from: 'experiment' (the builtin "
            "library value) or a materials.json property key",
        ),
        # ---- compute settings (qe_* backends) ----------------------------
        ParamSpec.float(
            "ecut_wfc_ry",
            60.0,
            unit="Ry",
            minimum=20.0,
            maximum=150.0,
            doc="plane-wave kinetic-energy cutoff; must match the table record "
            "or the table is treated as stale",
        ),
        ParamSpec.float(
            "kspacing_1_A",
            0.20,
            unit="1/angstrom",
            minimum=0.05,
            maximum=0.5,
            doc="k-point density (Monkhorst spacing); must match the table record",
        ),
        ParamSpec.string(
            "pseudopotential_family",
            "SSSP/PBEsol/efficiency@1.3.0",
            doc="pseudopotential set with version, recorded in provenance",
        ),
        ParamSpec.float(
            "strain_step",
            0.01,
            unit="1",
            minimum=0.001,
            maximum=0.05,
            doc="finite-strain step for piezoelectric coefficient extraction (qe_*)",
        ),
        ParamSpec.boolean(
            "spin_orbit",
            False,
            doc="include SOC in band-structure runs (raises cost ~4x; qe_*)",
        ),
        # ---- alloy modelling ----------------------------------------------
        ParamSpec.integer(
            "sqs_size",
            64,
            minimum=8,
            maximum=512,
            doc="supercell size for special-quasirandom-structure AlGaN runs (qe_slurm)",
        ),
        ParamSpec.integer(
            "sqs_seed",
            1,
            minimum=0,
            doc="SQS generation seed; enters the cache key, so tables stay reproducible",
        ),
        # ---- caching --------------------------------------------------------
        ParamSpec.string(
            "cache_dir",
            "dft_cache",
            doc="result cache directory (qe_* backends); key = hash(structure, "
            "functional, pseudopotential, ecut, kspacing, sqs_seed)",
        ),
        ParamSpec.boolean(
            "allow_stale_cache",
            False,
            doc="if True, serve results whose numerical settings differ from the "
            "parameters (recorded in provenance); False refuses with a clear error",
        ),
        # ---- output control --------------------------------------------------
        ParamSpec.choice(
            "override_policy",
            ("dft_wins", "reference_wins", "dft_only_gapless"),
            default="dft_wins",
            doc="which source supplies each material property in the merged DB; "
            "non-converged entries are flagged converged=false either way",
        ),
    ),
)

MODULE = DftMaterials
