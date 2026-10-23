#!/usr/bin/env python
"""End-to-end walkthrough of the Huntian skeleton.

    python demo.py

Exercises every deliverable in one run:

1.  plugin discovery and the module catalogue
2.  Deck loading and two-stage validation
3.  a full pipeline run with provenance and timing
4.  the physics actually computed by the bundled plugins
5.  swapping the quantum model for a self-consistent Schroedinger-Poisson
    solver - the acceptance test for plug-and-play
6.  reproducibility through the context fingerprint
7.  a runtime hot swap, including proof that version pinning protects a
    previously validated Deck
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _entry in (str(ROOT / "python"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from huntian import (  # noqa: E402
    Deck,
    Executor,
    MaterialDB,
    ModuleRegistry,
    Pipeline,
    SimContext,
    load_deck,
)

DECK_PATH = ROOT / "decks" / "demo_hemt.json"
SP_DECK_PATH = ROOT / "decks" / "demo_hemt_sp1d.json"
PLUGIN_DIR = ROOT / "python" / "huntian" / "plugins"
WIDTH = 78


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def sub(title: str) -> None:
    print()
    print(f"-- {title}")


def run(deck: Deck, registry: ModuleRegistry, materials: MaterialDB):
    ctx = SimContext(name=deck.name, bus=registry.bus)
    result = Executor(registry).run(
        Pipeline.from_deck(deck), ctx=ctx, deck=deck, materials=materials
    )
    return result, ctx


def print_stages(result) -> None:
    for stage in result.stages:
        flag = "ok  " if stage.status == "ok" else "FAIL"
        print(f"  [{flag}] {stage.slot:<14} {stage.module_id}@{stage.module_version}"
              f"  {stage.duration_s * 1000:8.2f} ms")
        for warning in stage.warnings:
            print(f"          warn: {warning}")


def main() -> int:
    # ------------------------------------------------------------------
    banner("Huntian: registry + context + deck schema + pluggable physics")

    sub("1. Plugin discovery")
    registry = ModuleRegistry()
    report = registry.load_from_paths([PLUGIN_DIR])
    for record in report.loaded:
        print(f"  + {record.manifest.module_id}@{record.manifest.module_version}"
              f"  [{record.manifest.category}]  from {Path(record.origin).name}")
    for path, error in report.errors:
        print(f"  ! {path}: {error}")
    print(f"  discovered {len(report.loaded)} module(s), {len(report.errors)} error(s)")

    sub("2. Registry health")
    print(json.dumps(registry.health(), indent=2, sort_keys=True))

    sub("3. Module catalogue (what the LLM layer is shown)")
    print(registry.render_catalog())

    # ------------------------------------------------------------------
    banner("Baseline: analytic triangular-well quantum model")

    materials = MaterialDB.builtin()

    sub("4. Load and describe the Deck")
    deck = load_deck(DECK_PATH)
    print(deck.describe())
    print()
    print(f"  material database : v{materials.version} "
          f"(verification: {materials.verification}, fingerprint {materials.fingerprint()})")

    sub("5. Two-stage validation")
    errors, warnings = deck.validate_all(registry, materials)
    print(f"  structural : parsed without exceptions")
    print(f"  contextual : {len(errors)} error(s), {len(warnings)} warning(s)")
    for warning in warnings:
        print(f"    warn: {warning}")
    for error in errors:
        print(f"    error: {error}")
    if errors:
        print("  cannot run an invalid deck")
        return 2

    sub("6. Pipeline execution")
    print(Pipeline.from_deck(deck).describe())
    print()
    result, ctx = run(deck, registry, materials)
    print_stages(result)
    print()
    print(f"  status      : {result.status}")
    print(f"  fingerprint : {result.fingerprint}")
    print(f"  artifacts   : {', '.join(result.artifacts)}")

    # ------------------------------------------------------------------
    banner("Physics from the analytic chain")

    pol = ctx.get("polarization")
    quantum = ctx.get("quantum")
    fom = ctx.get("results.fom")
    mesh = ctx.get("mesh")

    sub("7. Polarization (physics.polarization.ambacher)")
    print(f"  Al fraction x           : {pol['x_al']:.3f}")
    print(f"  barrier thickness       : {pol['barrier_thickness_nm']:.1f} nm")
    print(f"  lattice strain          : {pol['strain'] * 100:.4f} %")
    print(f"  P_sp (barrier)          : {pol['p_sp_barrier_C_m2']:+.5f} C/m^2")
    print(f"  P_pz (barrier)          : {pol['p_pz_barrier_C_m2']:+.5f} C/m^2")
    print(f"  sigma_pol               : {pol['sigma_pol_cm2']:.4e} cm^-2")
    print(f"  screening factor        : {pol['screening']:.2f}")
    print(f"  sigma_eff               : {pol['sigma_eff_cm2']:.4e} cm^-2")

    sub("8. 2DEG density (physics.quantum.analytic_tw)")
    print(f"  model                   : {quantum['model']}")
    print(f"  n_s                     : {quantum['ns_cm2']:.4e} cm^-2")
    print(f"  upper bound sigma_eff/e : {quantum['ns_upper_bound_cm2']:.4e} cm^-2")
    print(f"  occupancy               : {quantum['occupancy_fraction'] * 100:.1f} %")
    print(f"  E0 / E_F                : {quantum['e0_eV']:.4f} / {quantum['ef_eV']:.4f} eV")
    print(f"  iterations / converged  : {quantum['iterations']} / {quantum['converged']}")
    for note in quantum["notes"]:
        print(f"  note: {note}")

    sub("9. Figures of merit (extract.fom)")
    print(f"  V_th (charge control)   : {fom['vth_est_V']:+.4f} V")
    print(f"  barrier voltage drop    : {fom['barrier_drop_V']:.4f} V")
    print(f"  C_ox                    : {fom['cox_uF_cm2']:.4f} uF/cm^2")

    sub("10. Mesh (mesh.structured2d)")
    print(f"  grid                    : {mesh['nx']} x {mesh['nz']} nodes "
          f"({mesh['n_nodes']} total)")
    print(f"  lateral extent          : {mesh['lateral_extent_nm'][0]:.1f} .. "
          f"{mesh['lateral_extent_nm'][1]:.1f} nm")
    print(f"  stack depth             : {mesh['total_depth_nm']:.1f} nm")

    # ------------------------------------------------------------------
    banner("Plug-and-play: swapping the quantum model")

    sub("11. The only change is one binding in the Deck")
    sp_deck = load_deck(SP_DECK_PATH)
    print(f"  baseline   : physics.quantum -> "
          f"{deck.physics.quantum.module}{deck.physics.quantum.version}")
    print(f"  replacement: physics.quantum -> "
          f"{sp_deck.physics.quantum.module}{sp_deck.physics.quantum.version}")
    print()
    print("  both declare capability 'quantum' and produce 'quantum' + 'ns',")
    print("  so nothing else in the Deck, the pipeline or extract.fom changes.")
    print()
    sp_errors, sp_warnings = sp_deck.validate_all(registry, materials)
    print(f"  sp1d deck validates : {len(sp_errors)} error(s), "
          f"{len(sp_warnings)} warning(s)")

    sub("12. Running the replacement through the very same pipeline")
    sp_result, sp_ctx = run(sp_deck, registry, materials)
    print_stages(sp_result)
    print()
    print(f"  status      : {sp_result.status}")
    print(f"  fingerprint : {sp_result.fingerprint}")
    print(f"  extract.fom ran unchanged: "
          f"{sp_result.stage('extract:0').module_id} -> {sp_result.stage('extract:0').status}")
    print(f"  services configured       : {sp_ctx.services.configured()}")

    sub("13. Newton versus Anderson-accelerated mixing")
    raw = json.loads(SP_DECK_PATH.read_text(encoding="utf-8"))
    raw["physics"]["quantum"]["params"]["algorithm"] = "anderson"
    raw["physics"]["quantum"]["params"]["max_iterations"] = 800
    raw["physics"]["quantum"]["params"]["density_clamp"] = True
    anderson_deck = Deck(**raw)
    t0 = time.perf_counter()
    anderson_result, anderson_ctx = run(anderson_deck, registry, materials)
    anderson_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  {'algorithm':<12}{'iterations':>11}{'evals':>7}{'linear solves':>15}"
          f"{'stage (ms)':>12}{'n_s (cm^-2)':>17}")
    for label, res, ctx_i, ms in (
        ("newton", sp_result, sp_ctx, sp_result.stage("quantum").duration_s * 1000.0),
        ("anderson", anderson_result, anderson_ctx, anderson_result.stage("quantum").duration_s * 1000.0),
    ):
        entry = ctx_i.get("quantum")
        print(f"  {label:<12}{entry['iterations']:>11}{entry['function_evaluations']:>7}"
              f"{entry['linear_solves'] or 0:>15}{ms:>12.0f}"
              f"{entry['ns_cm2']:>17.6e}")
    print()
    print("  Same device, same discretisation, same answer to within the mixing")
    print("  tolerance - but Newton needs roughly an order of magnitude fewer")
    print("  iterations, and solves a dense linear system instead of mixing.")

    sub("14. Grid convergence order")
    values = []
    print(f"  {'h_fine (nm)':>12}{'n_s (cm^-2)':>18}{'change':>12}")
    previous = None
    for h_nm in (0.4, 0.2, 0.1):
        variant = json.loads(SP_DECK_PATH.read_text(encoding="utf-8"))
        variant["physics"]["quantum"]["params"]["h_fine_nm"] = h_nm
        variant["physics"]["quantum"]["params"]["fine_width_nm"] = 12.0
        result_i, ctx_i = run(Deck(**variant), registry, materials)
        ns = float(ctx_i.get("ns"))
        change = "" if previous is None else f"{abs(ns - previous):.3e}"
        print(f"  {h_nm:>12.2f}{ns:>18.9e}{change:>12}")
        values.append(ns)
        previous = ns
    ratio = abs(values[0] - values[1]) / abs(values[1] - values[2])
    observed = math.log(ratio) / math.log(2.0)
    print()
    print(f"  successive-difference ratio = {ratio:.3f}  ->  observed order p = {observed:.2f}")
    print("  The heterointerface sits on a cell FACE, so the band-offset step, the")
    print("  permittivity step, the mass step and the sheet charge all land on a")
    print("  boundary. Before that change the same study gave p ~ 1.0.")

    sub("15. Newton diagnostics from the artefact")
    sp_quantum = sp_ctx.get("quantum")
    print(f"  algorithm           : {sp_quantum['algorithm']}")
    print(f"  jacobian            : {sp_quantum['jacobian']}")
    print(f"  interface           : {sp_quantum['interface_discretisation']} discretisation")
    print(f"  residual history    : "
          f"{[f'{v:.1e}' for v in sp_quantum['residual_history'][:8]]}")
    print(f"  final residual      : {sp_quantum['residual_eV']:.3e} eV")
    print(f"  linear solves       : {sp_quantum['linear_solves']}")
    print(f"  function evaluations: {sp_quantum['function_evaluations']}")

    # ------------------------------------------------------------------
    banner("What the self-consistent solve adds")

    sq = sp_ctx.get("quantum")
    sfom = sp_ctx.get("results.fom")

    sub("16. Analytic estimate vs self-consistent Schroedinger-Poisson")
    header = f"  {'quantity':<30}{'analytic':>16}{'self-consistent':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = [
        ("n_s (cm^-2)", quantum["ns_cm2"], sq["ns_cm2"]),
        ("occupancy of sigma_eff/e", quantum["occupancy_fraction"], sq["occupancy_fraction"]),
        ("E_F above band edge (eV)", quantum["ef_eV"], sq["ef_eV"]),
        ("E_0 above band edge (eV)", quantum["e0_eV"], sq["e0_eV"]),
        ("E_F - E_0 (eV)", quantum["ef_eV"] - quantum["e0_eV"], sq["ef_eV"] - sq["e0_eV"]),
        ("V_th charge control (V)", fom["vth_est_V"], sfom["vth_est_V"]),
    ]
    for label, analytic_value, sp_value in rows:
        delta = (
            (sp_value - analytic_value) / analytic_value * 100.0
            if analytic_value
            else 0.0
        )
        print(f"  {label:<30}{analytic_value:>16.6g}{sp_value:>18.6g}   {delta:+6.2f} %")
    print()
    print("  The sheet density moves by a few percent: that is the size of the")
    print("  quantum correction the analytic triangular-well fit approximates away.")
    print()
    print("  Note the shape of the change: the self-consistent well comes out")
    print("  shallower, so E_F and E_0 both drop by roughly a third - but the")
    print("  subband spacing E_F - E_0, which is what actually sets n_s, barely")
    print("  moves. The two models disagree about the potential, not the statistics.")
    print()
    print("  V_th is identical on purpose: extract.fom computes it from the")
    print("  full-depletion charge control relation, which depends only on the")
    print("  barrier electrostatics. Its reported n_s does come from whichever")
    print("  quantum module ran (see results.fom).")

    sub("17. Subband ladder from the Schroedinger solve")
    print(f"  {'state':<7}{'offset (eV)':>13}{'m*/m0':>9}{'n (cm^-2)':>15}{'below E_F':>11}")
    for index, (offset, mass, density) in enumerate(
        zip(sq["subband_offsets_eV"], sq["subband_masses_rel"], sq["subband_densities_cm2"])
    ):
        below = "yes" if offset < sq["ef_eV"] else "no"
        print(f"  E{index:<6}{offset:>13.6f}{mass:>9.4f}{density:>15.4e}{below:>11}")
    print()
    print(f"  bound subbands (E_i < E_F) : {sq['n_bound_subbands']}")
    print(f"  converged                  : {sq['converged']} in {sq['iterations']} "
          f"iteration(s), residual {sq['residual_eV']:.2e} eV")
    print(f"  warm start (depletion)     : {sq['warm_start_ns_cm2']:.4e} cm^-2")
    print(f"  ground-state centroid      : {sq['centroid_nm']:.3f} nm below the interface")
    print(f"  ground-state rms width     : {sq['rms_width_nm']:.3f} nm")
    print(f"  self-consistent grid       : {sq['grid_nodes']} nodes")

    sub("18. Band profile (every few nodes)")
    profile = sq["profiles"]
    z_nm, ec_eV, n_cm3 = profile["z_nm"], profile["ec_eV"], profile["n_cm3"]
    stride = max(1, len(z_nm) // 18)
    print(f"  {'z (nm)':>9}{'E_c (eV)':>12}{'n (cm^-3)':>15}   band")
    ceiling = max(ec_eV)
    floor = min(ec_eV)
    span = max(ceiling - floor, 1e-9)
    for index in range(0, len(z_nm), stride):
        bar = "#" * max(0, int((ceiling - ec_eV[index]) / span * 34))
        print(f"  {z_nm[index]:>9.2f}{ec_eV[index]:>12.4f}{n_cm3[index]:>15.3e}   {bar}")
    print()
    print("  The barrier rises, the well bottom sits at the interface, and the")
    print("  channel band flattens once the 2DEG has screened the sheet charge.")

    sub("19. Assumptions and limitations recorded with the artefact")
    for note in sq["assumptions"]:
        print(f"  - {note}")

    # ------------------------------------------------------------------
    banner("Reproducibility")

    sub("20. Same inputs, same fingerprint")
    repeat, _ = run(deck, registry, materials)
    SP_repeat, _ = run(sp_deck, registry, materials)
    print(f"  analytic        run A : {result.fingerprint}")
    print(f"  analytic        run B : {repeat.fingerprint}")
    print(f"  identical             : {result.fingerprint == repeat.fingerprint}")
    print(f"  self-consistent run A : {sp_result.fingerprint}")
    print(f"  self-consistent run B : {SP_repeat.fingerprint}")
    print(f"  identical             : {sp_result.fingerprint == SP_repeat.fingerprint}")
    print()
    print("  Provenance recorded for the analytic run:")
    for record in ctx.provenance:
        print(f"    {record.stage:<14} {record.module_id}@{record.module_version} "
              f"status={record.status} {record.duration_s * 1000:7.2f} ms "
              f"out={list(record.outputs)}")

    # ------------------------------------------------------------------
    banner("Hot swap: versioning a module at runtime")

    base_record = registry.select("physics.quantum.sp1d")
    base_cls = base_record.module_cls
    base_manifest = base_record.manifest

    # A synthetic major version with identical physics. Only the version string
    # differs, which is already enough to change the reproducibility
    # fingerprint. It is a test double, not a physics upgrade.
    variant_manifest = dataclasses.replace(
        base_manifest,
        module_version="2.0.0",
        description=(
            "DEMO TEST DOUBLE: byte-identical solver with a bumped major version, "
            "purely so the hot swap is observable. Not a physics upgrade."
        ),
    )

    class Sp1dV2(base_cls):  # type: ignore[misc, valid-type]
        MANIFEST = variant_manifest

    sub("21. Before the swap")
    live = registry.instantiate("physics.quantum.sp1d", version="^1.0.0")
    print(f"  selected version            : "
          f"{registry.select('physics.quantum.sp1d').manifest.module_version}")
    print(f"  live instance h_fine_nm     : {live.param('h_fine_nm')}")

    sub("22. Register v2.0.0")
    swap = registry.hot_swap(Sp1dV2, origin="<demo>")
    print(f"  {swap}")
    print(f"  unpinned selection now picks: "
          f"{registry.select('physics.quantum.sp1d').manifest.module_version}")
    print(f"  live instance still running : v{base_manifest.module_version}, "
          f"state = {live.state.value}")

    sub("23. Version pinning protects the validated Deck")
    pinned, pinned_ctx = run(sp_deck, registry, materials)
    print(f"  the sp1d Deck pins '^1.0.0' -> v"
          f"{pinned.stage('quantum').module_version}")
    print(f"  n_s                         : {pinned_ctx.get('ns'):.6e} cm^-2")
    print(f"  fingerprint unchanged       : "
          f"{pinned.fingerprint == sp_result.fingerprint}")
    print()
    print(f"  versions available now      : "
          f"{[str(v) for v in registry.versions('physics.quantum.sp1d')]}")
    print("  Both versions coexist; nothing was reloaded or restarted.")

    # ------------------------------------------------------------------
    banner("Run metadata")
    metadata = ctx.save_metadata(ROOT / "out" / "run_metadata.json")
    print(f"  wrote {metadata}")
    print("  (metadata only - artifact values are deliberately not serialised)")

    banner("Done")
    print("  Test suite : python run_tests.py")
    print("  CLI        : python -m huntian.cli --help   (needs PYTHONPATH=python)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
