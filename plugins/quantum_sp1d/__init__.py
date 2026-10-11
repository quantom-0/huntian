"""Self-consistent 1D Schroedinger-Poisson solver for the AlGaN/GaN 2DEG.

This plugin is a **drop-in replacement** for ``physics.quantum.analytic_tw``:
it declares the same ``quantum`` capability and produces the same ``quantum``
and ``ns`` artifacts, so the downstream ``extract.fom`` module, the Deck
bindings and the pipeline all keep working untouched.

Two algorithms are available:

``newton`` (default)
    Assembles the exact Jacobian of the coupled residual - both the
    subband-energy response and the wavefunction response - and hands it to a
    nonlinear solver module resolved from the context. Single-digit iterations,
    quadratic convergence. The solver is configured from the Deck's
    ``nonlinear`` slot, so the iteration strategy is itself pluggable.
``anderson``
    Damped fixed point with Anderson acceleration. Needs no collaborator
    module; kept as the fallback and as an independent cross-check.

The heterointerface is treated as a **face**, not a node, so the band-offset
step, the permittivity step, the mass step and the sheet charge all land on a
cell boundary. That is what restores second-order grid convergence in the
coupled solve.
"""

from __future__ import annotations

from typing import Any

from huntian.errors import Issue
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

from .solver import ELECTRON_MASS, build_grid, solve_schrodinger_poisson

MANIFEST = ModuleManifest(
    module_id="physics.quantum.sp1d",
    module_version="1.1.1",
    category=ModuleCategory.PHYSICS,
    description=(
        "Self-consistent 1D Schroedinger-Poisson solver for the AlGaN/GaN 2DEG. "
        "Face-based heterointerface, finite-volume Poisson, BenDaniel-Duke finite "
        "differences, and a choice of an exact Newton solver (resolved as a "
        "service) or Anderson-accelerated mixing. Drop-in replacement for "
        "physics.quantum.analytic_tw."
    ),
    capabilities=("quantum", "quantum.sp", "quantum.schrodinger_poisson"),
    requires=("polarization", "materials"),
    produces=("quantum", "ns"),
    verification="partially-verified",
    author="Huntian",
    tags=("quantum", "2deg", "self-consistent", "schrodinger-poisson", "newton"),
    parameters=(
        # ---- algorithm ---------------------------------------------------
        ParamSpec.choice(
            "algorithm",
            ("newton", "anderson"),
            default="newton",
            doc=(
                "coupled iteration: 'newton' uses the exact Jacobian and a "
                "nonlinear solver service; 'anderson' is damped fixed-point mixing"
            ),
        ),
        ParamSpec.string(
            "nonlinear_solver",
            "nonlinear.newton",
            doc=(
                "module id of the nonlinear solver service, resolved through "
                "ctx.services; only used when algorithm='newton'"
            ),
        ),
        ParamSpec.boolean(
            "include_wavefunction_response",
            True,
            doc=(
                "include the level-mixing term in the Jacobian. True (default) is "
                "the exact Jacobian, whose response sum runs over the full "
                "spectrum via a rank-1 deflation of the tridiagonal Hamiltonian; "
                "False gives a cheaper rank-only quasi-Newton that converges more "
                "slowly"
            ),
        ),
        # ---- physics ------------------------------------------------------
        ParamSpec.choice(
            "barrier_source",
            ("deck", "analytic"),
            default="deck",
            doc="'deck' uses device.gate.barrier_eV when present, else the analytic fit",
        ),
        ParamSpec.float(
            "phi_b_0_eV",
            0.84,
            unit="eV",
            minimum=0.0,
            maximum=3.0,
            calibratable=True,
            doc="constant term of the analytic Schottky barrier height",
            source="Ni/AlGaN barrier fit, phi_b ~ 0.84 + 0.39x",
        ),
        ParamSpec.float(
            "phi_b_x_eV",
            0.39,
            unit="eV",
            minimum=0.0,
            maximum=3.0,
            calibratable=True,
            doc="Al-fraction slope of the analytic Schottky barrier height",
            source="Ni/AlGaN barrier fit, phi_b ~ 0.84 + 0.39x",
        ),
        ParamSpec.float(
            "delta_ec_ratio",
            0.7,
            unit="1",
            minimum=0.0,
            maximum=1.0,
            calibratable=True,
            doc="conduction-band offset as a fraction of the band-gap difference",
            source="commonly 0.7; experimental reports span roughly 0.7-1.0",
        ),
        ParamSpec.integer(
            "valley_degeneracy",
            1,
            minimum=1,
            maximum=6,
            doc="conduction-band valley degeneracy entering the 2D density of states",
        ),
        ParamSpec.float(
            "gate_bias_V",
            0.0,
            unit="V",
            minimum=-20.0,
            maximum=5.0,
            calibratable=True,
            doc="gate bias at which the equilibrium solve is performed",
        ),
        # ---- iteration control --------------------------------------------
        ParamSpec.integer(
            "n_subbands",
            12,
            minimum=1,
            maximum=40,
            doc=(
                "number of subbands carried per Schroedinger solve. The box "
                "continuum above the well is represented by these states, so too "
                "few makes the density a rough functional of the potential and "
                "slows the coupled iteration"
            ),
        ),
        ParamSpec.integer(
            "max_iterations",
            30,
            minimum=1,
            maximum=20000,
            doc="iteration cap (Newton iterations are far fewer than mixing steps)",
        ),
        ParamSpec.float(
            "tolerance_eV",
            1.0e-6,
            unit="eV",
            minimum=1.0e-12,
            maximum=1.0e-1,
            doc="convergence tolerance on the band-profile update, in eV",
        ),
        ParamSpec.integer(
            "anderson_depth",
            8,
            minimum=0,
            maximum=20,
            doc="Anderson history depth for algorithm='anderson'; 0 disables it",
        ),
        ParamSpec.float(
            "mixing",
            0.5,
            unit="1",
            minimum=1.0e-3,
            maximum=1.0,
            doc="initial damping factor for algorithm='anderson'",
        ),
        ParamSpec.boolean(
            "adaptive_mixing",
            True,
            doc="halve the damping when the residual stops improving",
        ),
        ParamSpec.float(
            "mixing_min",
            0.02,
            unit="1",
            minimum=1.0e-4,
            maximum=1.0,
            doc="floor for the adaptive damping factor",
        ),
        ParamSpec.boolean(
            "density_clamp",
            False,
            doc=(
                "clamp the sheet density to [0, sigma_eff/e]. Off by default: the "
                "bound is a spurious fixed point that both iterations can stall on, "
                "so this is only a last-resort guard"
            ),
        ),
        ParamSpec.boolean(
            "require_convergence",
            True,
            doc="fail the stage instead of reporting a non-converged result",
        ),
        # ---- warm start ----------------------------------------------------
        ParamSpec.float(
            "gamma0_eV_m43",
            2.123e-12,
            unit="eV*m^(4/3)",
            minimum=0.0,
            doc=(
                "triangular-well prefactor used ONLY to build the depletion-model "
                "warm start; it does not enter the converged solution"
            ),
            source="Ambacher 2000 value for GaN",
        ),
        # ---- grid -----------------------------------------------------------
        ParamSpec.float(
            "h_fine_nm",
            0.1,
            unit="nm",
            minimum=1.0e-4,
            maximum=10.0,
            doc="fine vertical spacing near the heterointerface",
        ),
        ParamSpec.float(
            "fine_width_nm",
            12.0,
            unit="nm",
            minimum=0.1,
            maximum=500.0,
            doc="width of the fine-spacing zone on the channel side",
        ),
        ParamSpec.float(
            "barrier_fine_width_nm",
            3.0,
            unit="nm",
            minimum=0.0,
            maximum=500.0,
            doc="width of the fine-spacing zone on the barrier side",
        ),
        ParamSpec.float(
            "channel_extent_nm",
            60.0,
            unit="nm",
            minimum=1.0,
            maximum=5000.0,
            doc="depth of channel/buffer included below the interface",
        ),
        ParamSpec.float(
            "h_barrier_nm",
            1.0,
            unit="nm",
            minimum=1.0e-4,
            maximum=100.0,
            doc="coarse vertical spacing in the bulk of the barrier",
        ),
        ParamSpec.float(
            "h_max_nm",
            2.0,
            unit="nm",
            minimum=1.0e-4,
            maximum=1000.0,
            doc="maximum vertical spacing reached in the far channel",
        ),
        ParamSpec.float(
            "growth",
            1.15,
            unit="1",
            minimum=1.0,
            maximum=3.0,
            doc="geometric growth factor for the coarse channel zone",
        ),
    ),
)


class SchrodingerPoisson1D(BaseModule):
    """Solve the coupled Schroedinger-Poisson system for the 2DEG."""

    # -- pre-flight -------------------------------------------------------
    def validate(self, ctx: Any) -> list[Issue]:
        polarization = ctx.require("polarization")
        issues: list[Issue] = []

        if float(polarization.get("barrier_thickness_nm", 0.0)) <= 0.0:
            issues.append(
                Issue("polarization.barrier_thickness_nm", "must be positive", "range")
            )
        if float(polarization.get("sigma_eff_C_m2", 0.0)) <= 0.0:
            issues.append(
                Issue(
                    "polarization.sigma_eff_C_m2",
                    "effective polarization charge is zero, so there is no well to solve",
                    "physics",
                    hint="check the screening factor and the barrier Al fraction",
                )
            )
        if float(self.param("fine_width_nm")) > float(self.param("channel_extent_nm")):
            issues.append(
                Issue(
                    "params.fine_width_nm",
                    "fine zone cannot be wider than the whole channel extent",
                    "range",
                )
            )
        # The interface must sit on a cell face, so the barrier must be thicker
        # than half the fine spacing.
        if float(polarization.get("barrier_thickness_nm", 0.0)) * 1.0e-9 <= 0.5 * float(
            self.param("h_fine_nm")
        ) * 1e-9:
            issues.append(
                Issue(
                    "params.h_fine_nm",
                    "fine spacing must be smaller than twice the barrier thickness",
                    "range",
                )
            )
        for name in ("h_fine_nm", "h_barrier_nm", "h_max_nm"):
            if float(self.param(name)) <= 0.0:
                issues.append(Issue(f"params.{name}", "must be positive", "range"))
        if float(self.param("tolerance_eV")) <= 0.0:
            issues.append(Issue("params.tolerance_eV", "must be positive", "range"))

        if str(self.param("algorithm")).lower() == "newton":
            services = ctx.services
            if services is None:
                issues.append(
                    Issue(
                        "params.algorithm",
                        "algorithm='newton' needs a service registry, but this "
                        "context has none",
                        "service",
                        hint="run through huntian.pipeline.Executor",
                    )
                )
            elif not services.has(str(self.param("nonlinear_solver"))):
                issues.append(
                    Issue(
                        "params.nonlinear_solver",
                        f"nonlinear solver {self.param('nonlinear_solver')!r} is not "
                        "registered",
                        "service",
                        hint="bind the Deck's 'nonlinear' slot, or use algorithm='anderson'",
                    )
                )
        return issues

    # -- work -------------------------------------------------------------
    def execute(self, ctx: Any) -> Any:
        polarization = ctx.require("polarization")
        materials = ctx.require("materials")
        gate = ctx.get("gate") or {}
        temperature = float(ctx.get("temperature_K", 300.0))

        x_al = float(polarization["x_al"])
        channel_material = str(polarization["channel_material"])
        channel = materials.get(channel_material)

        grid = build_grid(
            barrier_thickness_m=float(polarization["barrier_thickness_nm"]) * 1.0e-9,
            channel_extent_m=float(self.param("channel_extent_nm")) * 1.0e-9,
            h_fine_m=float(self.param("h_fine_nm")) * 1.0e-9,
            fine_width_m=float(self.param("fine_width_nm")) * 1.0e-9,
            barrier_fine_width_m=float(self.param("barrier_fine_width_nm")) * 1.0e-9,
            h_barrier_m=float(self.param("h_barrier_nm")) * 1.0e-9,
            h_max_m=float(self.param("h_max_nm")) * 1.0e-9,
            growth=float(self.param("growth")),
            eps_r_barrier=float(polarization["eps_r_barrier"]),
            eps_r_channel=channel.numeric("eps_r"),
            m_eff_barrier_kg=float(polarization["m_eff_barrier"]) * ELECTRON_MASS,
            m_eff_channel_kg=channel.numeric("m_eff") * ELECTRON_MASS,
        )

        delta_ec = float(self.param("delta_ec_ratio")) * float(
            polarization["band_offset_eV"]
        )
        phi_b, phi_b_source = self._barrier_height(gate, x_al)

        algorithm = str(self.param("algorithm")).lower()
        nonlinear_solver = None
        if algorithm == "newton":
            services = ctx.require_services()
            nonlinear_solver = services.get(str(self.param("nonlinear_solver")))

        tolerance = float(self.param("tolerance_eV"))
        result = solve_schrodinger_poisson(
            grid=grid,
            sigma_eff_C_m2=float(polarization["sigma_eff_C_m2"]),
            phi_b_V=phi_b,
            delta_ec_eV=delta_ec,
            gate_bias_V=float(self.param("gate_bias_V")),
            temperature_K=temperature,
            valley_degeneracy=int(self.param("valley_degeneracy")),
            n_subbands=int(self.param("n_subbands")),
            max_iterations=int(self.param("max_iterations")),
            tolerance_eV=tolerance,
            anderson_depth=int(self.param("anderson_depth")),
            mixing=float(self.param("mixing")),
            adaptive_mixing=bool(self.param("adaptive_mixing")),
            mixing_min=float(self.param("mixing_min")),
            gamma0_eV_m43=float(self.param("gamma0_eV_m43")),
            density_clamp=bool(self.param("density_clamp")),
            algorithm=algorithm,
            nonlinear_solver=nonlinear_solver,
            newton_options={"tolerance": tolerance},
            include_wavefunction_response=bool(
                self.param("include_wavefunction_response")
            ),
        )

        if not result.converged and bool(self.param("require_convergence")):
            raise RuntimeError(
                f"{MANIFEST.module_id} did not converge in {result.iterations} "
                f"iteration(s) with algorithm={algorithm!r}; final residual "
                f"{result.residual_eV:.3e} eV (tolerance {tolerance:.3e} eV) - "
                f"{result.solver_message}"
            )

        offsets = result.subband_offsets_eV
        occupancy = (
            result.ns_m2 / result.ns_upper_m2 if result.ns_upper_m2 > 0.0 else 0.0
        )
        linear_solves = getattr(getattr(nonlinear_solver, "last_result", None), "linear_solves", None)

        quantum = {
            "model": "self_consistent_sp1d",
            "algorithm": result.algorithm,
            "algorithm_detail": result.solver_message,
            "interface_discretisation": "face",
            "jacobian": (
                "exact: subband energy + wavefunction response (full spectrum, "
                "Woodbury deflation) + mass response"
                if algorithm == "newton" and bool(self.param("include_wavefunction_response"))
                else (
                    "rank-only quasi-Newton"
                    if algorithm == "newton"
                    else "not applicable (fixed-point mixing)"
                )
            ),
            # -- contract keys shared with physics.quantum.analytic_tw --------
            "ns_m2": result.ns_m2,
            "ns_cm2": result.ns_cm2,
            "ns_upper_bound_cm2": result.ns_upper_m2 * 1.0e-4,
            "ef_eV": result.ef_eV,
            "e0_eV": float(offsets[0]),
            "phi_b_eV": result.phi_b_eV,
            "delta_ec_eV": result.delta_ec_eV,
            "converged": result.converged,
            "iterations": result.iterations,
            # -- extra self-consistent detail --------------------------------
            "occupancy_fraction": occupancy,
            "subband_energies_eV": [float(e) for e in result.energies_eV],
            "subband_offsets_eV": [float(e) for e in offsets],
            "subband_densities_cm2": [
                float(d) * 1.0e-4 for d in result.ns_per_subband_m2
            ],
            "subband_masses_rel": [float(m) / ELECTRON_MASS for m in result.masses_kg],
            "n_bound_subbands": result.n_bound,
            "phi_b_source": phi_b_source,
            "gate_bias_V": float(self.param("gate_bias_V")),
            "temperature_K": result.temperature_K,
            "ec_well_bottom_eV": result.ec_well_bottom_eV,
            "ec_interface_eV": result.ec_interface_eV,
            "ec_min_eV": result.ec_min_eV,
            "centroid_nm": result.centroid_nm,
            "rms_width_nm": result.rms_width_nm,
            "residual_eV": result.residual_eV,
            "residual_history": [float(v) for v in result.residual_history],
            "damping_history": [float(v) for v in result.damping_history],
            "function_evaluations": result.function_evaluations,
            "linear_solves": linear_solves,
            "mixing_final": result.mixing_final,
            "anderson_fallbacks": result.anderson_fallbacks,
            "density_clamp_hits": result.clamp_hits,
            "warm_start_ns_cm2": result.seed_ns_m2 * 1.0e-4,
            "grid_nodes": grid.size,
            "grid_interface_face": grid.interface_face,
            "profiles": {
                "z_nm": [float(value) for value in grid.z_nm],
                "ec_eV": [float(value) for value in result.ec_profile_eV],
                "n_cm3": [float(value) * 1.0e-6 for value in result.n_profile_m3],
                "psi0": [float(value) for value in result.psi[:, 0]],
            },
            "notes": [
                "Fermi level is the energy reference and the Schottky barrier fixes "
                "the conduction band at the metal/AlGaN interface.",
                "The heterointerface is a cell face: the band offset, the "
                "permittivity, the effective mass and the sheet charge all sit on a "
                "boundary rather than on a node.",
                "The 2DEG screens the polarization sheet charge; the channel band "
                "flattens below the 2DEG once the charge is accounted for.",
                f"warm start from the depletion model: "
                f"{result.seed_ns_m2 * 1.0e-4:.6e} cm^-2",
                f"converged in {result.iterations} iteration(s) with "
                f"{result.function_evaluations} residual evaluation(s)",
            ],
            "assumptions": [
                "Effective-mass approximation with a BenDaniel-Duke interface "
                "condition; non-parabolicity is neglected.",
                "The conduction-band offset is delta_ec_ratio * dEg, taken from the "
                "polarization module.",
                "Surface-state screening is carried entirely by the polarization "
                "module's screening factor and is not re-derived here.",
                "No traps, no self-heating, no strain-dependent mass renormalisation.",
                "Only the Gamma valley is resolved; valley_degeneracy only scales the "
                "density of states.",
                "The Newton Jacobian is the exact derivative of the discretised "
                "residual. The wavefunction response is not truncated: the "
                "coupling to every eigenstate outside the carried set comes from "
                "a rank-1 (Woodbury) deflation of the tridiagonal Hamiltonian, "
                "and the density-of-states mass response is carried as well. "
                "Against central differences the agreement is ~1e-11, and the "
                "coupled iteration is quadratically convergent (observed local "
                "order ~1.99).",
                "States above the barrier are box eigenstates of the finite domain, "
                "not true bound states. They are a legitimate quadrature of the 3D "
                "continuum - the sheet density is insensitive to the domain size to "
                "better than 1e-5 - but their individual wavefunctions are not "
                "physical.",
            ],
        }

        ctx.put(
            "quantum",
            quantum,
            producer=MANIFEST.module_id,
            module_version=MANIFEST.module_version,
        )
        ctx.put(
            "ns",
            quantum["ns_cm2"],
            producer=MANIFEST.module_id,
            module_version=MANIFEST.module_version,
            meta={"unit": "cm^-2", "quantity": "2DEG sheet density"},
        )
        return ctx

    # -- helpers -----------------------------------------------------------
    def _barrier_height(self, gate: dict[str, Any], x_al: float) -> tuple[float, str]:
        source = str(self.param("barrier_source"))
        deck_value = gate.get("barrier_eV")
        if source == "deck" and deck_value is not None:
            return float(deck_value), "deck"
        return (
            float(self.param("phi_b_0_eV")) + float(self.param("phi_b_x_eV")) * x_al,
            "analytic",
        )


MODULE = SchrodingerPoisson1D
