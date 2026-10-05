"""Analytic triangular-well estimate of the 2DEG sheet density.

Solves the classic Ambacher charge-control relation self-consistently in
closed form:

    n_s = sigma_eff/e - (eps0*eps_r/(d*e^2)) * (e*phi_b + E_F - dE_C)
    E_F = E_0 + pi*hbar^2*n_s/(m* * g_v)
    E_0 = gamma_0 * n_s^(2/3)          (triangular-well ground state)

The fixed-point map has a contraction ratio well below one for realistic
barriers, so damped iteration converges in a handful of steps.

**Scope warning.** This is an *analytic* triangular-well estimate, not a
self-consistent Schroedinger-Poisson solve. It places the subband correctly to
first order but ignores wavefunction penetration, barrier band bending and
multi-subband occupation. A true SP solver declaring the same ``quantum``
capability can be dropped in as a replacement without touching any other
module - that is exactly the plug-and-play property this skeleton exists to
demonstrate.
"""

from __future__ import annotations

import math
from typing import Any

from nemo.errors import Issue
from nemo.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
REDUCED_PLANCK = 1.054571817e-34  # J*s
ELECTRON_MASS = 9.1093837015e-31  # kg

MANIFEST = ModuleManifest(
    module_id="physics.quantum.analytic_tw",
    module_version="0.1.0",
    category=ModuleCategory.PHYSICS,
    description=(
        "Analytic triangular-well (charge-control) estimate of the 2DEG sheet density "
        "from the polarization sheet charge. Not a self-consistent "
        "Schroedinger-Poisson solve; a drop-in replacement declaring capability "
        "'quantum' is planned."
    ),
    capabilities=("quantum", "quantum.analytic", "quantum.triangular_well"),
    requires=("polarization",),
    produces=("quantum", "ns"),
    verification="partially-verified",
    author="NemoTCAD",
    tags=("quantum", "2deg", "charge-control"),
    parameters=(
        ParamSpec.choice(
            "barrier_source",
            ("deck", "analytic"),
            default="deck",
            doc=(
                "'deck' uses device.gate.barrier_eV when present; 'analytic' always uses "
                "phi_b_0_eV + phi_b_x_eV * x_al"
            ),
        ),
        ParamSpec.float(
            "phi_b_0_eV",
            0.84,
            unit="eV",
            minimum=0.0,
            maximum=3.0,
            calibratable=True,
            doc="constant term of the analytic Schottky barrier height",
            source="Ni/AlGaN Schottky barrier fit, phi_b ~ 0.84 + 0.39x",
        ),
        ParamSpec.float(
            "phi_b_x_eV",
            0.39,
            unit="eV",
            minimum=0.0,
            maximum=3.0,
            calibratable=True,
            doc="Al-fraction slope of the analytic Schottky barrier height",
            source="Ni/AlGaN Schottky barrier fit, phi_b ~ 0.84 + 0.39x",
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
        ParamSpec.float(
            "gamma0_eV_m43",
            2.123e-12,
            unit="eV*m^(4/3)",
            minimum=0.0,
            calibratable=True,
            doc="triangular-well ground-state prefactor, E_0 = gamma_0 * n_s^(2/3)",
            source="Ambacher 2000 value for GaN",
        ),
        ParamSpec.boolean(
            "include_fermi",
            True,
            doc="include the Fermi-level rise above the ground subband (degenerate 2DEG)",
        ),
        ParamSpec.float(
            "relaxation",
            0.5,
            unit="1",
            minimum=1.0e-3,
            maximum=1.0,
            doc="fixed-point damping factor",
        ),
        ParamSpec.integer(
            "max_iterations",
            500,
            minimum=1,
            maximum=100000,
            doc="iteration cap before declaring non-convergence",
        ),
        ParamSpec.float(
            "tolerance",
            1.0e-10,
            unit="1",
            minimum=1.0e-16,
            maximum=1.0e-1,
            doc="relative convergence tolerance on n_s",
        ),
        ParamSpec.integer(
            "valley_degeneracy",
            1,
            minimum=1,
            maximum=6,
            doc="conduction-band valley degeneracy",
        ),
    ),
)


class AnalyticTriangularWell(BaseModule):
    """Populate ``ctx['quantum']`` and the scalar convenience artifact ``ns``."""

    def validate(self, ctx: Any) -> list[Issue]:
        polarization = ctx.require("polarization")
        issues: list[Issue] = []
        if polarization.get("barrier_thickness_nm", 0.0) <= 0.0:
            issues.append(
                Issue("polarization.barrier_thickness_nm", "barrier thickness must be positive", "range")
            )
        if polarization.get("sigma_eff_C_m2", 0.0) <= 0.0:
            issues.append(
                Issue(
                    "polarization.sigma_eff_C_m2",
                    "effective polarization charge is zero; there is no 2DEG to solve for",
                    "physics",
                    hint="check the screening factor and the barrier Al fraction",
                )
            )
        if self.param("gamma0_eV_m43") < 0.0:
            issues.append(Issue("params.gamma0_eV_m43", "must not be negative", "range"))
        return issues

    def execute(self, ctx: Any) -> Any:
        polarization = ctx.require("polarization")
        gate = ctx.get("gate") or {}

        thickness_m = float(polarization["barrier_thickness_nm"]) * 1.0e-9
        eps_r = float(polarization["eps_r_barrier"])
        m_eff = float(polarization["m_eff_barrier"])
        sigma_eff = float(polarization["sigma_eff_C_m2"])
        x_al = float(polarization["x_al"])

        delta_ec = float(self.param("delta_ec_ratio")) * float(polarization["band_offset_eV"])

        phi_b, phi_b_source = self._barrier_height(gate, x_al)

        gamma0 = float(self.param("gamma0_eV_m43"))
        valley = int(self.param("valley_degeneracy"))
        include_fermi = bool(self.param("include_fermi"))
        relaxation = float(self.param("relaxation"))
        tolerance = float(self.param("tolerance"))
        max_iterations = int(self.param("max_iterations"))

        n_upper = sigma_eff / ELEMENTARY_CHARGE  # m^-2, the no-barrier-drop limit
        capacitance_factor = VACUUM_PERMITTIVITY * eps_r / (thickness_m * ELEMENTARY_CHARGE**2)
        dos_coefficient = (
            math.pi * REDUCED_PLANCK**2 / (m_eff * ELECTRON_MASS * valley) / ELEMENTARY_CHARGE
        )  # eV*m^2

        n_s = 0.5 * n_upper
        e0 = 0.0
        e_f = 0.0
        converged = False
        residual = 0.0
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            e0 = gamma0 * n_s ** (2.0 / 3.0)
            e_f = e0 + dos_coefficient * n_s if include_fermi else e0
            bracket_joules = ELEMENTARY_CHARGE * (phi_b + e_f - delta_ec)
            n_next = n_upper - capacitance_factor * bracket_joules
            n_next = max(n_next, 0.0)
            residual = n_next - n_s
            if abs(residual) <= tolerance * max(n_upper, 1.0):
                n_s = n_next
                converged = True
                break
            n_s += relaxation * residual

        # Refresh the derived energies at the converged density.
        e0 = gamma0 * n_s ** (2.0 / 3.0)
        e_f = e0 + dos_coefficient * n_s if include_fermi else e0

        quantum = {
            "model": "analytic_triangular_well",
            "ns_m2": n_s,
            "ns_cm2": n_s * 1.0e-4,
            "ns_upper_bound_cm2": n_upper * 1.0e-4,
            "e0_eV": e0,
            "ef_eV": e_f,
            "phi_b_eV": phi_b,
            "phi_b_source": phi_b_source,
            "delta_ec_eV": delta_ec,
            "dos_coefficient_eV_m2": dos_coefficient,
            "capacitance_factor_per_V_m2": capacitance_factor,
            "valley_degeneracy": valley,
            "iterations": iterations,
            "converged": converged,
            "residual_m2": residual,
            "occupancy_fraction": (n_s / n_upper) if n_upper > 0 else 0.0,
            "notes": [
                "Analytic triangular-well estimate, NOT a self-consistent "
                "Schroedinger-Poisson solve.",
                "Fermi-Dirac degeneracy is handled through the explicit E_F term.",
                "Multi-subband occupation and wavefunction penetration into the barrier "
                "are neglected.",
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

    # -- helpers ---------------------------------------------------------
    def _barrier_height(self, gate: dict[str, Any], x_al: float) -> tuple[float, str]:
        source = str(self.param("barrier_source"))
        deck_value = gate.get("barrier_eV")
        if source == "deck" and deck_value is not None:
            return float(deck_value), "deck"
        return (
            float(self.param("phi_b_0_eV")) + float(self.param("phi_b_x_eV")) * x_al,
            "analytic",
        )


MODULE = AnalyticTriangularWell
