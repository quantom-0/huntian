"""First-order figure-of-merit extractor.

Packages the computed charge state into a result record and adds a closed-form
threshold estimate from charge control:

    V_th = phi_b - dE_C - sigma_eff * d / (eps0 * eps_r)

where every energy in electronvolts is numerically equal to a voltage.

**Scope warning.** This is *not* a substitute for extracting V_th from a real
Id-Vg sweep. It is the charge-control limit, which is a useful sanity anchor
and a placeholder for the full IV / C-V / S-parameter extraction chain.
"""

from __future__ import annotations

from typing import Any

from huntian.errors import Issue
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
BOLTZMANN = 1.380649e-23  # J/K

MANIFEST = ModuleManifest(
    module_id="extract.fom",
    module_version="0.1.0",
    category=ModuleCategory.EXTRACT,
    description=(
        "Packages the 2DEG charge state and a charge-control threshold estimate into "
        "results.fom. Placeholder for the full IV / C-V / S-parameter extraction chain."
    ),
    capabilities=("extract", "extract.fom"),
    requires=("ns", "quantum", "polarization"),
    produces=("results.fom",),
    verification="unverified",
    author="Huntian",
    tags=("extract", "fom"),
    parameters=(
        ParamSpec.float(
            "vth_correction_V",
            0.0,
            unit="V",
            minimum=-5.0,
            maximum=5.0,
            calibratable=True,
            doc=(
                "empirical offset absorbing the difference between the charge-control "
                "estimate and a measured threshold"
            ),
        ),
        ParamSpec.float(
            "interface_charge_cm2",
            0.0,
            unit="cm^-2",
            calibratable=True,
            doc="net fixed interface charge subtracted from the 2DEG density",
        ),
    ),
)


class FigureOfMeritExtractor(BaseModule):
    """Write ``ctx['results.fom']``."""

    def validate(self, ctx: Any) -> list[Issue]:
        issues: list[Issue] = []
        temperature = ctx.get("temperature_K", 300.0)
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            issues.append(Issue("temperature_K", "temperature must be positive", "range"))
        return issues

    def execute(self, ctx: Any) -> Any:
        quantum = ctx.require("quantum")
        polarization = ctx.require("polarization")
        temperature = float(ctx.get("temperature_K", 300.0))

        thickness_m = float(polarization["barrier_thickness_nm"]) * 1.0e-9
        eps_r = float(polarization["eps_r_barrier"])
        sigma_eff = float(polarization["sigma_eff_C_m2"])
        phi_b = float(quantum["phi_b_eV"])
        delta_ec = float(quantum["delta_ec_eV"])
        e_f = float(quantum["ef_eV"])

        barrier_drop = sigma_eff * thickness_m / (VACUUM_PERMITTIVITY * eps_r)  # volts
        vth = (
            phi_b
            - delta_ec
            - barrier_drop
            + float(self.param("vth_correction_V"))
        )

        interface_charge = float(self.param("interface_charge_cm2"))
        ns_cm2 = float(quantum["ns_cm2"]) - interface_charge
        cox = VACUUM_PERMITTIVITY * eps_r / thickness_m  # F/m^2

        fom = {
            "model": "charge_control_fom",
            "ns_cm2": ns_cm2,
            "ns_uncorrected_cm2": float(quantum["ns_cm2"]),
            "vth_est_V": vth,
            "barrier_drop_V": barrier_drop,
            "phi_b_eV": phi_b,
            "delta_ec_eV": delta_ec,
            "ef_eV": e_f,
            "sigma_pol_cm2": float(polarization["sigma_pol_cm2"]),
            "sigma_eff_cm2": float(polarization["sigma_eff_cm2"]),
            "screening": float(polarization["screening"]),
            "x_al": float(polarization["x_al"]),
            "barrier_nm": float(polarization["barrier_thickness_nm"]),
            "cox_F_m2": cox,
            "cox_uF_cm2": cox * 1.0e2,
            "thermal_voltage_V": BOLTZMANN * temperature / ELEMENTARY_CHARGE,
            "temperature_K": temperature,
            "assumptions": [
                "V_th is the charge-control (full-depletion) estimate, not an "
                "Id-Vg extraction.",
                "Series resistance, trap-induced dispersion and short-channel effects "
                "are not represented here.",
                "This extractor is marked 'unverified' by design.",
            ],
        }

        ctx.put(
            "results.fom",
            fom,
            producer=MANIFEST.module_id,
            module_version=MANIFEST.module_version,
        )
        return ctx


MODULE = FigureOfMeritExtractor
