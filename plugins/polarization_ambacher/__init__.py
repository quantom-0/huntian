"""Ambacher polarization model for Ga-face AlGaN/GaN heterostructures.

Computes the spontaneous and piezoelectric polarization of a strained AlGaN
barrier on a relaxed GaN channel, and the resulting bound sheet charge at the
interface. The sheet charge - not doping - is what induces the 2DEG in a GaN
HEMT, so this module is the physical starting point of the whole chain.

The *screening factor* is deliberately a calibratable parameter: only part of
the theoretical polarization charge survives surface-state screening, and no
universally agreed value exists in the literature. It should be fitted against
Hall data rather than assumed.

References
----------
Ambacher et al., J. Appl. Phys. 87, 334 (2000)
Ambacher et al., J. Appl. Phys. 92, 3202 (2002)
"""

from __future__ import annotations

from typing import Any

from nemo.errors import Issue
from nemo.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

ELEMENTARY_CHARGE = 1.602176634e-19  # C

MANIFEST = ModuleManifest(
    module_id="physics.polarization.ambacher",
    module_version="0.1.0",
    category=ModuleCategory.PHYSICS,
    description=(
        "Spontaneous + piezoelectric polarization of a strained AlGaN barrier on GaN, "
        "with a calibratable surface-state screening factor. Produces the bound sheet "
        "charge that induces the 2DEG."
    ),
    capabilities=("polarization", "polarization.ambacher"),
    requires=("structure", "materials"),
    produces=("polarization",),
    verification="partially-verified",
    author="NemoTCAD",
    tags=("polarization", "2deg", "iii-nitride"),
    parameters=(
        ParamSpec.float(
            "screening",
            0.8,
            unit="1",
            minimum=0.0,
            maximum=1.0,
            calibratable=True,
            doc=(
                "fraction of the theoretical polarization sheet charge that survives "
                "surface-state screening; the single largest polarization uncertainty"
            ),
            source=(
                "no universally agreed value - calibrate against Hall data. "
                "The unscreened sigma_pol/e overpredicts n_s in practice."
            ),
        ),
        ParamSpec.float(
            "relaxed_fraction",
            1.0,
            unit="1",
            minimum=0.0,
            maximum=1.0,
            calibratable=True,
            doc="degree to which the barrier is strained to the channel (1.0 = fully strained)",
            source="assume pseudomorphic growth unless TEM/XRD says otherwise",
        ),
        ParamSpec.integer(
            "valley_degeneracy",
            1,
            minimum=1,
            maximum=6,
            doc="conduction-band valley degeneracy carried on to the quantum module",
        ),
        ParamSpec.choice(
            "polarity",
            ("ga_face", "n_face"),
            default="ga_face",
            doc="growth polarity; only Ga-face is implemented in this version",
        ),
    ),
)


class AmbacherPolarization(BaseModule):
    """Compute ``ctx['polarization']`` from the structure and material library."""

    def validate(self, ctx: Any) -> list[Issue]:
        structure = ctx.require("structure")
        issues: list[Issue] = []
        roles = {layer.get("role") for layer in structure.get("layers", [])}
        if "barrier" not in roles:
            issues.append(Issue("structure.layers", "no barrier layer present", "physics"))
        if "channel" not in roles:
            issues.append(Issue("structure.layers", "no channel layer present", "physics"))
        if self.param("polarity") != "ga_face":
            issues.append(
                Issue(
                    "params.polarity",
                    "only Ga-face polarity is implemented in this module version",
                    "unsupported",
                )
            )
        return issues

    def execute(self, ctx: Any) -> Any:
        structure = ctx.require("structure")
        materials = ctx.require("materials")

        barrier = _layer_by_role(structure["layers"], "barrier")
        channel = _layer_by_role(structure["layers"], "channel")
        if barrier is None or channel is None:
            raise ValueError("structure must contain one barrier and one channel layer")

        x_al = float(barrier.get("x_al", 0.0))
        thickness_nm = float(barrier["thickness_nm"])
        channel_material = str(channel["material"])

        def alloy(key: str) -> float:
            return materials.alloy_from_spec("AlGaN", x_al, key)

        # -- spontaneous polarization --------------------------------
        p_sp_barrier = alloy("p_sp_C_m2")
        p_sp_channel = materials.get(channel_material).numeric("p_sp_C_m2")

        # -- piezoelectric polarization ------------------------------
        a_barrier = alloy("a_lattice_A")
        a_channel = materials.get(channel_material).numeric("a_lattice_A")
        relaxed_fraction = float(self.param("relaxed_fraction"))
        # AlGaN on GaN is under biaxial tension because a(AlGaN) < a(GaN).
        strain = relaxed_fraction * (a_channel - a_barrier) / a_barrier

        e31 = alloy("e31_C_m2")
        e33 = alloy("e33_C_m2")
        c13 = alloy("C13_GPa")
        c33 = alloy("C33_GPa")
        p_pz = 2.0 * strain * (e31 - e33 * (c13 / c33))

        delta_p = (p_sp_barrier + p_pz) - p_sp_channel
        sigma_pol = abs(delta_p)

        screening = float(self.param("screening"))
        sigma_eff = screening * sigma_pol

        band_offset = alloy("Eg_eV") - materials.get(channel_material).numeric("Eg_eV")

        polarization = {
            "model": "ambacher",
            "x_al": x_al,
            "polarity": self.param("polarity"),
            "barrier_material": str(barrier["material"]),
            "channel_material": channel_material,
            "barrier_thickness_nm": thickness_nm,
            "p_sp_barrier_C_m2": p_sp_barrier,
            "p_sp_channel_C_m2": p_sp_channel,
            "p_pz_barrier_C_m2": p_pz,
            "delta_p_C_m2": delta_p,
            "sigma_pol_C_m2": sigma_pol,
            "sigma_eff_C_m2": sigma_eff,
            "sigma_pol_cm2": sigma_pol / ELEMENTARY_CHARGE * 1.0e-4,
            "sigma_eff_cm2": sigma_eff / ELEMENTARY_CHARGE * 1.0e-4,
            "strain": strain,
            "relaxed_fraction": relaxed_fraction,
            "screening": screening,
            "eps_r_barrier": alloy("eps_r"),
            "m_eff_barrier": alloy("m_eff"),
            "lattice_a_barrier_A": a_barrier,
            "lattice_a_channel_A": a_channel,
            "band_offset_eV": band_offset,
            "valley_degeneracy": int(self.param("valley_degeneracy")),
            "notes": [
                "Polarization constants come from the material database and are marked "
                "unverified; validate against your own Hall and XRD data before trusting "
                "absolute numbers.",
                "The screening factor absorbs the missing surface-state physics.",
            ],
        }

        ctx.put(
            "polarization",
            polarization,
            producer=MANIFEST.module_id,
            module_version=MANIFEST.module_version,
        )
        return ctx


def _layer_by_role(layers: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for layer in layers:
        if layer.get("role") == role:
            return layer
    return None


MODULE = AmbacherPolarization
