"""``transport.dd`` - steady-state 2D drift-diffusion transport.

Turns the equilibrium charge picture (polarization + optional quantum) into a
**biased** one: finite-volume Poisson and electron continuity on the structured
mesh, Scharfetter-Gummel fluxes, damped Newton with pseudo-transient
continuation, swept over gate and drain bias, producing the ``iv`` contract that
a future ``extract.iv`` consumes.

Deviations from ``transport_dd_插件设计.md`` v0.1.1, recorded rather than hidden
-------------------------------------------------------------------------------
1. **No SciPy.** The design bound the sparse linear step to ``scipy.sparse``.
   The deployment target has NumPy only, so the linear step is a block
   tridiagonal direct solve written here (``plugins/linear_blocktridiag``) and
   resolved through the Deck's ``linear`` slot, which is what the design wanted
   architecturally anyway.
2. **Boltzmann statistics only.** The design defaulted to a Joyce-Dixon
   correction. A degeneracy correction has to change the Scharfetter-Gummel flux
   too (generalised Einstein relation); shipping one without the other would be
   quietly inconsistent, so ``statistics`` currently accepts ``boltzmann`` only
   and rejects the rest loudly.
3. **Field-dependent mobility.** ``vsat_cm_s`` / ``beta_field`` are exposed and
   *on* by default (Caughey-Thomas), with the exact derivative in the Jacobian -
   the Scharfetter-Gummel prefactor carries the mobility and the face field
   depends on the band edges, so leaving the derivative out would make the
   Jacobian inexact exactly in the high-field regime the model exists for. Set
   ``vsat_cm_s`` to 0 (or negative) to fall back to a constant mobility.
4. **Interface on a node.** The mesh artifact puts layer boundaries on nodes, so
   the permittivity and band-offset steps are smeared over one cell there - the
   first-order treatment the Schroedinger-Poisson solver deliberately avoided by
   building a face-based grid. Consuming the mesh as given is the design's own
   instruction (section 3); the consequence is documented in README, and its one
   measured effect is a 2DEG sheet density about 1.7x the polarization bound.

Why ``partially-verified`` and not ``unverified``
-------------------------------------------------
The design's own gate was "T0-T4 before anything else, and ``verification`` stays
``unverified`` until then". They are in: the uniform-resistor closed form to
1e-6, a manufactured-solution second-order check of the Poisson operator
(observed orders 2.008 / 2.002 / 2.001), the terminal-current continuity gate,
and continuation-path independence. What keeps it short of ``verified`` is that
the *numbers* rest on an uncalibrated mobility, ``statistics`` is Boltzmann only,
and two of the design's criteria cannot be met by this discretisation at all: the
1 meV band-edge agreement with a 1D Schroedinger-Poisson profile (the smeared
interface above) and drain-current saturation (no field-dependent mobility).
"""

from __future__ import annotations

from typing import Any

from huntian.errors import Issue
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

MANIFEST = ModuleManifest(
    module_id="transport.dd",
    module_version="0.1.0",
    category=ModuleCategory.TRANSPORT,
    description=(
        "Steady-state 2D drift-diffusion transport for the AlGaN/GaN HEMT: "
        "finite-volume discretisation with Scharfetter-Gummel fluxes on the "
        "structured mesh, damped Newton with pseudo-transient continuation, and "
        "a bias sweep producing the IV contract. The sparse linear step comes "
        "from the Deck's 'linear' service. Optional quantum warm start from the "
        "sp1d equilibrium solution."
    ),
    capabilities=("transport", "transport.dd", "current", "iv"),
    requires=("mesh", "structure", "materials", "polarization"),
    produces=("electrostatics", "carriers", "current", "iv"),
    verification="partially-verified",
    author="Huntian",
    tags=("transport", "drift-diffusion", "newton", "iv", "hemt"),
    parameters=(
        # ---- bias sweep --------------------------------------------------
        ParamSpec.string(
            "vds_sweep_V",
            "0.0,1.0,5.0,10.0,20.0",
            doc="comma-separated drain voltages; the outer sweep loop",
        ),
        ParamSpec.string(
            "vgs_sweep_V",
            "-4.0,-2.0,0.0",
            doc="comma-separated gate voltages; the inner sweep loop",
        ),
        ParamSpec.boolean(
            "sweep_reuse_previous",
            True,
            doc="start each Newton solve from the previous bias point's solution",
        ),
        ParamSpec.float(
            "max_step_V",
            0.1,
            unit="V",
            minimum=0.0,
            maximum=100.0,
            doc=(
                "largest internal drain-bias step; the sweep walks to each "
                "requested point in steps no bigger than this, because a cold "
                "jump of several volts stalls the line search. 0 disables "
                "sub-stepping and jumps straight to each requested point"
            ),
        ),
        # ---- physics ------------------------------------------------------
        ParamSpec.float(
            "temperature_K",
            300.0,
            unit="K",
            minimum=200.0,
            maximum=500.0,
            calibratable=True,
            doc="isothermal lattice temperature; thermal coupling is future work",
        ),
        ParamSpec.float(
            "mu0_cm2_Vs",
            1500.0,
            unit="cm^2/(V*s)",
            minimum=1.0,
            maximum=10000.0,
            calibratable=True,
            source="placeholder pending a mobility plugin and Hall calibration",
            doc="low-field electron mobility",
        ),
        ParamSpec.float(
            "vsat_cm_s",
            2.0e7,
            unit="cm/s",
            minimum=0.0,
            maximum=1.0e9,
            calibratable=True,
            source="placeholder pending a mobility plugin and Hall calibration",
            doc=(
                "Caughey-Thomas saturation velocity; 0 or negative disables the "
                "field dependence and restores a constant mobility"
            ),
        ),
        ParamSpec.float(
            "beta_field",
            1.0,
            unit="1",
            minimum=1.0,
            maximum=4.0,
            doc="Caughey-Thomas exponent; 1 is the standard form",
        ),
        ParamSpec.float(
            "delta_ec_ratio",
            0.7,
            unit="1",
            minimum=0.0,
            maximum=1.0,
            calibratable=True,
            doc=(
                "conduction-band offset as a fraction of the band-gap difference; "
                "only used when no quantum module provides delta_ec_eV"
            ),
        ),
        ParamSpec.choice(
            "statistics",
            ("boltzmann",),
            default="boltzmann",
            doc=(
                "carrier statistics; only Boltzmann is implemented in v0.1.0 - a "
                "degeneracy correction must change the flux as well (see the "
                "module docstring)"
            ),
        ),
        ParamSpec.choice(
            "quantum_warm_start",
            ("auto", "off"),
            default="auto",
            doc=(
                "auto: use ctx['quantum'] as the initial guess and as the "
                "equilibrium consistency check when present"
            ),
        ),
        # ---- Newton / PTC -------------------------------------------------
        ParamSpec.integer(
            "max_newton_iterations", 200, minimum=1, maximum=5000
        ),
        ParamSpec.float(
            "newton_tolerance",
            1.0e-9,
            minimum=1.0e-14,
            maximum=1.0e-2,
            doc="scaled residual infinity-norm tolerance",
        ),
        ParamSpec.integer(
            "damping_max_halvings",
            30,
            minimum=0,
            maximum=200,
            doc="line-search halvings per Newton step",
        ),
        ParamSpec.boolean(
            "ptc_enabled",
            True,
            doc=(
                "pseudo-transient continuation; the safety net for the strongly "
                "polarised 2DEG"
            ),
        ),
        ParamSpec.float(
            "ptc_tau0_s",
            1.0e-12,
            unit="s",
            minimum=1.0e-18,
            maximum=1.0,
            doc="initial pseudo-time step; grows multiplicatively on success",
        ),
        # ---- linear service and gates -------------------------------------
        ParamSpec.string(
            "linear_solver",
            "linear.blocktridiag",
            doc=(
                "module id of the linear solver service, resolved through "
                "ctx.services; must declare the 'linear' capability"
            ),
        ),
        ParamSpec.boolean(
            "require_continuity",
            True,
            doc=(
                "raise when the terminal-current continuity gate fails, instead "
                "of reporting it and carrying on"
            ),
        ),
        ParamSpec.float(
            "continuity_tol",
            1.0e-6,
            minimum=1.0e-14,
            maximum=1.0e-1,
            doc="tolerance on |Is+Id|/max(|Id|, eps) at every bias point",
        ),
        ParamSpec.float(
            "max_continuity_error",
            1.0e-4,
            minimum=1.0e-14,
            maximum=1.0,
            doc=(
                "absolute floor (A/m) below which the relative continuity error "
                "is dominated by floating-point noise and is not gated"
            ),
        ),
    ),
)


class TransportDD(BaseModule):
    """Bias sweep around the drift-diffusion core; see :mod:`transport_dd.plugin`."""

    def validate(self, ctx: Any) -> list[Issue]:
        from .plugin import validate as _validate

        return _validate(self, ctx)

    def execute(self, ctx: Any) -> Any:
        from .plugin import execute as _execute

        return _execute(self, ctx)


MODULE = TransportDD

__all__ = ["MANIFEST", "MODULE", "TransportDD"]
