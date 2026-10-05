"""Pipeline shell for ``transport.dd``: the bias sweep, its artefacts and gates.

The numerics live next door: :mod:`transport_dd.physics` (statistics, mobility,
Scharfetter-Gummel flux and its analytic derivatives), :mod:`transport_dd.assemble`
(finite-volume Poisson and electron continuity, the block Jacobian, the
equilibrium start) and :mod:`transport_dd.newton` (damped Newton with a
backtracking line search and pseudo-transient continuation). This module only
sequences them:

1. build the grid from the mesh/structure/materials/polarization artefacts;
2. solve the *equilibrium* problem (non-linear Poisson, then the density that
   goes with it) - at zero bias that pair is the exact solution, which is what
   makes it a usable anchor rather than a guess;
3. sweep drain bias (outer) and gate bias (inner) with continuation, one damped
   Newton solve per point, each starting from the previous point's solution;
4. gate every point on terminal-current continuity and write the artefacts.

Bias continuation is more than "reuse the last solution". A drain step moves the
drain contact's potential while the interior still holds the old one, and that
contact-local shock is enough to stall a line search. The interior potential is
therefore pre-shifted by ``dVds * ramp(x)`` - the same linear ramp the contact
data imposes - which is what turned a stalled solve into five-iteration solves
(see ``工作日志.md``).

Verification status: the module ships as ``partially-verified``. What *is*
checked in the suite is the flux law against its closed form, every flux partial
against central differences (with and without the field-dependent mobility), the
zero-current equilibrium invariant, the coupled Jacobian against central
differences row by row, exact zero terminal current at 0 V, terminal current
continuity at a biased point, the uniform-resistor closed form, a manufactured
solution for the Poisson operator, and drain-current saturation. Field-level
agreement with a reference device has not been demonstrated, and the remaining
deviations in the package docstring (Boltzmann only, node-smeared interfaces)
both weaken it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nemo.errors import Issue

__all__ = ["validate", "execute"]

#: artefact names this module publishes; also the output keys a Deck may ask for
ARTIFACTS = ("electrostatics", "carriers", "current", "iv")


def _parse_sweep(module: Any, name: str) -> list[float]:
    """Parse a comma-separated voltage list, keeping the order as written."""
    raw = str(module.param(name))
    values: list[float] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:  # pragma: no cover - guarded by validate()
            raise RuntimeError(
                f"transport.dd: parameter {name!r} is not a list of numbers: {raw!r}"
            ) from exc
    return values


def validate(module: Any, ctx: Any) -> list[Issue]:
    """Check the artefacts, the linear service and the sweep before running."""
    issues: list[Issue] = []
    for key in ("mesh", "structure", "materials", "polarization"):
        if not ctx.has(key):
            issues.append(
                Issue(
                    f"ctx.{key}",
                    f"transport.dd requires the {key!r} artifact",
                    "required",
                    hint="bind the mesh/polarization stages upstream",
                )
            )
    services = getattr(ctx, "services", None)
    solver = str(module.param("linear_solver"))
    if services is None:
        issues.append(
            Issue(
                "params.linear_solver",
                "transport.dd needs a service registry to resolve its linear step",
                "service",
                hint="run through nemo.pipeline.Executor",
            )
        )
    elif not services.has(solver):
        issues.append(
            Issue(
                "params.linear_solver",
                f"linear solver {solver!r} is not registered",
                "service",
                hint="bind the Deck's 'linear' slot, e.g. linear.blocktridiag",
            )
        )
    polarization = ctx.get("polarization") or {}
    if float(polarization.get("sigma_eff_C_m2", 0.0)) <= 0.0:
        issues.append(
            Issue(
                "polarization.sigma_eff_C_m2",
                "no polarization charge: there is no 2DEG to transport",
                "physics",
            )
        )
    for name in ("vds_sweep_V", "vgs_sweep_V"):
        if not _parse_sweep(module, name):
            issues.append(
                Issue(
                    f"params.{name}",
                    f"{name} is empty",
                    "value",
                    hint="give at least one bias point, e.g. '0.0,1.0'",
                )
            )
    structure = ctx.get("structure") or {}
    layers = structure.get("layers") or []
    roles = {str(layer.get("role", "")) for layer in layers}
    if "channel" not in roles or "barrier" not in roles:
        issues.append(
            Issue(
                "structure.layers",
                "transport.dd needs one barrier layer and one channel layer",
                "structure",
                hint="the polarization sheet sits at the barrier/channel interface",
            )
        )
    return issues


def execute(module: Any, ctx: Any) -> Any:
    from . import assemble as A
    from . import newton as N

    grid_module = ctx.require("mesh")
    structure = ctx.require("structure")
    materials = ctx.require("materials")
    polarization = ctx.require("polarization")
    services = ctx.require_services()
    linear = services.get(str(module.param("linear_solver")))

    temperature = float(module.param("temperature_K"))
    mu0 = float(module.param("mu0_cm2_Vs")) * 1.0e-4  # cm^2/(V*s) -> m^2/(V*s)
    statistics = str(module.param("statistics"))
    gate = structure.get("gate") or {}
    barrier_eV = float(gate.get("barrier_eV", 0.94))
    delta_ec_ratio = float(module.param("delta_ec_ratio"))

    grid = A.build_grid2d(
        grid_module,
        structure,
        materials,
        polarization,
        temperature_K=temperature,
        mu0_m2_Vs=mu0,
        delta_ec_ratio=delta_ec_ratio,
        gate_barrier_eV=barrier_eV,
    )
    boundary = A.make_boundary(grid, vds_V=0.0, vgs_V=0.0)
    vsat_cm_s = float(module.param("vsat_cm_s"))
    system = A.DriftDiffusionSystem(
        grid,
        temperature_K=temperature,
        boundary=boundary,
        mu0_m2_Vs=mu0,
        statistics=statistics,
        # A non-positive saturation velocity switches the field dependence off.
        vsat_m_s=(vsat_cm_s * 1.0e-2) if vsat_cm_s > 0.0 else None,
        beta_field=float(module.param("beta_field")),
    )

    # -- equilibrium anchor ------------------------------------------------
    psi_equilibrium = system.poisson_start(
        linear=linear, tolerance=1.0e-11, statistics=statistics
    )
    system.adopt_equilibrium(psi_equilibrium)
    equilibrium_state = None
    if _quantum_warm_start(module, ctx):
        equilibrium_state = _quantum_state(ctx, grid, system)

    vds_values = _parse_sweep(module, "vds_sweep_V")
    vgs_values = _parse_sweep(module, "vgs_sweep_V")
    reuse = bool(module.param("sweep_reuse_previous"))
    tolerance = float(module.param("newton_tolerance"))
    max_iterations = int(module.param("max_newton_iterations"))
    halvings = int(module.param("damping_max_halvings"))
    ptc = (
        N.PseudoTransient(tau_s=float(module.param("ptc_tau0_s")))
        if bool(module.param("ptc_enabled"))
        else None
    )
    continuity_tol = float(module.param("continuity_tol"))
    absolute_floor = float(module.param("max_continuity_error"))
    require_continuity = bool(module.param("require_continuity"))

    points: list[dict[str, Any]] = []
    state: np.ndarray | None = equilibrium_state
    previous_vds: float | None = None
    failed: list[str] = []
    max_step = abs(float(module.param("max_step_V")))
    for vds in vds_values:
        # Walk to the requested drain bias in internal steps. A sweep that jumps
        # straight to 5 V from 0 V is not a continuation, it is a cold start with
        # the wrong initial guess; the intermediate points are not reported, they
        # only carry the solution forward.
        if previous_vds is None or previous_vds == vds:
            ladder = [vds]
        else:
            ladder = _ladder(previous_vds, vds, max_step)
        for index, step_vds in enumerate(ladder):
            report = index == len(ladder) - 1
            for vgs in vgs_values:
                state, point = _solve_point(
                    module,
                    system,
                    N,
                    linear,
                    state,
                    psi_equilibrium,
                    reuse=reuse,
                    vds=step_vds,
                    vgs=vgs,
                    previous_vds=previous_vds,
                    report=report,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    halvings=halvings,
                    ptc=ptc,
                    continuity_tol=continuity_tol,
                    absolute_floor=absolute_floor,
                )
                if point is None:
                    continue
                points.append(point)
                if not point["continuity_passed"] or not point["converged"]:
                    failed.append(
                        f"Vds={vds:g} V, Vgs={vgs:g} V: "
                        + (
                            ""
                            if point["converged"]
                            else f"Newton did not converge ({point['message']}); "
                        )
                        + (
                            "terminal current not conserved "
                            f"(imbalance {point['current_imbalance_A_per_m']:.3e} A/m, "
                            f"relative {point['continuity_error']:.3e})"
                            if not point["continuity_passed"]
                            else "continuity gate passed"
                        )
                    )
            previous_vds = step_vds
        previous_vds = vds

    # Publish before gating, so a failed run still leaves the evidence behind
    # instead of only a message.
    _publish(module, ctx, grid, system, state, points, failed, temperature)
    if failed and require_continuity:
        raise RuntimeError(
            "transport.dd: the terminal-current continuity gate failed at "
            + "; ".join(failed)
            + ". Set require_continuity=false to get the artefacts anyway, or "
            "loosen continuity_tol / max_continuity_error knowingly."
        )

    return ctx


def _ladder(start: float, stop: float, max_step: float) -> list[float]:
    """Intermediate drain biases from ``start`` to ``stop``, ``stop`` last."""
    if max_step <= 0.0 or abs(stop - start) <= max_step:
        return [stop]
    count = int(np.ceil(abs(stop - start) / max_step))
    values = [start + (stop - start) * (index / count) for index in range(1, count + 1)]
    values[-1] = stop
    return values


def _solve_point(
    module: Any,
    system: Any,
    N: Any,
    linear: Any,
    state: np.ndarray | None,
    psi_equilibrium: np.ndarray,
    *,
    reuse: bool,
    vds: float,
    vgs: float,
    previous_vds: float | None,
    report: bool,
    tolerance: float,
    max_iterations: int,
    halvings: int,
    ptc: Any,
    continuity_tol: float,
    absolute_floor: float,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """One bias point: continuation start, damped Newton, terminal currents."""
    system.boundary = system.boundary_for(vds_V=vds, vgs_V=vgs)
    system._dirichlet = system.boundary.unknown_mask()
    system._targets = system.boundary.unknown_values()
    if state is not None and reuse:
        guess = np.array(state, dtype=float, copy=True)
        if previous_vds is not None and vds != previous_vds:
            guess = system.bias_shift(guess, d_vds_V=vds - previous_vds)
    else:
        guess = system.initial_guess(psi_field=psi_equilibrium, vds_V=vds)
    guess = np.where(system._dirichlet, system._targets, guess)
    system.equilibrate(guess)

    outcome = N.solve(
        system,
        guess,
        linear=linear,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping_max_halvings=halvings,
        ptc=ptc,
    )
    if not report:
        return outcome.x, None

    currents = system.terminal_currents(outcome.x)
    imbalance = abs(currents["source"] + currents["drain"])
    scale = max(abs(currents["drain"]), abs(currents["source"]), 0.0)
    relative = imbalance / scale if scale > 0.0 else float("inf")
    gated = imbalance > absolute_floor
    passed = (relative <= continuity_tol) if (gated and scale > 0.0) else True
    point = {
        "vds_V": float(vds),
        "vgs_V": float(vgs),
        "converged": bool(outcome.converged),
        "iterations": int(outcome.iterations),
        "residual_norm": float(outcome.residual_norm),
        "message": str(outcome.message),
        "id_A_per_m": float(currents["drain"]),
        "is_A_per_m": float(currents["source"]),
        "ig_A_per_m": float(currents["gate"]),
        "current_imbalance_A_per_m": float(imbalance),
        "continuity_error": float(relative),
        "continuity_gated": bool(gated),
        "continuity_passed": bool(passed),
    }
    return outcome.x, point


# -- helpers ---------------------------------------------------------------
def _quantum_warm_start(module: Any, ctx: Any) -> bool:
    mode = str(module.param("quantum_warm_start"))
    return mode == "auto" and ctx.has("quantum")


def _quantum_state(ctx: Any, grid: Any, system: Any) -> np.ndarray | None:
    """Project a 1D quantum potential onto the 2D grid as a better start.

    The quantum artefact carries an equilibrium band profile on its own vertical
    grid. Used as a *start* only, never as a constraint, and only when the number
    of vertical samples is compatible enough to interpolate; otherwise the
    Poisson start is used and nothing is claimed.
    """
    quantum = ctx.get("quantum") or {}
    values = quantum.get("potential_eV") or quantum.get("psi_V")
    if not values:
        return None
    try:
        profile = np.asarray(values, dtype=float)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if profile.ndim != 1 or profile.size < 2:
        return None
    try:
        psi = np.interp(
            grid.z_m,
            np.linspace(grid.z_m[0], grid.z_m[-1], profile.size),
            profile,
        )
    except Exception:  # pragma: no cover - defensive
        return None
    return system.initial_guess(psi_field=np.repeat(psi[:, None], grid.nx, axis=1))


def _publish(
    module: Any,
    ctx: Any,
    grid: Any,
    system: Any,
    state: np.ndarray | None,
    points: list[dict[str, Any]],
    failed: list[str],
    temperature: float,
) -> Any:
    """Write the artefacts. Field arrays are the *last* solved bias point."""
    last = points[-1] if points else {}
    psi = state[:, :, 0] if state is not None else np.zeros((grid.nz, grid.nx))
    density = (
        state[:, :, 1] * system.density_scale
        if state is not None
        else np.zeros((grid.nz, grid.nx))
    )
    # Sheet density in cm^-2, per unit device width: integrate n dz, then average
    # over the lateral nodes. Summing over the columns instead would report nx
    # times the real value, which is a factor of twenty on the demo mesh.
    profile = (density * grid.wz_m[:, None]).sum(axis=0)
    sheet = float(np.mean(profile)) * 1.0e-4  # cm^-2
    converged = [point for point in points if point["converged"]]
    worst_continuity = max(
        (point["continuity_error"] for point in points), default=0.0
    )

    electrostatics = {
        "psi_V": psi.tolist(),
        "psi_min_V": float(np.min(psi)),
        "psi_max_V": float(np.max(psi)),
        "x_nm": [float(value * 1.0e9) for value in grid.x_m],
        "z_nm": [float(value * 1.0e9) for value in getattr(grid, "z_m", [])]
        if hasattr(grid, "z_m")
        else [],
        "gate_reference_V": float(grid.gate_reference_V),
        "temperature_K": float(temperature),
        "bias_V": {"vds": last.get("vds_V"), "vgs": last.get("vgs_V")},
        "note": (
            "field arrays are the last solved bias point; every point's summary "
            "and the full IV curve are in the current/iv artefacts"
        ),
    }
    carriers = {
        "density_m3": density.tolist(),
        "density_max_m3": float(np.max(density)),
        "density_min_m3": float(np.min(density)),
        "ns_cm2": sheet,
        "density_scale_m3": float(system.density_scale),
        "statistics": str(module.param("statistics")),
        "mobility_m2_Vs": float(system.mu0_m2_Vs),
    }
    current = {
        "points": points,
        "converged_points": len(converged),
        "total_points": len(points),
        "worst_continuity_error": float(worst_continuity),
        "failed": failed,
        "unit": "A/m",
        "note": "per metre of device width; multiply by the drawn width in metres",
    }
    iv = {
        "curves": points,
        "vds_V": [point["vds_V"] for point in points],
        "vgs_V": [point["vgs_V"] for point in points],
        "id_A_per_m": [point["id_A_per_m"] for point in points],
        "unit": "A/m",
        "gate_current_included": False,
        "note": (
            "gate current is not modelled in v0.1.0 (no gate leakage path), so "
            "ig_A_per_m is zero by construction rather than by solution"
        ),
    }

    version = type(module).MANIFEST.module_version
    for name, payload in (
        ("electrostatics", electrostatics),
        ("carriers", carriers),
        ("current", current),
        ("iv", iv),
    ):
        ctx.put(name, payload, producer="transport.dd", module_version=version)
    ctx.put(
        "ns",
        sheet,
        producer="transport.dd",
        module_version=version,
        meta={"unit": "cm^-2", "quantity": "2DEG sheet density at the last bias point"},
    )
    return ctx
