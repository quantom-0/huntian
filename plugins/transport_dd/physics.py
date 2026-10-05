"""Drift-diffusion physics kernels: statistics, fluxes, mobility.

Pure functions, no I/O, no imports beyond NumPy - so the tests can load this
file directly as well as through the plugin package (the same split the
Schroedinger-Poisson solver uses).

Formulation
-----------
Electron-only steady-state drift-diffusion in the **quasi-Fermi** variables
``(psi, phi_n)``:

    n      = N_C_eff(z) * exp( q (psi - phi_n) / kT )
    E_C    = ec0(z) - q psi                       (band edge, eV)
    J_n    = -q mu n grad(phi_n)                  (exact, any statistics)

``ec0`` carries the heterojunction conduction-band offset, so the offset is
folded into ``N_C_eff = N_C exp(-ec0/kT)`` for the density and enters the flux
through the *band-edge* difference. That is what makes the Scharfetter-Gummel
flux below vanish identically at equilibrium across a heterointerface, which is
the property the zero-bias test asserts.

The flux discretisation is Scharfetter-Gummel: the Bernoulli form is the exact
solution of the one-dimensional drift-diffusion equation with a constant field,
so it cannot produce the spurious oscillations a centred difference gives once
the mesh Peclet number exceeds two. Both the flux and every partial derivative
it needs are analytic, so the assembled Jacobian can be checked against central
differences rather than trusted.

A note on statistics
--------------------
``carrier_density`` implements **Boltzmann** statistics only. The design note
called for a Joyce-Dixon correction to cover the degenerate 2DEG; that is
recorded as a deviation rather than half-implemented, because a degeneracy
correction also changes the Scharfetter-Gummel flux (the generalised Einstein
relation) and doing one without the other would be quietly inconsistent. The
choice is explicit in the function signature and validated, so a later version
can add it without changing any caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "ELEMENTARY_CHARGE",
    "BOLTZMANN",
    "VACUUM_PERMITTIVITY",
    "PLANCK",
    "ELECTRON_MASS",
    "EXPONENT_CAP",
    "effective_dos_m3",
    "bernoulli",
    "carrier_density",
    "carrier_density_with_statistics",
    "density_partials",
    "sg_face_current",
    "constant_mobility",
    "band_edge_eV",
]

#: CODATA 2018.
ELEMENTARY_CHARGE = 1.602176634e-19  # C
BOLTZMANN = 1.380649e-23  # J/K
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
PLANCK = 6.62607015e-34  # J*s
ELECTRON_MASS = 9.1093837015e-31  # kg

#: Numerical guard on ``q(psi - phi_n)/kT``. Physical operating points stay
#: below ~150 (a 4 V forward bias at 300 K); the cap only catches runaway trial
#: states inside a Newton line search. Above the cap the implemented residual is
#: *exactly* exponential-in-cap, so the Jacobian keeps being its true derivative.
EXPONENT_CAP = 100.0


def effective_dos_m3(m_eff_kg: float | np.ndarray, temperature_K: float) -> np.ndarray:
    """Conduction-band effective density of states ``N_C`` in m^-3.

    ``N_C = 2 (2 pi m* kT / h^2)^{3/2}``.
    """
    m = np.asarray(m_eff_kg, dtype=float)
    return 2.0 * (2.0 * np.pi * m * BOLTZMANN * float(temperature_K) / PLANCK**2) ** 1.5


def _bernoulli_pair(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``B(x) = x/(exp(x)-1)`` and ``B'(x)``, evaluated without overflow.

    ``B`` is the Scharfetter-Gummel weighting. Naively ``x/(exp(x)-1)`` overflows
    for ``x`` above ~700 and loses all precision (0/0) at ``x = 0``, so both
    halves are evaluated with the *decaying* exponential and the reflection
    identities

        ``B(-y) = B(y) + y``,   ``B'(-y) = -B'(y) - 1``   (y > 0)

    carry the sign. Getting that reflection wrong leaves a flux that is
    perfectly self-consistent - so a finite-difference Jacobian check still
    passes - while being wrong everywhere the band edge rises along the flux
    direction. It is a good reminder that consistency is not correctness.

    ``x -> 0`` falls back on the Taylor series, where the exponential form is
    0/0.
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1.0e-6
    safe = np.where(small, 0.0, x)
    magnitude = np.abs(safe)
    positive = safe > 0.0

    t = np.exp(-magnitude)
    denominator = 1.0 - t
    with np.errstate(divide="ignore", invalid="ignore"):
        value_abs = magnitude * t / denominator  # B(|x|) >= 0
        slope_abs = t * (1.0 - t - magnitude) / denominator**2

    value = np.where(positive, value_abs, value_abs + magnitude)
    slope = np.where(positive, slope_abs, -slope_abs - 1.0)

    # -- x ~ 0: B = 1 - x/2 + x^2/12 - x^4/720, B' = -1/2 + x/6 - x^3/180
    x2 = x * x
    value_small = 1.0 - 0.5 * x + x2 / 12.0 - x2 * x2 / 720.0
    slope_small = -0.5 + x / 6.0 - x * x2 / 180.0
    return (
        np.where(small, value_small, value),
        np.where(small, slope_small, slope),
    )


def bernoulli(x: Any) -> np.ndarray:
    """``B(x) = x/(exp(x)-1)`` with the numerically safe branches above."""
    return _bernoulli_pair(np.asarray(x, dtype=float))[0]


def _carrier_exponent(psi: Any, phi_n: Any, kT_eV: float) -> np.ndarray:
    return (np.asarray(psi, dtype=float) - np.asarray(phi_n, dtype=float)) / kT_eV


def carrier_density(
    psi: Any,
    phi_n: Any,
    *,
    nc_eff_m3: Any,
    kT_eV: float,
) -> np.ndarray:
    """Electron density in m^-3, ``n = N_C_eff exp(q(psi-phi_n)/kT)``.

    ``kT_eV`` is ``k T / q`` in volts, which is how the exponent stays
    dimensionless with ``psi`` and ``phi_n`` in volts. The exponent is capped at
    :data:`EXPONENT_CAP`; see the constant for why that keeps the Jacobian
    exactly consistent with the residual.
    """
    exponent = np.minimum(_carrier_exponent(psi, phi_n, kT_eV), EXPONENT_CAP)
    return np.asarray(nc_eff_m3, dtype=float) * np.exp(exponent)


def carrier_density_with_statistics(
    psi: Any,
    phi_n: Any,
    *,
    nc_eff_m3: Any,
    kT_eV: float,
    statistics: str = "boltzmann",
) -> np.ndarray:
    """Statistics-aware entry point; only Boltzmann is implemented in v0.1.0."""
    if statistics != "boltzmann":
        raise NotImplementedError(
            f"statistics={statistics!r} is not implemented; v0.1.0 solves "
            "Boltzmann statistics only (a degeneracy correction has to change "
            "the flux as well, see the module docstring)"
        )
    return carrier_density(psi, phi_n, nc_eff_m3=nc_eff_m3, kT_eV=kT_eV)


def density_partials(
    psi: Any,
    phi_n: Any,
    *,
    nc_eff_m3: Any,
    kT_eV: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n, dn/dpsi, dn/dphi_n)``, all in SI units."""
    psi = np.asarray(psi, dtype=float)
    exponent = np.minimum(_carrier_exponent(psi, phi_n, kT_eV), EXPONENT_CAP)
    unclipped = exponent < EXPONENT_CAP
    density = np.asarray(nc_eff_m3, dtype=float) * np.exp(exponent)
    scale = np.where(unclipped, density / kT_eV, 0.0)
    return density, scale, -scale


def field_mobility(
    mu0_m2_Vs: Any,
    e_field_V_m: Any,
    *,
    vsat_m_s: Any,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Caughey-Thomas field-dependent mobility, and its derivative.

    ``mu(E) = mu0 * [1 + (mu0 E / vsat)^beta]^(-1/beta)``, which tends to ``mu0``
    at low field and to ``vsat / E`` at high field - the velocity-saturation
    branch a HEMT needs beyond roughly 1 V of drain bias.

    Returns ``(mu, dmu/dE)``. The derivative is what keeps the Jacobian exact:
    the Scharfetter-Gummel prefactor carries ``mu``, and the field itself depends
    on the band edges, so the flux gains a term proportional to ``dmu/dE``.
    Written in the closed form that stays finite at ``E = 0`` for ``beta >= 1``.
    """
    mu0 = np.asarray(mu0_m2_Vs, dtype=float)
    field = np.abs(np.asarray(e_field_V_m, dtype=float))
    beta = float(beta)
    if beta < 1.0:
        raise ValueError(f"beta must be >= 1, got {beta}")
    ratio = mu0 * field / float(vsat_m_s)
    factor = 1.0 + ratio**beta
    mu = mu0 * factor ** (-1.0 / beta)
    # d(mu)/dE = -mu0 * a * (aE)^(beta-1) * (1 + u)^(-1/beta - 1), with a = mu0/vsat
    slope = (
        -mu0
        * (mu0 / float(vsat_m_s))
        * ratio ** (beta - 1.0)
        * factor ** (-1.0 / beta - 1.0)
    )
    return mu, slope


def sg_face_current(
    *,
    band_left: Any,
    band_right: Any,
    n_left: Any,
    n_right: Any,
    mu_m2_Vs: Any,
    kT_eV: float,
    length_m: Any,
    statistics: str = "boltzmann",
    vsat_m_s: Any = None,
    beta_field: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scharfetter-Gummel electron current density from left to right.

    Returns ``(J, dJ/dpsi_L, dJ/dn_L, dJ/dpsi_R, dJ/dn_R)`` with ``J`` in A/m^2,
    the potential partials in A/m^2 per volt and the density partials in
    A/m per (m^-3) - a length per second, so a caller carrying ``n`` as its own
    unknown can chain straight through.

    **The density partials do not depend on the density.** ``dJ/dn_L =
    -q mu kT B(-delta)/h`` is set by the geometry, the mobility and the band-edge
    difference alone. That is the property that makes a ``(psi, n)`` formulation
    well posed in a wide-gap barrier or a depleted buffer, where the density has
    underflowed to zero and a quasi-Fermi formulation would leave the continuity
    equation identically empty. It is why this module solves for the density.

    ``band_*`` are the conduction-band edges in eV (``ec0 - psi`` numerically,
    with ``psi`` in volts), so the Bernoulli argument ``delta = (E_C,L -
    E_C,R)/kT`` carries both the electrostatic and the heterojunction step.
    """
    if statistics != "boltzmann":
        raise NotImplementedError(
            f"statistics={statistics!r} is not implemented in v0.1.0"
        )
    kT_eV = float(kT_eV)
    band_left = np.asarray(band_left, dtype=float)
    band_right = np.asarray(band_right, dtype=float)
    n_left = np.asarray(n_left, dtype=float)
    n_right = np.asarray(n_right, dtype=float)
    mu = np.asarray(mu_m2_Vs, dtype=float)
    length = np.asarray(length_m, dtype=float)

    delta = (band_left - band_right) / kT_eV
    weight_right, slope_right = _bernoulli_pair(delta)
    weight_left, slope_left = _bernoulli_pair(-delta)

    # The mobility acts on the face's own field, E = |E_C,L - E_C,R| / h. With
    # Caughey-Thomas it is no longer a constant, and because the field depends on
    # the band edges the flux picks up a term proportional to dmu/dE - leaving it
    # out would make the Jacobian inexact in exactly the high-field regime the
    # model exists to describe.
    if vsat_m_s is None:
        mu = np.asarray(mu_m2_Vs, dtype=float)
        dmu_dE = None
    else:
        field = np.abs(band_left - band_right) / length
        mu, dmu_dE = field_mobility(
            mu_m2_Vs, field, vsat_m_s=vsat_m_s, beta=float(beta_field)
        )

    prefactor = ELEMENTARY_CHARGE * mu * kT_eV / length  # A/m^2
    bracket = n_right * weight_right - n_left * weight_left
    current = prefactor * bracket

    # The Bernoulli argument depends on *both* potentials, so both terms of J
    # contribute to each potential partial, and they must come out antisymmetric
    # - shifting both potentials by the same constant leaves delta, and hence the
    # current, unchanged. Writing only the term that carries n_left (the shape a
    # quasi-Fermi formulation invites, where dn/dpsi = n/kT) is 100 % wrong while
    # the density partials stay exact, which is exactly the kind of error a
    # system-level Jacobian check taken only at the reference state can miss.
    scale = 1.0 / kT_eV
    potential_partial = prefactor * scale * (
        n_right * slope_right + n_left * slope_left
    )
    if dmu_dE is not None:
        # dE/dpsi_L = -sign(delta)/h, dE/dpsi_R = +sign(delta)/h; the mobility
        # term is antisymmetric like the Bernoulli one, so the constant-shift
        # invariance survives.
        direction = np.sign(band_left - band_right) / length
        mobility_partial = (
            ELEMENTARY_CHARGE * dmu_dE * kT_eV / length
        ) * bracket * direction
        potential_partial = potential_partial + mobility_partial
    return (
        current,
        -potential_partial,
        -prefactor * weight_left,
        potential_partial,
        prefactor * weight_right,
    )


def band_edge_eV(psi: Any, offset_eV: Any) -> np.ndarray:
    """Conduction-band edge ``E_C = ec0 - q psi`` in eV (``psi`` in volts)."""
    return np.asarray(offset_eV, dtype=float) - np.asarray(psi, dtype=float)


def constant_mobility(mu0_m2_Vs: float) -> float:
    """Field-independent mobility in m^2/(V*s).

    v0.1.0 solves with a constant mobility. A field-dependent model
    (Caughey-Thomas, ``vsat``/``beta_field``) is deliberately *not* exposed as a
    parameter until it is implemented end to end, including the derivative that
    keeps the Jacobian exact - see the deviation log in the design note.
    """
    mu = float(mu0_m2_Vs)
    if mu <= 0.0:
        raise ValueError(f"mobility must be positive, got {mu}")
    return mu
