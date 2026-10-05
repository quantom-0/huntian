"""One-dimensional Schroedinger-Poisson numerics for an AlGaN/GaN 2DEG.

Self-contained: it imports nothing but NumPy, so it can be unit-tested directly
from the file system as well as loaded through the plugin registry.

Geometry and conventions
------------------------
``z`` increases downwards from the gate::

    z = 0            metal / AlGaN interface   (Schottky boundary condition)
    0 < z < d        AlGaN barrier             (undoped, eps_b, m_b)
    z = d            polarization sheet charge + conduction-band offset step
    d < z < L        GaN channel and buffer    (undoped, eps_c, m_c)
    z = L            back face, zero field     (insulating buffer)

**The heterointerface is a face, not a node.** Nodes sit at ``d - h/2`` and
``d + h/2``, so the band-offset step, the permittivity step, the effective-mass
step and the sheet charge all live exactly on a cell boundary. Putting them on
a node instead resolves all four at O(h) and costs an order of convergence in
the coupled solve. Cell ``i`` therefore sees the source quadrature of a region
that lies entirely inside one material, which is what makes the midpoint rule
second order here.

Conventions
-----------
* The Fermi level is the energy zero, ``E_F = 0`` everywhere at equilibrium.
* The unknown of the Poisson operator is the **conduction-band edge** ``E_c``
  in eV. At the metal interface ``E_c(0) = q(phi_b - V_g)``, numerically equal
  to ``phi_b - V_g`` volts.
* A positive sheet charge at ``z = d`` makes ``E_c`` dip there: that dip is the
  2DEG well.

Discretisation
--------------
Finite volume on a non-uniform grid. With ``c_i = eps_face_i * eps0 / h_i`` the
operator ``L`` has rows::

    row 0      : Dirichlet, coefficient 1
    row i      : -c_{i-1} E_{i-1} + (c_{i-1}+c_i) E_i - c_i E_{i+1}
    row N-1    : -c_{N-2} E_{N-2} + c_{N-2} E_{N-1}        (zero field)

``c_i`` uses the length-weighted series average of the two adjacent
permittivities, which is exact when the material boundary sits on the face.

The coupled residual is ``R(E_c) = L E_c - q W n[E_c] + b`` with
``W = diag(w_i)`` (zero on the Dirichlet row). Its Jacobian is
``J = L - q W dn/dE_c``, where ``dn/dE_c`` is built from first-order
perturbation theory - both the subband-energy response and the wavefunction
response.

Jacobian accuracy, stated precisely
-----------------------------------
``J`` is the exact derivative of the *discretised* residual. Three terms carry
``dn/dE_c``, and all three are exact:

* **subband-energy response** - ``<psi_i|dE_c|psi_i>`` moves the Fermi-Dirac
  occupation. A rank-``n`` update, and the dominant physics.
* **wavefunction response** - ``dpsi_i = sum_{j != i} psi_j <psi_j|dE_c|psi_i> /
  (E_i - E_j)`` formally runs over the *whole* spectrum. It is not truncated:
  the sum is the resolvent of the tridiagonal Hamiltonian orthogonal to the
  carried state, ``sum_{k != i} psi_k psi_k^T w / (E_i - E_k) =
  -(A - E_i)^dagger``, and that pseudo-inverse is obtained by a rank-1
  (Woodbury) deflation of the tridiagonal operator, so it costs O(N^2) per
  subband instead of the O(N^3) eigenvector sum - see
  :func:`deflated_resolvent`. The eigenvector sum is kept as
  :func:`quantum_response_jacobian_reference`, and the tests assert the two
  agree.
* **density-of-states mass response** - ``m*_i = 1 / sum_z |psi_i|^2 w / m``
  depends on ``psi_i`` too, so it carries a piece of ``dn/dE_c``. Omitting it
  is not a small approximation but a *missing term*, and it has the signature
  central differences are designed to expose: an error floor that does not fall
  as the difference step shrinks (measured at ~1.3e-5 relative here).

Against central differences the assembled Jacobian agrees to ~1e-11 relative,
i.e. to the noise of the difference formula, and the coupled Newton iteration
recovers its quadratic order: the local order estimated from three consecutive
residual norms climbs to ~1.99 before round-off takes over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "ELEMENTARY_CHARGE",
    "VACUUM_PERMITTIVITY",
    "REDUCED_PLANCK",
    "ELECTRON_MASS",
    "BOLTZMANN",
    "HBAR2_OVER_2M0_EV_M2",
    "Grid",
    "build_grid",
    "cell_widths",
    "face_coefficients",
    "solve_tridiagonal",
    "PoissonOperator",
    "conduction_band_edge",
    "schrodinger_bands",
    "solve_schrodinger",
    "subband_masses",
    "subband_densities",
    "volume_density",
    "QuantumState",
    "quantum_state",
    "deflated_resolvent",
    "quantum_response_jacobian",
    "quantum_response_jacobian_reference",
    "SchrodingerPoissonSystem",
    "fang_howard_profile",
    "depletion_seed_ns",
    "AndersonMixer",
    "SPResult",
    "solve_schrodinger_poisson",
]

# -- physical constants (CODATA 2018) ---------------------------------------
ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
REDUCED_PLANCK = 1.054571817e-34  # J*s
ELECTRON_MASS = 9.1093837015e-31  # kg
BOLTZMANN = 1.380649e-23  # J/K

#: hbar^2 / (2*m0) in eV*m^2, so the Schroedinger operator assembles in eV.
HBAR2_OVER_2M0_EV_M2 = (REDUCED_PLANCK**2) / (2.0 * ELECTRON_MASS) / ELEMENTARY_CHARGE

_PIVOT_EPS = 1.0e-300


# ---------------------------------------------------------------------------
# Grid: the interface is a face
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A non-uniform 1D grid whose interfaces fall exactly on cell faces."""

    z_m: np.ndarray
    barrier_index: int
    eps_r: np.ndarray
    m_eff_kg: np.ndarray
    eps_barrier: float
    eps_channel: float
    m_barrier_kg: float
    m_channel_kg: float
    d_m: float

    @property
    def size(self) -> int:
        return int(self.z_m.size)

    @property
    def spacing_m(self) -> np.ndarray:
        return np.diff(self.z_m)

    @property
    def z_nm(self) -> np.ndarray:
        return self.z_m * 1.0e9

    @property
    def interface_face(self) -> int:
        """Index of the face that lies exactly on the heterointerface."""
        return self.barrier_index

    @property
    def channel_index(self) -> int:
        """First node on the channel side, at ``d + h/2``."""
        return self.barrier_index + 1

    def interface_position_m(self) -> float:
        """Midpoint of the interface face."""
        return 0.5 * (self.z_m[self.barrier_index] + self.z_m[self.barrier_index + 1])

    def __repr__(self) -> str:
        return (
            f"Grid(nodes={self.size}, d={self.d_m * 1e9:.1f} nm, "
            f"L={self.z_m[-1] * 1e9:.1f} nm, interface_face={self.barrier_index})"
        )


def _uniform(a: float, b: float, h: float) -> np.ndarray:
    length = b - a
    if length <= 0.0:
        return np.array([a], dtype=float)
    count = max(1, int(round(length / h)))
    return np.linspace(a, b, count + 1)


def _growing(a: float, b: float, h0: float, h_max: float, growth: float) -> np.ndarray:
    nodes = [a]
    z = a
    h = min(h0 * growth, h_max)
    guard = 0
    while z < b and guard < 1_000_000:
        step = min(h, b - z)
        if step <= 0.0:
            break
        z += step
        nodes.append(z)
        h = min(h * growth, h_max)
        guard += 1
    return np.asarray(nodes, dtype=float)


def build_grid(
    *,
    barrier_thickness_m: float,
    channel_extent_m: float,
    h_fine_m: float,
    fine_width_m: float,
    barrier_fine_width_m: float,
    h_barrier_m: float,
    h_max_m: float,
    growth: float,
    eps_r_barrier: float,
    eps_r_channel: float,
    m_eff_barrier_kg: float,
    m_eff_channel_kg: float,
) -> Grid:
    """Build the graded grid with the heterointerface exactly on a face."""
    d = float(barrier_thickness_m)
    if d <= 0.0:
        raise ValueError("barrier thickness must be positive")
    if channel_extent_m <= 0.0:
        raise ValueError("channel extent must be positive")

    half = 0.5 * float(h_fine_m)
    if d <= half:
        raise ValueError(
            f"barrier thickness ({d * 1e9:.3f} nm) must exceed half the fine spacing "
            f"({half * 1e9:.3f} nm) for the interface to sit on a face"
        )

    barrier_fine = min(float(barrier_fine_width_m), d - half)
    fine_width = min(float(fine_width_m), float(channel_extent_m))

    segments: list[np.ndarray] = []
    start = d - half - barrier_fine
    if start > 0.0:
        segments.append(_uniform(0.0, start, h_barrier_m))
    segments.append(_uniform(start, d - half, h_fine_m))
    segments.append(_uniform(d + half, d + half + fine_width, h_fine_m))
    if channel_extent_m > fine_width:
        segments.append(
            _growing(d + half + fine_width, d + half + channel_extent_m, h_fine_m, h_max_m, growth)
        )

    # The barrier and channel segments deliberately do *not* touch: they end and
    # begin half a fine spacing either side of the interface, so the interface
    # lands on a face. Concatenate and de-duplicate shared endpoints instead of
    # assuming continuity.
    z = np.unique(np.concatenate(segments))
    if z.size < 5 or np.any(np.diff(z) <= 0.0):
        raise AssertionError("the grid construction produced a non-monotonic axis")

    barrier_index = int(np.searchsorted(z, d, side="left")) - 1
    if barrier_index < 0 or barrier_index + 1 >= z.size:
        raise AssertionError("the heterointerface did not land between two grid nodes")
    midpoint = 0.5 * (z[barrier_index] + z[barrier_index + 1])
    if abs(midpoint - d) > 1.0e-12 * max(1.0, d):
        raise AssertionError(
            f"the interface face is at {midpoint * 1e9:.6f} nm, not at {d * 1e9:.6f} nm"
        )

    eps_r = np.where(z < d, float(eps_r_barrier), float(eps_r_channel))
    m_eff = np.where(z < d, float(m_eff_barrier_kg), float(m_eff_channel_kg))

    return Grid(
        z_m=z,
        barrier_index=barrier_index,
        eps_r=eps_r,
        m_eff_kg=m_eff,
        eps_barrier=float(eps_r_barrier),
        eps_channel=float(eps_r_channel),
        m_barrier_kg=float(m_eff_barrier_kg),
        m_channel_kg=float(m_eff_channel_kg),
        d_m=d,
    )


def cell_widths(grid: Grid) -> np.ndarray:
    """Finite-volume control-volume widths for every node."""
    h = np.diff(grid.z_m)
    w = np.empty(grid.z_m.size, dtype=float)
    w[0] = 0.5 * h[0]
    w[-1] = 0.5 * h[-1]
    if grid.z_m.size > 2:
        w[1:-1] = 0.5 * (h[:-1] + h[1:])
    return w


def face_coefficients(grid: Grid) -> np.ndarray:
    """``c_i = eps_face_i * eps0 / h_i`` for every face.

    The face permittivity is the length-weighted series average of the two
    adjacent nodes. Because the interface sits exactly on a face, this is the
    exact series combination of the two half-cells there; everywhere else the
    two permittivities agree and it reduces to a plain average.
    """
    h = np.diff(grid.z_m)
    eps_i = grid.eps_r[:-1]
    eps_j = grid.eps_r[1:]
    # (h/2 + h/2) / ((h/2)/eps_i + (h/2)/eps_j)
    eps_face = 2.0 / (1.0 / eps_i + 1.0 / eps_j)
    return eps_face * VACUUM_PERMITTIVITY / h


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm; ``lower[0]`` and ``upper[-1]`` are ignored.

    ``rhs`` is either a single right-hand side of shape ``(n,)`` or a stack of
    them of shape ``(n, m)``. The elimination coefficients depend only on the
    matrix, so a stack costs one factorisation plus O(n*m) - which is what makes
    the block pseudo-inverse in :func:`deflated_resolvent` affordable.
    """
    rhs = np.asarray(rhs, dtype=float)
    n = diag.size
    if n == 0:
        return np.empty_like(rhs)
    c = np.zeros(n, dtype=float)
    dd = np.zeros_like(rhs)

    pivot = diag[0]
    if abs(pivot) < _PIVOT_EPS:
        raise ZeroDivisionError("singular tridiagonal system (zero first pivot)")
    c[0] = upper[0] / pivot
    dd[0] = rhs[0] / pivot

    for i in range(1, n):
        factor = diag[i] - lower[i] * c[i - 1]
        if abs(factor) < _PIVOT_EPS:
            raise ZeroDivisionError(f"singular tridiagonal system at row {i}")
        c[i] = upper[i] / factor if i < n - 1 else 0.0
        dd[i] = (rhs[i] - lower[i] * dd[i - 1]) / factor

    x = np.empty_like(rhs)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - c[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# Poisson operator in conduction-band-edge form
# ---------------------------------------------------------------------------


@dataclass
class PoissonOperator:
    """Discrete ``R(E_c) = L E_c - q^2 W n[E_c] + b`` for the HEMT stack."""

    grid: Grid
    widths: np.ndarray
    coefficients: np.ndarray
    charge_weight: np.ndarray
    sheet: np.ndarray
    gate_ec_eV: float
    offset_eV: np.ndarray
    _dense: np.ndarray | None = field(default=None, repr=False)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        grid: Grid,
        *,
        sheet_charge_C_m2: float,
        gate_ec_eV: float,
        band_offset_eV: float = 0.0,
    ) -> "PoissonOperator":
        """Assemble the operator.

        The unknown is the **electrostatic** part of the conduction-band edge in
        volts (numerically equal to eV), so the space-charge source is
        ``q * n * w`` and the sheet enters as ``sigma`` itself - no additional
        factor of ``q``. Treating ``E_c`` as an energy in joules instead would
        scale the source by ``q`` and collapse the 2DEG.

        The heterojunction band offset is *not* folded into the unknown: it is a
        material step, not a charge, and differencing it would inject a spurious
        interface dipole. It is carried separately and added by
        :meth:`potential` when the Schroedinger equation needs the real band
        profile.
        """
        widths = cell_widths(grid)
        charge_weight = ELEMENTARY_CHARGE * widths
        charge_weight[0] = 0.0  # the Dirichlet row carries no space charge

        # A sheet sitting on a face is handled by eliminating the interface
        # potential between the two one-sided flux relations
        #     G^- = eps_b eps0 (E_I - E_k) / (h/2)
        #     G^+ = eps_c eps0 (E_{k+1} - E_I) / (h/2),   G^+ - G^- = sigma
        # which leaves the linear operator unchanged and moves the sheet into
        # the constant vector with the permittivity weights below. A plain
        # 50/50 split is the equal-permittivity limit of this and is only
        # approximate; the exact weights sum to sigma as they must.
        eps_b = float(grid.eps_barrier)
        eps_c = float(grid.eps_channel)
        sheet = np.zeros(grid.size, dtype=float)
        sigma = float(sheet_charge_C_m2)
        sheet[grid.barrier_index] += sigma * eps_b / (eps_b + eps_c)
        sheet[grid.channel_index] += sigma * eps_c / (eps_b + eps_c)

        # Reference is the AlGaN band edge at the gate, so the conduction band on
        # the GaN side of the heterojunction sits dE_C *below* it. Applying the
        # step the other way round raises the channel instead of confining it,
        # which destroys the well entirely.
        offset = np.where(grid.z_m < grid.d_m, 0.0, -float(band_offset_eV))

        return cls(
            grid=grid,
            widths=widths,
            coefficients=face_coefficients(grid),
            charge_weight=charge_weight,
            sheet=sheet,
            gate_ec_eV=float(gate_ec_eV),
            offset_eV=offset,
        )

    def potential(self, ec_electrostatic: np.ndarray) -> np.ndarray:
        """Full conduction-band profile, offset step included."""
        return ec_electrostatic + self.offset_eV

    # -- operator ---------------------------------------------------------
    def dense(self) -> np.ndarray:
        if self._dense is None:
            n = self.grid.size
            c = self.coefficients
            matrix = np.zeros((n, n), dtype=float)
            matrix[0, 0] = 1.0
            if n > 2:
                idx = np.arange(1, n - 1)
                matrix[idx, idx - 1] = -c[idx - 1]
                matrix[idx, idx] = c[idx - 1] + c[idx]
                matrix[idx, idx + 1] = -c[idx]
            matrix[n - 1, n - 2] = -c[n - 2]
            matrix[n - 1, n - 1] = c[n - 2]
            self._dense = matrix
        return self._dense

    def constant(self) -> np.ndarray:
        """Right-hand constant ``b``: the sheet charge plus the gate value."""
        b = self.sheet.copy()
        b[0] = -self.gate_ec_eV
        return b

    def solve(self, n_volume_m3: np.ndarray) -> np.ndarray:
        """Solve ``L E_c = q W n - b`` by a tridiagonal elimination."""
        n = self.grid.size
        c = self.coefficients
        lower = np.zeros(n, dtype=float)
        diag = np.zeros(n, dtype=float)
        upper = np.zeros(n, dtype=float)
        rhs = self.charge_weight * n_volume_m3 - self.constant()

        lower[0], diag[0], upper[0] = 0.0, 1.0, 0.0
        if n > 2:
            idx = np.arange(1, n - 1)
            lower[idx] = -c[idx - 1]
            diag[idx] = c[idx - 1] + c[idx]
            upper[idx] = -c[idx]
        lower[n - 1] = -c[n - 2]
        diag[n - 1] = c[n - 2]
        upper[n - 1] = 0.0

        return solve_tridiagonal(lower, diag, upper, rhs)

    def residual(self, ec_eV: np.ndarray, n_volume_m3: np.ndarray) -> np.ndarray:
        return self.dense() @ ec_eV - self.charge_weight * n_volume_m3 + self.constant()

    # -- legacy phi form, kept for the closed-form verification tests ------
    def phi_operator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tridiagonal bands of the ``-L`` operator (the phi-space system)."""
        n = self.grid.size
        c = self.coefficients
        lower = np.zeros(n, dtype=float)
        diag = np.zeros(n, dtype=float)
        upper = np.zeros(n, dtype=float)
        lower[0], diag[0], upper[0] = 0.0, 1.0, 0.0
        if n > 2:
            idx = np.arange(1, n - 1)
            lower[idx] = c[idx - 1]
            diag[idx] = -(c[idx - 1] + c[idx])
            upper[idx] = c[idx]
        lower[n - 1] = c[n - 2]
        diag[n - 1] = -c[n - 2]
        upper[n - 1] = 0.0
        return lower, diag, upper


# ---------------------------------------------------------------------------
# Closed-form verification path (phi form, Dirichlet/Neumann selectable)
# ---------------------------------------------------------------------------


def solve_poisson(
    grid: Grid,
    n_volume_m3: np.ndarray,
    sheet_charge_C_m2: np.ndarray,
    *,
    gate_value_V: float = 0.0,
    back_bc: str = "neumann",
    back_value_V: float = 0.0,
) -> np.ndarray:
    """Solve ``d/dz(eps*eps0*dphi/dz) = q*n - sigma*delta`` for ``phi``.

    Kept as the verification path: the finite-volume integrated form of Gauss's
    law is exact for the piecewise-linear solution of a two-dielectric parallel
    plate, so this is checked against closed form to machine precision.
    """
    n_nodes = grid.size
    h = np.diff(grid.z_m)
    w = cell_widths(grid)
    c = face_coefficients(grid)

    lower = np.zeros(n_nodes, dtype=float)
    diag = np.zeros(n_nodes, dtype=float)
    upper = np.zeros(n_nodes, dtype=float)
    rhs = np.zeros(n_nodes, dtype=float)

    interior = np.arange(1, n_nodes - 1)
    if interior.size:
        lower[interior] = c[interior - 1]
        upper[interior] = c[interior]
        diag[interior] = -(c[interior - 1] + c[interior])
        rhs[interior] = (
            ELEMENTARY_CHARGE * n_volume_m3[interior] * w[interior]
            - sheet_charge_C_m2[interior]
        )

    lower[0], diag[0], upper[0], rhs[0] = 0.0, 1.0, 0.0, float(gate_value_V)

    if back_bc == "dirichlet":
        lower[-1], diag[-1], upper[-1], rhs[-1] = 0.0, 1.0, 0.0, float(back_value_V)
    elif back_bc == "neumann":
        lower[-1] = c[-1]
        diag[-1] = -c[-1]
        upper[-1] = 0.0
        rhs[-1] = (
            ELEMENTARY_CHARGE * n_volume_m3[-1] * w[-1] - sheet_charge_C_m2[-1]
        )
    else:
        raise ValueError(f"unsupported back boundary condition {back_bc!r}")

    return solve_tridiagonal(lower, diag, upper, rhs)


def conduction_band_edge(
    phi_V: np.ndarray,
    grid: Grid,
    *,
    phi_b_V: float,
    gate_bias_V: float,
    delta_ec_eV: float,
) -> np.ndarray:
    """Conduction-band edge in eV, with the heterojunction offset step."""
    ec = -phi_V + (float(phi_b_V) - float(gate_bias_V))
    return np.where(grid.z_m < grid.d_m, ec, ec - float(delta_ec_eV))


# ---------------------------------------------------------------------------
# Schroedinger
# ---------------------------------------------------------------------------


def schrodinger_bands(
    grid: Grid, potential_eV: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symmetrised tridiagonal bands of the interior BenDaniel-Duke operator.

    Returns ``(interior, widths, diagonal, off_diagonal)``, where the operator
    ``A = tridiag(off_diagonal, diagonal, off_diagonal)`` acts on ``v =
    sqrt(w) psi``. That change of variable is what makes ``A`` symmetric, and a
    symmetric tridiagonal operator is what lets the wavefunction response be
    read off one tridiagonal elimination - see :func:`deflated_resolvent`.

    This is the single place the Schroedinger discretisation is written down.
    :func:`solve_schrodinger` and both response paths consume it, so the
    derivative cannot drift away from the operator it differentiates.
    """
    z = grid.z_m
    n_nodes = z.size
    if n_nodes < 5:
        raise ValueError("the Schroedinger solve needs at least 5 grid nodes")

    h = np.diff(z)
    w = cell_widths(grid)

    inv_m_face = 0.5 * (1.0 / grid.m_eff_kg[:-1] + 1.0 / grid.m_eff_kg[1:])
    t = HBAR2_OVER_2M0_EV_M2 * (ELECTRON_MASS * inv_m_face) / h

    interior = np.arange(1, n_nodes - 1)
    widths = w[interior]
    diagonal = (t[interior - 1] + t[interior]) / widths + potential_eV[interior]
    if widths.size > 1:
        off_diagonal = -t[interior][:-1] / np.sqrt(widths[:-1] * widths[1:])
    else:
        off_diagonal = np.empty(0, dtype=float)
    return interior, widths, diagonal, off_diagonal


def _tridiagonal_matrix(diagonal: np.ndarray, off_diagonal: np.ndarray) -> np.ndarray:
    """Dense symmetric tridiagonal matrix from its two bands."""
    matrix = np.diag(diagonal)
    if off_diagonal.size:
        matrix = matrix + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return matrix


def solve_schrodinger(
    grid: Grid, potential_eV: np.ndarray, *, n_states: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``n_states`` eigenpairs of the BenDaniel-Duke Hamiltonian.

    Returns ``(energies_eV, psi)`` with ``psi`` of shape ``(n_nodes, n_states)``
    and ``sum(psi**2 * w) == 1`` for every column.
    """
    interior, widths, diagonal, off_diagonal = schrodinger_bands(grid, potential_eV)
    n_states = max(1, min(int(n_states), interior.size))

    values, vectors = np.linalg.eigh(_tridiagonal_matrix(diagonal, off_diagonal))
    values = values[:n_states]
    vectors = vectors[:, :n_states]

    psi = np.zeros((grid.z_m.size, n_states), dtype=float)
    psi[interior, :] = vectors / np.sqrt(widths)[:, None]
    return values, psi


def subband_masses(grid: Grid, psi: np.ndarray) -> np.ndarray:
    """Effective density-of-states mass of each subband, in kg."""
    w = cell_widths(grid)
    inv_m = (np.abs(psi) ** 2 * w[:, None] / grid.m_eff_kg[:, None]).sum(axis=0)
    return 1.0 / inv_m


def subband_densities(
    *,
    masses_kg: np.ndarray,
    energies_eV: np.ndarray,
    ef_eV: float,
    temperature_K: float,
    valley_degeneracy: int,
) -> np.ndarray:
    """Sheet density of each subband in m^-2 (Fermi-Dirac, numerically stable)."""
    kT_J = BOLTZMANN * float(temperature_K)
    prefactor = float(valley_degeneracy) * masses_kg * kT_J / (np.pi * REDUCED_PLANCK**2)
    argument = (float(ef_eV) - energies_eV) * ELEMENTARY_CHARGE / kT_J
    return prefactor * np.logaddexp(0.0, argument)


def volume_density(psi: np.ndarray, densities_m2: np.ndarray) -> np.ndarray:
    """Reconstruct ``n(z)`` in m^-3 from the subband occupations."""
    return (np.abs(psi) ** 2 * densities_m2[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# Quantum state and its response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumState:
    """Everything a Schroedinger solve yields, bundled for the Newton step.

    ``potential_eV`` is the conduction-band profile the eigenpairs belong to.
    The exact Jacobian needs it to rebuild the tridiagonal operator whose
    response it is taking; :func:`quantum_state` always fills it in, so only a
    hand-assembled state can be missing it.
    """

    energies_eV: np.ndarray
    psi: np.ndarray
    masses_kg: np.ndarray
    densities_m2: np.ndarray
    n_volume_m3: np.ndarray
    clamped: bool = False
    potential_eV: np.ndarray | None = None

    @property
    def ns_m2(self) -> float:
        return float(self.densities_m2.sum())


def quantum_state(
    grid: Grid,
    ec_eV: np.ndarray,
    *,
    n_subbands: int,
    temperature_K: float,
    valley_degeneracy: int,
    ns_upper_m2: float | None = None,
    density_clamp: bool = True,
) -> QuantumState:
    """Solve Schroedinger for ``ec_eV`` and return the populated state."""
    energies, psi = solve_schrodinger(grid, ec_eV, n_states=n_subbands)
    masses = subband_masses(grid, psi)
    densities = subband_densities(
        masses_kg=masses,
        energies_eV=energies,
        ef_eV=0.0,
        temperature_K=temperature_K,
        valley_degeneracy=valley_degeneracy,
    )
    clamped = False
    if density_clamp and ns_upper_m2 is not None and ns_upper_m2 > 0.0:
        total = float(densities.sum())
        if total > ns_upper_m2:
            densities = densities * (ns_upper_m2 / total)
            clamped = True
    return QuantumState(
        energies_eV=energies,
        psi=psi,
        masses_kg=masses,
        densities_m2=densities,
        n_volume_m3=volume_density(psi, densities),
        clamped=clamped,
        potential_eV=np.asarray(ec_eV, dtype=float),
    )


def deflated_resolvent(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    energy: float,
    null_vector: np.ndarray,
) -> np.ndarray:
    """``(A - energy)^dagger`` for a symmetric tridiagonal ``A``, in O(n^2).

    ``A - energy`` is singular: ``null_vector`` spans its null space (the
    carried state, expressed in the ``v = sqrt(w) psi`` variable). What the
    wavefunction response needs is the resolvent *orthogonal to that state*,

        ``(A - energy)^dagger = sum_{k != i} v_k v_k^T / (E_k - energy)``,

    which is exactly the sum the truncated Jacobian used to cut short.

    The rank-1 Woodbury (Sherman-Morrison) regularisation

        ``B = A - energy + sigma v v^T``  =>  ``B^{-1} = (1/sigma) v v^T -
        (A - energy)^dagger``

    removes the singularity, and the only remaining question is how to apply
    ``B^{-1}`` without ever forming ``A - energy``. Fixing the gauge answers it:
    drop the row and the column at the node where ``|v|`` is largest (every
    component of a bound-state null vector is non-zero, so any node will do) and
    hold that entry of the solution at zero. The reduced tridiagonal system is
    nonsingular, its solution ``x_g`` solves ``(A - energy) x_g = b - (v^T b) v``
    exactly, and subtracting the null component gives

        ``(A - energy)^dagger b = x_g - (v^T x_g) v``.

    One Thomas factorisation therefore serves all ``n`` columns of the
    pseudo-inverse at once, which is why :func:`solve_tridiagonal` accepts a
    stack of right-hand sides. Cost is O(n^2) against O(n^3) for the eigenvector
    sum of :func:`quantum_response_jacobian_reference`.
    """
    n = diagonal.size
    if null_vector.size != n:
        raise ValueError(
            f"null_vector has length {null_vector.size}, expected {n}"
        )

    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    if n > 1:
        lower[1:] = off_diagonal
        upper[:-1] = off_diagonal
    diagonal_shifted = diagonal - float(energy)

    # Gauge: row `gauge` becomes e_gauge^T (right-hand side zero), and the
    # column is dropped because that unknown is held at zero.
    gauge = int(np.argmax(np.abs(null_vector)))
    diagonal_shifted[gauge] = 1.0
    lower[gauge] = 0.0
    upper[gauge] = 0.0
    if gauge > 0:
        upper[gauge - 1] = 0.0
    if gauge < n - 1:
        lower[gauge + 1] = 0.0

    # Column j of the projected identity is e_j - v_j v.
    rhs = np.eye(n, dtype=float) - np.outer(null_vector, null_vector)
    rhs[gauge, :] = 0.0

    gauge_solution = solve_tridiagonal(lower, diagonal_shifted, upper, rhs)
    return gauge_solution - np.outer(null_vector, null_vector @ gauge_solution)


def _require_eigenpair(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    vector: np.ndarray,
    energy: float,
    index: int,
) -> None:
    """Refuse a state that is not an eigenpair of the operator it is used with.

    The deflation assumes ``null_vector`` really does annihilate ``A - energy``.
    A state carrying a different band profile would otherwise produce a
    plausible-looking but meaningless Jacobian, so this fails loudly instead.
    """
    applied = diagonal * vector
    if off_diagonal.size:
        applied[:-1] += off_diagonal * vector[1:]
        applied[1:] += off_diagonal * vector[:-1]
    applied -= float(energy) * vector
    scale = max(1.0, abs(float(energy)))
    if float(np.max(np.abs(applied))) > 1.0e-8 * scale:
        raise ValueError(
            f"carried state {index} is not an eigenpair of the operator rebuilt "
            "from QuantumState.potential_eV, so its response is not defined"
        )


def _response_matrix(
    grid: Grid,
    state: QuantumState,
    *,
    temperature_K: float,
    valley_degeneracy: int,
    include_wavefunction_response: bool,
    resolvent: str,
) -> np.ndarray:
    """Assemble ``dn/dE_c`` in node space.

    ``resolvent`` selects how the orthogonal resolvent ``(A - E_i)^dagger`` is
    obtained: ``"deflated"`` is the production rank-1 deflation, ``"eigenvectors"``
    is the full eigenvector sum. Everything else - the subband-energy term, both
    psi-dependent terms, the node-space bookkeeping, the eigenpair guard - is
    shared, so the two paths can differ only in the resolvent itself.
    """
    psi = state.psi
    energies = state.energies_eV
    densities = state.densities_m2
    masses = state.masses_kg
    n_nodes, n_states = psi.shape

    kT_J = BOLTZMANN * float(temperature_K)
    w = cell_widths(grid)

    # A_i = g_v m* kT / (pi hbar^2)  [m^-2];  N_i = A_i ln(1 + exp(-E_i/kT))
    # with E_i in eV, so dN_i/dE_i = -(g_v m* q / (pi hbar^2)) f_i, in m^-2/eV.
    # This matches the operator, whose unknown is the band edge in eV.
    amplitudes = (
        float(valley_degeneracy) * masses * ELEMENTARY_CHARGE / (np.pi * REDUCED_PLANCK**2)
    )
    scaled = np.clip(energies * ELEMENTARY_CHARGE / kT_J, -700.0, 700.0)
    occupation = 1.0 / (1.0 + np.exp(scaled))
    beta = amplitudes * occupation  # [m^-2 / eV]

    exact = bool(include_wavefunction_response)
    interior = widths = diagonal = off_diagonal = v = None
    values = vectors = None
    if exact:
        if resolvent not in ("deflated", "eigenvectors"):
            raise ValueError(
                f"unknown resolvent {resolvent!r}; expected 'deflated' or 'eigenvectors'"
            )
        if state.potential_eV is None:
            raise ValueError(
                "the wavefunction response needs the band profile the state was "
                "solved for; build the state with quantum_state() (or set "
                "QuantumState.potential_eV) instead of assembling one by hand"
            )
        interior, widths, diagonal, off_diagonal = schrodinger_bands(
            grid, state.potential_eV
        )
        if widths.size != psi.shape[0] - 2:
            raise ValueError("state.psi is not compatible with this grid")
        v = psi[interior, :] * np.sqrt(widths)[:, None]  # Euclidean-orthonormal
        inv_m = 1.0 / grid.m_eff_kg[interior]
        if resolvent == "eigenvectors":
            values, vectors = np.linalg.eigh(_tridiagonal_matrix(diagonal, off_diagonal))

    jacobian = np.zeros((n_nodes, n_nodes), dtype=float)

    for index in range(n_states):
        u = psi[:, index] ** 2  # |psi_i|^2, units 1/m

        # -- subband-energy response ---------------------------------
        # <psi_i|dE_c|psi_i> shifts the Fermi-Dirac occupation: a rank-n update.
        jacobian -= beta[index] * np.outer(u, u * w)

        if not exact:
            continue

        # -- wavefunction and mass response ---------------------------
        v_index = v[:, index]
        _require_eigenpair(diagonal, off_diagonal, v_index, energies[index], index)
        if resolvent == "deflated":
            resolvent_matrix = deflated_resolvent(
                diagonal, off_diagonal, energies[index], v_index
            )
        else:
            # (A - E_i)^dagger = sum_{k != i} v_k v_k^T / (E_k - E_i)
            weights = np.zeros(values.size, dtype=float)
            other = np.arange(values.size) != index
            weights[other] = 1.0 / (values[other] - energies[index])
            resolvent_matrix = vectors @ (weights[:, None] * vectors.T)

        # Both psi-dependent terms take the same matrix form for either
        # resolvent, so the reference path is a genuine cross-check of the
        # deflation rather than a paraphrase of it. The 1/w_z is the node-space
        # measure: the row index is a density, the column index a point value.
        block = -2.0 * densities[index] * (np.outer(v_index, v_index) * resolvent_matrix)
        block += (
            2.0
            * densities[index]
            * masses[index]
            * np.outer(
                v_index**2, v_index * (resolvent_matrix @ (v_index * inv_m))
            )
        )
        jacobian[np.ix_(interior, interior)] += block / widths[:, None]

    return jacobian


def quantum_response_jacobian(
    grid: Grid,
    state: QuantumState,
    *,
    temperature_K: float,
    valley_degeneracy: int,
    include_wavefunction_response: bool = True,
) -> np.ndarray:
    """Exact ``dn/dE_c`` for the coupled Newton step, as a dense matrix.

    Three contributions, all of them exact:

    * **subband-energy response** - shifting ``E_c`` moves each subband by
      ``<psi_i|dE_c|psi_i>``, which changes the Fermi-Dirac occupation:
      ``-beta_i * outer(|psi_i|^2, |psi_i|^2 w)``. This carries the dominant
      physics (the well-depth / charge feedback) and is a rank-``n`` update.
    * **wavefunction response** - the perturbation also mixes the states,
      ``dpsi_i = sum_{j!=i} psi_j <psi_j|dE_c|psi_i> / (E_i - E_j)``, which
      contributes ``2 N_i psi_i dpsi_i``. The sum runs over the **whole**
      spectrum here, via the rank-1 deflation in :func:`deflated_resolvent`;
      nothing is truncated.
    * **mass response** - ``m*_i = 1 / sum_z |psi_i|^2 w / m`` is built from
      ``psi_i`` too, so it carries ``-2 N_i m*_i |psi_i|^2 sum_z psi_i w
      dpsi_i / m``. Dropping it leaves a residual inconsistency of ~1.3e-5
      relative that finite differences expose and that no amount of extra
      subbands removes.

    Set ``include_wavefunction_response=False`` for the cheaper rank-only
    quasi-Newton: only the first term survives. It is still contractive, but it
    converges more slowly - it is kept as a cross-check, not as the default.

    Both psi-dependent terms need the operator the state was solved for, which
    :class:`QuantumState` carries as ``potential_eV``.

    The optional density clamp is *not* differentiated: it is a discontinuous
    last-resort guard (off by default), and pinning the density on its bound
    leaves the residual non-smooth there.
    """
    return _response_matrix(
        grid,
        state,
        temperature_K=temperature_K,
        valley_degeneracy=valley_degeneracy,
        include_wavefunction_response=include_wavefunction_response,
        resolvent="deflated",
    )


def quantum_response_jacobian_reference(
    grid: Grid,
    state: QuantumState,
    *,
    temperature_K: float,
    valley_degeneracy: int,
) -> np.ndarray:
    """The same ``dn/dE_c`` by the O(N^3)-per-state eigenvector sum.

    Kept as the verification path, in the same spirit as the phi-form Poisson
    operator: it shares the whole assembly with
    :func:`quantum_response_jacobian` and differs only in how the orthogonal
    resolvent is obtained, so any disagreement between the two isolates the
    rank-1 deflation. The test suite asserts they agree to round-off, and that
    both match central differences.
    """
    return _response_matrix(
        grid,
        state,
        temperature_K=temperature_K,
        valley_degeneracy=valley_degeneracy,
        include_wavefunction_response=True,
        resolvent="eigenvectors",
    )


class SchrodingerPoissonSystem:
    """Residual and exact Jacobian of the coupled system, for a Newton solver.

    Callables are shaped for a generic Newton driver::

        R, state = system.residual(ec)
        J        = system.jacobian(ec, state)
    """

    def __init__(
        self,
        operator: PoissonOperator,
        *,
        n_subbands: int = 6,
        temperature_K: float = 300.0,
        valley_degeneracy: int = 1,
        density_clamp: bool = True,
        ns_upper_m2: float | None = None,
        include_wavefunction_response: bool = True,
    ) -> None:
        self.operator = operator
        self.n_subbands = int(n_subbands)
        self.temperature_K = float(temperature_K)
        self.valley_degeneracy = int(valley_degeneracy)
        self.density_clamp = bool(density_clamp)
        self.ns_upper_m2 = ns_upper_m2
        self.include_wavefunction_response = bool(include_wavefunction_response)
        self.evaluations = 0

    def state(self, ec_electrostatic: np.ndarray) -> QuantumState:
        return quantum_state(
            self.operator.grid,
            self.operator.potential(ec_electrostatic),
            n_subbands=self.n_subbands,
            temperature_K=self.temperature_K,
            valley_degeneracy=self.valley_degeneracy,
            ns_upper_m2=self.ns_upper_m2,
            density_clamp=self.density_clamp,
        )

    def residual(self, ec_eV: np.ndarray) -> tuple[np.ndarray, QuantumState]:
        self.evaluations += 1
        state = self.state(ec_eV)
        return self.operator.residual(ec_eV, state.n_volume_m3), state

    def jacobian(self, ec_eV: np.ndarray, state: QuantumState) -> np.ndarray:
        response = quantum_response_jacobian(
            self.operator.grid,
            state,
            temperature_K=self.temperature_K,
            valley_degeneracy=self.valley_degeneracy,
            include_wavefunction_response=self.include_wavefunction_response,
        )
        return self.operator.dense() - self.operator.charge_weight[:, None] * response


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


def fang_howard_profile(grid: Grid, ns_m2: float) -> np.ndarray:
    """Normalised Fang-Howard variational shape, used as the initial guess."""
    if ns_m2 <= 0.0:
        return np.zeros(grid.size, dtype=float)
    m = grid.m_channel_kg
    eps = grid.eps_channel
    b = (
        33.0
        * m
        * ELEMENTARY_CHARGE**2
        * ns_m2
        / (8.0 * eps * VACUUM_PERMITTIVITY * REDUCED_PLANCK**2)
    ) ** (1.0 / 3.0)

    relative = np.clip(grid.z_m - grid.d_m, 0.0, None)
    shape = 0.5 * b**3 * relative**2 * np.exp(-b * relative)

    w = cell_widths(grid)
    norm = float((shape * w).sum())
    if norm <= 0.0:
        return np.zeros(grid.size, dtype=float)
    return ns_m2 * shape / norm


def depletion_seed_ns(
    *,
    sigma_eff_C_m2: float,
    barrier_thickness_m: float,
    eps_r_barrier: float,
    phi_b_V: float,
    delta_ec_eV: float,
    m_eff_channel_kg: float,
    valley_degeneracy: int,
    gamma0_eV_m43: float,
    bisection_steps: int = 80,
) -> float:
    """Depletion-approximation estimate of ``n_s``, used only as a warm start.

    Solves the scalar charge-control relation by bisection::

        n_s = dos * ( W(n_s) - E_0(n_s) )
        W   = q * (sigma_eff - q*n_s) * d / (eps_b*eps0) - q*(phi_b - dE_C)
        E_0 = gamma_0 * n_s^(2/3)

    The right-hand side decreases monotonically in ``n_s``, so bisection always
    brackets a root. The result is the analytic Ambacher answer - a good place
    to start from, and emphatically *not* what the self-consistent solver
    returns.
    """
    upper = float(sigma_eff_C_m2) / ELEMENTARY_CHARGE
    if not np.isfinite(upper) or upper <= 0.0:
        return 0.0

    dos_per_m2_per_eV = (
        float(valley_degeneracy)
        * m_eff_channel_kg
        * ELEMENTARY_CHARGE
        / (np.pi * REDUCED_PLANCK**2)
    )
    capacitance = barrier_thickness_m / (eps_r_barrier * VACUUM_PERMITTIVITY)
    threshold = float(phi_b_V) - float(delta_ec_eV)

    def predicted(n: float) -> float:
        well = (float(sigma_eff_C_m2) - ELEMENTARY_CHARGE * n) * capacitance - threshold
        if well <= 0.0:
            return 0.0
        e0 = float(gamma0_eV_m43) * n ** (2.0 / 3.0)
        return dos_per_m2_per_eV * max(well - e0, 0.0)

    lo, hi = 0.0, upper
    for _ in range(max(1, int(bisection_steps))):
        mid = 0.5 * (lo + hi)
        if predicted(mid) > mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Anderson (Pulay) mixing, kept as the fallback iteration
# ---------------------------------------------------------------------------


class AndersonMixer:
    """Anderson acceleration for a fixed-point map ``x -> g(x)``."""

    def __init__(self, *, depth: int = 8, beta: float = 0.5, max_gamma: float = 1.0e6) -> None:
        self.depth = max(0, int(depth))
        self.beta = float(beta)
        self.max_gamma = float(max_gamma)
        self._xs: list[np.ndarray] = []
        self._rs: list[np.ndarray] = []
        self.fallbacks = 0

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def reset(self) -> None:
        self._xs.clear()
        self._rs.clear()

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        step = self.beta
        r = gx - x
        if self.depth <= 0:
            return x + step * r

        self._xs.append(np.array(x, dtype=float, copy=True))
        self._rs.append(np.array(r, dtype=float, copy=True))
        if len(self._xs) > self.depth + 1:
            self._xs.pop(0)
            self._rs.pop(0)

        m = len(self._rs) - 1
        if m <= 0:
            return x + step * r

        dX = np.column_stack([self._xs[i + 1] - self._xs[i] for i in range(m)])
        dR = np.column_stack([self._rs[i + 1] - self._rs[i] for i in range(m)])

        try:
            gamma, *_ = np.linalg.lstsq(dR, r, rcond=None)
        except np.linalg.LinAlgError:
            gamma = None

        if (
            gamma is None
            or not np.all(np.isfinite(gamma))
            or float(np.max(np.abs(gamma))) > self.max_gamma
        ):
            self.fallbacks += 1
            self.reset()
            return x + step * r

        return x + step * r - (dX + step * dR) @ gamma


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SPResult:
    """Everything the caller may want to inspect after a self-consistent solve."""

    converged: bool
    iterations: int
    residual_eV: float
    residual_history: list[float]
    ns_m2: float
    ns_cm2: float
    ns_upper_m2: float
    ns_per_subband_m2: np.ndarray
    energies_eV: np.ndarray
    masses_kg: np.ndarray
    ef_eV: float
    ec_well_bottom_eV: float
    ec_interface_eV: float
    ec_min_eV: float
    ec_profile_eV: np.ndarray
    n_profile_m3: np.ndarray
    psi: np.ndarray
    grid: Grid
    temperature_K: float
    delta_ec_eV: float
    phi_b_eV: float
    mixing_final: float
    anderson_fallbacks: int
    n_bound: int
    centroid_nm: float
    rms_width_nm: float
    seed_ns_m2: float = 0.0
    clamp_hits: int = 0
    algorithm: str = "anderson"
    damping_history: list[float] = field(default_factory=list)
    function_evaluations: int = 0
    solver_message: str = ""

    @property
    def subband_offsets_eV(self) -> np.ndarray:
        """Subband energies measured from the channel band minimum."""
        return self.energies_eV - self.ec_well_bottom_eV


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _finalize(
    *,
    grid: Grid,
    ec: np.ndarray,
    state: QuantumState,
    converged: bool,
    residual: float,
    history: list[float],
    seed_ns_m2: float,
    mixing_final: float,
    anderson_fallbacks: int,
    clamp_hits: int,
    algorithm: str,
    damping_history: Sequence[float],
    function_evaluations: int,
    solver_message: str,
    temperature_K: float,
    delta_ec_eV: float,
    phi_b_eV: float,
    sigma_eff_C_m2: float,
) -> SPResult:
    channel = grid.z_m >= grid.d_m
    ec_min = float(ec[channel].min()) if channel.any() else float(ec.min())

    w = cell_widths(grid)
    ground = np.abs(state.psi[:, 0]) ** 2
    weight = float((ground * w).sum())
    if weight > 0.0:
        centroid = float((grid.z_m * ground * w).sum() / weight)
        variance = float(((grid.z_m - centroid) ** 2 * ground * w).sum() / weight)
        rms_width_nm = float(np.sqrt(max(variance, 0.0)) * 1.0e9)
    else:  # pragma: no cover - only when there is no bound state at all
        centroid = grid.d_m
        rms_width_nm = 0.0

    return SPResult(
        converged=converged,
        iterations=len(history),
        residual_eV=residual,
        residual_history=list(history),
        ns_m2=state.ns_m2,
        ns_cm2=state.ns_m2 * 1.0e-4,
        ns_upper_m2=float(sigma_eff_C_m2) / ELEMENTARY_CHARGE,
        ns_per_subband_m2=state.densities_m2,
        energies_eV=state.energies_eV,
        masses_kg=state.masses_kg,
        ef_eV=-ec_min,
        ec_well_bottom_eV=ec_min,
        ec_interface_eV=ec_min,
        ec_min_eV=ec_min,
        ec_profile_eV=ec,
        n_profile_m3=state.n_volume_m3,
        psi=state.psi,
        grid=grid,
        temperature_K=float(temperature_K),
        delta_ec_eV=float(delta_ec_eV),
        phi_b_eV=float(phi_b_eV),
        mixing_final=float(mixing_final),
        anderson_fallbacks=int(anderson_fallbacks),
        n_bound=int(np.count_nonzero(state.energies_eV < 0.0)),
        centroid_nm=float((centroid - grid.d_m) * 1.0e9),
        rms_width_nm=rms_width_nm,
        seed_ns_m2=float(seed_ns_m2),
        clamp_hits=int(clamp_hits),
        algorithm=algorithm,
        damping_history=[float(x) for x in damping_history],
        function_evaluations=int(function_evaluations),
        solver_message=solver_message,
    )


def solve_schrodinger_poisson(
    *,
    grid: Grid,
    sigma_eff_C_m2: float,
    phi_b_V: float,
    delta_ec_eV: float,
    gate_bias_V: float = 0.0,
    temperature_K: float = 300.0,
    valley_degeneracy: int = 1,
    n_subbands: int = 12,
    max_iterations: int = 200,
    tolerance_eV: float = 1.0e-6,
    anderson_depth: int = 8,
    mixing: float = 0.5,
    adaptive_mixing: bool = True,
    mixing_min: float = 0.02,
    gamma0_eV_m43: float = 2.123e-12,
    density_clamp: bool = False,
    seed_ns_m2: float | None = None,
    algorithm: str = "anderson",
    nonlinear_solver: Any = None,
    newton_options: Mapping[str, Any] | None = None,
    include_wavefunction_response: bool = True,
) -> SPResult:
    """Solve the coupled Schroedinger-Poisson system self-consistently.

    ``algorithm`` selects the iteration:

    ``"newton"``
        Build the Jacobian of the coupled residual and solve it with the
        supplied ``nonlinear_solver`` (a module exposing ``solve(residual,
        jacobian, x0, options=...)``). Roughly an order of magnitude fewer
        iterations than mixing.
    ``"anderson"``
        Damped fixed point with Anderson acceleration. Needs no collaborator
        module, and is kept as the fallback and cross-check.

    The sheet density can optionally be clamped to ``[0, sigma_eff/e]``, which
    is the point beyond which the barrier field would reverse and no 2DEG could
    exist. It is **off by default**: pinning the density exactly on the bound
    creates a spurious fixed point that both iterations can get stuck on, so it
    is only worth enabling as a last-resort guard for a badly conditioned case.
    """
    algorithm = str(algorithm).lower()
    if algorithm not in ("newton", "anderson"):
        raise ValueError(f"unknown algorithm {algorithm!r}; expected 'newton' or 'anderson'")

    ns_upper = float(sigma_eff_C_m2) / ELEMENTARY_CHARGE
    operator = PoissonOperator.build(
        grid,
        sheet_charge_C_m2=float(sigma_eff_C_m2),
        gate_ec_eV=float(phi_b_V) - float(gate_bias_V),
        band_offset_eV=float(delta_ec_eV),
    )

    # -- warm start: depletion estimate, then one linear Poisson solve -----
    if seed_ns_m2 is None:
        ns_seed = depletion_seed_ns(
            sigma_eff_C_m2=float(sigma_eff_C_m2),
            barrier_thickness_m=grid.d_m,
            eps_r_barrier=grid.eps_barrier,
            phi_b_V=float(phi_b_V),
            delta_ec_eV=float(delta_ec_eV),
            m_eff_channel_kg=grid.m_channel_kg,
            valley_degeneracy=int(valley_degeneracy),
            gamma0_eV_m43=float(gamma0_eV_m43),
        )
    else:
        ns_seed = max(0.0, min(float(seed_ns_m2), ns_upper))

    # The iteration unknown is the *electrostatic* part of the band edge. The
    # heterojunction offset is added by operator.potential() wherever the
    # Schroedinger equation needs the real profile.
    x = operator.solve(fang_howard_profile(grid, ns_seed))

    if algorithm == "newton":
        if nonlinear_solver is None:
            raise RuntimeError(
                "algorithm='newton' needs a nonlinear solver module; bind the "
                "Deck's 'nonlinear' slot to e.g. 'nonlinear.newton'"
            )
        system = SchrodingerPoissonSystem(
            operator,
            n_subbands=n_subbands,
            temperature_K=temperature_K,
            valley_degeneracy=valley_degeneracy,
            density_clamp=density_clamp,
            ns_upper_m2=ns_upper,
            include_wavefunction_response=include_wavefunction_response,
        )
        options = dict(newton_options or {})
        options.setdefault("tolerance", float(tolerance_eV))
        options.setdefault("max_iterations", max(1, int(max_iterations)))
        outcome = nonlinear_solver.solve(
            system.residual, system.jacobian, x, options=options
        )
        x = np.asarray(outcome.x, dtype=float)
        state = system.state(x)
        residual = float(getattr(outcome, "residual_norm", float("inf")))
        history = [float(v) for v in getattr(outcome, "residual_history", [])]
        damping = [float(v) for v in getattr(outcome, "damping_history", [])]
        converged = bool(getattr(outcome, "converged", False))
        message = str(getattr(outcome, "message", ""))

        return _finalize(
            grid=grid,
            ec=operator.potential(x),
            state=state,
            converged=converged,
            residual=residual,
            history=history,
            seed_ns_m2=ns_seed,
            mixing_final=0.0,
            anderson_fallbacks=0,
            clamp_hits=0,
            algorithm="newton",
            damping_history=damping,
            function_evaluations=int(getattr(system, "evaluations", 0)),
            solver_message=message,
            temperature_K=temperature_K,
            delta_ec_eV=delta_ec_eV,
            phi_b_eV=phi_b_V,
            sigma_eff_C_m2=sigma_eff_C_m2,
        )

    # -- Anderson-accelerated fixed point ---------------------------------
    def quantum(x_in: np.ndarray) -> QuantumState:
        return quantum_state(
            grid,
            operator.potential(x_in),
            n_subbands=n_subbands,
            temperature_K=temperature_K,
            valley_degeneracy=valley_degeneracy,
            ns_upper_m2=ns_upper,
            density_clamp=density_clamp,
        )

    beta = float(mixing)
    mixer = AndersonMixer(depth=anderson_depth, beta=beta)
    history: list[float] = []
    converged = False
    residual = float("inf")
    best = float("inf")
    stall = 0
    clamp_hits = 0

    for _iteration in range(1, max(1, int(max_iterations)) + 1):
        state = quantum(x)
        if state.clamped:
            clamp_hits += 1
        x_out = operator.solve(state.n_volume_m3)
        residual = float(np.max(np.abs(x_out - x)))
        history.append(residual)

        if residual <= tolerance_eV:
            x = x_out
            converged = True
            break

        if adaptive_mixing and residual > best:
            stall += 1
            if stall >= 3:
                beta = max(beta * 0.5, float(mixing_min))
                mixer.set_beta(beta)
                stall = 0
        else:
            stall = 0
        best = min(best, residual)

        x = mixer.step(x, x_out)

    state = quantum(x)
    return _finalize(
        grid=grid,
        ec=operator.potential(x),
        state=state,
        converged=converged,
        residual=residual,
        history=history,
        seed_ns_m2=ns_seed,
        mixing_final=beta,
        anderson_fallbacks=mixer.fallbacks,
        clamp_hits=clamp_hits,
        algorithm="anderson",
        damping_history=[],
        function_evaluations=len(history) + 1,
        solver_message="anderson-accelerated fixed point",
        temperature_K=temperature_K,
        delta_ec_eV=delta_ec_eV,
        phi_b_eV=phi_b_V,
        sigma_eff_C_m2=sigma_eff_C_m2,
    )
