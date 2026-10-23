"""Finite-volume assembly of the coupled drift-diffusion system.

Unknowns, ordering
------------------
Two unknowns per node - the electrostatic potential ``psi`` [V] and the electron
quasi-Fermi potential ``phi_n`` [V] - stored as an ``(nz, nx, 2)`` array. A
vertical row ``j`` therefore flattens to ``2*nx`` unknowns in the interleaved
order ``psi_0, phi_n_0, psi_1, phi_n_1, ...``, which is exactly the block size
of the Jacobian: node row ``j`` couples only to rows ``j-1`` and ``j+1`` on a
tensor-product grid, so the Jacobian is **block tridiagonal** with ``m = 2*nx``
dense blocks. That is the structure ``linear.blocktridiag`` factorises.

Discretisation
--------------
Finite volume with a five-point stencil.

* **Poisson** - flux form ``sum_faces eps_face A/h (psi_i - psi_neighbour)``
  with the face permittivity taken as the harmonic (series) mean of the two
  node values, which is the correct treatment of a permittivity step that sits
  *between* nodes. The space charge is ``q (N_net - n) V_cell`` plus the
  polarization sheet on the interface row.
* **Continuity** - ``sum_faces J_n . A = 0`` with the Scharfetter-Gummel flux
  from :mod:`physics`, so the scheme is monotone at any mesh Peclet number and
  vanishes identically at equilibrium across a heterointerface.

Both blocks are assembled with analytic derivatives, so the Jacobian is the
exact derivative of the assembled residual and can be checked against central
differences - which is how this file is verified.

Row scaling
-----------
The two blocks live in different units (charge per unit width, and current per
unit width) and their magnitudes differ by many orders. Each non-Dirichlet row
is therefore divided by its own Jacobian diagonal magnitude, frozen at the
initial state - a diagonal row equilibration. It changes no solution (it is a
left multiplication by a positive diagonal) and it makes the residual norm a
meaningful convergence measure for both blocks at once.

Honest limits
-------------
The mesh arrives with layer boundaries **on nodes**, not on faces, so the
permittivity and band-offset steps are smeared over one cell there. That is the
first-order treatment the Schroedinger-Poisson solver deliberately replaced with
a face-based grid; here the mesh is consumed as given (design note section 3), so
the interface is a documented first-order point rather than a hidden one. Away
from interfaces the scheme is second order, which the manufactured-solution test
asserts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from . import physics as P

__all__ = [
    "Grid2D",
    "Boundary",
    "DriftDiffusionSystem",
    "build_grid2d",
    "control_volumes",
]

#: Reference density for the polarization/charge bookkeeping diagnostics.
CM3_TO_M3 = 1.0e6


def control_volumes(axis_m: np.ndarray) -> np.ndarray:
    """Midpoint control-volume widths: half a cell at each end of the axis."""
    widths = np.empty(axis_m.size, dtype=float)
    spacing = np.diff(axis_m)
    widths[0] = 0.5 * spacing[0]
    widths[-1] = 0.5 * spacing[-1]
    if axis_m.size > 2:
        widths[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return widths


def harmonic_face(values: np.ndarray, axis: int) -> np.ndarray:
    """Series (harmonic) mean of adjacent nodes along ``axis``."""
    left = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
    right = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
    return 2.0 * left * right / (left + right)


@dataclass(frozen=True)
class Grid2D:
    """Static geometry, materials and boundary masks of the device."""

    nx: int
    nz: int
    x_m: np.ndarray  # (nx,)
    z_m: np.ndarray  # (nz,)
    dx_m: np.ndarray  # (nx-1,)
    dz_m: np.ndarray  # (nz-1,)
    wx_m: np.ndarray  # (nx,) control-volume widths
    wz_m: np.ndarray  # (nz,)
    eps_r: np.ndarray  # (nz, nx)
    mu_m2_Vs: np.ndarray  # (nz, nx)
    n_net_m3: np.ndarray  # (nz, nx), donors minus acceptors
    ec0_eV: np.ndarray  # (nz, nx), conduction-band edge at psi = 0
    band_offset_eV: np.ndarray  # (nz, nx), band edge as the flux must see it
    nc_eff_m3: np.ndarray  # (nz, nx)
    layer_of_row: np.ndarray  # (nz,) layer index per vertical node
    sheet_row: int | None
    sheet_C_m2: float
    ohmic_mask: np.ndarray  # (nz, nx) bool - Dirichlet on both unknowns
    gate_mask: np.ndarray  # (nz, nx) bool - Dirichlet on psi (Schottky)
    ground_mask: np.ndarray  # (nz, nx) bool - Dirichlet on psi (backside)
    surface_mask: np.ndarray  # (nz, nx) bool - Dirichlet on psi (pinned surface)
    blocked_mask: np.ndarray  # (nz, nx) bool - carriers pinned to zero
    equilibrium_mask: np.ndarray  # (nz, nx) bool - continuity closed by local equilibrium
    gate_reference_V: float  # psi at the gate for vgs = 0
    source_reference_V: np.ndarray  # (nz,) equilibrium psi on the left contact
    drain_reference_V: np.ndarray  # (nz,)
    contact_density_m3: np.ndarray  # (nz,) equilibrium density at a contact

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def unknowns(self) -> int:
        return self.nz * self.nx * 2


@dataclass(frozen=True)
class Boundary:
    """Dirichlet data for one bias point, in the ``(nz, nx)`` node layout."""

    psi_mask: np.ndarray
    psi_value: np.ndarray
    density_mask: np.ndarray
    density_value: np.ndarray

    def unknown_mask(self) -> np.ndarray:
        """``(nz, nx, 2)`` mask of the constrained unknowns."""
        mask = np.zeros(self.psi_mask.shape + (2,), dtype=bool)
        mask[:, :, 0] = self.psi_mask
        mask[:, :, 1] = self.density_mask
        return mask

    def unknown_values(self) -> np.ndarray:
        values = np.zeros(self.psi_mask.shape + (2,), dtype=float)
        values[:, :, 0] = self.psi_value
        values[:, :, 1] = self.density_value
        return values


class DriftDiffusionSystem:
    """Residual, block Jacobian and terminal currents of the DD system."""

    def __init__(
        self,
        grid: Grid2D,
        *,
        temperature_K: float,
        boundary: Boundary,
        mu0_m2_Vs: float,
        statistics: str = "boltzmann",
        vsat_m_s: float | None = None,
        beta_field: float = 1.0,
        density_scale_m3: float = 1.0e23,
        column_floor: float = 1.0e-30,
    ) -> None:
        self.grid = grid
        self.temperature_K = float(temperature_K)
        self.kT_eV = P.BOLTZMANN * self.temperature_K / P.ELEMENTARY_CHARGE
        self.statistics = statistics
        self.mu0_m2_Vs = P.constant_mobility(mu0_m2_Vs)
        #: Caughey-Thomas saturation velocity (m/s), or None for a constant
        #: mobility. The face field is what enters, so the current saturates
        #: where it should instead of growing without bound.
        self.vsat_m_s = None if vsat_m_s is None else float(vsat_m_s)
        self.beta_field = float(beta_field)
        #: The second unknown is ``n / density_scale_m3``: a number of order one
        #: in the active region, which keeps the block factorisation well scaled
        #: across a density range of thirty decades.
        self.density_scale = float(density_scale_m3)
        #: Floor on the unknown magnitude used by the column scaling.
        self.column_floor = float(column_floor)
        #: Reference density for the fixed non-dimensionalisation below.
        self.reference_density = 1.0e23  # 1e17 cm^-3

        self.boundary = boundary
        self.evaluations = 0
        #: Ohmic reservoirs: the equilibrium potential profile on each contact
        #: column. Filled by :meth:`adopt_equilibrium`; pinning the contact
        #: density to the *matching* potential is what keeps the neighbouring
        #: faces at zero current instead of inventing a contact-side flux.
        self.source_reference = np.zeros(grid.nz, dtype=float)
        self.drain_reference = np.zeros(grid.nz, dtype=float)
        #: Residual history of the nonlinear-Poisson start, for diagnostics.
        self.poisson_history: list[float] = []

        eps = grid.eps_r
        self.eps_face_x = harmonic_face(eps, axis=1)  # (nz, nx-1)
        self.eps_face_z = harmonic_face(eps, axis=0)  # (nz-1, nx)

        # Poisson flux coefficients: eps_face * eps0 * A / h, per unit width.
        self.cx = (
            self.eps_face_x * P.VACUUM_PERMITTIVITY * grid.wz_m[:, None]
            / grid.dx_m[None, :]
        )  # (nz, nx-1)
        self.cz = (
            self.eps_face_z * P.VACUUM_PERMITTIVITY * grid.wx_m[None, :]
            / grid.dz_m[:, None]
        )  # (nz-1, nx)
        self.volume = grid.wx_m[None, :] * grid.wz_m[:, None]  # (nz, nx)
        self.sheet_source = np.zeros((grid.nz, grid.nx), dtype=float)
        if grid.sheet_row is not None and grid.sheet_C_m2:
            self.sheet_source[grid.sheet_row, :] = grid.sheet_C_m2 * grid.wx_m

        # Fixed (solution-independent) equation scaling, chosen so that each row
        # reads as a *physical error*:
        #
        #   Poisson row / (local capacitance)      -> volts (a potential error)
        #   continuity row / (reference current)   -> a relative current error
        #
        # Scaling rows by their own Jacobian diagonal - the obvious thing - makes
        # the norm in a wide-gap or depleted row dominated by cancellation noise:
        # those rows are weak, so they get a huge factor, and the Newton loop then
        # chases a floor it can never lower while the line search refuses every
        # step. Physical reference scales have no such failure mode: the two
        # blocks end up on a common O(1) footing and one tolerance means the same
        # thing for the charge balance and for the current continuity.
        capacitance = np.zeros((grid.nz, grid.nx), dtype=float)
        if grid.nx > 1:
            capacitance[:, :-1] += self.cx
            capacitance[:, 1:] += self.cx
        if grid.nz > 1:
            capacitance[:-1, :] += self.cz
            capacitance[1:, :] += self.cz
        self.poisson_scale = 1.0 / capacitance

        # A device-level current per metre of width: q * n_s * mu * (V/L) at the
        # polarization sheet density. It only has to be the right order, and it
        # makes the continuity residual a relative current error.
        sheet = abs(float(grid.sheet_C_m2)) or P.ELEMENTARY_CHARGE * 1.0e16
        length = max(float(grid.x_m[-1] - grid.x_m[0]), 1.0e-9)
        self.reference_current = (
            sheet * self.mu0_m2_Vs * 1.0 / length
        )  # A/m
        self.continuity_scale = 1.0 / self.reference_current

        self.row_scale = np.ones((grid.nz, grid.nx, 2), dtype=float)
        self.row_scale[:, :, 0] = self.poisson_scale
        self.row_scale[:, :, 1] = self.continuity_scale
        #: Rows whose continuity equation is closed by local equilibrium instead.
        self.equilibrium = np.array(grid.equilibrium_mask, dtype=bool)
        if self.equilibrium.any():
            self.row_scale[self.equilibrium, 1] = 1.0 / self.reference_density

        # Carrier-free nodes (a neutral buffer, or a wide-gap row closed by local
        # equilibrium) are walls for the continuity equation: the
        # Scharfetter-Gummel flux has no way of knowing that a neighbour holds no
        # carriers, so it happily computes a diffusion current into a zero-density
        # node, and the channel row next to it sees an imbalance of thousands of
        # amps per metre before the solver has taken a single step. Setting those
        # faces to zero is both the physical statement ("no carriers flow into
        # the insulator") and what makes the assembled system start near its own
        # solution.
        inactive = self.equilibrium | np.array(grid.blocked_mask, dtype=bool)
        face_open_x = ~(inactive[:, :-1] | inactive[:, 1:])
        face_open_z = ~(inactive[:-1, :] | inactive[1:, :]) if grid.nz > 1 else None
        self.face_open_x = face_open_x
        self.face_open_z = face_open_z
        #: Frozen unknown (column) scaling, filled by :meth:`equilibrate`. The
        #: density spans twenty-odd decades across the stack, so rows alone are
        #: not enough: without column scaling the factorisation sees entries
        #: whose ratio is that whole range.
        self.column_scale = np.ones((grid.nz, grid.nx, 2), dtype=float)
        self._dirichlet = boundary.unknown_mask()
        self._targets = boundary.unknown_values()
        # Pinned rows carry the constraint itself, not a physical equation.
        self.row_scale[self._dirichlet] = 1.0

    # -- geometry helpers -------------------------------------------------
    def _row_scale_flat(self) -> np.ndarray:
        return self.row_scale.reshape(self.grid.nz, 2 * self.grid.nx)

    # -- residual ---------------------------------------------------------
    def residual(
        self, u: np.ndarray, ptc: tuple[np.ndarray, float] | None = None
    ) -> tuple[np.ndarray, Any]:
        """Scaled residual, shaped like ``u`` (``(nz, nx, 2)``).

        ``ptc`` optionally carries ``(n_previous, tau_s)`` and adds the physical
        pseudo-transient term ``(n - n_prev) V / tau`` to the continuity rows.
        The *steady* residual is returned alongside it in ``aux['physical']`` so
        the Newton loop can monitor the equations it actually wants to solve.
        """
        self.evaluations += 1
        grid = self.grid
        nx, nz = grid.nx, grid.nz
        u = np.asarray(u, dtype=float).reshape(nz, nx, 2)
        psi = u[:, :, 0]
        density = u[:, :, 1] * self.density_scale

        # -- Poisson ------------------------------------------------------
        poisson = np.zeros((nz, nx), dtype=float)
        flux_x = self.cx * (psi[:, :-1] - psi[:, 1:])
        poisson[:, :-1] += flux_x
        poisson[:, 1:] -= flux_x
        if nz > 1:
            flux_z = self.cz * (psi[:-1, :] - psi[1:, :])
            poisson[:-1, :] += flux_z
            poisson[1:, :] -= flux_z
        poisson -= P.ELEMENTARY_CHARGE * (grid.n_net_m3 - density) * self.volume
        poisson -= self.sheet_source

        # -- continuity ---------------------------------------------------
        continuity = np.zeros((nz, nx), dtype=float)
        band = P.band_edge_eV(psi, grid.band_offset_eV)
        current_x, _djl_p, _djl_f, djr_p, djr_f = P.sg_face_current(
            band_left=band[:, :-1],
            band_right=band[:, 1:],
            n_left=density[:, :-1],
            n_right=density[:, 1:],
            mu_m2_Vs=self.mu0_m2_Vs,
            kT_eV=self.kT_eV,
            length_m=grid.dx_m[None, :],
            statistics=self.statistics,
            vsat_m_s=self.vsat_m_s,
            beta_field=self.beta_field,
        )
        area_x = grid.wz_m[:, None]  # per unit device width
        current_x = current_x * self.face_open_x
        continuity[:, :-1] += current_x * area_x
        continuity[:, 1:] -= current_x * area_x
        if nz > 1:
            current_z, _djl_p_z, _djl_f_z, djr_p_z, djr_f_z = P.sg_face_current(
                band_left=band[:-1, :],
                band_right=band[1:, :],
                n_left=density[:-1, :],
                n_right=density[1:, :],
                mu_m2_Vs=self.mu0_m2_Vs,
                kT_eV=self.kT_eV,
                length_m=grid.dz_m[:, None],
                statistics=self.statistics,
            vsat_m_s=self.vsat_m_s,
            beta_field=self.beta_field,
            )
            area_z = grid.wx_m[None, :]
            current_z = current_z * self.face_open_z
            continuity[:-1, :] += current_z * area_z
            continuity[1:, :] -= current_z * area_z

        if self.equilibrium.any():
            # Local-equilibrium closure on the current-free rows: the continuity
            # residual becomes n - n_eq(psi), and the row scaling below turns it
            # into a relative density error.
            n_equilibrium = grid.nc_eff_m3 * np.exp(
                np.clip(psi / self.kT_eV, -P.EXPONENT_CAP, P.EXPONENT_CAP)
            )
            continuity = np.where(
                self.equilibrium, density - n_equilibrium, continuity
            )

        physical = np.stack([poisson, continuity], axis=2) * self.row_scale
        if ptc is None:
            total = physical
        else:
            n_previous, tau_s = ptc
            pseudo = continuity + (density - np.asarray(n_previous, dtype=float)) * (
                self.volume / float(tau_s)
            )
            pseudo = np.where(self.equilibrium, continuity, pseudo)
            total = np.stack([poisson, pseudo], axis=2) * self.row_scale

        total = np.where(self._dirichlet, u - self._targets, total)
        physical = np.where(self._dirichlet, u - self._targets, physical)
        return total, {"u": u, "density": density, "physical": physical}

    # -- Jacobian ---------------------------------------------------------
    def jacobian(
        self,
        u: np.ndarray,
        aux: Any = None,
        *,
        ptc: tuple[np.ndarray, float] | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Block bands ``(lower, diagonal, upper)`` of the Jacobian.

        ``ptc`` adds ``V/tau * dn/du`` to the continuity diagonal blocks, i.e.
        the derivative of the pseudo-transient term in :meth:`residual`. It
        changes the *step*, never the steady equations being solved.
        """
        grid = self.grid
        nx, nz = grid.nx, grid.nz
        u = np.asarray(u, dtype=float).reshape(nz, nx, 2)
        psi = u[:, :, 0]
        # The second unknown is the *density*, in units of density_scale, and it
        # has to be read the same way here as in :meth:`residual`. It once was
        # not: this line went through ``density_partials``, which reads its second
        # argument as a quasi-Fermi potential, so the Jacobian evaluated the flux
        # at a density seventeen orders of magnitude away from the residual's.
        # The flux's density partials are independent of the density and stayed
        # exact, which hid it; only the potential partials, which are proportional
        # to the density, were wrong - so the continuity rows lost their entire
        # coupling to psi and the coupled system went nearly singular in exactly
        # the direction the Newton step needed.
        density = np.asarray(u[:, :, 1], dtype=float) * self.density_scale
        band = P.band_edge_eV(psi, grid.band_offset_eV)

        size = 2 * nx
        diagonal = np.zeros((nz, size, size), dtype=float)
        upper = np.zeros((max(nz - 1, 0), size, size), dtype=float)
        lower = np.zeros((max(nz - 1, 0), size, size), dtype=float)

        # ---- lateral (same row) couplings -------------------------------
        # With the density as its own unknown, the Poisson row couples to the
        # local potential (Laplacian) and to the local density; both coupling
        # constants are pure geometry, which is what makes this block linear.
        self_psi_psi = np.zeros((nz, nx), dtype=float)
        self_psi_psi[:, :-1] += self.cx
        self_psi_psi[:, 1:] += self.cx
        if nz > 1:
            self_psi_psi[:-1, :] += self.cz
            self_psi_psi[1:, :] += self.cz
        self_psi_density = P.ELEMENTARY_CHARGE * self.density_scale * self.volume

        flux_x, djl_psi, djl_n, djr_psi, djr_n = P.sg_face_current(
            band_left=band[:, :-1],
            band_right=band[:, 1:],
            n_left=density[:, :-1],
            n_right=density[:, 1:],
            mu_m2_Vs=self.mu0_m2_Vs,
            kT_eV=self.kT_eV,
            length_m=grid.dx_m[None, :],
            statistics=self.statistics,
            vsat_m_s=self.vsat_m_s,
            beta_field=self.beta_field,
        )
        djl_d = djl_n * self.density_scale
        djr_d = djr_n * self.density_scale
        open_x = self.face_open_x
        djl_psi = djl_psi * open_x
        djl_n = djl_n * open_x
        djr_psi = djr_psi * open_x
        djr_n = djr_n * open_x
        djl_d = djl_d * open_x
        djr_d = djr_d * open_x
        area_x = grid.wz_m[:, None]
        # continuity self blocks (node i as the left node of face i, and as the
        # right node of face i-1)
        self_cont_psi = np.zeros((nz, nx), dtype=float)
        self_cont_density = np.zeros((nz, nx), dtype=float)
        self_cont_psi[:, :-1] += area_x * djl_psi
        self_cont_density[:, :-1] += area_x * djl_d
        self_cont_psi[:, 1:] -= area_x * djr_psi
        self_cont_density[:, 1:] -= area_x * djr_d

        # ---- vertical couplings ------------------------------------------
        # Kept in separate arrays: the Poisson row couples to the neighbour's
        # psi only, the continuity row couples to both of its unknowns.
        up_poisson_psi = np.zeros((nz, nx), dtype=float)
        down_poisson_psi = np.zeros((nz, nx), dtype=float)
        up_cont_psi = np.zeros((nz, nx), dtype=float)
        up_cont_density = np.zeros((nz, nx), dtype=float)
        down_cont_psi = np.zeros((nz, nx), dtype=float)
        down_cont_density = np.zeros((nz, nx), dtype=float)
        local_from_up_psi = np.zeros((nz, nx), dtype=float)
        local_from_up_density = np.zeros((nz, nx), dtype=float)
        local_from_down_psi = np.zeros((nz, nx), dtype=float)
        local_from_down_density = np.zeros((nz, nx), dtype=float)

        if nz > 1:
            _jz, djl_psi_z, djl_n_z, djr_psi_z, djr_n_z = P.sg_face_current(
                band_left=band[:-1, :],
                band_right=band[1:, :],
                n_left=density[:-1, :],
                n_right=density[1:, :],
                mu_m2_Vs=self.mu0_m2_Vs,
                kT_eV=self.kT_eV,
                length_m=grid.dz_m[:, None],
                statistics=self.statistics,
            vsat_m_s=self.vsat_m_s,
            beta_field=self.beta_field,
            )
            djl_d_z = djl_n_z * self.density_scale * self.face_open_z
            djr_d_z = djr_n_z * self.density_scale * self.face_open_z
            djl_psi_z = djl_psi_z * self.face_open_z
            djl_n_z = djl_n_z * self.face_open_z
            djr_psi_z = djr_psi_z * self.face_open_z
            djr_n_z = djr_n_z * self.face_open_z
            area_z = grid.wx_m[None, :]
            # face (j -> j+1): residual row j sees +A*J, row j+1 sees -A*J
            local_from_up_psi[:-1, :] += area_z * djl_psi_z
            local_from_up_density[:-1, :] += area_z * djl_d_z
            up_cont_psi[:-1, :] += area_z * djr_psi_z
            up_cont_density[:-1, :] += area_z * djr_d_z

            local_from_down_psi[1:, :] -= area_z * djr_psi_z
            local_from_down_density[1:, :] -= area_z * djr_d_z
            down_cont_psi[1:, :] -= area_z * djl_psi_z
            down_cont_density[1:, :] -= area_z * djl_d_z

            # Poisson couplings along z
            up_poisson_psi[:-1, :] -= self.cz
            down_poisson_psi[1:, :] -= self.cz

        self_cont_psi += local_from_up_psi + local_from_down_psi
        self_cont_density += local_from_up_density + local_from_down_density

        if ptc is not None:
            _n_previous, tau_s = ptc
            self_cont_density = self_cont_density + (
                self.density_scale * self.volume / float(tau_s)
            )

        for row in range(nz):
            diagonal[row] = self._row_block(
                nx,
                self_psi_psi[row],
                self_psi_density[row],
                self_cont_psi[row],
                self_cont_density[row],
                self.cx[row] if nx > 1 else None,
                djr_psi[row] * area_x[row, 0] if nx > 1 else None,
                djr_d[row] * area_x[row, 0] if nx > 1 else None,
                -djl_psi[row] * area_x[row, 0] if nx > 1 else None,
                -djl_d[row] * area_x[row, 0] if nx > 1 else None,
            )
            if row < nz - 1:
                upper[row] = self._diagonal_block(
                    nx, up_poisson_psi[row], up_cont_psi[row], up_cont_density[row]
                )
            if row > 0:
                lower[row - 1] = self._diagonal_block(
                    nx, down_poisson_psi[row], down_cont_psi[row], down_cont_density[row]
                )

        if self.equilibrium.any():
            # Replace the continuity row by the derivative of the closure: no
            # coupling to the neighbours (the closure is local), d/d(d) =
            # density_scale and d/d(psi) = -(q/kT) n_eq.
            n_equilibrium = grid.nc_eff_m3 * np.exp(
                np.clip(psi / self.kT_eV, -P.EXPONENT_CAP, P.EXPONENT_CAP)
            )
            slope = (P.ELEMENTARY_CHARGE / self.kT_eV) * n_equilibrium
            for row in range(nz):
                for i in np.flatnonzero(self.equilibrium[row]):
                    index = 2 * i + 1
                    diagonal[row][index, :] = 0.0
                    if row < nz - 1:
                        upper[row][index, :] = 0.0
                    if row > 0:
                        lower[row - 1][index, :] = 0.0
                    diagonal[row][index, 2 * i] = -slope[row, i]
                    diagonal[row][index, 2 * i + 1] = self.density_scale

        flat_scale = self._row_scale_flat()
        diagonal *= flat_scale[:, :, None]
        if nz > 1:
            upper *= flat_scale[:-1, :, None]
            lower *= flat_scale[1:, :, None]

        self._apply_dirichlet_rows(diagonal, upper, lower)
        return lower, diagonal, upper

    # -- block construction helpers ---------------------------------------
    @staticmethod
    def _row_block(
        nx: int,
        psi_psi: np.ndarray,
        psi_density: np.ndarray,
        cont_psi: np.ndarray,
        cont_density: np.ndarray,
        poisson_cx: np.ndarray | None,
        cont_right_psi: np.ndarray | None,
        cont_right_density: np.ndarray | None,
        cont_left_psi: np.ndarray | None,
        cont_left_density: np.ndarray | None,
    ) -> np.ndarray:
        """Interleaved ``2nx x 2nx`` block with 2x2 entries per node pair."""
        blocks = np.zeros((nx, nx, 2, 2), dtype=float)
        idx = np.arange(nx)
        blocks[idx, idx, 0, 0] = psi_psi
        blocks[idx, idx, 0, 1] = psi_density
        blocks[idx, idx, 1, 0] = cont_psi
        blocks[idx, idx, 1, 1] = cont_density
        if nx > 1:
            left = idx[:-1]
            right = idx[1:]
            # Poisson: neighbour psi with -c ; continuity: the face partials
            blocks[left, right, 0, 0] = -poisson_cx
            blocks[left, right, 1, 0] = cont_right_psi
            blocks[left, right, 1, 1] = cont_right_density
            blocks[right, left, 0, 0] = -poisson_cx
            blocks[right, left, 1, 0] = cont_left_psi
            blocks[right, left, 1, 1] = cont_left_density
        return blocks.transpose(0, 2, 1, 3).reshape(2 * nx, 2 * nx)

    @staticmethod
    def _diagonal_block(
        nx: int,
        poisson_psi: np.ndarray,
        cont_psi: np.ndarray,
        cont_density: np.ndarray,
    ) -> np.ndarray:
        """Interleaved ``2nx x 2nx`` block that is diagonal in the node index."""
        blocks = np.zeros((nx, nx, 2, 2), dtype=float)
        idx = np.arange(nx)
        blocks[idx, idx, 0, 0] = poisson_psi
        blocks[idx, idx, 1, 0] = cont_psi
        blocks[idx, idx, 1, 1] = cont_density
        return blocks.transpose(0, 2, 1, 3).reshape(2 * nx, 2 * nx)

    def _apply_dirichlet_rows(
        self,
        diagonal: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
    ) -> None:
        mask = self._dirichlet.reshape(self.grid.nz, 2 * self.grid.nx)
        positions = np.arange(2 * self.grid.nx)
        for row in range(self.grid.nz):
            constrained = mask[row]
            if not constrained.any():
                continue
            diagonal[row][constrained, :] = 0.0
            diagonal[row][constrained, positions[constrained]] = 1.0
            if row < self.grid.nz - 1:
                upper[row][constrained, :] = 0.0
            if row > 0:
                lower[row - 1][constrained, :] = 0.0

    # -- equilibrium start -------------------------------------------------
    def poisson_start(
        self,
        *,
        linear: Any,
        max_iterations: int = 60,
        tolerance: float = 1.0e-10,
        statistics: str = "boltzmann",
    ) -> np.ndarray:
        """Nonlinear-Poisson equilibrium start, one unknown per node.

        The coupled system is far easier to converge from a charge-neutral
        starting point than from flat bands, and this is the cheapest honest way
        to get one: solve Poisson alone with the density as an explicit function
        of the potential, ``n = N_C_eff exp(q psi/kT)``.

        It must be a *nonlinear* solve - a naive Gummel loop that pins the
        density and re-evaluates it diverges, because the well-depth feedback
        has a gain of order minus thirty. Keeping ``dn/dpsi`` in the Jacobian is
        what makes the iteration contract, and that is exactly what this method
        does; the degeneracy that made the quasi-Fermi formulation unusable in
        the barrier is harmless here, because the density is not an unknown.

        The resulting potential is returned as an ``(nz, nx)`` array.
        """
        if statistics != "boltzmann":
            raise NotImplementedError(
                f"statistics={statistics!r} is not implemented in v0.1.0"
            )
        grid = self.grid
        nz, nx = grid.nz, grid.nx
        mask = self.boundary.psi_mask
        targets = self.boundary.psi_value
        psi = np.where(mask, targets, 0.0).astype(float)
        residual_history: list[float] = []

        for _ in range(max(1, int(max_iterations))):
            exponent = np.clip(psi / self.kT_eV, -P.EXPONENT_CAP, P.EXPONENT_CAP)
            # The carrier-free rows must be excluded here as well, or the start
            # solves a different charge balance than the coupled system does (the
            # buffer would come out charged, and the first thing the coupled
            # solve does is undo that difference).
            density = grid.nc_eff_m3 * np.exp(exponent)
            density = np.where(grid.blocked_mask, 0.0, density)

            residual = np.zeros((nz, nx), dtype=float)
            flux_x = self.cx * (psi[:, :-1] - psi[:, 1:])
            residual[:, :-1] += flux_x
            residual[:, 1:] -= flux_x
            if nz > 1:
                flux_z = self.cz * (psi[:-1, :] - psi[1:, :])
                residual[:-1, :] += flux_z
                residual[1:, :] -= flux_z
            residual -= (
                P.ELEMENTARY_CHARGE * (grid.n_net_m3 - density) * self.volume
            )
            residual -= self.sheet_source
            residual = np.where(mask, psi - targets, residual)
            norm = float(np.max(np.abs(residual)))
            residual_history.append(norm)
            if norm <= tolerance:
                break

            # One unknown per node: the row blocks are nx x nx.
            diagonal = np.zeros((nz, nx, nx), dtype=float)
            upper = np.zeros((max(nz - 1, 0), nx, nx), dtype=float)
            lower = np.zeros((max(nz - 1, 0), nx, nx), dtype=float)
            idx = np.arange(nx)
            # dn/dpsi = n / kT_eV, so the space-charge stiffness is q*n*V/kT_eV.
            # It is this term - not the Laplacian - that makes the nonlinear
            # Poisson iterate contract; dropping the q here turns Newton back
            # into the naive fixed point that diverges.
            space_charge = (
                P.ELEMENTARY_CHARGE * density * self.volume / self.kT_eV
            )
            for row in range(nz):
                block = np.zeros((nx, nx), dtype=float)
                block[idx, idx] = space_charge[row]
                if nx > 1:
                    block[idx[:-1], idx[:-1]] += self.cx[row]
                    block[idx[1:], idx[1:]] += self.cx[row]
                    block[idx[:-1], idx[1:]] -= self.cx[row]
                    block[idx[1:], idx[:-1]] -= self.cx[row]
                if row > 0:
                    block[idx, idx] += self.cz[row - 1]
                if row < nz - 1:
                    block[idx, idx] += self.cz[row]
                diagonal[row] = block
                if row < nz - 1:
                    vertical = np.zeros((nx, nx), dtype=float)
                    vertical[idx, idx] = -self.cz[row]
                    upper[row] = vertical
                if row > 0:
                    vertical = np.zeros((nx, nx), dtype=float)
                    vertical[idx, idx] = -self.cz[row - 1]
                    lower[row - 1] = vertical

            positions = np.arange(nx)
            for row in range(nz):
                constrained = mask[row]
                if not constrained.any():
                    continue
                diagonal[row][constrained, :] = 0.0
                diagonal[row][constrained, positions[constrained]] = 1.0
                if row < nz - 1:
                    upper[row][constrained, :] = 0.0
                if row > 0:
                    lower[row - 1][constrained, :] = 0.0

            step = linear.solve(lower, diagonal, upper, -residual)
            psi = psi + np.asarray(step, dtype=float).reshape(nz, nx)

        self.poisson_history = residual_history
        return psi

    # -- ohmic reservoirs --------------------------------------------------
    def contact_density_m3(
        self,
        reference_V: np.ndarray | None = None,
        *,
        column: int = 0,
    ) -> np.ndarray:
        """Equilibrium density on a contact column: ``N_C_eff exp(q psi_ref/kT)``."""
        reference = (
            self.source_reference if reference_V is None else np.asarray(reference_V)
        )
        density = self.grid.nc_eff_m3[:, column] * np.exp(
            np.clip(reference / self.kT_eV, -P.EXPONENT_CAP, P.EXPONENT_CAP)
        )
        return np.where(self.grid.blocked_mask[:, column], 0.0, density)

    def adopt_equilibrium(self, psi_field: np.ndarray) -> None:
        """Use an equilibrium potential's contact columns as the reservoirs."""
        psi = np.asarray(psi_field, dtype=float).reshape(self.grid.nz, self.grid.nx)
        self.source_reference = psi[:, 0].copy()
        self.drain_reference = psi[:, -1].copy()

    def boundary_for(self, *, vds_V: float, vgs_V: float) -> Boundary:
        """Dirichlet data at one bias point, consistent with the reservoirs.

        The contacts pin the density **and** the potential that goes with it, so
        ``psi`` and ``n`` shift together with the terminal bias and the contact
        faces carry no spurious current. The drain rides ``vds`` on top of its
        equilibrium profile; the gate is a Schottky Dirichlet on ``psi`` alone
        (no gate current in v0.1.0); the backside is the grounded substrate.
        """
        grid = self.grid
        nz, nx = grid.nz, grid.nx
        psi_mask = np.zeros((nz, nx), dtype=bool)
        psi_value = np.zeros((nz, nx), dtype=float)
        density_mask = np.zeros((nz, nx), dtype=bool)
        density_value = np.zeros((nz, nx), dtype=float)

        left = grid.ohmic_mask[:, 0]
        right = grid.ohmic_mask[:, -1]
        psi_mask[left, 0] = True
        psi_value[left, 0] = self.source_reference[left]
        density_mask[left, 0] = True
        density_value[left, 0] = (
            self.contact_density_m3(self.source_reference, column=0)[left]
            / self.density_scale
        )

        psi_mask[right, -1] = True
        psi_value[right, -1] = self.drain_reference[right] + float(vds_V)
        density_mask[right, -1] = True
        density_value[right, -1] = (
            self.contact_density_m3(self.drain_reference, column=-1)[right]
            / self.density_scale
        )

        psi_mask |= grid.gate_mask
        psi_value[grid.gate_mask] = grid.gate_reference_V + float(vgs_V)
        psi_mask |= grid.ground_mask
        psi_value[grid.ground_mask] = 0.0
        psi_mask |= grid.surface_mask
        psi_value[grid.surface_mask] = grid.gate_reference_V
        density_mask |= grid.blocked_mask
        density_value[grid.blocked_mask] = 0.0
        return Boundary(
            psi_mask=psi_mask,
            psi_value=psi_value,
            density_mask=density_mask,
            density_value=density_value,
        )

    # -- linear step adapter ----------------------------------------------
    def solve_linear(
        self, linear: Any, lower: Any, diagonal: Any, upper: Any, rhs_flat: np.ndarray
    ) -> np.ndarray:
        """Solve the block system for a flattened residual, return it flattened.

        Applies the frozen column (unknown) scaling on the way in and undoes it
        on the way out, so the solver sees a two-sided equilibrated matrix:
        ``(D_row A D_col) y = D_row b`` with ``delta_u = D_col y``. Keeps the
        Newton loop free of the ``(nz, 2*nx)`` block layout.
        """
        blocks, size = self.grid.nz, 2 * self.grid.nx
        scale = self.column_scale.reshape(blocks, size)
        scaled_lower = lower * scale[:-1, None, :] if blocks > 1 else lower
        scaled_diagonal = diagonal * scale[:, None, :]
        scaled_upper = upper * scale[1:, None, :] if blocks > 1 else upper
        solution = linear.solve(
            scaled_lower,
            scaled_diagonal,
            scaled_upper,
            np.asarray(rhs_flat, dtype=float).reshape(blocks, size),
        )
        return (np.asarray(solution, dtype=float) * scale).reshape(-1)

    # -- equilibration ----------------------------------------------------
    def equilibrate(self, u: np.ndarray) -> np.ndarray:
        """Reset the unknown (column) scaling - currently a deliberate no-op.

        Tried, measured, rejected: in the ``(psi, n)`` formulation the matrix
        entries do **not** scale with the unknowns - ``dJ/dn`` is independent of
        ``n`` and the Poisson Laplacian is independent of ``psi`` - so dividing a
        column by its own unknown cannot improve the conditioning, and applied
        blindly it destroys it: a near-zero density in the buffer asks for a
        column factor of ~1e19, the eliminated pivot overflows and the block
        factorisation reports a singular pivot at the last row. The mechanism is
        kept (and the call site unchanged) because a future formulation whose
        entries *do* carry unknown factors will want it.
        """
        self.column_scale = np.ones((self.grid.nz, self.grid.nx, 2), dtype=float)
        return self.column_scale

    # -- currents ---------------------------------------------------------
    def terminal_currents(self, u: np.ndarray) -> dict[str, float]:
        """Terminal currents per metre of device width, entering the device."""
        grid = self.grid
        u = np.asarray(u, dtype=float).reshape(grid.nz, grid.nx, 2)
        psi = u[:, :, 0]
        density = u[:, :, 1] * self.density_scale
        band = P.band_edge_eV(psi, grid.band_offset_eV)
        current_x, *_ = P.sg_face_current(
            band_left=band[:, :-1],
            band_right=band[:, 1:],
            n_left=density[:, :-1],
            n_right=density[:, 1:],
            mu_m2_Vs=self.mu0_m2_Vs,
            kT_eV=self.kT_eV,
            length_m=grid.dx_m[None, :],
            statistics=self.statistics,
            vsat_m_s=self.vsat_m_s,
            beta_field=self.beta_field,
        )
        area_x = grid.wz_m[:, None]
        # Only faces that actually carry current, i.e. the same mask the residual
        # uses. Without it the barrier rows - whose carrier faces are closed - are
        # counted as if they pushed current into the contact, and the terminal
        # currents disagree with the continuity balance the solver just satisfied,
        # by orders of magnitude more than the residual.
        current_x = current_x * self.face_open_x
        source = float(np.sum(current_x[:, 0] * area_x[:, 0]))
        drain = float(-np.sum(current_x[:, -1] * area_x[:, 0]))
        return {
            "source": source,
            "drain": drain,
            "gate": 0.0,
            "continuity_error": abs(source + drain) / max(abs(drain), 1.0e-30),
        }

    # -- initial guess -----------------------------------------------------
    def bias_shift(self, state: np.ndarray, *, d_vds_V: float) -> np.ndarray:
        """Shift a converged solution onto a new drain bias (continuation).

        A drain step moves the drain contact's potential while the interior still
        holds the old one; that contact-local shock is enough to stall a line
        search even with an exact Jacobian. Pre-shifting the interior potential by
        ``dVds * ramp(x)`` - the same linear ramp the contact data imposes - keeps
        the first iterate within a few kT of the new solution, which is the
        difference between a stalled solve and a handful of Newton iterations.
        """
        shifted = np.array(state, dtype=float, copy=True)
        span = float(self.grid.x_m[-1] - self.grid.x_m[0])
        if span <= 0.0 or d_vds_V == 0.0:
            return shifted
        ramp = (self.grid.x_m - self.grid.x_m[0]) / span
        shifted[:, :, 0] = shifted[:, :, 0] + float(d_vds_V) * ramp[None, :]
        return shifted

    def initial_guess(
        self,
        *,
        psi_field: np.ndarray | None = None,
        vds_V: float = 0.0,
    ) -> np.ndarray:
        """Starting point: a reference potential and a linear quasi-Fermi ramp.

        ``psi_field`` normally comes from the Poisson-only pre-solve, which is
        what keeps the first coupled Newton step in a sane basin. The quasi-Fermi
        level starts as a straight ramp between the contacts - the standard start
        for a current-carrying solution - and is turned into a density unknown
        through the Boltzmann relation.
        """
        grid = self.grid
        guess = np.zeros((grid.nz, grid.nx, 2), dtype=float)
        if psi_field is not None:
            guess[:, :, 0] = np.asarray(psi_field, dtype=float).reshape(
                grid.nz, grid.nx
            )
        else:
            guess[:, :, 0] = self.boundary.psi_value
        phi_n = np.zeros((grid.nz, grid.nx), dtype=float)
        if vds_V and grid.x_m[-1] > grid.x_m[0]:
            ramp = (grid.x_m - grid.x_m[0]) / (grid.x_m[-1] - grid.x_m[0])
            phi_n = float(vds_V) * ramp[None, :]
        guess[:, :, 1] = (
            P.carrier_density(
                guess[:, :, 0], phi_n, nc_eff_m3=grid.nc_eff_m3, kT_eV=self.kT_eV
            )
            / self.density_scale
        )
        guess[:, :, 1] = np.where(grid.blocked_mask, 0.0, guess[:, :, 1])
        return np.where(self._dirichlet, self._targets, guess)


# ---------------------------------------------------------------------------
# Grid construction from the pipeline artifacts
# ---------------------------------------------------------------------------


def build_grid2d(
    mesh: Mapping[str, Any],
    structure: Mapping[str, Any],
    materials: Any,
    polarization: Mapping[str, Any],
    *,
    temperature_K: float,
    mu0_m2_Vs: float,
    delta_ec_ratio: float,
    delta_ec_eV: float | None = None,
    gate_barrier_eV: float | None = None,
    left_reference_V: np.ndarray | None = None,
    right_reference_V: np.ndarray | None = None,
    ground_backside: bool = True,
    surface_pinning: bool = True,
) -> Grid2D:
    """Turn the mesh/structure/material artifacts into a :class:`Grid2D`.

    ``delta_ec_eV`` (the conduction-band offset, e.g. from the quantum module)
    wins over ``delta_ec_ratio * dEg`` when given. Contact nodes are the left
    and right boundary columns restricted to the electrically active layers:
    an acceptor-doped or explicitly semi-insulating layer (the buffer) is left
    floating, because "ohmic to a depleted p-type buffer" is not a device.
    """
    x_m = np.asarray(mesh["x_nm"], dtype=float) * 1.0e-9
    z_m = np.asarray(mesh["z_nm"], dtype=float) * 1.0e-9
    nx, nz = x_m.size, z_m.size
    if mesh.get("nx") not in (None, nx) or mesh.get("nz") not in (None, nz):
        raise ValueError("mesh artifact disagrees with its own axis lengths")
    dx_m, dz_m = np.diff(x_m), np.diff(z_m)
    wx_m, wz_m = control_volumes(x_m), control_volumes(z_m)

    layers = list(mesh["layers"])
    specs = list(structure["layers"])
    if len(layers) != len(specs):
        raise ValueError("mesh and structure disagree on the layer count")
    channel = next((layer for layer in specs if layer.get("role") == "channel"), None)
    channel_material = str(channel["material"]) if channel else "GaN"
    eg_channel = materials.get(channel_material).numeric("Eg_eV")

    kT_eV = P.BOLTZMANN * float(temperature_K) / P.ELEMENTARY_CHARGE
    eps_r = np.ones((nz, nx), dtype=float)
    m_eff_rel = np.ones((nz, nx), dtype=float)
    n_net = np.zeros((nz, nx), dtype=float)
    ec0 = np.zeros((nz, nx), dtype=float)
    layer_of_row = np.full(nz, -1, dtype=int)
    active_row = np.zeros(nz, dtype=bool)
    blocked_row = np.zeros(nz, dtype=bool)
    sheet_row: int | None = None
    sheet_charge = 0.0

    for index, (entry, spec) in enumerate(zip(layers, specs)):
        start, end = entry["node_slice"]
        rows = slice(int(start), int(end) + 1)
        x_al = float(spec.get("x_al", 0.0))

        def alloy(key: str) -> float:
            if x_al > 0.0:
                return float(materials.alloy_from_spec("AlGaN", x_al, key))
            return float(materials.get(str(spec["material"])).numeric(key))

        eps_row = alloy("eps_r")
        mass_row = alloy("m_eff")
        eg_row = alloy("Eg_eV")
        eps_r[rows, :] = eps_row
        m_eff_rel[rows, :] = mass_row
        # Later layers overwrite the shared interface node, which puts the
        # interface node and the 2DEG peak on the channel side of the step.
        layer_of_row[rows] = index

        offset = float(delta_ec_eV) if delta_ec_eV is not None else (
            float(delta_ec_ratio) * (eg_row - eg_channel)
        )
        ec0[rows, :] = max(offset, 0.0)

        doping = float(spec.get("doping_cm3", 0.0)) * CM3_TO_M3
        kind = str(spec.get("doping_kind", "unintentional"))
        if kind == "donor":
            n_net[rows, :] = doping
        elif kind == "acceptor":
            n_net[rows, :] = -doping

        role = str(spec.get("role", "channel"))
        semi_insulating = kind == "acceptor" or role in ("buffer", "substrate", "nucleation")
        active_row[rows] |= not semi_insulating
        if semi_insulating:
            # A monopolar (electron-only) v1 has no holes with which to deplete an
            # acceptor, so treating the acceptors as fixed ionised charge puts an
            # enormous unphysical field across the buffer, and the electron
            # density the Poisson solution then implies makes the "semi-
            # insulating" buffer a conductor - which shows up as a terminal
            # current that is an order of magnitude larger than the channel
            # current, flowing straight down into the substrate. The standard
            # monopolar abstraction is a neutral dielectric spacer with no
            # carriers: no space charge, and the density pinned to zero.
            n_net[rows, :] = 0.0
            blocked_row[rows] = True

        if role == "channel" and index > 0 and layers[index - 1]["role"] == "barrier":
            sheet_row = int(start)
            sheet_charge = float(polarization.get("sigma_eff_C_m2", 0.0))
        if role == "barrier":
            # The interface with the channel sits at the barrier's last node.
            sheet_row = int(end)
            sheet_charge = float(polarization.get("sigma_eff_C_m2", 0.0))

    if np.any(layer_of_row < 0):
        raise ValueError("the mesh contains nodes that belong to no layer")

    nc_dos = P.effective_dos_m3(m_eff_rel * P.ELECTRON_MASS, temperature_K)
    # The Scharfetter-Gummel flux must be at equilibrium across a heterojunction,
    # which needs the *density of states* step folded into the band edge: with
    # n = N_C exp((psi - ec0)/kT), a face is at equilibrium only when
    # delta = (band_i - band_j)/kT equals ln(n_j/n_i) exactly, and that holds for
    # band = ec0 + kT ln(N_ref/N_C) rather than for ec0 itself. Dropping the
    # ln(N_C) term leaves a 19 % jump in the electron density across AlGaN/GaN -
    # invisible in the potential, but a contact-side current that never converges.
    reference_dos = 1.0e24
    band_offset = ec0 + kT_eV * np.log(reference_dos / nc_dos)
    nc_eff = reference_dos * np.exp(-band_offset / kT_eV)

    ohmic = np.zeros((nz, nx), dtype=bool)
    ohmic[:, 0] = active_row
    ohmic[:, -1] = active_row
    blocked = np.zeros((nz, nx), dtype=bool)
    blocked[blocked_row, :] = True

    # Wide-gap rows carry no current, and their carrier density is negligible
    # next to the channel's. Leaving the continuity equation there leaves the
    # density a nearly-free variable - the discrete flux divergence is linear and
    # homogeneous in n, so it has modes the equation cannot pin down - and the
    # Jacobian ends up with a singular value ten orders below its largest, which
    # is what makes a Newton step explode from an already-converged start. The
    # standard closure for a current-free insulator is local equilibrium:
    # replace the continuity row by n = N_C_eff exp(q psi/kT), which is exact
    # physics where no current flows and removes the free mode by construction.
    threshold = 1.0e-3 * float(np.max(nc_eff)) if nc_eff.size else 1.0
    equilibrium = (nc_eff < threshold) & ~blocked

    z_nm = np.asarray(mesh["z_nm"], dtype=float)
    gate_row = int(np.argmin(np.abs(z_nm)))  # the surface is z = 0
    contacts = mesh.get("contacts_nm", {})
    gate_left = float(contacts.get("gate_left", 0.0)) * 1.0e-9
    gate_right = float(contacts.get("gate_right", 0.0)) * 1.0e-9
    gate = np.zeros((nz, nx), dtype=bool)
    under_gate = (x_m >= gate_left - 1e-18) & (x_m <= gate_right + 1e-18)
    gate[gate_row, under_gate] = True

    # The backside is a grounded substrate contact. Without it the
    # semi-insulating buffer has Neumann conditions on every side, so its
    # potential is only weakly coupled and the Jacobian is ill conditioned
    # enough that the first Newton step leaves the physical range. Grounding the
    # back face is the standard device-simulation anchor; a genuinely floating
    # buffer needs a substrate-leakage model and is future work.
    ground = np.zeros((nz, nx), dtype=bool)
    if ground_backside:
        ground[-1, :] = True

    # Surface pinning. The ungated surface of a GaN HEMT is pinned by donor
    # states, and pinning it is also what stops the barrier potential from being
    # a nearly-free variable: with charge ~0 in the barrier, Poisson there is a
    # pure Laplacian whose sides are Neumann and whose only anchor is the gate,
    # so a lateral mode of the barrier potential is resisted by almost nothing.
    # That shows up as a Jacobian singular value twelve orders below the largest
    # and a Newton step that stops moving while the equations are still violated.
    # A proper surface-donor model belongs to trap.surface; the pinned value here
    # is the Schottky reference, i.e. the surface is assumed to sit at the same
    # energy as the gate metal.
    surface = np.zeros((nz, nx), dtype=bool)
    if surface_pinning:
        surface[gate_row, :] = True
        surface[gate_row, under_gate] = False

    if gate_barrier_eV is not None:
        surface_offset = float(ec0[gate_row, int(np.argmax(under_gate))])
        gate_reference = surface_offset - float(gate_barrier_eV)
    else:
        gate_reference = 0.0

    zero = np.zeros(nz, dtype=float)
    left_reference = zero.copy() if left_reference_V is None else np.asarray(left_reference_V, dtype=float)
    right_reference = zero.copy() if right_reference_V is None else np.asarray(right_reference_V, dtype=float)
    #: Ohmic contacts are reservoirs: pinning the density to its equilibrium
    #: value (from the reference potential, with phi_n = 0 there) makes the
    #: contact density bias-independent, so a drain bias shifts psi and phi_n
    #: together and leaves psi - phi_n - and hence n - untouched.
    contact_density = nc_eff[:, 0] * np.exp(
        np.clip(left_reference / kT_eV, -P.EXPONENT_CAP, P.EXPONENT_CAP)
    )
    return Grid2D(
        nx=nx,
        nz=nz,
        x_m=x_m,
        z_m=z_m,
        dx_m=dx_m,
        dz_m=dz_m,
        wx_m=wx_m,
        wz_m=wz_m,
        eps_r=eps_r,
        mu_m2_Vs=np.full((nz, nx), P.constant_mobility(mu0_m2_Vs)),
        n_net_m3=n_net,
        ec0_eV=ec0,
        band_offset_eV=band_offset,
        nc_eff_m3=nc_eff,
        layer_of_row=layer_of_row,
        sheet_row=sheet_row,
        sheet_C_m2=sheet_charge,
        ohmic_mask=ohmic,
        gate_mask=gate,
        ground_mask=ground,
        surface_mask=surface,
        blocked_mask=blocked,
        equilibrium_mask=equilibrium,
        gate_reference_V=float(gate_reference),
        source_reference_V=left_reference,
        drain_reference_V=right_reference,
        contact_density_m3=contact_density,
    )


def make_boundary(
    grid: Grid2D,
    *,
    vds_V: float,
    vgs_V: float,
    density_scale_m3: float = 1.0e23,
) -> Boundary:
    """Ohmic source/drain plus Schottky gate Dirichlet data for one bias point.

    The source is the potential reference and the drain rides ``vds`` on top of
    its equilibrium column profile, so ``psi`` at the contact keeps its
    equilibrium relation to the pinned density: that is what makes a contact an
    ohmic reservoir rather than a Dirichlet guess.
    """
    nz, nx = grid.nz, grid.nx
    psi_mask = np.zeros((nz, nx), dtype=bool)
    psi_value = np.zeros((nz, nx), dtype=float)
    density_mask = np.zeros((nz, nx), dtype=bool)
    density_value = np.zeros((nz, nx), dtype=float)

    left = grid.ohmic_mask[:, 0]
    right = grid.ohmic_mask[:, -1]
    contact_density = grid.contact_density_m3 / float(density_scale_m3)

    psi_mask[left, 0] = True
    psi_value[left, 0] = grid.source_reference_V[left]
    density_mask[left, 0] = True
    density_value[left, 0] = contact_density[left]

    psi_mask[right, -1] = True
    psi_value[right, -1] = grid.drain_reference_V[right] + float(vds_V)
    density_mask[right, -1] = True
    density_value[right, -1] = contact_density[right]

    psi_mask |= grid.gate_mask
    psi_value[grid.gate_mask] = grid.gate_reference_V + float(vgs_V)
    psi_mask |= grid.ground_mask
    psi_value[grid.ground_mask] = 0.0
    psi_mask |= grid.surface_mask
    psi_value[grid.surface_mask] = grid.gate_reference_V
    density_mask |= grid.blocked_mask
    density_value[grid.blocked_mask] = 0.0
    return Boundary(
        psi_mask=psi_mask,
        psi_value=psi_value,
        density_mask=density_mask,
        density_value=density_value,
    )
