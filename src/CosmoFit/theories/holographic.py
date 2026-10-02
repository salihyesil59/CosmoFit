r"""
The holographic family, with radiation.

All three set ``rho_de = 3 M_p^2 / L^2`` for some infrared length ``L``
and are flat only. The old models solved them in a matter-only
universe; with photons and neutrinos present, each equation picks up
the standard fluids' pressure. Writing ``Omega = Omega_DE(z)`` and
``' = d/d ln a``, with ``E^2 = rho_std / (1 - Omega)`` in a flat
universe:

**HDE** (future event horizon, ``rho = 3 c^2 / L^2``)::

    Omega' = Omega (1 - Omega) [1 + 2 sqrt(Omega)/c + 3 p_std/rho_std]

**ADE** (conformal age, ``rho = 3 n^2 / eta^2``)::

    Omega' = Omega (1 - Omega) [3 - 2 sqrt(Omega)/(n a) + 3 p_std/rho_std]

**RDE** (Ricci scalar, ``rho = 3 gamma (Hdot + 2 H^2)``)::

    (gamma/2) (E^2)' + (2 gamma - 1) E^2 + rho_std = 0

Each is derived from ``rho_de`` and the continuity equation, with
``-2 H'/H = 3 (1 + w_total)`` carrying the standard fluids' pressure;
with only dust (``p_std = 0``) they are the old models' equations
exactly. Radiation's ``3p/rho = 1`` changes the early-time behaviour:
ADE's dark energy tracks ``n^2 a^2`` in the radiation era, not the
``n^2 a^2 / 4`` of the matter era, and the transition between the two
depends on when matter and radiation are equal -- which is why ADE's
``Omega_m``, still fixed by ``n``, is now found by iteration.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.integrate import solve_ivp

from .dark_sector import N_MIN, DarkSector, solve_linear


__all__ = ["HDE", "ADE", "RDE"]


def _pressure_ratio(ctx, N):
    """``3 p_std / rho_std`` at ``N = ln a``."""

    z = np.expm1(-np.asarray(N, dtype=float))

    return 3.0 * ctx.p_std(z) / ctx.rho_std(z)


def _omega_function(solution, n_lo: float, floor_slope: float):
    """
    ``Omega_DE(z)`` from an ODE solution in ``x = ln Omega`` over
    ``[n_lo, 0]``; below ``n_lo`` the dark energy is extrapolated along
    ``x = x(n_lo) + floor_slope (N - n_lo)`` -- its early-time power law.
    """

    x_lo = float(solution.sol(n_lo)[0])

    def omega(z):

        N = -np.log1p(np.asarray(z, dtype=float))

        inside = N >= n_lo

        x = np.empty_like(N)

        if np.any(inside):
            x[inside] = solution.sol(N[inside])[0]

        x[~inside] = x_lo + floor_slope * (N[~inside] - n_lo)

        return np.exp(x)

    return omega


class HDE(DarkSector):
    """Holographic dark energy, future-event-horizon cutoff, ``c_hde``."""

    name = "hde"
    params = ("c_hde",)
    flat_only = True

    def solve(self, ctx, c_hde):

        c = float(c_hde)

        if c <= 0.0:
            return None

        Omega0 = 1.0 - ctx.rho_std0

        if not 0.0 < Omega0 < 1.0:
            return None

        def rhs(N, x):

            omega = math.exp(x[0])

            return [(1.0 - omega) * (
                1.0 + 2.0 * math.sqrt(omega) / c + float(_pressure_ratio(ctx, N))
            )]

        solution = solve_ivp(
            rhs, (0.0, N_MIN), [math.log(Omega0)],
            rtol=1e-10, atol=1e-12, dense_output=True,
        )

        if not solution.success:
            return None

        # Deep in the radiation era Omega' -> Omega (1 + 1): slope 2.
        omega = _omega_function(solution, N_MIN, 2.0 if ctx.radiation else 1.0)

        def E2(z):
            return ctx.rho_std(z) / (1.0 - omega(z))

        return E2


class ADE(DarkSector):
    """
    New agegraphic dark energy (conformal age), ``n_ade``. Fixes the
    matter density from ``n_ade``: the background does not take
    ``Omega_m`` with it.
    """

    name = "ade"
    params = ("n_ade",)
    flat_only = True
    derives_matter = True

    def _forward(self, ctx, n):
        """Integrate ``ln Omega`` from deep in the past to today."""

        a0 = math.exp(N_MIN)

        # The early attractor: n^2 a^2 in the radiation era, n^2 a^2/4
        # in a matter-only universe.
        start = (n * a0) ** 2 if ctx.radiation else (n * a0) ** 2 / 4.0

        def rhs(N, x):

            omega = math.exp(x[0])
            a = math.exp(N)

            return [(1.0 - omega) * (
                3.0 - 2.0 * math.sqrt(omega) / (n * a)
                + float(_pressure_ratio(ctx, N))
            )]

        solution = solve_ivp(
            rhs, (N_MIN, 0.0), [math.log(start)],
            rtol=1e-10, atol=1e-12, dense_output=True,
        )

        return solution if solution.success else None

    def matter_density(self, make_context, n_ade):

        n = float(n_ade)

        # Without radiation Omega_DE(today) depends on n alone; with it,
        # on when matter overtakes radiation too, so iterate.
        ctx = make_context(0.3)

        Omega_cb = 0.3

        for _ in range(50):

            solution = self._forward(ctx, n)

            if solution is None:
                return math.nan

            omega_today = math.exp(float(solution.sol(0.0)[0]))

            updated = 1.0 - omega_today - ctx.rho_rel0

            if abs(updated - Omega_cb) < 1e-13:
                return updated

            Omega_cb = updated
            ctx = make_context(Omega_cb)

            if not ctx.radiation:
                return Omega_cb

        return Omega_cb

    def solve(self, ctx, n_ade):

        solution = self._forward(ctx, float(n_ade))

        if solution is None:
            return None

        # Below N_MIN, Omega_DE ~ a^2.
        omega = _omega_function(solution, N_MIN, 2.0)

        def E2(z):
            return ctx.rho_std(z) / (1.0 - omega(z))

        return E2


class RDE(DarkSector):
    """Ricci dark energy, ``gamma_rde``."""

    name = "rde"
    params = ("gamma_rde",)
    flat_only = True

    def solve(self, ctx, gamma_rde):

        g = float(gamma_rde)

        if g <= 0.0:
            return None

        # (E^2)' = k E^2 + s,  k = 2(1 - 2 gamma)/gamma,  s = -2 rho_std/gamma
        k = 2.0 * (1.0 - 2.0 * g) / g

        def source(N):
            return -2.0 * ctx.rho_std(np.expm1(-N)) / g

        return solve_linear(k, source)
