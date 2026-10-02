r"""
The running vacuum, ``Lambda(H) = c0 + 3 nu H^2``, with radiation.

The vacuum exchanges energy with cold matter; photons and neutrinos are
conserved. In units of today's critical density, ``E^2 (1 - nu) =
rho_cb + rho_rel + c + Omega_k a^-2``, and the continuity equations
combine into one linear equation for ``y = E^2``::

    y' = -3 (1 - nu) y + 3 c + Omega_k a^-2 - 3 p_rel,
    c  = (1 - nu) - Omega_cb - rho_rel(0) - Omega_k,

with ``p_rel`` the photon and neutrino pressure and ``y(0) = 1``.
Without radiation this is the old model's closed form -- curvature term
``Omega_k (1+z)^2 / (1 - 3 nu)`` included -- and with it, radiation
enters with ``(1+z)^4 / (1 + 3 nu)``: the vacuum responds to radiation
in ``H^2`` as it does to everything else there. Massive neutrinos, whose
pressure is not a fixed fraction of their density, are why this is
integrated rather than written out.
"""

from __future__ import annotations

import numpy as np

from .dark_sector import DarkSector, solve_linear


__all__ = ["RunningVacuum"]


class RunningVacuum(DarkSector):
    """Running vacuum model, ``nu``."""

    name = "rvm"
    params = ("nu",)

    def solve(self, ctx, nu):

        nu = float(nu)

        c = (1.0 - nu) - ctx.Omega_cb - ctx.rho_rel0 - ctx.Omega_k

        def source(N):

            z = np.expm1(-N)

            return 3.0 * c + ctx.Omega_k * np.exp(-2.0 * N) - 3.0 * ctx.p_rel(z)

        return solve_linear(-3.0 * (1.0 - nu), source)

    @staticmethod
    def _c(ctx, nu):
        return (1.0 - nu) - ctx.Omega_cb - ctx.rho_rel0 - ctx.Omega_k

    def clustering_matter(self, z, growth, nu):
        """``rho_m = (1 - nu) E^2 - rho_rel - c - Omega_k a^-2``."""

        ctx = growth.ctx

        return (
            (1.0 - nu) * growth.E2(z) - ctx.rho_rel(z)
            - self._c(ctx, nu) - ctx.curvature(z)
        )

    def matter_exchange(self, z, growth, nu):
        """
        ``psi = [3 nu rho_m - nu (rho_rel' - 2 Omega_k a^-2)] / rho_m``,
        ``rho_rel' = -3 (rho_rel + p_rel)`` -- ``3 nu`` when flat and
        radiation-free.
        """

        ctx = growth.ctx

        rho_m = self.clustering_matter(z, growth, nu)

        rho_rel_prime = -3.0 * (ctx.rho_rel(z) + ctx.p_rel(z))

        return (
            3.0 * nu * rho_m - nu * (rho_rel_prime - 2.0 * ctx.curvature(z))
        ) / rho_m
