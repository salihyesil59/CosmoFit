"""
Expansion from a modified Friedmann equation rather than a dark fluid.

Each of these maps the density of the standard fluids onto ``E^2`` its
own way. The old models wrote that map with ``Omega_m (1+z)^3`` as the
only density; here it is applied to *all* the standard fluids -- the
modification is to how gravity responds to energy density, and
radiation is energy density. With radiation switched off every formula
reduces to the old model's, and the tests check that it does.

* **DGP** (self-accelerating branch): ``E = sqrt(Omega_rc + rho) +
  sqrt(Omega_rc)``, curvature added outside.
* **Cardassian** (modified polytropic): ``E^2 = rho [1 + (Omega_*^-q - 1)
  (rho/rho_0)^{q(n-1)}]^{1/q}``.
* **f(Q)** exponential, ``(E^2 - 2 lambda) e^{lambda/E^2} = rho``, and
  **f(T)** power law, ``E^2 + c E^{2n} = rho``: transcendental,
  solved by Newton's method, flat only.
* **f(R,T)** linear: the trace ``T = rho - 3p`` enters the source, so
  each fluid contributes ``rho + beta (3 rho - p)`` -- dust with ``1 +
  3 beta``, radiation (traceless) with ``1 + (8/3) beta``.
"""

from __future__ import annotations

import numpy as np

from scipy.special import lambertw

from cosmofit.cosmology.core.utils import coupling_from_derivative

from .dark_sector import DarkSector


__all__ = ["DGP", "Cardassian", "FQExponential", "FTPowerLaw", "FRTLinear"]


class DGP(DarkSector):
    """Dvali-Gabadadze-Porrati braneworld, self-accelerating branch."""

    name = "dgp"

    def solve(self, ctx):

        one_minus_k = 1.0 - ctx.Omega_k
        excess = one_minus_k - ctx.rho_std0

        # sqrt(Omega_rc) = excess / (2 sqrt(1 - Omega_k)) before squaring;
        # a non-positive excess has no self-accelerating solution.
        if excess <= 0.0:
            return None

        Omega_rc = excess ** 2 / (4.0 * one_minus_k)

        def E2(z):

            brane = (np.sqrt(Omega_rc + ctx.rho_std(z)) + np.sqrt(Omega_rc)) ** 2

            return brane + ctx.curvature(z)

        return E2

    def mu(self, z, k, growth):
        """
        ``mu = 1 + 1/(3 beta)``, ``beta = 1 - 2 H r_c [1 + Hdot/(3 H^2)]``,
        ``2 H r_c = E / sqrt(Omega_rc)``.
        """

        ctx = growth.ctx

        one_minus_k = 1.0 - ctx.Omega_k
        Omega_rc = (one_minus_k - ctx.rho_std0) ** 2 / (4.0 * one_minus_k)

        E = np.sqrt(growth.E2(z))

        beta = 1.0 - E / np.sqrt(Omega_rc) * (1.0 + growth.dlnH_dN(z) / 3.0)

        return 1.0 + 1.0 / (3.0 * beta)


class Cardassian(DarkSector):
    """Modified polytropic Cardassian expansion."""

    name = "cardassian"
    params = ("n_card", "q_card")

    def solve(self, ctx, n_card, q_card):

        star = ctx.rho_std0 / (1.0 - ctx.Omega_k)

        def E2(z):

            rho = ctx.rho_std(z)

            bracket = 1.0 + (star ** (-q_card) - 1.0) * (
                rho / ctx.rho_std0
            ) ** (q_card * (n_card - 1.0))

            with np.errstate(invalid="ignore"):
                value = ctx.curvature(z) + rho * bracket ** (1.0 / q_card)

            return np.where(bracket > 0.0, value, np.nan)

        return E2


class FQExponential(DarkSector):
    """
    f(Q) = Q exp(lambda Q0/Q), with lambda fixed by ``E(0) = 1``:
    ``lambda = 1/2 + W0(-rho_std(0) / (2 sqrt(e)))``.
    """

    name = "fq_exponential"
    flat_only = True

    @staticmethod
    def _lambda(ctx) -> float:
        return float(np.real(0.5 + lambertw(-ctx.rho_std0 / (2.0 * np.sqrt(np.e)), k=0)))

    def mu(self, z, k, growth):
        """``mu = 1/f_Q``, ``f_Q = e^{lambda/E^2} (1 - lambda/E^2)``."""

        lam = self._lambda(growth.ctx)

        x = 1.0 / growth.E2(z)

        return coupling_from_derivative(
            np.exp(lam * x) * (1.0 - lam * x), model="FQExponential",
        )

    #: Fixed Newton steps; the map is smooth and monotonic over the
    #: physical range, as in the old model.
    newton_iterations = 30

    def solve(self, ctx):

        lam = float(np.real(0.5 + lambertw(-ctx.rho_std0 / (2.0 * np.sqrt(np.e)), k=0)))

        def E2(z):

            rhs = ctx.rho_std(z)

            x = rhs + (1.0 - ctx.rho_std0)

            for _ in range(self.newton_iterations):

                expo = np.exp(lam / x)
                g = (x - 2.0 * lam) * expo
                dg = expo * (1.0 - lam * (x - 2.0 * lam) / x ** 2)

                x = x - (g - rhs) / dg

            return x

        return E2


class FTPowerLaw(DarkSector):
    """
    f(T) = T + alpha T^n_ft, with the amplitude fixed by ``E(0) = 1``:
    ``E^2 + (rho_std(0) - 1) E^{2 n_ft} = rho_std``. The root is checked,
    and NaN (no solution) returned where Newton's method did not find
    one -- for ``n_ft > 1/2`` the equation stops having a real solution
    above some redshift.
    """

    name = "ft_power_law"
    params = ("n_ft",)
    flat_only = True

    newton_iterations = 30
    residual_tolerance = 1.0e-10

    def mu(self, z, k, growth, n_ft):
        """
        ``mu = 1/f_T = 1/(1 + n A E^{2n-2})``, ``A = (rho_std(0) - 1)/(2n - 1)``;
        the ``n = 1/2`` pole refused.
        """

        from cosmofit.cosmology.core.errors import ModelConfigurationError

        n = float(n_ft)

        if abs(2.0 * n - 1.0) < 1e-3:
            raise ModelConfigurationError(
                f"FTPowerLaw: n_ft = {n!r} sits on the n = 1/2 pole of the "
                f"effective gravitational coupling."
            )

        A = (growth.ctx.rho_std0 - 1.0) / (2.0 * n - 1.0)

        return coupling_from_derivative(
            1.0 + n * A * growth.E2(z) ** (n - 1.0), model="FTPowerLaw",
        )

    def solve(self, ctx, n_ft):

        n = float(n_ft)
        c = ctx.rho_std0 - 1.0

        def E2(z):

            rhs = ctx.rho_std(z)

            x = rhs + (1.0 - ctx.rho_std0)

            with np.errstate(invalid="ignore", divide="ignore"):

                for _ in range(self.newton_iterations):

                    xn = x ** n
                    x = x - (x + c * xn - rhs) / (1.0 + n * c * xn / x)
                    x = np.where(x > 0.0, x, np.nan)

                residual = np.abs(x + c * x ** n - rhs)

            return np.where(
                residual <= self.residual_tolerance * np.maximum(rhs, 1.0), x, np.nan,
            )

        return E2


class FRTLinear(DarkSector):
    """
    f(R,T) = R + 2 lambda T, ``beta = lambda / 8 pi``:

        E^2 = Omega_k (1+z)^2 + (1 + 3 beta) rho_std - beta p_std
              + (1 + 4 beta) Omega_L

    with ``Omega_L`` from ``E(0) = 1``.
    """

    name = "frt_linear"
    params = ("beta",)

    def mu(self, z, k, growth, beta):
        """``mu = 1 + 3 beta``, the old model's stated simplification."""

        return np.full_like(np.asarray(z, dtype=float), 1.0 + 3.0 * beta)

    def solve(self, ctx, beta):

        def source(z):
            return (1.0 + 3.0 * beta) * ctx.rho_std(z) - beta * ctx.p_std(z)

        Omega_L = (1.0 - ctx.Omega_k - float(source(0.0))) / (1.0 + 4.0 * beta)

        def E2(z):
            return ctx.curvature(z) + source(z) + (1.0 + 4.0 * beta) * Omega_L

        return E2
