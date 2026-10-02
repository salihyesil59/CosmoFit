r"""
The linear growth of structure, on the native background.

The sub-horizon, quasi-static equation for the density contrast of
the clustering matter, in ``N = ln a``:

    D'' + (2 + dlnH/dN + psi) D'
        - [3/2 Omega_m(a) mu(a, k) - psi (2 + dlnH/dN) - psi'] D = 0

``Omega_m(a)`` is the clustering matter's share of ``H^2``, ``mu`` the
gravitational coupling a modified-gravity sector reports, ``psi`` the
energy a coupled sector feeds the matter per Hubble time -- each read
from the dark sector's hooks (:class:`~theories.dark_sector.DarkSector`),
``1`` and ``0`` by default.

What is new against the old growth calculator is the background it
sits on. That one had no radiation, so ``D proportional to a`` at
``a = 1e-4`` was its growing mode. With radiation the growing mode at
early times is Meszaros's, ``D proportional to 1 + 3y/2``,
``y = rho_m / rho_rel`` -- the exact growing solution of this equation
in a matter-plus-radiation universe -- and that is where integration
starts. Without radiation the old initial condition is kept, and the
old results come back.

The clustering matter is the cold matter: massive neutrinos free-stream
on the scales growth data measure (``k = 0.1 h/Mpc`` is well above the
free-streaming scale of a 0.06 eV neutrino). The old calculator counted
them in, a ~0.5% difference in ``Omega_m(a)`` at that mass.

Where ``E(z)`` jumps, the equation is integrated in pieces and matched
by continuity of ``H dD/dN``, as in the old calculator.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.interpolate import PPoly

from CosmoFit.core.component import ComponentError, Theory
from CosmoFit.cosmology.calculators.growth import GrowthCalculator
from CosmoFit.cosmology.numerics.hermite import hermite_spline

from .dark_sector import GrowthContext


__all__ = ["Growth"]


#: Finite-difference step in ``N`` for the logarithmic slope of ``H``
#: and of the exchange rate.
_STEP = 1.0e-5


def _slope(f, N, lo, hi):
    """
    ``df/dN`` at ``N`` by second-order differences that stay inside
    ``[lo, hi]`` -- one-sided at the ends, so a jump at either end is
    never differenced across.
    """

    N = np.asarray(N, dtype=float)

    out = np.empty_like(N)

    centre = (N - _STEP >= lo) & (N + _STEP <= hi)
    low = (N - _STEP < lo)
    high = ~centre & ~low

    if np.any(centre):
        x = N[centre]
        out[centre] = (f(x + _STEP) - f(x - _STEP)) / (2 * _STEP)

    if np.any(low):
        x = N[low]
        out[low] = (-3 * f(x) + 4 * f(x + _STEP) - f(x + 2 * _STEP)) / (2 * _STEP)

    if np.any(high):
        x = N[high]
        out[high] = (3 * f(x) - 4 * f(x - _STEP) + f(x - 2 * _STEP)) / (2 * _STEP)

    return out


class Growth(Theory):
    """
    Options
    -------
    k : float
        Wavenumber [h/Mpc] for a scale-dependent ``mu``; default 0.1.

    Parameters
    ----------
    ``sigma8``: today's amplitude, which ``D(z)`` scales.

    Provides
    --------
    ``growth_factor`` (``D``, normalized to 1 today), ``growth_rate``
    (``f = dlnD/dlna``), ``sigma8_z`` and ``fsigma8`` -- each ``z=...``.

    Derived parameters
    ------------------
    ``S8 = sigma8 sqrt(Omega_m / 0.3)``.
    """

    #: Fixed RK4 steps per unit of ``ln a``. The old calculator's 300
    #: steps over 9.2 e-folds, which it checked against an adaptive
    #: solver to 1e-8.
    steps_per_efold = 300.0 / 9.21

    def initialize(self) -> None:

        unknown = set(self.info) - {"k"}

        if unknown:
            raise ComponentError(f"{self.name}: unknown option(s) {sorted(unknown)}.")

        self.k = float(self.info.get("k", 0.1))

    def accepts(self, name: str) -> bool:
        return name == "sigma8"

    def get_default_params(self) -> dict:
        return {}

    def get_requirements(self) -> dict:
        return {"expansion": None, "background_densities": None}

    def get_required_params(self) -> list[str]:
        return ["sigma8"]

    def get_can_provide(self) -> list[str]:
        return ["growth_factor", "growth_rate", "sigma8_z", "fsigma8"]

    def get_derived_params(self) -> list[str]:
        return ["S8"]

    # ---------------------------------------------------------

    def calculate(self, state: dict, want_derived: bool = True, sigma8=None):

        expansion = self.provider.get_expansion()
        densities = self.provider.get_background_densities()

        sector = expansion["sector"]
        p = expansion["sector_params"]
        ctx = expansion["context"]
        E2 = expansion["E2"]

        radiation = ctx.radiation

        # Old initial condition without radiation; Meszaros with it, from
        # deep enough that the dark sector is negligible.
        a_init = 1.0e-5 if radiation else 1.0e-4
        N_init = math.log(a_init)

        jumps = sorted(
            -math.log1p(z) for z in expansion["jumps"]
            if 0.0 < z < 1.0 / a_init - 1.0
        )

        edges = [N_init, *jumps, 0.0]

        def lnE2(N):
            return np.log(E2(np.expm1(-np.asarray(N, dtype=float))))

        def coefficients(N, lo, hi):

            z = np.expm1(-N)

            dlnH = 0.5 * _slope(lnE2, N, lo, hi)

            growth = GrowthContext(
                ctx=ctx, E2=E2,
                dlnH_dN=lambda x: 0.5 * _slope(lnE2, -np.log1p(x), lo, hi),
            )

            e2 = E2(z)

            Om = sector.clustering_matter(z, growth, **p) / e2

            mu = sector.mu(z, self.k, growth, **p)

            if mu is None:
                mu = 1.0

            friction = 2.0 + dlnH
            source = 1.5 * Om * mu

            if sector.matter_exchange(np.array([0.0]), growth, **p) is not None:

                def psi_of_N(x):
                    return sector.matter_exchange(np.expm1(-x), growth, **p)

                psi = psi_of_N(N)
                dpsi = _slope(psi_of_N, N, lo, hi)

                friction = friction + psi
                source = source - psi * (2.0 + dlnH) - dpsi

            return friction, source

        if radiation:

            y = float(ctx.rho_cb(1.0 / a_init - 1.0) / ctx.rho_rel(1.0 / a_init - 1.0))
            start = np.array([1.0 + 1.5 * y, 1.5 * y])

        else:

            start = np.array([a_init, a_init])

        D_coefficients, P_coefficients, xs = [], [], []

        inset = 1.0e-10

        for index, (N0, N1) in enumerate(zip(edges[:-1], edges[1:])):

            n = max(16, int(round(self.steps_per_efold * (N1 - N0))))

            h = (N1 - N0) / n

            fine = N0 + 0.5 * h * np.arange(2 * n + 1)

            evaluate = fine.copy()

            if index > 0:
                evaluate[0] += inset

            if index < len(edges) - 2:
                evaluate[-1] -= inset

            friction, source = coefficients(evaluate, N0, N1)

            if not (np.all(np.isfinite(friction)) and np.all(np.isfinite(source))):
                return False

            D, P = GrowthCalculator._step_by_prefix_product(
                friction[0:-1:2], source[0:-1:2],
                friction[1::2], source[1::2],
                friction[2::2], source[2::2],
                h, n, start=start,
            )

            nodes = fine[::2]
            second = -friction[::2] * P + source[::2] * D

            D_coefficients.append(hermite_spline(nodes, D, P).c)
            P_coefficients.append(hermite_spline(nodes, P, second).c)
            xs.append(nodes if index == 0 else nodes[1:])

            if index < len(edges) - 2:

                # Continuity of H dD/dN across the jump.
                before = math.sqrt(float(E2(np.expm1(-(N1 - inset)))))
                after = math.sqrt(float(E2(np.expm1(-(N1 + inset)))))

                start = np.array([D[-1], P[-1] * before / after])

        x = np.concatenate(xs)

        D_spline = PPoly.construct_fast(np.concatenate(D_coefficients, axis=1), x)
        P_spline = PPoly.construct_fast(np.concatenate(P_coefficients, axis=1), x)

        D0 = float(D_spline(0.0))

        state.update(D_spline=D_spline, P_spline=P_spline, D0=D0, sigma8=float(sigma8))

        if want_derived:
            state["derived"] = {
                "S8": float(sigma8) * math.sqrt(densities["Omega_m"] / 0.3),
            }

        return True

    # ---------------------------------------------------------

    def get_growth_factor(self, z):

        state = self.current_state
        N = -np.log1p(np.asarray(z, dtype=float))

        return state["D_spline"](N) / state["D0"]

    def get_growth_rate(self, z):

        state = self.current_state
        N = -np.log1p(np.asarray(z, dtype=float))

        return state["P_spline"](N) / state["D_spline"](N)

    def get_sigma8_z(self, z):
        return self.current_state["sigma8"] * self.get_growth_factor(z)

    def get_fsigma8(self, z):
        return self.get_growth_rate(z) * self.get_sigma8_z(z)
