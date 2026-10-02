"""
What a model adds to the standard fluids, and the context it is given.

:class:`~theories.background.Background` owns the standard content of
the universe -- cold matter, photons, neutrinos -- and curvature. A
*dark sector* turns that into an expansion history: it receives an
:class:`ExpansionContext` holding the standard fluids' density and
pressure as functions of redshift, and returns ``E(z)^2``.

How it does that is the model's business:

* a dark energy on top of general relativity adds ``Omega_de f(z)``
  (:mod:`theories.dark_energy`);
* a modified Friedmann equation maps the fluids' density onto ``E^2``
  some other way (:mod:`theories.modified_gravity`);
* a holographic or Ricci dark energy, or a running vacuum, solves an
  ODE for it (:mod:`theories.holographic`, :mod:`theories.running_vacuum`).

Every one of them sees *all* the standard fluids, radiation included,
through the context -- which is what lets one background carry
photons and neutrinos for every model without each model writing them
out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scipy.integrate import cumulative_simpson


__all__ = ["DarkSector", "ExpansionContext", "GrowthContext", "solve_linear", "N_MIN"]


#: How far back (``ln a``) ODE-based sectors tabulate their solution:
#: ``a = 1e-13``, below everything the early-universe integrals reach.
N_MIN = -30.0


@dataclass
class ExpansionContext:
    """
    The standard content of the universe, in units of today's critical
    density.

    Attributes
    ----------
    rho_std, p_std : callable(z)
        Density and pressure of all standard fluids together.
    rho_cb, rho_rel, p_rel : callable(z)
        Cold matter alone; photons and neutrinos alone.
    rho_std0, rho_rel0, Omega_cb : float
        Their values today.
    Omega_k : float
    radiation : bool
        Whether photons and neutrinos are present at all.
    H0 : float
        Only for wrapped old-style models, whose ``E(z)`` may read it.
    """

    rho_std: Callable
    p_std: Callable
    rho_cb: Callable
    rho_rel: Callable
    p_rel: Callable
    rho_std0: float
    rho_rel0: float
    Omega_cb: float
    Omega_k: float
    radiation: bool
    H0: float = 70.0

    def curvature(self, z):
        """``Omega_k (1+z)^2``."""

        return self.Omega_k * (1.0 + np.asarray(z, dtype=float)) ** 2


class DarkSector:
    """
    A model's contribution to the expansion.

    Declared through class attributes: ``name``; ``params``, the
    parameters it requires; ``defaults``, optional parameters with
    their default values; ``flat_only``, defined for ``Omega_k = 0``
    only; ``radiation_ok``, has a form with radiation (``False`` for a
    wrapped model that only ever had ``E(z)`` without it); and
    ``derives_matter``, fixes the cold-matter density itself, so the
    background does not take ``Omega_m`` (or ``omega_cdm``).
    """

    name = "dark_sector"
    params: tuple = ()
    defaults: dict = {}
    flat_only = False
    radiation_ok = True
    derives_matter = False

    def solve(self, ctx: ExpansionContext, **p) -> Callable | None:
        """
        ``E(z)^2`` as a function, for this context and these parameters,
        or ``None`` where the model has no solution.
        """

        raise NotImplementedError

    def matter_density(self, make_context: Callable, **p) -> float:
        """
        For ``derives_matter`` sectors: today's cold-matter density
        ``Omega_cb``, given ``make_context(Omega_cb)``.
        """

        raise NotImplementedError

    def jumps(self, **p) -> tuple:
        """Redshifts at which ``E(z)`` is discontinuous."""

        return ()

    # ---------------------------------------------------------
    # Growth of structure (see theories.growth)
    # ---------------------------------------------------------

    def clustering_matter(self, z, growth: "GrowthContext", **p):
        """
        Density of the matter that clusters, in units of today's
        critical density. Cold matter by default: massive neutrinos
        free-stream on the scales growth data measure.
        """

        return growth.ctx.rho_cb(z)

    def matter_exchange(self, z, growth: "GrowthContext", **p):
        """
        ``psi = Q / (H rho_m)``, energy gained by clustering matter per
        Hubble time -- ``None`` for conserved matter.
        """

        return None

    def mu(self, z, k: float, growth: "GrowthContext", **p):
        """
        ``G_eff / G_N`` at wavenumber ``k`` [h/Mpc] -- ``None`` for
        general relativity.
        """

        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass
class GrowthContext:
    """
    What a dark sector's growth hooks may read: the standard fluids,
    the expansion, and its logarithmic slope.

    Attributes
    ----------
    ctx : ExpansionContext
    E2 : callable(z)
    dlnH_dN : callable(z)
        ``d ln H / d ln a``.
    """

    ctx: ExpansionContext
    E2: Callable
    dlnH_dN: Callable


def solve_linear(k: float, source: Callable, y0: float = 1.0,
                 n_min: float = N_MIN, points: int = 30001) -> Callable:
    r"""
    Solve ``dy/dN = k y + s(N)`` from ``y(0) = y0`` back to ``N = n_min``,
    ``N = ln a``, through its integrating factor:

        y(N) = e^{kN} [ y0 + int_0^N e^{-kN'} s(N') dN' ]

    Returns ``y`` as a function of ``z``, interpolated in ``N`` -- a
    cubic spline of ``ln |y|`` would assume a sign; the solution here is
    interpolated directly, on a grid fine enough that the cubic error is
    far below the 1e-9 the rest of the background works to.
    """

    from scipy.interpolate import CubicSpline

    # From N = 0 backwards, so the integral is accumulated from where it
    # is zero. Accumulating from n_min and subtracting the total would
    # take the difference of two numbers as large as e^30 near N = 0.
    N = np.linspace(0.0, n_min, points)

    weight = np.exp(-k * N) * source(N)

    # int_0^N dN' = -int_0^{-N} dt with t = -N' increasing, which is
    # what cumulative_simpson integrates over.
    integral = -cumulative_simpson(weight, x=-N, initial=0.0)

    # y e^{-kN} = y0 + integral: smooth, and what is interpolated.
    spline = CubicSpline(N[::-1], (y0 + integral)[::-1])

    def y_of_z(z):

        n = -np.log1p(np.asarray(z, dtype=float))

        inside = n >= n_min

        out = np.empty_like(n)
        out[inside] = np.exp(k * n[inside]) * spline(n[inside])
        out[~inside] = np.nan

        return out

    return y_of_z

