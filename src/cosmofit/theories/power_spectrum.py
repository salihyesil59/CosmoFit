"""
A matter power spectrum ``P(z, k)``, interpolated from a grid.

What a Boltzmann theory returns for ``Pk_interpolator``. It is built
here, from the grid, rather than handed over from the Boltzmann code,
so that a likelihood reads the same object whichever code computed
the spectrum.

Units are those of the Boltzmann codes' own outputs, and of cobaya:
``k`` in 1/Mpc and ``P`` in Mpc^3 -- no factors of ``h``.

Outside the grid it raises instead of extrapolating. A spline carried
past the last ``k`` it was given returns numbers that look like a
spectrum; a likelihood integrating to a higher ``k`` than it asked for
would never notice.
"""

from __future__ import annotations

import numpy as np

from scipy.interpolate import RectBivariateSpline, make_interp_spline


__all__ = ["PowerSpectrumInterpolator"]


#: Relative slack on the grid's edges, for a ``z`` or ``k`` that is the
#: grid's own end point after a round trip through ``log``.
_EDGE = 1.0e-10


class PowerSpectrumInterpolator:
    """
    ``P(z, k)`` from a grid ``P[z_i, k_j]``.

    Interpolated in ``ln k`` -- and in ``ln P`` when the spectrum is
    positive everywhere, as an auto-spectrum is; a cross-spectrum that
    changes sign is interpolated in ``P``. Cubic in both directions,
    or of the highest degree the number of redshifts allows (linear in
    ``z`` for two, none for one).

    Attributes ``z``, ``k`` and ``values`` (``P[z, k]``) are the grid;
    ``zmin``, ``zmax``, ``kmin`` and ``kmax`` its extent.
    """

    def __init__(self, z, k, P):

        z = np.asarray(z, dtype=float)
        k = np.asarray(k, dtype=float)
        P = np.asarray(P, dtype=float)

        if P.shape != (z.size, k.size):
            raise ValueError(f"P has shape {P.shape}, not (len(z), len(k)) = {(z.size, k.size)}.")

        if np.any(np.diff(z) <= 0) or np.any(np.diff(k) <= 0) or np.any(k <= 0):
            raise ValueError("z and k must be increasing, and k positive.")

        self.z, self.k, self.values = z, k, P

        self.islog = bool(np.all(P > 0))

        values = np.log(P) if self.islog else P

        if z.size == 1:
            self._spline = make_interp_spline(np.log(k), values[0], k=3)
        else:
            self._spline = RectBivariateSpline(
                z, np.log(k), values, kx=min(3, z.size - 1), ky=3, s=0,
            )

    @property
    def zmin(self) -> float:
        return float(self.z[0])

    @property
    def zmax(self) -> float:
        return float(self.z[-1])

    @property
    def kmin(self) -> float:
        return float(self.k[0])

    @property
    def kmax(self) -> float:
        return float(self.k[-1])

    # ---------------------------------------------------------

    def _check(self, z, k) -> None:

        def outside(x, lo, hi):
            return np.any(x < lo * (1 - _EDGE) - _EDGE) or np.any(x > hi * (1 + _EDGE) + _EDGE)

        if outside(z, self.zmin, self.zmax):
            raise ValueError(
                f"z = {np.min(z):g}..{np.max(z):g} is outside the spectrum's "
                f"{self.zmin:g}..{self.zmax:g}: ask the theory for these "
                f"redshifts (option 'z')."
            )

        if outside(k, self.kmin, self.kmax):
            raise ValueError(
                f"k = {np.min(k):g}..{np.max(k):g} 1/Mpc is outside the "
                f"spectrum's {self.kmin:g}..{self.kmax:g}: ask the theory "
                f"for a larger 'k_max'."
            )

    def logP(self, z, k, grid: bool = False):
        """``ln P``; only for a spectrum that is positive everywhere."""

        if not self.islog:
            raise ValueError("This spectrum changes sign; it has no logarithm.")

        return self._evaluate(z, k, grid)

    def P(self, z, k, grid: bool = False):
        """
        ``P(z, k)`` [Mpc^3], ``k`` in 1/Mpc. ``z`` and ``k`` broadcast
        against each other; with ``grid=True`` the result is the outer
        grid instead, of shape ``(len(z), len(k))``.
        """

        values = self._evaluate(z, k, grid)

        return np.exp(values) if self.islog else values

    __call__ = P

    def _evaluate(self, z, k, grid):

        z = np.asarray(z, dtype=float)
        k = np.asarray(k, dtype=float)

        self._check(z, k)

        if grid:
            z, k = np.meshgrid(np.atleast_1d(z), np.atleast_1d(k), indexing="ij")

        z, k = np.broadcast_arrays(z, k)

        x = np.log(k)

        if self.z.size == 1:
            return self._spline(x)

        return self._spline.ev(z, x)
