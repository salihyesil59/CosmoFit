"""
Cubic Hermite interpolation without the construction overhead.

``scipy.interpolate.CubicHermiteSpline`` validates its inputs,
sorts them, normalizes axes and allocates working arrays before
assembling the piecewise coefficients. That is the right default
for a public API and the wrong one for a table rebuilt on every
MCMC step from a grid this library generated itself.

The coefficients are four lines of algebra, so they are written
out here and handed to :meth:`scipy.interpolate.PPoly.construct_fast`,
which skips the checks. Measured on a 505-point grid: **43 to 13
microseconds**, with coefficients identical to what
``CubicHermiteSpline`` produces.

What is given up is the error message: a non-monotonic ``x`` or a
non-finite ``y`` produces nonsense rather than an exception. Both
callers here build ``x`` with ``numpy.linspace`` and get ``y`` from
a cosmology whose non-finite values are already refused upstream
(see :meth:`stats.posterior.LogPosterior.chi2`), and a NaN that did
slip through would propagate to a NaN ``chi2`` and be rejected
there rather than crashing.
"""

from __future__ import annotations

import numpy as np

from scipy.interpolate import PPoly


def hermite_spline(x, y, dydx, extrapolate=True):
    """
    Piecewise cubic through ``(x, y)`` with prescribed slopes.

    Parameters
    ----------
    x : ndarray
        Strictly increasing nodes. Not checked.
    y, dydx : ndarray
        Values and first derivatives at those nodes.
    extrapolate : bool, optional
        Whether to evaluate outside ``[x[0], x[-1]]``. ``False``
        returns NaN there, which is how the distance table signals
        a redshift beyond the one it was built for -- a silently
        extrapolated cubic would be worse than no answer.

    Returns
    -------
    scipy.interpolate.PPoly
        The same object ``CubicHermiteSpline(x, y, dydx)`` would
        return, built directly.
    """

    h = np.diff(x)

    delta = np.diff(y)

    slope = delta / h

    coefficients = np.empty((4, len(x) - 1))

    coefficients[0] = (dydx[:-1] + dydx[1:] - 2.0 * slope) / h ** 2

    coefficients[1] = (3.0 * slope - 2.0 * dydx[:-1] - dydx[1:]) / h

    coefficients[2] = dydx[:-1]

    coefficients[3] = y[:-1]

    return PPoly.construct_fast(

        coefficients,

        np.asarray(x, dtype=float),

        extrapolate=extrapolate,

    )
