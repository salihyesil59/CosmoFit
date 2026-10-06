"""
Small integer powers, written the fast way.

NumPy special-cases ``x ** 2`` and ``x ** 0.5`` into a multiply and
a ``sqrt``, but not ``x ** 3`` -- that goes through the general
``pow``, which is transcendental and roughly six times slower.
Measured on a 505-element array:

=================  ========
expression           time
=================  ========
``x ** 3``          5.02 us
``x * x * x``       0.79 us
``x ** 2``          0.40 us
``x ** -3``         4.87 us
=================  ========

``Omega_m * (1 + z) ** 3`` is the most-evaluated expression in this
library -- it is in every model's ``E(z)``, which the distance
table, the growth coefficients and every likelihood call in turn --
so the difference is worth a helper.

The two forms differ by **one unit in the last place** (2.2e-16
relative, measured), because ``x * x * x`` rounds twice where
``pow`` rounds once. That is thirteen orders of magnitude below the
convergence tolerance of anything here.
"""

from __future__ import annotations


def cube(x):
    """``x ** 3``, without going through ``pow``."""

    return x * x * x


def reciprocal_powers(x):
    """
    ``(1/x, 1/x^2, 1/x^3, 1/x^4)`` from a single division.

    Negative exponents are the same story as positive ones and
    worse: ``a ** -4`` is a ``pow`` call. The early-universe
    expansion rate needs three of these at once, so they are built
    from one reciprocal and three multiplies.
    """

    inverse = 1.0 / x

    inverse_2 = inverse * inverse

    return inverse, inverse_2, inverse_2 * inverse, inverse_2 * inverse_2
