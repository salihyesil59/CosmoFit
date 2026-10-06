"""
Errors that mean a model cannot answer *at all*.

:meth:`~stats.posterior.LogPosterior.chi2` deliberately swallows
``ValueError``, ``RuntimeError`` and ``FloatingPointError`` into an
infinite chi-squared. That is right for a *parameter* the model
cannot represent -- a negative square root, a runaway field, a
Boltzmann code refusing an extreme point -- because a sampler that
merely proposed such a point should not crash, and treating it as
the worst possible fit is exactly what excluding it means.

It is wrong for a *configuration* the model cannot represent. A
scalar-field action asked for ``E(z)`` beyond the redshift its
initial conditions were set at will fail at every parameter value,
not some of them; swallowing that leaves a fit whose chi-squared is
infinite everywhere, with nothing to say why. The error is about
how the model was built, and the fix is to build it differently.

So this exception derives from ``Exception`` rather than from any
of the three caught there, and reaches the caller.
"""

from __future__ import annotations


class ModelConfigurationError(Exception):
    """
    The model cannot answer this question at any parameter values.

    Raised where the obstacle is a choice made when the model was
    constructed rather than the point being evaluated -- so unlike
    an ordinary failure it is not turned into an infinite
    chi-squared, and a fit stops with the reason instead of running
    to completion having learned nothing.
    """
