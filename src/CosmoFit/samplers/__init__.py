"""
Samplers of the new core (see :mod:`CosmoFit.core`).

``evaluate``
    The model at one point: log-prior, every chi2, every derived value.
``minimize``
    The maximum of the posterior or the likelihood, in scaled
    coordinates, from one or more starts.
``mcmc``
    Adaptive Metropolis-Hastings, stopped by the Gelman-Rubin ``R - 1``,
    writing getdist-format chains.

Nested sampling comes in a later phase; until then it is ``Fitter``'s.
"""

from .base import Sampler
from .evaluate import Evaluate
from .mcmc import MCMC
from .minimize import Minimize

__all__ = ["Sampler", "Evaluate", "MCMC", "Minimize"]
