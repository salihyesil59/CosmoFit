"""
Samplers of the new core (see :mod:`CosmoFit.core`).

``evaluate``
    The model at one point: log-prior, every chi2, every derived value.
``minimize``
    The maximum of the posterior or the likelihood, in scaled
    coordinates, from one or more starts.

MCMC and nested sampling come in a later phase; until then they are
``Fitter``'s.
"""

from .base import Sampler
from .evaluate import Evaluate
from .minimize import Minimize

__all__ = ["Sampler", "Evaluate", "Minimize"]
