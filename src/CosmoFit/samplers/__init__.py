"""
Samplers of the new core (see :mod:`CosmoFit.core`).

``evaluate``
    The model at one point: log-prior, every chi2, every derived value.
``minimize``
    The maximum of the posterior or the likelihood, in scaled
    coordinates, from one or more starts.
``profile``
    A profile likelihood: one parameter fixed at each of a list of
    values, the others minimized over.
``fisher``
    The curvature at the best fit, with steps chosen by the likelihood,
    and its inverse as a Gaussian approximation to the posterior.
``mcmc``
    Adaptive Metropolis-Hastings, stopped by the Gelman-Rubin ``R - 1``,
    writing getdist-format chains.
``emcee``
    emcee's affine-invariant ensemble, stopped by the autocorrelation
    time.
``nested``
    dynesty's nested sampling: the evidence, and weighted samples.
"""

from .base import Sampler
from .emcee import Emcee
from .evaluate import Evaluate
from .fisher import Fisher
from .mcmc import MCMC
from .minimize import Minimize
from .nested import Nested
from .profile import Profile

__all__ = [
    "Sampler", "Evaluate", "Minimize", "Profile", "Fisher", "MCMC", "Emcee", "Nested",
]
