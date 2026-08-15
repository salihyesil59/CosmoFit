"""
Statistical tools: priors, posteriors, the high-level Fitter, its
MCMC sampling backend and consolidated result objects, model
comparison (AIC/BIC/LRT), CPL-specific posterior diagnostics
(w(z) bands, w(z)=-1 crossing, Mahalanobis distance from LCDM),
and expansion-history derived posteriors (transition redshift z_t,
q0) that apply to every model.
"""

from .priors import UniformPrior
from .posterior import LogPosterior
from .sampler import BaseSampler, EnsembleSampler
from .results import FitResult, BestFitResult, MCMCResult
from .fitter import Fitter, DATASET_REGISTRY
from . import model_comparison
from . import cpl_diagnostics
from . import derived

__all__ = [
    "UniformPrior",
    "LogPosterior",
    "BaseSampler",
    "EnsembleSampler",
    "FitResult",
    "BestFitResult",
    "MCMCResult",
    "Fitter",
    "DATASET_REGISTRY",
    "model_comparison",
    "cpl_diagnostics",
    "derived",
]