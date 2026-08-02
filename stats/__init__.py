"""
Statistical tools: priors, posteriors, the high-level Fitter,
model comparison (AIC/BIC/LRT), and CPL-specific posterior
diagnostics (w(z) bands, w(z)=-1 crossing, Mahalanobis distance
from LCDM).
"""

from .priors import UniformPrior
from .posterior import LogPosterior
from .fitter import Fitter, DATASET_REGISTRY
from . import model_comparison
from . import cpl_diagnostics

__all__ = [
    "UniformPrior",
    "LogPosterior",
    "Fitter",
    "DATASET_REGISTRY",
    "model_comparison",
    "cpl_diagnostics",
]