"""
CosmoFit: a modular Python library for cosmological parameter
estimation.

Unified public API -- everything most users need is importable
directly from the top-level package:

>>> from CosmoFit import CPL, Fitter
>>>
>>> fit = Fitter(
...     model=CPL,
...     datasets=["cc", "desi", "pantheon"],
...     free_params=["H0", "Omega_m", "w0", "wa"],
...     initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "wa": 0.0,
...              "rd": 147.1},
... )
>>> fit.run_mcmc(nwalkers=48, nsteps=6000, burnin=1000)
>>> fit.best_fit()
>>> fit.summary()
>>> fit.plots.corner()

Pass ``save="chains/cpl.h5"`` to ``run_mcmc`` and the chain is
written as it is sampled, then reused instead of re-sampled the
next time -- or reopened in a later session with
``Fitter.from_chain("chains/cpl.h5")``. See ``CosmoFit.stats.chains``.

The underlying subpackages (``CosmoFit.cosmology``, ``CosmoFit.data``,
``CosmoFit.likelihoods``, ``CosmoFit.stats``, ``CosmoFit.plots``) are
still available for anything not re-exported here -- e.g.
``CosmoFit.stats.model_comparison``, ``CosmoFit.stats.cpl_diagnostics``,
or ``CosmoFit.data.loader.load_pantheon`` for direct dataset access.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# ============================================================
# Cosmological models and parameters
# ============================================================

from CosmoFit.cosmology import (
    Cosmology,
    CosmologyParameters,
    constants,
    LCDM,
    WCDM,
    CPL,
    JBP,
    BA,
    LogarithmicDE,
    PEDE,
    GEDE,
    LsCDM,
    GCG,
    IDE,
    RunningVacuum,
    Cardassian,
    DGP,
    FQExponential,
    FRTLinear,
    FRHuSawicki,
    define_model,
    model_from_expression,
)

# ============================================================
# Likelihoods
# ============================================================

from CosmoFit.likelihoods import (
    BaseLikelihood,
    CCLikelihood,
    DESILikelihood,
    SDSSBAOLikelihood,
    BAOLowZLikelihood,
    PantheonLikelihood,
    DESSN5YRLikelihood,
    Union3Likelihood,
    PlanckLikelihood,
    PlanckLiteLikelihood,
    H0Likelihood,
    OmegaBLikelihood,
    TauLikelihood,
    FSigma8Likelihood,
    S8Likelihood,
    JointLikelihood,
)

# ============================================================
# Fitting and plotting
# ============================================================

from CosmoFit.stats.fitter import Fitter
from CosmoFit.stats.sampler import BaseSampler, EnsembleSampler
from CosmoFit.stats.results import FitResult, BestFitResult, MCMCResult
from CosmoFit.stats.chains import ChainFile, StoredSampler, open_chain, chain_info
from CosmoFit.plots import FitPlotter

# ============================================================
# Dataset discovery
# ============================================================

from CosmoFit.data.loader import available_datasets, available_versions

# ============================================================
# Version
# ============================================================

try:
    __version__ = version("cosmofit")
except PackageNotFoundError:
    # Package not installed (e.g. running from a source checkout
    # without `pip install -e .`).
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # Cosmology
    "Cosmology",
    "CosmologyParameters",
    "constants",
    "LCDM",
    "WCDM",
    "CPL",
    "JBP",
    "BA",
    "LogarithmicDE",
    "PEDE",
    "GEDE",
    "LsCDM",
    "GCG",
    "IDE",
    "RunningVacuum",
    "Cardassian",
    "DGP",
    "FQExponential",
    "FRTLinear",
    "FRHuSawicki",
    "define_model",
    "model_from_expression",
    # Likelihoods
    "BaseLikelihood",
    "CCLikelihood",
    "DESILikelihood",
    "SDSSBAOLikelihood",
    "BAOLowZLikelihood",
    "PantheonLikelihood",
    "DESSN5YRLikelihood",
    "Union3Likelihood",
    "PlanckLikelihood",
    "PlanckLiteLikelihood",
    "H0Likelihood",
    "OmegaBLikelihood",
    "TauLikelihood",
    "FSigma8Likelihood",
    "S8Likelihood",
    "JointLikelihood",
    # Fitting / plotting
    "Fitter",
    "BaseSampler",
    "EnsembleSampler",
    "FitResult",
    "BestFitResult",
    "MCMCResult",
    "FitPlotter",
    # Saved chains
    "ChainFile",
    "StoredSampler",
    "open_chain",
    "chain_info",
    # Datasets
    "available_datasets",
    "available_versions",
]
