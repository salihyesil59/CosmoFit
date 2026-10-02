"""
The new core: theories, likelihoods and samplers assembled from an
input.

**Status: phase 1 of the 2.0 rewrite.** The core runs every model and
dataset the library has, through the wrappers in :mod:`core.legacy`,
with the ``evaluate`` and ``minimize`` samplers. ``Fitter`` remains the
complete interface for now -- MCMC, nested sampling, chains and plots
move here in later phases.

::

    from CosmoFit.core import run

    info, sampler = run('''
    theory:
      legacy_cosmology: {model: LCDM, compute_rd: true}
    likelihood:
      desi:
      omega_b:
    params:
      H0:      {prior: {min: 50, max: 90}, latex: H_0}
      Omega_m: {prior: {min: 0.1, max: 0.6}}
      Omega_b: {prior: {dist: norm, loc: 0.049, scale: 0.005, min: 0.03, max: 0.07}}
    sampler:
      minimize: {starts: 3}
    ''')

    sampler.products()["point"]

What each part is:

* :mod:`core.priors` -- normalized priors (uniform, Gaussian and
  truncated Gaussian, log-uniform).
* :mod:`core.parameters` -- sampled, fixed, dependent and derived
  parameters.
* :mod:`core.component` -- the ``Theory``/``Likelihood`` base classes,
  requirements and the parameter-keyed cache.
* :mod:`core.model` -- assembly and the log-posterior.
* :mod:`core.registry` -- names to classes: built-ins, import paths,
  entry points.
* :mod:`core.info` -- the input, as a dict or YAML.
"""

from .component import Component, ComponentError, Likelihood, Provider, Theory
from .info import load_info, validate_info
from .model import PosteriorPoint, Model, get_model
from .parameters import ParameterSet, ParamSpec, parse_param
from .priors import Gaussian, LogUniform, Prior, Uniform, make_prior
from .registry import resolve
from .run import run

__all__ = [
    "Component",
    "ComponentError",
    "Gaussian",
    "Likelihood",
    "PosteriorPoint",
    "LogUniform",
    "Model",
    "ParamSpec",
    "ParameterSet",
    "Prior",
    "Provider",
    "Theory",
    "Uniform",
    "get_model",
    "load_info",
    "make_prior",
    "parse_param",
    "resolve",
    "run",
    "validate_info",
]
