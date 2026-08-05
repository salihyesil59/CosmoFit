"""
MCMC sampling backends.

Splits "how to explore a posterior" out of :class:`~stats.fitter.Fitter`
(which owns "what the posterior *is*" -- a model, data, and a prior)
and into its own, swappable piece. :class:`EnsembleSampler` (emcee's
affine-invariant ensemble sampler) is the only backend implemented
today and is what ``Fitter.run_mcmc`` uses internally, but any object
following :class:`BaseSampler` can be dropped in without touching
``Fitter``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ============================================================
# Sampler interface
# ============================================================

class BaseSampler(ABC):
    """
    Interface a sampling backend must implement to be usable by
    :class:`~stats.fitter.Fitter`.
    """

    @abstractmethod
    def run(self, logpost, prior, theta0, **kwargs):
        """
        Run the sampler and return the backend-native result
        object (e.g. an ``emcee.EnsembleSampler`` for
        :class:`EnsembleSampler`).
        """

        raise NotImplementedError


# ============================================================
# emcee ensemble sampler
# ============================================================

class EnsembleSampler(BaseSampler):
    """
    ``emcee.EnsembleSampler`` backend: affine-invariant ensemble
    MCMC, the default (and currently only) way ``Fitter.run_mcmc``
    explores a posterior.

    Parameters
    ----------
    moves : emcee move or list of (move, weight), optional
        Passed straight through to ``emcee.EnsembleSampler``. Lets
        the proposal strategy be swapped out (e.g.
        ``emcee.moves.DEMove()`` for strongly correlated
        posteriors) without changing anything else about the fit.
        Defaults to emcee's own default (``StretchMove``).
    """

    def __init__(self, moves=None):

        self.moves = moves

    # ------------------------------------------------------------

    def run(
        self,
        logpost,
        prior,
        theta0,
        nwalkers: int = 48,
        nsteps: int = 6000,
        initial_scatter=None,
        seed: int = 42,
        progress: bool = True,
    ):
        """
        Initialize walkers around ``theta0`` and run the sampler.

        Parameters
        ----------
        logpost : callable
            Log-posterior function of a single ``theta`` vector,
            e.g. a :class:`~stats.posterior.LogPosterior`.

        prior : stats.priors.UniformPrior
            Defines the parameter order and bounds walkers are
            initialized and clipped within.

        theta0 : ndarray
            Center of the initial walker cloud.

        nwalkers, nsteps : int
            Standard emcee settings. ``nwalkers`` should be at
            least ``2 * prior.ndim``.

        initial_scatter : dict[str, float], optional
            Per-parameter Gaussian scatter used to initialize the
            walkers around ``theta0``. Defaults to 1% of each
            parameter's prior width.

        seed : int
            Seed for the walker initialization RNG.

        progress : bool
            Show an emcee progress bar.

        Returns
        -------
        emcee.EnsembleSampler
            Already run for ``nsteps`` steps.
        """

        import emcee

        ndim = prior.ndim

        if initial_scatter is None:
            scatter = 0.01 * (prior.upper - prior.lower)
        else:
            scatter = np.array(
                [initial_scatter[n] for n in prior.names],
                dtype=float,
            )

        rng = np.random.default_rng(seed)

        pos = theta0 + scatter * rng.normal(size=(nwalkers, ndim))

        # Keep the initial walker positions safely inside the prior.
        eps = 1e-6 * (prior.upper - prior.lower)
        pos = np.clip(pos, prior.lower + eps, prior.upper - eps)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost, moves=self.moves,
        )

        sampler.run_mcmc(pos, nsteps, progress=progress)

        return sampler
