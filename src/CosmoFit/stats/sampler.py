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

import time
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
        callback=None,
        pool=None,
    ):
        """
        Initialize walkers around ``theta0`` and run the sampler.

        Parameters
        ----------
        logpost : callable
            Log-posterior function of a single ``theta`` vector,
            e.g. a :class:`~stats.posterior.LogPosterior`. Must be
            picklable if ``pool`` is given.

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
            Show an emcee (tqdm) progress bar. Ignored if
            ``callback`` is given.

        callback : callable(step, nsteps, elapsed), optional
            Called after every completed step with the step index
            (1-based), the total step count, and elapsed wall time
            in seconds -- e.g. to drive a GUI progress bar. Runs
            the sampler step-by-step instead of in one call, which
            is slightly slower than ``progress=True`` for very fast
            likelihoods, but negligible next to typical per-step
            cost.

        pool : optional
            Passed straight through to ``emcee.EnsembleSampler`` to
            evaluate walkers in parallel, e.g. a
            ``multiprocessing.Pool``. Not built here -- see
            :meth:`~stats.fitter.Fitter.run_mcmc`'s ``n_processes``,
            which builds one correctly (naively pairing a raw
            ``multiprocessing.Pool`` with a heavy, data-carrying
            ``logpost`` re-sends that data on every step, which is
            usually *slower* than a single process).

        Returns
        -------
        emcee.EnsembleSampler
            Already run for ``nsteps`` steps.
        """

        import emcee

        ndim = prior.ndim

        # emcee doesn't enforce this itself, but with fewer walkers
        # than free parameters the initial walker cloud can't span
        # the parameter space -- it ends up linearly dependent, which
        # surfaces later as emcee's context-free "large condition
        # number" error. `nwalkers >= 2 * ndim` is emcee's own
        # documented recommendation; catch a violation here instead.
        if nwalkers < 2 * ndim:
            raise ValueError(
                f"nwalkers={nwalkers} is too small for {ndim} free "
                f"parameter(s); emcee needs at least 2x as many "
                f"walkers as free parameters (>= {2 * ndim} here) to "
                f"initialize a non-degenerate walker cloud."
            )

        # `theta0` outside the prior gets clipped to the boundary a
        # few lines down (`np.clip`), which -- like zero scatter --
        # collapses every walker onto the same edge value and trips
        # the same opaque emcee error. Name the offender instead.
        outside = [
            f"{prior.names[i]}={theta0[i]:g} (bounds "
            f"[{prior.lower[i]:g}, {prior.upper[i]:g}])"
            for i in range(ndim)
            if not (prior.lower[i] <= theta0[i] <= prior.upper[i])
        ]
        if outside:
            raise ValueError(
                f"Initial value outside its prior bounds for "
                f"parameter(s): {'; '.join(outside)}. Fix `initial=` "
                f"(or the GUI's Initial column) for these."
            )

        if initial_scatter is None:
            scatter = 0.01 * (prior.upper - prior.lower)
        else:
            scatter = np.array(
                [initial_scatter[n] for n in prior.names],
                dtype=float,
            )

        # A parameter with zero (or negative) scatter puts every
        # walker at the exact same value along that axis, which
        # makes the initial walker cloud linearly dependent -- emcee
        # then refuses to start, with a "large condition number"
        # error that doesn't say which parameter caused it. Usually
        # means that parameter's prior bounds are equal (e.g. a
        # `bounds=` override, or the GUI's Lower/Upper columns, with
        # lower == upper); catch it here with a message that does.
        degenerate = [
            prior.names[i] for i in range(ndim) if not scatter[i] > 0
        ]
        if degenerate:
            raise ValueError(
                f"Zero (or negative) initial walker scatter for "
                f"parameter(s) {degenerate}. This usually means "
                f"their prior bounds have lower == upper (check "
                f"`bounds=`, or the GUI's Lower/Upper columns) -- "
                f"widen them, or pass an explicit `initial_scatter=` "
                f"for these parameter(s)."
            )

        rng = np.random.default_rng(seed)

        pos = theta0 + scatter * rng.normal(size=(nwalkers, ndim))

        # Keep the initial walker positions safely inside the prior.
        eps = 1e-6 * (prior.upper - prior.lower)
        pos = np.clip(pos, prior.lower + eps, prior.upper - eps)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, logpost, moves=self.moves, pool=pool,
        )

        # `EnsembleSampler` otherwise seeds its own internal proposal
        # RNG from OS entropy (`np.random.mtrand.RandomState()` with
        # no argument, in emcee's own `__init__`) -- `seed` above
        # only fixes the *initial walker positions*, not the
        # step-by-step move/acceptance randomness that actually
        # drives the chain. Without this, `run_mcmc(..., seed=42)`
        # produces a different chain every time it's called, single-
        # or multi-process alike (the pool only ever evaluates
        # log-probabilities in parallel; the proposal RNG always
        # lives in this process).
        sampler.random_state = np.random.RandomState(seed).get_state()

        if callback is None:
            sampler.run_mcmc(pos, nsteps, progress=progress)
        else:
            start = time.monotonic()
            for step, _ in enumerate(
                sampler.sample(pos, iterations=nsteps, store=True), start=1,
            ):
                callback(step, nsteps, time.monotonic() - start)

        return sampler
