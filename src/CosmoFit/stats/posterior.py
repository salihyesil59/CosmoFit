"""
Posterior probability.

Glues a JointLikelihood, a prior, and a mutable set of
cosmological parameters together into a single callable
``log_probability(theta)`` function of the kind samplers like
``emcee`` expect.
"""

from __future__ import annotations

import numpy as np


class LogPosterior:
    """
    Callable log-posterior for a subset of "free" cosmological
    parameters, with the rest held fixed.

    Parameters
    ----------
    joint : likelihoods.joint.JointLikelihood
        Joint likelihood (already built from one or more
        ``BaseLikelihood`` instances sharing a single
        ``cosmology`` object).

    cosmology : cosmology.core.base.Cosmology
        The (mutable) cosmology instance the likelihoods above
        were built with. Its ``.params`` will be updated in
        place at every evaluation.

    prior : statistics.priors.UniformPrior
        Prior over the free parameters. ``prior.names`` defines
        which entries of ``theta`` map to which parameter.
    """

    def __init__(self, joint, cosmology, prior):

        self.joint = joint
        self.cosmology = cosmology
        self.prior = prior

    # ---------------------------------------------------------

    @property
    def ndim(self) -> int:
        return self.prior.ndim

    @property
    def names(self):
        return self.prior.names

    # ---------------------------------------------------------

    def _apply(self, theta) -> None:
        """
        Write ``theta`` into the shared parameter object and
        refresh the cosmology's cached distance table.
        """

        kwargs = dict(zip(self.prior.names, theta))

        self.cosmology.params.update(**kwargs)
        self.cosmology.refresh()

    # ---------------------------------------------------------

    def log_prior(self, theta) -> float:

        return self.prior.log_prior(theta)

    # ---------------------------------------------------------

    def log_likelihood(self, theta) -> float:

        self._apply(theta)

        try:
            chi2 = self.joint.chi2()
        except (ValueError, FloatingPointError, RuntimeError):
            return -np.inf

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    # ---------------------------------------------------------

    def chi2(self, theta) -> float:
        """
        Convenience wrapper returning chi2(theta) directly --
        useful for ``scipy.optimize.minimize`` best-fit searches.

        Some models have prior-bounded regions where the background
        is unphysical (e.g. E(z)^2 < 0 for a modified-gravity model
        at extreme coupling values) -- ``log_likelihood`` already
        treats that as -inf log-probability during MCMC; mirror that
        here as +inf chi2 (worst possible fit) rather than letting
        ``scipy.optimize.minimize`` crash on a NaN/exception when
        its search steps into that region.
        """

        self._apply(theta)

        try:
            chi2 = self.joint.chi2()
        except (ValueError, FloatingPointError, RuntimeError):
            return np.inf

        if not np.isfinite(chi2):
            return np.inf

        return chi2

    # ---------------------------------------------------------

    def __call__(self, theta) -> float:
        """
        Full log-posterior, log_prior(theta) + log_likelihood(theta).

        This is the function to hand to ``emcee.EnsembleSampler``.
        """

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta)
