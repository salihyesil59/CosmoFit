"""
Posterior probability.

Glues a JointLikelihood, a prior, and a mutable set of
cosmological parameters together into a single callable
``log_probability(theta)`` function of the kind samplers like
``emcee`` expect.
"""

from __future__ import annotations

import warnings

import numpy as np

from CosmoFit.cosmology.boltzmann import BoltzmannError


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

        #: How many evaluations were rejected because the Boltzmann
        #: solver failed, rather than because the parameters were
        #: unphysical. See :meth:`solver_failures`.
        self._solver_failures = 0

        #: The first point it happened at, kept so the count comes
        #: with somewhere to start looking.
        self._first_solver_failure = None

    # ---------------------------------------------------------

    @property
    def solver_failures(self) -> int:
        """
        Evaluations rejected because the Boltzmann code failed.

        These are **not** the same as evaluations rejected for being
        unphysical, and the difference matters. An unphysical point
        should be rejected: that is the prior and the model doing
        their job. A point where CAMB failed at parameters that are
        perfectly reasonable should not be, and every one of them
        distorts the posterior a little, in a direction nothing in
        the output records.

        Both used to land in the same ``-inf`` with nothing to tell
        them apart. This counts the second kind so a run can say how
        much of it happened. A handful in a long chain is noise; a
        large fraction means the result should not be trusted until
        the cause is found.
        """

        return self._solver_failures

    @property
    def first_solver_failure(self):
        """
        The parameter vector of the first solver failure, or None.
        """

        return self._first_solver_failure

    # ---------------------------------------------------------

    def _record_solver_failure(self, theta, exc) -> None:
        """
        Count a Boltzmann failure, and say so the first time.

        Warning once rather than every time is deliberate: a chain
        can evaluate this millions of times, and a warning per
        rejection would bury the run. The count is the number to
        read afterwards.
        """

        self._solver_failures += 1

        if self._solver_failures == 1:

            self._first_solver_failure = np.array(theta, dtype=float, copy=True)

            warnings.warn(
                "The Boltzmann solver failed at a point the sampler "
                "proposed, and that point has been rejected. This is "
                "not the same as rejecting an unphysical point: the "
                "parameters may be perfectly reasonable and the "
                "failure the solver's. Rejections of this kind bias "
                "a posterior in a way nothing else in the output "
                "records, so the running total is kept in "
                "`solver_failures` -- check it before trusting the "
                f"result. First failure: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

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

        if not np.all(np.isfinite(theta)):
            return -np.inf

        try:
            self._apply(theta)
            chi2 = self.joint.chi2()
        except BoltzmannError as exc:
            # A solver failure, not a statement about the parameters.
            # Rejected like anything else so a chain keeps running,
            # but counted, because it is not the same thing.
            self._record_solver_failure(theta, exc)
            return -np.inf
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

        Two guards, and the first one exists because the second was
        not enough. A ``theta`` that is itself **not finite** is
        rejected before it reaches the cosmology at all: L-BFGS-B
        started at a point where chi2 is already inf computes a
        finite-difference gradient of ``inf - inf = nan``, takes a
        nan search direction, and evaluates the objective at
        ``[nan, nan, nan]``. Writing that into the parameters builds
        an interpolation table full of nan, and the interpolator
        raises rather than returning anything -- from inside
        ``refresh()``, which used to sit *outside* the try below.

        So the try now covers ``_apply`` as well. A cosmology that
        cannot even be constructed is as excluded as one that fits
        badly, and neither should be able to crash a sampler that
        merely proposed it.
        """

        if not np.all(np.isfinite(theta)):
            return np.inf

        try:
            self._apply(theta)
            chi2 = self.joint.chi2()
        except BoltzmannError as exc:
            self._record_solver_failure(theta, exc)
            return np.inf
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
