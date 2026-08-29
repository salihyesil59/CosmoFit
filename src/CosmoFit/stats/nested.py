"""
The nested-sampling run itself, and its result object.

Kept apart from :mod:`stats.evidence`, which only compares finished
runs and needs no optional dependency to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NestedResult:
    """
    What a nested-sampling run produced.

    Attributes
    ----------
    log_evidence, log_evidence_error : float
        ``ln Z`` and its uncertainty, which for nested sampling is
        an honest sampling error rather than a scatter estimate.
    samples : ndarray
        Posterior samples, ``(n, ndim)``, already resampled to
        equal weight -- so they can be summarized exactly like MCMC
        output.
    free_params : list of str
    prior_volume : float
        Volume of the uniform prior box the evidence was computed
        against. Reported because ``ln Z`` moves with it, and a
        Bayes factor that does not say which box it used is not a
        reproducible number.
    n_live : int
    n_evaluations : int
    """

    log_evidence: float
    log_evidence_error: float
    samples: np.ndarray
    free_params: list
    prior_volume: float
    n_live: int
    n_evaluations: int
    information: float = field(default=float("nan"))

    def summary(self, percentiles=(2.5, 16, 50, 84, 97.5)) -> dict:
        """
        Per-parameter posterior percentiles, in the same shape
        :meth:`stats.fitter.Fitter.summary` returns.
        """

        out = {}

        for index, name in enumerate(self.free_params):

            column = self.samples[:, index]

            low2, low1, median, high1, high2 = np.percentile(
                column, percentiles,
            )

            out[name] = {
                "median": float(median),
                "mean": float(column.mean()),
                "std": float(column.std(ddof=1)),
                "plus": float(high1 - median),
                "minus": float(median - low1),
                "ci95": (float(low2), float(high2)),
            }

        return out

    def __repr__(self):

        return (
            f"NestedResult(ln Z = {self.log_evidence:.3f} "
            f"+- {self.log_evidence_error:.3f}, "
            f"{len(self.samples)} samples, "
            f"{self.n_evaluations} evaluations)"
        )


def run_nested(
    logpost,
    prior,
    free_params,
    n_live: int = 500,
    dlogz: float = 0.05,
    seed: int | None = 42,
    progress: bool = True,
    **dynesty_kwargs,
) -> NestedResult:
    """
    Integrate the posterior with ``dynesty``.

    Parameters
    ----------
    logpost : LogPosterior
        Only its ``log_likelihood`` is used. The prior enters
        through the unit-cube transform instead, which is what
        makes the integral an evidence rather than an unnormalized
        posterior.
    prior : UniformPrior
    free_params : list of str
    n_live : int, optional
        Live points. The evidence error scales roughly as
        ``sqrt(information / n_live)``, so this is the accuracy
        knob.
    dlogz : float, optional
        Stopping criterion: the estimated remaining evidence.
    seed : int, optional
    progress : bool, optional

    Returns
    -------
    NestedResult
    """

    try:
        from dynesty import NestedSampler
        from dynesty.utils import resample_equal
    except ImportError as error:  # pragma: no cover

        raise ImportError(
            "Nested sampling needs dynesty: "
            'pip install "cosmofit[evidence]".'
        ) from error

    lower = np.asarray(prior.lower, dtype=float)
    upper = np.asarray(prior.upper, dtype=float)
    width = upper - lower

    def prior_transform(unit):
        """Unit hypercube to parameters, for a uniform prior."""

        return lower + unit * width

    def log_likelihood(theta):
        """
        dynesty needs a finite value everywhere inside the prior.

        The library's likelihoods return ``-inf`` for cosmologies
        the data excludes, and that is correct -- but a live point
        cannot be initialized there, so it is floored to a very
        negative finite number instead. The floor is far below any
        reachable likelihood, so it changes no result; it only
        keeps the sampler able to move.
        """

        value = logpost.log_likelihood(theta)

        return value if np.isfinite(value) else -1.0e300

    sampler = NestedSampler(
        log_likelihood,
        prior_transform,
        len(free_params),
        nlive=n_live,
        rstate=np.random.default_rng(seed),
        **dynesty_kwargs,
    )

    sampler.run_nested(dlogz=dlogz, print_progress=progress)

    results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])

    samples = resample_equal(results.samples, weights / weights.sum())

    return NestedResult(
        log_evidence=float(results.logz[-1]),
        log_evidence_error=float(results.logzerr[-1]),
        samples=np.asarray(samples, dtype=float),
        free_params=list(free_params),
        prior_volume=float(np.prod(width)),
        n_live=int(n_live),
        n_evaluations=int(results.ncall.sum()),
        information=float(results.information[-1]),
    )
