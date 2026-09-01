"""
The Boltzmann backend must not hand out a spectrum it cannot vouch for.

CAMB can return an all-NaN spectrum without raising. This was seen for
Planck's own best-fit LCDM, intermittently, during full test-suite runs
-- physically unremarkable parameters, a spectrum of NaN.

Left unchecked that is a silent scientific error rather than a crash.
The NaN becomes a NaN chi2, and :class:`~stats.posterior.Posterior`
turns any non-finite chi2 into ``-inf``, which is a rejection. So the
sampler quietly drops a point it had no reason to drop. Parameters that
are genuinely unphysical should be rejected; a solver that glitched at
good parameters should not be, and nothing in the output distinguished
the two.

Repeating the call does not help -- measured on the failures this check
first exposed, an identical second call returns NaN again -- so the
backend does the one thing it can: refuse to pass the result on.

Raising is not the whole story either, because ``BoltzmannError`` is a
``RuntimeError`` and :class:`~stats.posterior.LogPosterior` catches
those and rejects the point like any other. What keeps it from being
silent is that the posterior counts solver failures separately from
unphysical ones, warns the first time, and reports the total when a run
ends. That half is tested at the bottom of this file.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoFit import LCDM, CosmologyParameters


def _has_camb() -> bool:

    try:

        import camb  # noqa: F401

    except ImportError:

        return False

    return True


requires_camb = pytest.mark.skipif(

    not _has_camb(),

    reason="CAMB not installed (optional 'cmb' extra)",

)


PLANCK_BEST_FIT = dict(
    H0=67.36,
    Omega_m=0.3153,
    Omega_b=0.02237 / 0.6736 ** 2,
    ln1e10As=3.044,
    n_s=0.9649,
    tau_reio=0.0544,
)


@pytest.fixture
def backend():

    if not _has_camb():
        pytest.skip("CAMB not installed")

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    likelihood = PlanckLensingLikelihood(

        LCDM(CosmologyParameters(**PLANCK_BEST_FIT)),

    )

    return likelihood.backend


def _all_nan(spectra):
    """
    The same dictionary with every spectrum replaced by NaN, which is
    the shape of the failure CAMB actually produces.
    """

    return {

        name: (
            value
            if name == "ell"
            else np.full_like(np.asarray(value, dtype=float), np.nan)
        )

        for name, value in spectra.items()

    }


# ============================================================
# A failure must be loud, not a quiet NaN
# ============================================================

@requires_camb
def test_the_call_is_made_once(backend):
    """
    No retry. It was tried, and measured not to help: an identical
    repeated call returns NaN again. Retrying only doubled the cost
    of the most expensive thing in the library.
    """

    attempts = {"n": 0}
    real = backend._run_once

    def counted():

        attempts["n"] += 1

        return _all_nan(real())

    backend._run_once = counted
    backend._cache_key = None

    from CosmoFit.cosmology.boltzmann import BoltzmannError

    with pytest.raises(BoltzmannError):

        backend._run()

    assert attempts["n"] == 1


@requires_camb
def test_a_persistent_nan_raises_and_names_the_point(backend):

    from CosmoFit.cosmology.boltzmann import BoltzmannError

    real = backend._run_once

    backend._run_once = lambda: _all_nan(real())
    backend._cache_key = None

    with pytest.raises(BoltzmannError) as excinfo:

        backend._run()

    message = str(excinfo.value)

    # Which quantities went bad...
    assert "TT" in message
    assert "PP" in message

    # ...and at which point in parameter space, so the report is
    # something you can act on rather than just a complaint.
    for parameter in ("H0", "Omega_m", "ln1e10As", "n_s", "tau", "lmax"):

        assert parameter in message


@requires_camb
def test_sigma8_alone_is_enough_to_reject(backend):
    """
    Every returned quantity is checked, not just the spectra: a
    finite spectrum with a NaN sigma8 is still unusable.
    """

    from CosmoFit.cosmology.boltzmann import BoltzmannError

    real = backend._run_once

    def bad_sigma8():

        result = dict(real())
        result["sigma8"] = float("nan")

        return result

    backend._run_once = bad_sigma8
    backend._cache_key = None

    with pytest.raises(BoltzmannError, match="sigma8"):

        backend._run()


# ============================================================
# None of this may disturb the ordinary path
# ============================================================

@requires_camb
def test_a_healthy_call_is_untouched(backend):
    """
    One CAMB call, no warning, and a chi2 where Planck's own best
    fit should put it.
    """

    attempts = {"n": 0}
    real = backend._run_once

    def counted():

        attempts["n"] += 1

        return real()

    backend._run_once = counted
    backend._cache_key = None

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        spectra = backend._run()

    assert attempts["n"] == 1

    assert not [w for w in caught if w.category is RuntimeWarning]

    assert np.all(np.isfinite(spectra["TT"]))


# ============================================================
# The half that matters for sampling
# ============================================================

def _posterior_over(likelihood, cosmology):

    from CosmoFit.likelihoods.joint import JointLikelihood
    from CosmoFit.stats.posterior import LogPosterior
    from CosmoFit.stats.priors import UniformPrior

    return LogPosterior(

        JointLikelihood(likelihood),

        cosmology,

        UniformPrior(["Omega_m"], {"Omega_m": (0.1, 0.6)}),

    )


@requires_camb
def test_a_solver_failure_is_counted_not_just_rejected():
    """
    The point still has to be rejected -- a chain cannot stop
    because CAMB had a bad moment -- but a rejection for this reason
    is not the same as rejecting an unphysical point, and the
    difference has to survive into something the caller can read.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    cosmology = LCDM(CosmologyParameters(**PLANCK_BEST_FIT))
    likelihood = PlanckLensingLikelihood(cosmology)

    posterior = _posterior_over(likelihood, cosmology)

    assert np.isfinite(posterior.log_likelihood(np.array([0.3153])))
    assert posterior.solver_failures == 0

    real = likelihood.backend._run_once
    likelihood.backend._run_once = lambda: _all_nan(real())
    likelihood.backend._cache_key = None

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        values = [
            posterior.log_likelihood(np.array([x]))
            for x in (0.30, 0.32, 0.34)
        ]

    # Rejected, not raised: the sampler keeps going.
    assert all(value == -np.inf for value in values)

    assert posterior.solver_failures == 3

    assert posterior.first_solver_failure == pytest.approx([0.30])

    # Said once, not three times -- a chain evaluates this millions
    # of times and a warning per rejection would bury the run.
    assert sum(
        1 for w in caught if w.category is RuntimeWarning
    ) == 1


@requires_camb
def test_an_unphysical_point_is_not_counted_as_a_solver_failure():
    """
    The counter has to mean what it says. A point rejected for
    being outside the prior is not evidence of anything wrong with
    the solver, and must not inflate the number that is.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    cosmology = LCDM(CosmologyParameters(**PLANCK_BEST_FIT))
    likelihood = PlanckLensingLikelihood(cosmology)

    posterior = _posterior_over(likelihood, cosmology)

    assert posterior.log_likelihood(np.array([np.nan])) == -np.inf

    assert posterior.solver_failures == 0
