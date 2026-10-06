"""
Chain diagnostics and posterior intervals.

Before these, a fit's summary was its 16/50/84 percentiles whatever
the chain looked like: a fixed 1000-step burn-in, no Gelman-Rubin
test, nothing that warned when the autocorrelation time said the
chain was far too short, and no way to quote a one-sided limit for a
parameter piled against a bound.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cosmofit import LCDM, Fitter
from cosmofit.stats.diagnostics import (
    hpd_interval,
    one_sided_limit,
    posterior_summary,
    split_rhat,
    walker_group_rhat,
)


# ============================================================
# The statistics themselves
# ============================================================

def test_split_rhat_is_one_for_identical_distributions():

    rng = np.random.default_rng(0)

    chains = rng.normal(size=(4, 5000, 2))

    rhat = split_rhat(chains)

    assert np.all(np.abs(rhat - 1.0) < 0.01)


def test_split_rhat_flags_chains_in_different_places():

    rng = np.random.default_rng(1)

    chains = rng.normal(size=(4, 2000, 1))
    chains[0] += 3.0

    assert split_rhat(chains)[0] > 1.5


def test_split_rhat_flags_a_drifting_chain_against_itself():
    """
    The split form's reason to exist: one chain still trending
    disagrees with its own second half.
    """

    rng = np.random.default_rng(2)

    drift = np.linspace(0.0, 5.0, 4000)[None, :, None]

    chains = rng.normal(size=(2, 4000, 1)) + drift

    assert split_rhat(chains)[0] > 1.5


def test_walker_groups_that_never_mixed_are_caught():

    rng = np.random.default_rng(3)

    chain = rng.normal(size=(3000, 32, 1))

    # A quarter of the walkers stuck in another mode.
    chain[:, :8] += 4.0

    assert walker_group_rhat(chain)[0] > 1.5


def test_hpd_reaches_a_bound_where_percentiles_do_not():
    """
    An exponential posterior against zero: the shortest 68% interval
    starts at zero, while the 16th percentile is well above it.
    """

    rng = np.random.default_rng(4)

    x = rng.exponential(size=200_000)

    low, high = hpd_interval(x, 0.68)

    assert low == pytest.approx(0.0, abs=1e-3)
    assert high == pytest.approx(-np.log(0.32), rel=0.01)
    assert np.percentile(x, 16) > 0.15


def test_hpd_matches_percentiles_for_a_gaussian():

    rng = np.random.default_rng(5)

    x = rng.normal(size=400_000)

    low, high = hpd_interval(x, 0.6827)

    assert low == pytest.approx(-1.0, abs=0.01)
    assert high == pytest.approx(1.0, abs=0.01)


def test_one_sided_limit():

    rng = np.random.default_rng(6)

    x = rng.normal(size=400_000)

    assert one_sided_limit(x, 0.95, "upper") == pytest.approx(1.645, abs=0.01)
    assert one_sided_limit(x, 0.95, "lower") == pytest.approx(-1.645, abs=0.01)


def test_summary_refuses_an_empty_chain():

    with pytest.raises(ValueError, match="burn-in"):

        posterior_summary(np.empty((0, 2)), ["a", "b"])


# ============================================================
# Through a fit
# ============================================================

@pytest.fixture(scope="module")
def short_fit():

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 70.0, "Omega_m": 0.3},
    )

    fit.run_mcmc(nwalkers=16, nsteps=120, burnin=20, progress=False,
                 n_processes=1)

    return fit


def test_a_short_chain_warns_on_summary(short_fit):

    with pytest.warns(UserWarning, match="not converged"):

        short_fit.summary()


def test_the_warning_can_be_silenced(short_fit):

    with warnings.catch_warnings():

        warnings.simplefilter("error")

        short_fit.summary(check=False)


def test_convergence_reports_acceptance_and_rhat(short_fit):

    diagnostics = short_fit.convergence()

    assert 0.0 < diagnostics["acceptance_fraction"] < 1.0
    assert set(diagnostics["rhat_minus_1"]) == {"H0", "Omega_m"}
    assert diagnostics["converged"] is False


def test_summary_carries_the_interval_ends(short_fit):

    summary = short_fit.summary(check=False, interval="hpd")

    for entry in summary.values():

        assert entry["low"] <= entry["median"] <= entry["high"]
        assert entry["plus"] == pytest.approx(entry["high"] - entry["median"])


def test_limit(short_fit):

    upper = short_fit.limit("Omega_m", 0.95)
    lower = short_fit.limit("Omega_m", 0.95, side="lower")

    assert lower < upper

    with pytest.raises(ValueError, match="free parameter"):
        short_fit.limit("w0")


def test_auto_burnin_is_measured_from_the_chain():

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 70.0, "Omega_m": 0.3},
    )

    fit.run_mcmc(nwalkers=16, nsteps=200, burnin="auto", progress=False,
                 n_processes=1)

    assert isinstance(fit.burnin, int)
    assert 0 < fit.burnin <= 100
