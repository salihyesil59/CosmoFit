"""
The samplers beyond Metropolis-Hastings: profile, Fisher, emcee and
nested sampling, on the correlated Gaussian of ``test_mcmc``, whose
every answer is known in closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from CosmoFit.core import run
from CosmoFit.core.output import OutputError, load_covmat

from test_mcmc import COV, CORR, MEAN, SIGMA, Gaussian, gaussian_input, moments


def with_sampler(name, output=None, **options):

    info = gaussian_input(output)
    info["sampler"] = {name: options}

    return info


# ============================================================
# Minimize and profile
# ============================================================

def test_minimize_writes_its_minimum(tmp_path):

    prefix = tmp_path / "gauss"

    _, sampler = run(with_sampler("minimize", prefix), seed=0)

    rows = np.loadtxt(tmp_path / "gauss.minimum.txt", ndmin=2)

    assert rows.shape[0] == 1
    np.testing.assert_allclose(rows[0, 2:5], MEAN, atol=1e-4)

    header = (tmp_path / "gauss.minimum.txt").read_text().splitlines()[0]

    assert header.lstrip("#").split()[:5] == ["weight", "minuslogpost", "a", "b", "c"]


def test_a_profile_of_a_gaussian_is_its_marginal():
    """
    Profiling one parameter of a Gaussian over the others gives
    ``((x - mean) / sigma)^2`` with the *marginal* sigma, and puts the
    others on their conditional means.
    """

    values = np.linspace(-1.0, 3.0, 5)

    _, sampler = run(with_sampler("profile", param="a", values=list(values)), seed=0)
    products = sampler.products()

    np.testing.assert_allclose(
        products["delta_chi2"], ((values - MEAN[0]) / SIGMA[0]) ** 2, atol=1e-6,
    )

    slope = COV[1, 0] / COV[0, 0]

    np.testing.assert_allclose(
        products["params"]["b"], MEAN[1] + slope * (values - MEAN[0]), atol=1e-3,
    )

    assert all(products["success"])
    assert products["maximized"] == "likelihood"


def test_a_profile_is_written(tmp_path):

    prefix = tmp_path / "gauss"

    run(with_sampler("profile", prefix, param="c",
                     values={"min": 0.3, "max": 0.7, "n": 3}), seed=0)

    lines = (tmp_path / "gauss.profile.txt").read_text().splitlines()

    assert lines[0].lstrip("#").split() == ["c", "chi2", "delta_chi2", "a", "b"]

    rows = np.loadtxt(tmp_path / "gauss.profile.txt")

    np.testing.assert_allclose(rows[:, 0], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(rows[:, 2], [4.0, 0.0, 4.0], atol=1e-6)


@pytest.mark.parametrize("options, match", [
    ({"param": "zz", "values": [0.0]}, "sampled parameters"),
    ({"param": "a", "values": [100.0]}, "outside the prior"),
    ({"param": "a", "values": {"min": 0, "max": 1}}, "min, max and n"),
    ({"param": "a"}, "give 'values'"),
])
def test_a_profile_says_what_is_wrong(options, match):

    with pytest.raises(ValueError, match=match):
        run(with_sampler("profile", **options))


# ============================================================
# Fisher
# ============================================================

def test_fisher_recovers_the_covariance():

    _, sampler = run(with_sampler("fisher"), seed=0)
    products = sampler.products()

    np.testing.assert_allclose(products["covariance"], COV, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        [products["errors"][n] for n in "abc"], SIGMA, rtol=1e-6,
    )

    assert products["positive_definite"]
    assert not products["unstable_steps"]


def test_the_automatic_step_is_about_one_sigma():
    """Whatever the parameter's declared scale."""

    info = with_sampler("fisher")

    info["params"]["a"]["proposal"] = 1e-4
    info["params"]["b"]["proposal"] = 10.0

    _, sampler = run(info, seed=0)
    steps = sampler.products()["steps"]

    # One *conditional* standard deviation: 1 / sqrt(F_ii).
    conditional = 1.0 / np.sqrt(np.diag(np.linalg.inv(COV)))

    for name, expected in zip("abc", conditional):
        assert 0.5 < steps[name] / expected < 2.0


def test_fisher_writes_a_covmat_mcmc_can_start_from(tmp_path):

    prefix = tmp_path / "gauss"

    run(with_sampler("fisher", prefix), seed=0)

    names, covmat = load_covmat(tmp_path / "gauss.covmat")

    assert names == ["a", "b", "c"]
    np.testing.assert_allclose(covmat, COV, rtol=1e-6, atol=1e-10)

    assert (tmp_path / "gauss.minimum.txt").exists()


def test_fisher_at_a_given_point_skips_the_minimization():

    point = dict(zip("abc", MEAN))

    _, sampler = run(with_sampler("fisher", point=point), seed=0)

    assert sampler.minimizer is None
    np.testing.assert_allclose(sampler.products()["covariance"], COV, rtol=1e-6, atol=1e-10)


def test_fisher_away_from_the_minimum_is_still_the_curvature():
    """A Gaussian's curvature is the same everywhere."""

    point = dict(zip("abc", MEAN + 0.5 * SIGMA))

    _, sampler = run(with_sampler("fisher", point=point), seed=0)

    np.testing.assert_allclose(sampler.products()["covariance"], COV, rtol=1e-6, atol=1e-10)


class Unconstrained(Gaussian):
    """The same Gaussian, taking a parameter it does not depend on."""

    params = {**Gaussian.params, "flat": None}

    def logp(self, a, b, c, flat):
        return super().logp(a, b, c)


def test_a_flat_direction_is_named_not_inverted():

    info = with_sampler("fisher", point={"a": 1.0, "b": -2.0, "c": 0.5, "flat": 0.0})
    info["likelihood"] = {"gauss": {"class": Unconstrained}}
    info["params"]["flat"] = {"prior": {"min": -1, "max": 1}, "proposal": 0.1}

    with pytest.warns(UserWarning, match="not positive definite.*flat"):
        _, sampler = run(info, seed=0)

    products = sampler.products()

    assert not products["positive_definite"]
    assert math.isnan(products["errors"]["flat"])


def test_a_point_on_the_prior_edge_is_refused():

    point = {"a": 20.0, "b": -2.0, "c": 0.5}

    with pytest.raises(ValueError, match="prior edge"):
        run(with_sampler("fisher", point=point))


def test_a_point_missing_a_parameter_is_refused():

    with pytest.raises(ValueError, match="missing"):
        run(with_sampler("fisher", point={"a": 1.0}))


# ============================================================
# emcee
# ============================================================

def test_emcee_recovers_a_correlated_gaussian():

    _, sampler = run(with_sampler("emcee"), seed=1)
    products = sampler.products()

    assert products["converged"]

    mean, cov = moments(products)

    assert np.all(np.abs((mean - MEAN) / SIGMA) < 0.15)
    np.testing.assert_allclose(cov / np.outer(SIGMA, SIGMA), CORR, atol=0.1)

    assert 0.1 < products["acceptance"] < 0.9


def test_emcee_writes_and_resumes(tmp_path):

    prefix = tmp_path / "gauss"

    _, first = run(with_sampler("emcee", prefix, walkers=12, max_samples=100,
                                check_every=50), seed=2)

    rows = np.loadtxt(tmp_path / "gauss.1.txt")

    assert rows.shape == (1200, 2 + 3 + 1 + 1 + 1)
    np.testing.assert_allclose(rows, np.concatenate(first.rows), rtol=1e-9)

    info = with_sampler("emcee", prefix, walkers=12, max_samples=300, check_every=50)
    info["resume"] = True

    _, second = run(info, seed=3)

    after = np.loadtxt(tmp_path / "gauss.1.txt")

    assert after.shape[0] == 3600

    # The first 100 steps are the first run's, untouched.
    np.testing.assert_allclose(after[:1200], rows, rtol=1e-9)

    # And the second run began where the first left its walkers.
    assert len(second.rows) == 300


def test_emcee_needs_enough_walkers():

    with pytest.raises(ValueError, match="at least 2 d"):
        run(with_sampler("emcee", walkers=4))


def test_walkers_that_cannot_spread_are_refused():

    info = with_sampler("emcee")
    info["params"]["a"]["ref"] = 1.0

    with pytest.raises(ValueError, match="subspace"):
        run(info, seed=0)


# ============================================================
# Nested sampling
# ============================================================

#: ln Z of the Gaussian under the uniform prior on [-20, 20]^3, with the
#: likelihood unnormalized: ln[(2 pi)^(3/2) sqrt(det C)] - 3 ln 40.
LOG_EVIDENCE = (
    0.5 * np.log(np.linalg.det(2.0 * np.pi * COV)) - 3.0 * np.log(40.0)
)


def test_nested_finds_the_evidence(tmp_path):

    pytest.importorskip("dynesty")

    prefix = tmp_path / "gauss"

    _, sampler = run(with_sampler("nested", prefix, nlive=300), seed=4)
    products = sampler.products()

    error = products["log_evidence_error"]

    assert abs(products["log_evidence"] - LOG_EVIDENCE) < 4.0 * error

    mean, cov = moments(products)

    assert np.all(np.abs((mean - MEAN) / SIGMA) < 0.2)
    np.testing.assert_allclose(cov / np.outer(SIGMA, SIGMA), CORR, atol=0.15)

    # Three parameters, all constrained.
    assert products["dimensionality"] == pytest.approx(3.0, abs=0.6)

    evidence = yaml.safe_load((tmp_path / "gauss.evidence.yaml").read_text())

    assert evidence["log_evidence"] == pytest.approx(products["log_evidence"])

    rows = np.loadtxt(tmp_path / "gauss.1.txt")

    np.testing.assert_allclose(rows, products["chains"][0], rtol=1e-9)

    # minuslogpost is -ln L plus the prior's -ln pi.
    chi2 = rows[:, -1]
    minuslogprior = rows[:, -2]

    np.testing.assert_allclose(rows[:, 1], 0.5 * chi2 + minuslogprior, rtol=1e-9)


def test_nested_respects_a_gaussian_prior():
    """Through the inverse CDF, not the box around it."""

    pytest.importorskip("dynesty")

    info = with_sampler("nested", nlive=200)
    info["params"]["c"]["prior"] = {"dist": "norm", "loc": 0.5, "scale": 0.1}

    _, sampler = run(info, seed=5)
    products = sampler.products()

    # Prior and likelihood both N(0.5, 0.1) in c (marginally): the
    # posterior's marginal width narrows below 0.1.
    _, cov = moments(products)

    assert np.sqrt(cov[2, 2]) < 0.09


def test_nested_cannot_resume(tmp_path):

    pytest.importorskip("dynesty")

    prefix = tmp_path / "gauss"

    run(with_sampler("nested", prefix, nlive=50, dlogz=5.0), seed=6)

    info = with_sampler("nested", prefix, nlive=50, dlogz=5.0)
    info["resume"] = True

    with pytest.raises(OutputError, match="cannot resume"):
        run(info, seed=6)
