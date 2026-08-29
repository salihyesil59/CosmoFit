"""
The prior, the posterior, and the joint likelihood.

Three small modules between the datasets and the sampler, and every
MCMC step in the library goes through all three. Coverage stood at
73%, 85% and 70%, and what had not run was almost entirely the
*rejection* paths -- which is where these three earn their place,
because their job is to turn a parameter point the model cannot
represent into a finite answer rather than a crash.

The `chi2` guards below are not hypothetical. Both were added after a
real failure: L-BFGS-B started where chi-squared was already infinite,
computed `inf - inf` for its gradient, and evaluated the objective at
`[nan, nan, nan]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM, CosmologyParameters, Fitter, JointLikelihood
from CosmoFit.likelihoods import CCLikelihood
from CosmoFit.stats.posterior import LogPosterior
from CosmoFit.stats.priors import UniformPrior


BOUNDS = {"H0": (60.0, 80.0), "Omega_m": (0.1, 0.5)}


@pytest.fixture
def prior():

    return UniformPrior(["H0", "Omega_m"], BOUNDS)


# ============================================================
# UniformPrior
# ============================================================


def test_inside_the_box_is_zero_and_outside_is_minus_infinity(prior):
    """
    A top-hat prior contributes nothing to the posterior where it is
    satisfied -- it is not normalized, deliberately, since the
    normalization is a constant the sampler never sees.
    """

    assert prior.log_prior([70.0, 0.3]) == 0.0

    assert prior([70.0, 0.3]) == 0.0  # __call__ is log_prior

    assert prior.log_prior([50.0, 0.3]) == -np.inf

    assert prior.log_prior([70.0, 0.9]) == -np.inf


def test_the_edges_are_inside(prior):
    """
    A closed interval. An open one would reject the boundary, and a
    parameter pinned at its bound by the data -- which happens, and
    is what `profile_likelihood_and_fisher.ipynb` is about -- would
    be unreachable.
    """

    assert prior.log_prior([60.0, 0.1]) == 0.0

    assert prior.log_prior([80.0, 0.5]) == 0.0


def test_a_missing_bound_is_refused():
    """
    Silently defaulting to (-inf, inf) would give an improper prior
    on one parameter and nothing would say so.
    """

    with pytest.raises(ValueError, match="Missing bounds.*Omega_m"):
        UniformPrior(["H0", "Omega_m"], {"H0": (60.0, 80.0)})


def test_an_inverted_or_empty_interval_is_refused():

    with pytest.raises(ValueError, match="upper > lower"):
        UniformPrior(["H0"], {"H0": (80.0, 60.0)})

    with pytest.raises(ValueError, match="upper > lower"):
        UniformPrior(["H0"], {"H0": (70.0, 70.0)})


def test_a_vector_of_the_wrong_length_is_refused(prior):

    with pytest.raises(ValueError, match="length-2"):
        prior.log_prior([70.0])


def test_sampling_stays_inside_the_box(prior):

    draws = prior.sample(500, rng=np.random.default_rng(0))

    assert draws.shape == (500, 2)

    assert np.all(draws >= prior.lower)

    assert np.all(draws <= prior.upper)

    # And every draw is accepted by the prior that produced it.
    assert all(prior.log_prior(theta) == 0.0 for theta in draws)


def test_sampling_without_a_generator_still_works(prior):

    assert prior.sample(3).shape == (3, 2)


def test_repr_shows_the_bounds(prior):

    assert "H0=(60, 80)" in repr(prior)


# ============================================================
# LogPosterior
# ============================================================


@pytest.fixture
def posterior():

    model = LCDM(CosmologyParameters(H0=70.0, Omega_m=0.3))

    joint = JointLikelihood(CCLikelihood(model))

    return LogPosterior(
        joint, model, UniformPrior(["H0", "Omega_m"], BOUNDS),
    )


def test_the_posterior_is_the_prior_plus_the_likelihood(posterior):

    theta = [70.0, 0.3]

    assert posterior(theta) == pytest.approx(
        posterior.log_prior(theta) + posterior.log_likelihood(theta)
    )

    assert posterior.log_likelihood(theta) == pytest.approx(
        -0.5 * posterior.chi2(theta)
    )

    assert posterior.ndim == 2

    assert posterior.names == ["H0", "Omega_m"]


def test_a_point_outside_the_prior_never_reaches_the_likelihood(posterior):
    """
    The short circuit matters for cost as much as for correctness: a
    rejected walker proposal must not pay for a Boltzmann code.
    """

    calls = []

    original = posterior.joint.chi2

    posterior.joint.chi2 = lambda: calls.append(1) or original()

    assert posterior([200.0, 0.3]) == -np.inf

    assert calls == []

    posterior.joint.chi2 = original


def test_a_non_finite_parameter_vector_is_rejected_before_the_cosmology(posterior):
    """
    The first of the two guards, and it exists because the second was
    not enough. Writing nan into the parameters builds an
    interpolation table full of nan, and the interpolator raises from
    inside `refresh()` -- which used to sit outside the try block.
    """

    for theta in ([np.nan, 0.3], [np.inf, 0.3], [70.0, np.nan]):

        assert posterior.log_likelihood(theta) == -np.inf

        assert posterior.chi2(theta) == np.inf

    # The cosmology was not touched by any of that.
    assert posterior.cosmology.H0 == 70.0


def test_a_likelihood_that_raises_becomes_the_worst_possible_fit(posterior):
    """
    A sampler that merely *proposed* an unphysical point should not
    crash, and treating it as infinitely bad is exactly what
    excluding it means.
    """

    def unphysical():
        raise ValueError("E(z)^2 < 0")

    posterior.joint.chi2 = unphysical

    assert posterior.chi2([70.0, 0.3]) == np.inf

    assert posterior.log_likelihood([70.0, 0.3]) == -np.inf

    assert posterior([70.0, 0.3]) == -np.inf


def test_a_non_finite_chi_squared_is_also_the_worst_possible_fit(posterior):

    posterior.joint.chi2 = lambda: np.nan

    assert posterior.chi2([70.0, 0.3]) == np.inf

    assert posterior.log_likelihood([70.0, 0.3]) == -np.inf


def test_applying_a_vector_moves_the_shared_cosmology(posterior):

    posterior._apply([72.0, 0.28])

    assert posterior.cosmology.H0 == 72.0

    assert posterior.cosmology.Omega_m == 0.28


# ============================================================
# JointLikelihood
# ============================================================


@pytest.fixture
def model():

    return LCDM(CosmologyParameters(H0=70.0, Omega_m=0.3))


def test_the_joint_chi_squared_is_the_sum(model):

    cc = CCLikelihood(model)

    joint = JointLikelihood(cc)

    assert len(joint) == 1

    assert joint.chi2() == pytest.approx(cc.chi2())

    assert joint.n_data == cc.n_data

    assert joint.names == [cc.name]

    assert list(joint) == [cc]

    assert cc.name in repr(joint)


def test_adding_a_second_likelihood_adds_its_chi_squared_and_its_points(model):

    joint = JointLikelihood(CCLikelihood(model))

    before_chi2 = joint.chi2()
    before_n = joint.n_data

    second = CCLikelihood(model)

    joint.add(second)

    assert len(joint) == 2

    assert joint.chi2() == pytest.approx(2.0 * before_chi2)

    assert joint.n_data == 2 * before_n


def test_loglike_is_minus_half_chi_squared(model):

    joint = JointLikelihood(CCLikelihood(model))

    assert joint.loglike() == pytest.approx(-0.5 * joint.chi2())


def test_the_information_criteria(model):

    joint = JointLikelihood(CCLikelihood(model))

    assert joint.aic(3) == pytest.approx(joint.chi2() + 6.0)

    assert joint.bic(3) == pytest.approx(
        joint.chi2() + 3.0 * np.log(joint.n_data)
    )


def test_the_summary_carries_every_component(model):

    joint = JointLikelihood(CCLikelihood(model), CCLikelihood(model))

    summary = joint.summary()

    assert summary["n_likelihoods"] == 2

    assert summary["n_data"] == joint.n_data

    assert summary["chi2"] == pytest.approx(joint.chi2())

    assert summary["loglike"] == pytest.approx(joint.loglike())

    assert len(summary["likelihoods"]) == 2


def test_an_empty_joint_likelihood_is_zero():
    """
    `sum()` of nothing is 0, so this is well defined -- and it is
    what `Fitter` briefly holds before any dataset is attached.
    """

    joint = JointLikelihood()

    assert len(joint) == 0

    assert joint.chi2() == 0.0

    assert joint.n_data == 0

    assert joint.names == []


# ============================================================
# The three together, as a Fitter builds them
# ============================================================


def test_a_fitter_wires_the_three_up_consistently():

    fitter = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 70.0, "Omega_m": 0.3},
    )

    theta = fitter.theta0

    assert fitter.logpost.ndim == len(theta)

    assert fitter.logpost.names == fitter.free_params

    assert fitter.logpost.chi2(theta) == pytest.approx(
        fitter.joint.chi2()
    )

    # And the prior the fitter built accepts its own starting point.
    assert fitter.logpost.log_prior(theta) == 0.0
