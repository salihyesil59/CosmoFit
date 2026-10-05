"""
Fast and slow parameters in the Metropolis-Hastings sampler.

A theory computes from ``a`` and ``b`` (slow); a likelihood adds a
nuisance ``c`` (fast) strongly correlated with ``a``. Oversampling and
dragging must both leave the posterior -- a known Gaussian -- where it
is, while the theory is computed far less often than the likelihood.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from CosmoFit.core import Likelihood, Theory, get_model, run


MEAN = np.array([1.0, -2.0, 0.5])
SIGMA = np.array([1.0, 2.0, 0.1])
CORR = np.array([[1.0, 0.3, 0.9], [0.3, 1.0, 0.0], [0.9, 0.0, 1.0]])
COV = CORR * np.outer(SIGMA, SIGMA)
ICOV = np.linalg.inv(COV)


class Pair(Theory):
    """Provides ``(a, b)``; slow when ``delay`` says so."""

    params = {"a": None, "b": None}

    delay = 0.0

    def get_can_provide(self):
        return ["pair"]

    def calculate(self, state, want_derived=True, *, a, b):

        if self.delay:
            time.sleep(self.delay)

        state["pair"] = np.array([a, b])

    def get_pair(self):
        return self.current_state["pair"]


class Joint(Likelihood):
    """The Gaussian in ``(a, b, c)``, ``c`` its own nuisance parameter."""

    params = {"c": None}

    evaluations = 0

    def get_requirements(self):
        return {"pair": None}

    def logp(self, c):

        type(self).evaluations += 1

        d = np.append(self.provider.get_pair(), c) - MEAN

        return -0.5 * d @ ICOV @ d


def info(**sampler):

    params = {
        name: {
            "prior": {"min": -20, "max": 20},
            "ref": {"dist": "norm", "loc": m, "scale": 0.5 * s},
            "proposal": s,
        }
        for name, m, s in zip("abc", MEAN, SIGMA)
    }

    return {
        "theory": {"pair": {"class": Pair}},
        "likelihood": {"joint": {"class": Joint}},
        "params": params,
        "sampler": {"mcmc": {"Rminus1_stop": 0.01, **sampler}},
    }


def moments(products):

    x, w = products["samples"], products["weights"]

    return np.average(x, axis=0, weights=w), np.cov(x.T, aweights=w)


# ============================================================
# The split
# ============================================================

def test_the_split_follows_what_reaches_a_theory():

    model = get_model({k: v for k, v in info().items() if k != "sampler"})

    assert model.slow_and_fast() == (["a", "b"], ["c"])


def test_a_parameter_reaching_a_theory_through_a_dependent_one_is_slow():

    spec = info()
    del spec["sampler"]

    spec["params"]["x"] = spec["params"].pop("a")
    spec["params"]["x"]["drop"] = True
    spec["params"]["a"] = {"value": "lambda x: 2 * x"}

    assert get_model(spec).slow_and_fast() == (["b", "x"], ["c"])


def test_without_a_theory_nothing_is_blocked():

    from test_mcmc import gaussian_input

    _, sampler = run({**gaussian_input(), "sampler": {"mcmc": {"max_samples": 1}}})

    assert not sampler.blocked


# ============================================================
# Sampling
# ============================================================

@pytest.mark.parametrize("drag", [False, True])
def test_the_posterior_is_unchanged(drag):

    Joint.evaluations = 0

    _, sampler = run(info(fast_steps=4, drag=drag, Rminus1_stop=0.005), seed=11)
    products = sampler.products(skip=0.3)

    assert sampler.blocked
    assert products["converged"]

    mean, cov = moments(products)

    assert np.all(np.abs((mean - MEAN) / SIGMA) < 0.15)
    np.testing.assert_allclose(cov / np.outer(SIGMA, SIGMA), CORR, atol=0.1)

    # The fast moves were served from the theory's cache.
    theory = sampler.model.theories["pair"]

    assert theory.n_calculations < 0.45 * Joint.evaluations


def test_oversampling_steps_the_fast_block_given_the_slow():
    """
    The fast block's proposal is the conditional width of ``c`` given
    ``(a, b)``, not its marginal width: with c 0.9-correlated with a,
    that is under half of it.
    """

    _, sampler = run(info(fast_steps=2, max_samples=1), seed=12)

    sampler.covariance = COV.copy()

    slow, fast = sampler._proposal_factors()

    conditional = np.sqrt(1.0 / ICOV[2, 2])

    np.testing.assert_allclose(
        fast[2, 0], sampler.options["proposal_scale"] * conditional, rtol=1e-12,
    )
    assert fast[0, 0] == fast[1, 0] == 0.0

    # A slow move shifts c along its regression on (a, b).
    covariance = (slow @ slow.T) / (sampler.options["proposal_scale"] ** 2 / 2)

    np.testing.assert_allclose(covariance, COV - np.outer([0, 0, 1], [0, 0, 1]) / ICOV[2, 2],
                               atol=1e-12)


def test_fast_steps_follow_the_cost_ratio():

    Pair.delay = 0.005

    try:
        _, sampler = run(info(max_samples=1, learn_every=1), seed=13)
    finally:
        Pair.delay = 0.0

    # A 5 ms theory against a likelihood of tens of microseconds: a
    # ratio of hundreds, to the power 0.4.
    assert sampler.fast_steps >= 4


def test_fast_steps_must_be_positive():

    with pytest.raises(ValueError, match="at least 1"):
        run(info(fast_steps=0, max_samples=1), seed=14)
