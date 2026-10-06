"""
Importance reweighting of a finished run.

The run is the correlated Gaussian of ``test_mcmc``. Adding an
independent Gaussian constraint on ``a`` gives a posterior that is
again Gaussian, with a mean and covariance written down in closed
form; the reweighted samples must find it.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from cosmofit.core import Likelihood, run
from cosmofit.core.output import OutputError
from cosmofit.core.post import PostError

from test_mcmc import COV, ICOV, MEAN, gaussian_input


class OnA(Likelihood):
    """A Gaussian measurement of ``a``: options ``loc`` and ``scale``."""

    params = {"a": None}

    def logp(self, a):
        return -0.5 * ((a - self.info["loc"]) / self.info["scale"]) ** 2


class RefusesLargeA(Likelihood):
    """No prediction above a = 2."""

    params = {"a": None}

    def logp(self, a):

        if a > 2.0:
            raise ValueError("no solution")

        return 0.0


def constrained(loc, scale):
    """Mean and covariance of the Gaussian times N(a; loc, scale)."""

    precision = ICOV.copy()
    precision[0, 0] += 1.0 / scale ** 2

    covariance = np.linalg.inv(precision)
    mean = covariance @ (ICOV @ MEAN + np.array([loc / scale ** 2, 0.0, 0.0]))

    return mean, covariance


def moments(products):

    x, w = products["samples"], products["weights"]

    return np.average(x, axis=0, weights=w), np.cov(x.T, aweights=w)


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    """A converged run with output, to post-process."""

    prefix = tmp_path_factory.mktemp("post") / "gauss"

    run(gaussian_input(prefix, Rminus1_stop=0.003), seed=31)

    return prefix


def post(prefix, force=True, **block):
    return run({"output": str(prefix), "post": block, "force": force})


# ============================================================

def test_adding_a_likelihood_moves_the_posterior_where_it_should(chains):

    loc, scale = 1.5, 0.7

    _, result = post(chains, suffix="ona", skip=0.3, add={
        "likelihood": {"ona": {"class": OnA, "loc": loc, "scale": scale}},
    })

    products = result.products()

    mean, covariance = moments(products)
    expected_mean, expected_cov = constrained(loc, scale)

    sigma = np.sqrt(np.diag(expected_cov))

    assert np.all(np.abs((mean - expected_mean) / sigma) < 0.2)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)) / sigma, 1.0, atol=0.15)

    # Narrower than what was sampled: fewer effective samples.
    assert products["ess_after"] < products["ess_before"]
    assert products["n_rejected"] == 0


def test_the_output_reads_like_a_run(chains):

    post(chains, suffix="files", add={
        "likelihood": {"ona": {"class": OnA, "loc": 1.5, "scale": 0.7}},
        "params": {"diff": {"derived": "lambda a, c: a - c"}},
    })

    folder = chains.parent

    for index in range(1, 5):
        assert (folder / f"gauss.post.files.{index}.txt").exists()

    header = (folder / "gauss.post.files.1.txt").read_text().splitlines()[0].lstrip("#").split()

    assert header[:5] == ["weight", "minuslogpost", "a", "b", "c"]
    assert "diff" in header and "chi2__ona" in header and "chi2__gauss" in header

    rows = np.loadtxt(folder / "gauss.post.files.1.txt")
    column = {name: i for i, name in enumerate(header)}

    np.testing.assert_allclose(
        rows[:, column["diff"]], rows[:, column["a"]] - rows[:, column["c"]], rtol=1e-9,
    )
    assert rows[:, 0].max() <= 1.0

    paramnames = (folder / "gauss.post.files.paramnames").read_text()

    assert "diff*" in paramnames

    updated = yaml.safe_load((folder / "gauss.post.files.updated.yaml").read_text())

    assert updated["post"]["from"] == str(chains)
    assert set(updated["likelihood"]) == {"gauss", "ona"}
    assert updated["params"]["diff"] == {"derived": "lambda a, c: a - c"}

    # The run's own output is untouched, and still the run's.
    assert not (folder / "gauss.post.files.covmat").exists()


def test_a_derived_parameter_alone_leaves_the_weights(chains):

    _, result = post(chains, suffix="derived", add={
        "params": {"diff": {"derived": "lambda a, c: a - c"}},
    })

    products = result.products()

    original = np.concatenate([np.loadtxt(chains.parent / f"gauss.{i}.txt")[:, 0]
                               for i in range(1, 5)])

    np.testing.assert_allclose(products["weights"] / products["weights"].max(),
                               original / original.max(), rtol=1e-9)


def test_removing_a_likelihood_takes_it_out_again(tmp_path):
    """
    Sampled with a weak constraint on a, then without it: back to the
    plain Gaussian.
    """

    prefix = tmp_path / "both"

    info = gaussian_input(prefix, Rminus1_stop=0.003)
    info["likelihood"]["ona"] = {"class": OnA, "loc": 3.0, "scale": 2.0}

    run(info, seed=32)

    _, result = post(prefix, suffix="without", skip=0.3, remove={"likelihood": ["ona"]})

    mean, _ = moments(result.products())

    assert np.all(np.abs((mean - MEAN) / np.sqrt(np.diag(COV))) < 0.2)


def test_points_the_new_model_rejects_are_dropped(chains):

    _, result = post(chains, suffix="rejects", add={
        "likelihood": {"refuses": {"class": RefusesLargeA}},
    })

    products = result.products()

    assert products["n_rejected"] > 0
    assert products["samples"][:, 0].max() <= 2.0


def test_a_poor_overlap_is_warned_about(chains):

    with pytest.warns(UserWarning, match="effective sample size"):
        post(chains, suffix="tail", add={
            "likelihood": {"ona": {"class": OnA, "loc": 3.5, "scale": 0.05}},
        })


# ============================================================
# Refused
# ============================================================

@pytest.mark.parametrize("block, match", [
    ({"suffix": "x", "add": {"params": {"new": {"prior": {"min": 0, "max": 1}}}}},
     "never explored"),
    ({"suffix": "x", "add": {"params": {"a": {"derived": "lambda b: b"}}}}, "sampled"),
    ({"suffix": "x", "remove": {"likelihood": ["nope"]}}, "Cannot remove 'nope'"),
    ({"suffix": "x", "remove": {"likelihood": ["gauss"]}}, "nothing to weight by"),
    ({"suffix": "x", "remove": {"theory": ["t"]}}, "Only likelihoods"),
    ({"suffix": "x", "add": {"likelihood": {"gauss": None}}}, "already a likelihood"),
    ({"suffix": "a.b"}, "plain name"),
    ({"suffix": "x", "skip": 1.0}, "skip"),
    ({"suffix": "x", "thin": 0}, "thin"),
])
def test_what_cannot_be_post_processed_is_refused(chains, block, match):

    with pytest.raises(PostError, match=match):
        run({"output": str(chains), "post": block})


def test_a_run_without_output_cannot_be_post_processed(tmp_path):

    with pytest.raises(PostError, match="missing"):
        post(tmp_path / "nothing", suffix="x")


def test_a_post_does_not_overwrite_another_by_accident(chains):

    post(chains, suffix="twice")

    with pytest.raises(OutputError, match="already exists"):
        post(chains, force=False, suffix="twice")
