"""
The adaptive Metropolis-Hastings sampler and what it writes.

On a correlated Gaussian, whose answer is known, the chains must stop
by their own R - 1 and recover the mean and covariance. On disk they
must be what getdist reads, and a run must neither overwrite another's
output by accident nor resume one written by a different input.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from cosmofit.core import Likelihood, run
from cosmofit.core.output import OutputError, load_covmat
from cosmofit.core.parameters import parse_param
from cosmofit.samplers.mcmc import gelman_rubin


MEAN = np.array([1.0, -2.0, 0.5])
SIGMA = np.array([1.0, 2.0, 0.1])
CORR = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.3], [0.0, 0.3, 1.0]])
COV = CORR * np.outer(SIGMA, SIGMA)
ICOV = np.linalg.inv(COV)


class Gaussian(Likelihood):

    params = {"a": None, "b": None, "c": None}

    def logp(self, a, b, c):
        d = np.array([a, b, c]) - MEAN
        return -0.5 * d @ ICOV @ d


def gaussian_input(output=None, **sampler):

    info = {
        "likelihood": {"gauss": {"class": Gaussian}},
        "params": {
            name: {
                "prior": {"min": -20, "max": 20},
                "ref": {"dist": "norm", "loc": 0.0, "scale": 0.5},
                "proposal": 0.5,
                "latex": rf"\theta_{name}",
            }
            for name in "abc"
        },
        "sampler": {"mcmc": {"Rminus1_stop": 0.01, **sampler}},
    }

    info["params"]["sum"] = {"derived": "lambda a, b: a + b"}

    if output is not None:
        info["output"] = str(output)

    return info


def moments(products):

    x, w = products["samples"], products["weights"]

    return np.average(x, axis=0, weights=w), np.cov(x.T, aweights=w)


# ============================================================
# Sampling
# ============================================================

def test_recovers_a_correlated_gaussian():
    """
    At R - 1 < 0.005: at 0.01 the chains' means agree but a variance can
    still be 10% off, which is what R - 1 does not measure.
    """

    _, sampler = run(gaussian_input(Rminus1_stop=0.005), seed=1)
    products = sampler.products(skip=0.3)

    assert products["converged"]
    assert products["Rminus1"] < 0.005

    mean, cov = moments(products)

    np.testing.assert_allclose(mean, MEAN, atol=0.15 * SIGMA.max())
    assert np.all(np.abs((mean - MEAN) / SIGMA) < 0.15)

    np.testing.assert_allclose(cov / np.outer(SIGMA, SIGMA), CORR, atol=0.1)

    # The learned proposal is the posterior's covariance.
    learned = products["covmat"] / np.outer(SIGMA, SIGMA)
    np.testing.assert_allclose(learned, CORR, atol=0.15)

    assert 0.15 < products["acceptance"] < 0.5


def test_one_chain_is_compared_with_itself():

    _, sampler = run(gaussian_input(chains=1, Rminus1_stop=0.03), seed=2)
    products = sampler.products(skip=0.3)

    assert products["converged"]
    assert np.all(np.abs((moments(products)[0] - MEAN) / SIGMA) < 0.25)


def test_gelman_rubin():

    rng = np.random.default_rng(0)

    same = [(rng.standard_normal((4000, 2)), np.ones(4000)) for _ in range(4)]
    apart = [(x + shift, w) for (x, w), shift in zip(same, (0.0, 0.0, 0.0, 1.0))]

    assert gelman_rubin(same) < 0.01
    assert gelman_rubin(apart) > 0.1


def test_a_stuck_chain_says_why():

    info = gaussian_input(max_tries=50)

    for name in "abc":
        info["params"][name]["proposal"] = 1e4

    with pytest.raises(RuntimeError, match="too wide"):
        run(info, seed=3)


def test_unknown_options_are_named():

    with pytest.raises(ValueError, match="Rminus_stop"):
        run(gaussian_input(Rminus_stop=0.01))


# ============================================================
# Output
# ============================================================

def test_writes_what_getdist_reads(tmp_path):

    prefix = tmp_path / "chains" / "gauss"

    _, sampler = run(gaussian_input(prefix), seed=4)
    products = sampler.products()

    files = {p.name for p in prefix.parent.iterdir()}

    assert {
        "gauss.input.yaml", "gauss.updated.yaml", "gauss.paramnames",
        "gauss.ranges", "gauss.covmat", "gauss.progress",
        "gauss.1.txt", "gauss.2.txt", "gauss.3.txt", "gauss.4.txt",
    } <= files

    header = (prefix.parent / "gauss.1.txt").read_text().splitlines()[0]

    assert header.lstrip("#").split() == products["columns"]

    rows = np.loadtxt(prefix.parent / "gauss.1.txt")

    np.testing.assert_allclose(rows, products["chains"][0], rtol=1e-9)

    # Sampled unstarred, derived and the chi2 columns starred.
    paramnames = dict(
        line.split("\t") for line in
        (prefix.parent / "gauss.paramnames").read_text().splitlines()
    )

    assert paramnames["a"] == r"\theta_a"
    assert "sum*" in paramnames and "chi2__gauss*" in paramnames

    assert (prefix.parent / "gauss.ranges").read_text().splitlines()[0] == "a\t-20.0\t20.0"

    names, covmat = load_covmat(prefix.parent / "gauss.covmat")

    assert names == ["a", "b", "c"]
    np.testing.assert_allclose(covmat, products["covmat"])

    updated = yaml.safe_load((prefix.parent / "gauss.updated.yaml").read_text())

    assert updated["params"]["a"]["prior"] == {"min": -20.0, "max": 20.0}
    assert updated["sampler"]["mcmc"]["Rminus1_stop"] == 0.01
    assert updated["sampler"]["mcmc"]["chains"] == 4

    getdist = pytest.importorskip("getdist")

    samples = getdist.loadMCSamples(str(prefix), settings={"ignore_rows": 0.3})

    np.testing.assert_allclose(
        samples.getMeans()[:3], MEAN, atol=0.2 * SIGMA.max(),
    )


def test_will_not_overwrite_by_accident(tmp_path):

    prefix = tmp_path / "gauss"

    run(gaussian_input(prefix), seed=5)

    with pytest.raises(OutputError, match="already exists"):
        run(gaussian_input(prefix), seed=5)

    info = gaussian_input(prefix)
    info["force"] = True

    run(info, seed=6)

    with pytest.raises(OutputError, match="contradict"):
        run({**info, "resume": True}, seed=6)


def test_resumes_where_it_stopped(tmp_path):

    prefix = tmp_path / "gauss"

    _, first = run(gaussian_input(prefix, Rminus1_stop=0.5), seed=7)

    before = [len(chain) for chain in first.products()["chains"]]

    info = gaussian_input(prefix)
    info["resume"] = True

    _, second = run(info, seed=8)

    after = [len(np.loadtxt(prefix.parent / f"gauss.{i}.txt")) for i in range(1, 5)]

    assert all(a > b for a, b in zip(after, before))
    assert second.products()["Rminus1"] < 0.01

    # The proposal it learned is where the second run started.
    np.testing.assert_allclose(
        load_covmat(prefix.parent / "gauss.covmat")[1], second.products()["covmat"],
    )


def test_will_not_resume_a_different_input(tmp_path):

    prefix = tmp_path / "gauss"

    run(gaussian_input(prefix, Rminus1_stop=0.5), seed=9)

    info = gaussian_input(prefix, Rminus1_stop=0.5)
    info["params"]["a"]["prior"] = {"min": -10, "max": 10}
    info["resume"] = True

    with pytest.raises(OutputError, match="different input"):
        run(info, seed=9)


def test_starts_from_a_covmat(tmp_path):
    """Only the parameters the file names; the others keep their width."""

    path = tmp_path / "start.covmat"

    np.savetxt(path, COV[:2, :2], header="a b", comments="# ")

    _, sampler = run(gaussian_input(covmat=str(path), learn_proposal=False,
                                    max_samples=1), seed=10)

    covariance = sampler.products()["covmat"]

    np.testing.assert_allclose(covariance[:2, :2], COV[:2, :2])
    assert covariance[2, 2] == pytest.approx(0.5 ** 2)
    assert covariance[0, 2] == 0.0


# ============================================================
# Declarations written back
# ============================================================

@pytest.mark.parametrize("declaration", [
    {"prior": {"min": 0.0, "max": 1.0}, "ref": 0.5, "proposal": 0.01, "latex": "x"},
    {"prior": {"dist": "norm", "loc": 1.0, "scale": 0.1, "min": 0.0},
     "ref": {"dist": "norm", "loc": 1.0, "scale": 0.05}},
    {"prior": {"dist": "loguniform", "min": 1e-3, "max": 1.0}, "drop": True},
    {"derived": True, "latex": r"\Omega"},
    {"derived": "lambda a, b: a + b"},
    {"value": "lambda H0: H0 / 100"},
    3.5,
])
def test_a_declaration_reads_back_as_itself(declaration):

    spec = parse_param("x", declaration)

    assert spec.declaration() == declaration
    assert parse_param("x", spec.declaration()).declaration() == declaration


def test_a_converged_run_is_not_sampled_again(tmp_path):

    prefix = tmp_path / "gauss"

    run(gaussian_input(prefix, Rminus1_stop=0.5), seed=11)

    before = [len(np.loadtxt(prefix.parent / f"gauss.{i}.txt")) for i in range(1, 5)]

    info = gaussian_input(prefix, Rminus1_stop=0.5)
    info["resume"] = True

    _, again = run(info, seed=12)

    after = [len(np.loadtxt(prefix.parent / f"gauss.{i}.txt")) for i in range(1, 5)]

    assert after == before
    assert again.products()["converged"]
    assert again.progress[0]["steps"] == 0


def test_progress_is_reported_at_every_check():

    seen = []

    _, sampler = run(gaussian_input(Rminus1_stop=0.05), seed=13, callback=seen.append)

    assert len(seen) == len(sampler.progress) > 1
    assert seen[-1]["Rminus1"] == sampler.products()["Rminus1"]
    assert seen[-1]["Rminus1_stop"] == 0.05 and "max_samples" in seen[-1]
