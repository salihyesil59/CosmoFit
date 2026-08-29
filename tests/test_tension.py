"""
Tension statistics, against cases where the answer is known.

This library quoted tensions by hand -- `np.hypot` and a
subtraction -- in three notebooks. That formula is right when both
posteriors are Gaussian, one-dimensional and independent, and the
point of these functions is to make each of those an assumption
someone had to make rather than one nobody noticed.

So the tests check the agreements *and* the disagreements: where
the Gaussian formula is right the sample-based one must reproduce
it, and where it is wrong they must part company.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit.stats.tension import (
    gaussian_tension,
    gaussian_tension_nd,
    sample_tension,
    suspiciousness,
)


def has_dynesty():

    try:
        import dynesty  # noqa: F401
    except ImportError:
        return False

    return True


needs_dynesty = pytest.mark.skipif(
    not has_dynesty(),
    reason="dynesty not installed (optional 'evidence' extra)",
)


# ============================================================
# Against the published tensions this library talks about
# ============================================================

def test_reproduces_the_hubble_tension():
    """
    SH0ES 2022 (73.04 +- 1.04) against Planck 2018 (67.36 +- 0.54),
    which the literature quotes at about 5 sigma.
    """

    result = gaussian_tension(73.04, 1.04, 67.36, 0.54)

    assert result["n_sigma"] == pytest.approx(4.85, abs=0.05)

    assert result["p_value"] < 2.0e-6


def test_reproduces_the_s8_tension():
    """
    Planck's S8 = 0.832 +- 0.013 against KiDS-1000's
    0.759 +- 0.024 -- the comparison `s8_tension_cmb.ipynb` makes,
    where it came out at 2.9 sigma against a slightly different
    Planck value.
    """

    result = gaussian_tension(0.832, 0.013, 0.759, 0.024)

    assert result["n_sigma"] == pytest.approx(2.67, abs=0.05)


def test_zero_uncertainty_is_refused():

    with pytest.raises(ValueError, match="no scale"):
        gaussian_tension(1.0, 0.0, 2.0, 0.0)


# ============================================================
# Samples versus the Gaussian formula
# ============================================================

@pytest.mark.parametrize("separation", [0.5, 1.5, 3.0])
def test_sample_tension_reproduces_the_gaussian_formula(separation):
    """
    Where the assumption holds, the two must agree -- otherwise the
    sample-based one is not measuring what it claims.
    """

    rng = np.random.default_rng(0)

    a = rng.normal(0.0, 1.0, 400_000)
    b = rng.normal(separation, 1.0, 400_000)

    analytic = gaussian_tension(0.0, 1.0, separation, 1.0)["n_sigma"]

    sampled = sample_tension(a, b, seed=1)["n_sigma"]

    assert sampled == pytest.approx(analytic, abs=0.05)


def test_the_two_part_company_on_a_skewed_posterior():
    """
    And where it does not hold, they must differ -- otherwise there
    would be no reason to have both.

    Log-normal posteriors, which is the shape a positive-definite
    parameter with a long tail actually takes.
    """

    rng = np.random.default_rng(0)

    a = rng.lognormal(0.0, 0.6, 400_000)
    b = rng.lognormal(0.9, 0.6, 400_000)

    analytic = gaussian_tension(a.mean(), a.std(), b.mean(), b.std())

    sampled = sample_tension(a, b, seed=1)

    assert abs(sampled["n_sigma"] - analytic["n_sigma"]) > 0.05


def test_sample_tension_needs_samples():

    with pytest.raises(ValueError, match="at least two"):
        sample_tension([1.0], [1.0, 2.0])


# ============================================================
# More than one dimension
# ============================================================

def test_a_joint_tension_can_exceed_both_its_projections():
    """
    The reason the multi-dimensional version exists.

    Two posteriors elongated along the same direction, offset along
    the *narrow* one. Each parameter separately shows 0.35 sigma --
    nothing. Together they are 1.74 sigma, because the offset is
    exactly where neither posterior has any room.
    """

    covariance = np.array([[1.0, 0.95], [0.95, 1.0]])

    mean_a = np.array([0.0, 0.0])
    mean_b = np.array([0.5, -0.5])

    for i in range(2):

        projected = gaussian_tension(
            mean_a[i], 1.0, mean_b[i], 1.0,
        )["n_sigma"]

        assert projected == pytest.approx(0.35, abs=0.02)

    joint = gaussian_tension_nd(mean_a, covariance, mean_b, covariance)

    assert joint["chi2"] == pytest.approx(5.0, rel=1e-6)
    assert joint["dof"] == 2
    assert joint["n_sigma"] == pytest.approx(1.74, abs=0.02)


def test_nd_reduces_to_the_1d_formula():

    result = gaussian_tension_nd([0.0], [[1.0]], [2.0], [[3.0]])

    scalar = gaussian_tension(0.0, 1.0, 2.0, np.sqrt(3.0))

    assert result["n_sigma"] == pytest.approx(scalar["n_sigma"], rel=1e-9)


def test_nd_refuses_mismatched_parameters():

    with pytest.raises(ValueError, match="different shapes"):
        gaussian_tension_nd([0.0, 1.0], np.eye(2), [0.0], [[1.0]])


# ============================================================
# Suspiciousness, end to end through the nested sampler
# ============================================================

@needs_dynesty
@pytest.mark.parametrize(
    "ndim,separation", [(1, 2.0), (1, 4.0), (2, 3.0)],
)
def test_suspiciousness_recovers_the_analytic_chi2(ndim, separation):
    """
    For two Gaussian likelihoods of unit width separated by ``s``,
    the parameter-difference chi2 is ``s^2 / 2`` exactly. The
    suspiciousness has to find it -- through three nested-sampling
    runs, their evidences and their KL divergences, none of which
    knows the answer.

    This is the check that the prior dependence really is being
    divided out: ``ln R`` alone would move with the box below, and
    ``ln S`` does not.
    """

    from CosmoFit.stats.nested import run_nested
    from CosmoFit.stats.priors import UniformPrior

    class Gaussians:

        def __init__(self, centres):
            self.centres = [np.asarray(c, dtype=float) for c in centres]

        def log_likelihood(self, theta):
            theta = np.asarray(theta, dtype=float)
            return float(
                sum(-0.5 * np.sum((theta - c) ** 2) for c in self.centres)
            )

    first_centre = np.zeros(ndim)
    second_centre = np.full(ndim, separation / np.sqrt(ndim))

    names = [f"p{i}" for i in range(ndim)]

    prior = UniformPrior(
        names, {n: (-12.0, 12.0) for n in names},
    )

    def evidence_for(centres):
        return run_nested(
            Gaussians(centres), prior, names,
            n_live=600, dlogz=0.01, seed=2, progress=False,
        )

    result = suspiciousness(
        evidence_for([first_centre, second_centre]),
        evidence_for([first_centre]),
        evidence_for([second_centre]),
    )

    assert result["chi2"] == pytest.approx(
        separation ** 2 / 2.0, rel=0.08,
    )

    assert result["d"] == ndim
