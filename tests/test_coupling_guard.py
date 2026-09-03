"""
``mu = 1/f'`` in the teleparallel and symmetric-teleparallel
sectors, and the two ways it stops meaning anything.

The metric sector got this treatment first: `quasi_static_mu`
refuses to return a value from the far side of the scalaron pole,
and `viability` reports `f_R > 0`. The other sector had neither,
and it has more models in it.

What was actually happening, measured before the guard existed --
`FTPowerLaw` at `n = 0.6`, across `z = 0, 0.5, 1, 2`::

    mu = -0.91, -1.79, -4.88, +5.08

Negative is repulsive gravity. The sign change between the third
and fourth is the denominator passing through zero. Every one of
those numbers is finite, the growth equation integrates them
without complaint, and `fsigma8` comes out smooth. A fit sampling
`n` in that region would have been fitting nonsense and could not
have known.

`f' > 0` is the standard viability condition here, the counterpart
of `f_R > 0` in the metric sector.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import FQExponential, FTPowerLaw
from CosmoFit.cosmology.core.utils import coupling_from_derivative


Z = np.array([0.0, 0.5, 1.0, 2.0])
A = 1.0 / (1.0 + Z)


# ============================================================
# 1. The helper itself
# ============================================================


def test_a_positive_derivative_is_simply_inverted():

    got = coupling_from_derivative(np.array([1.0, 2.0, 0.5]))

    assert np.allclose(got, [1.0, 0.5, 2.0])


def test_a_negative_derivative_is_refused_as_repulsive():

    with pytest.raises(ValueError, match="repulsive gravity"):
        coupling_from_derivative(np.array([1.0, -0.5]))


def test_a_vanishing_derivative_is_refused_as_singular():

    with pytest.raises(ValueError, match="singular"):
        coupling_from_derivative(np.array([1.0, 0.0]))


def test_a_non_finite_derivative_is_reported_as_its_own_fault():
    """
    Separately from the sign, because the numbers do not describe
    each other: quoting the smallest finite value when the real
    problem is a NaN names a perfectly good number as the culprit,
    which a first version of this did.
    """

    with pytest.raises(ValueError, match="non-finite"):
        coupling_from_derivative(np.array([1.0, np.nan]))

    with pytest.raises(ValueError, match="2 of 3 points"):
        coupling_from_derivative(np.array([1.0, np.nan, np.inf]))


# ============================================================
# 2. Through the models that had the problem
# ============================================================


@pytest.mark.parametrize("n", [-1.0, 0.25, 0.45, 0.8])
def test_the_healthy_branches_are_untouched(n):
    """
    The guard must not cost anything where nothing was wrong.
    These are the same numbers as before it existed.
    """

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=n))

    mu = np.asarray(model.mu(A), dtype=float)

    assert np.all(np.isfinite(mu))

    assert np.all(mu > 0.0)


def test_the_repulsive_branch_is_refused():
    """
    ``n = 0.6`` is the case that motivated all of this.
    """

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=0.6))

    with pytest.raises(ValueError, match="repulsive gravity"):
        model.mu(A)


def test_growth_cannot_be_computed_on_the_repulsive_branch():
    """
    The point of refusing in `mu` rather than downstream: the
    growth calculator consumes it, and a negative coupling there
    produces a perfectly smooth fsigma8.
    """

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(
        H0=70.0, Omega_m=0.3, n=0.6, sigma8=0.81,
    ))

    with pytest.raises(ValueError, match="repulsive|non-finite"):
        model.background.fsigma8(Z)


def test_the_exponential_fq_model_is_healthy_and_stays_so():
    """
    `FQExponential` at its defaults is well behaved, and goes
    through the same helper -- so this checks the wiring as much as
    the model.
    """

    model = FQExponential(FQExponential.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    mu = np.asarray(model.mu(A), dtype=float)

    assert np.all(mu > 0.0)

    assert np.all(np.isfinite(mu))


# ============================================================
# 3. And through the action compiler, which shares the fault
# ============================================================


def test_the_compiled_teleparallel_path_agrees_with_the_written_one():
    """
    Both routes reach `mu = 1/f'`, so both had the same hole. This
    pins that they now agree *and* that they agree with the
    hand-written model's numbers.
    """

    pytest.importorskip("sympy")

    from CosmoFit.theory import Action

    model = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={"A0": {"default": 1.0}, "b": {"default": 0.45}},
        closure="A0",
        growth="quasi_static",
    ).build("FTCompiled")

    compiled = model(model.PARAMS_CLASS(H0=70.0, Omega_m=0.3, b=0.45))

    written = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=0.45))

    assert np.allclose(
        np.asarray(compiled.mu(A), dtype=float),
        np.asarray(written.mu(A), dtype=float),
        rtol=1e-10,
    )


# ============================================================
# 4. Reporting, as opposed to refusing
# ============================================================
#
# `mu` raises, because a caller asking for a number must not be
# handed a meaningless one. But "is this model admissible" is a
# different question and deserves an answer rather than an
# exception -- which is what the metric sector's `viability()`
# already gave, and this sector did not.


@pytest.mark.parametrize("n", [-0.5, 0.2, 0.45])
def test_a_healthy_model_reports_viable(n):

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=n))

    verdict = model.viability(np.linspace(0.0, 3.0, 100))

    assert verdict["ok"]

    assert verdict["failed"] == []


@pytest.mark.parametrize("n", [0.6, 1.5])
def test_an_unphysical_model_reports_why_instead_of_raising(n):
    """
    The same two cases `mu` refuses. Here they come back as a
    verdict, so a caller can ask before committing to a fit.
    """

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=n))

    verdict = model.viability(np.linspace(0.0, 3.0, 100))

    assert not verdict["ok"]

    assert verdict["failed"] == ["f_prime"]

    assert "repulsive" in verdict["reasons"][0]


def test_the_coupling_itself_is_reachable():
    """
    `coupling(z)` returns `f'`, the counterpart of the metric
    sector's `scalaron(z)`. Its reciprocal is `mu` where `mu` is
    defined at all.
    """

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=0.2))

    f_prime = np.asarray(model.coupling(Z), dtype=float)

    assert np.all(f_prime > 0.0)

    assert np.allclose(
        1.0 / f_prime, np.asarray(model.mu(A), dtype=float), rtol=1e-12,
    )


def test_the_exponential_model_reports_too():

    model = FQExponential(FQExponential.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    assert model.viability()["ok"]

    assert np.all(np.asarray(model.coupling(Z), dtype=float) > 0.0)


def test_the_compiled_path_gives_the_same_verdict():
    """
    Both routes into `mu = 1/f'` now report as well as refuse, and
    they have to agree about which models are admissible.
    """

    pytest.importorskip("sympy")

    from CosmoFit.theory import Action

    model = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={"A0": {"default": 1.0}, "b": {"default": 0.45}},
        closure="A0",
        growth="quasi_static",
    ).build("FTVerdict")

    grid = np.linspace(0.0, 3.0, 100)

    for b, expected in ((0.45, True), (0.6, False)):

        compiled = model(model.PARAMS_CLASS(H0=70.0, Omega_m=0.3, b=b))

        written = FTPowerLaw(
            FTPowerLaw.PARAMS_CLASS(H0=70.0, Omega_m=0.3, n=b)
        )

        assert compiled.viability(grid)["ok"] is expected

        assert written.viability(grid)["ok"] is expected
