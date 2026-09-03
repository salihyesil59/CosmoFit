"""
The gate on the hand-written `FRHuSawicki`.

The compiled f(R) models answer `viability()` and `screening()`;
the canonical hand-written one, which is the f(R) most likely to be
fitted against growth data, answered neither. It is also the model
where the question is easiest to ask, because `f_R0` *is* the
quantity the Solar System bound is written on -- it is
`f_R(0) - 1`, the fractional departure of the gravitational
coupling today.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import FRHuSawicki
from CosmoFit.cosmology.core.utils import SOLAR_SYSTEM_BOUND


def build(f_R0=-1e-6, n=1.0):

    return FRHuSawicki(
        FRHuSawicki.PARAMS_CLASS(H0=70.0, Omega_m=0.3, f_R0=f_R0, n=n)
    )


# ============================================================
# Screening
# ============================================================


def test_the_deviation_is_the_parameter_itself():

    assert build(f_R0=-3e-5).screening()["deviation"] == pytest.approx(3e-5)


def test_the_class_default_sits_exactly_on_the_bound():
    """
    ``f_R0 = -1e-6`` is the default and 1e-6 is the bound, so this
    is the one comparison where the arithmetic has to be exact --
    and where a first version got it wrong.

    Forming ``1 + f_R0`` and subtracting 1 again returns
    ``1.0000000000287e-06``, three parts in 1e17 above the bound.
    That was enough to fail the default model, and to give
    ``f_R0 = +1e-6`` the opposite verdict to ``-1e-6`` for the same
    magnitude. The deviation is taken from ``f_R0`` directly now.
    """

    assert build(f_R0=-SOLAR_SYSTEM_BOUND).screening()["ok"]

    assert not build(f_R0=-1.0000001e-6).screening()["ok"]

    # same magnitude, same answer, whichever sign
    assert (
        build(f_R0=+SOLAR_SYSTEM_BOUND).screening()["ok"]
        == build(f_R0=-SOLAR_SYSTEM_BOUND).screening()["ok"]
    )


def test_a_model_big_enough_to_matter_is_excluded():
    """
    The tension these models live in: `f_R0` large enough to change
    growth measurably is far above what local tests allow.
    """

    verdict = build(f_R0=-1e-4).screening()

    assert not verdict["ok"]

    assert verdict["deviation"] / verdict["bound"] == pytest.approx(100.0)


def test_the_bound_can_be_relaxed_for_weaker_environments():

    assert build(f_R0=-1e-4).screening(bound=1e-3)["ok"]


# ============================================================
# Viability, which is a different question
# ============================================================


def test_the_default_is_a_consistent_theory():

    assert build().viability()["ok"]


def test_a_positive_scalaron_amplitude_is_tachyonic():
    """
    The scalaron mass squared goes as ``-1/f_R0`` here, so positive
    ``f_R0`` makes it tachyonic. The class bounds forbid that;
    constructing the model directly does not, which is why the
    check exists.
    """

    verdict = build(f_R0=+1e-6).viability()

    assert not verdict["ok"]

    assert verdict["failed"] == ["f_R0"]

    assert "tachyonic" in verdict["reasons"][0]


def test_consistent_and_allowed_are_asked_separately():
    """
    The same model can pass one and fail the other, which is the
    whole reason they are two methods. At ``f_R0 = -1e-2`` the
    theory is perfectly consistent and excluded by four orders of
    magnitude.
    """

    model = build(f_R0=-1e-2)

    assert model.viability()["ok"]

    assert not model.screening()["ok"]


def test_the_bounds_already_exclude_the_tachyonic_branch():
    """
    Worth pinning: the guard added here is a second line, not the
    first. A fit cannot reach positive `f_R0` because the prior
    forbids it.
    """

    low, high = FRHuSawicki.EXTRA_PARAMS["f_R0"]["bounds"]

    assert low < 0.0 and high < 0.0


# ============================================================
# And the growth it feeds
# ============================================================


def test_mu_stays_between_one_and_four_thirds_on_the_viable_branch():
    """
    Hu-Sawicki's ``mu`` runs from 1 far outside the Compton
    wavelength to 4/3 far inside. Values outside that range mean
    the formula has left its domain -- which is what the tachyonic
    branch produces, and part of why it is refused.
    """

    a = np.array([1.0, 0.6, 0.3])

    for f_R0 in (-1e-8, -1e-6, -1e-4, -1e-2):

        mu = np.asarray(build(f_R0=f_R0).mu(a, 0.1), dtype=float)

        assert np.all(mu >= 1.0 - 1e-12)

        assert np.all(mu <= 4.0 / 3.0 + 1e-12)
