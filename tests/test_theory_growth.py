"""
The scale-dependent ``mu`` a general ``f(R)`` gets from
:class:`theory.Action`.

Until this, ``growth="quasi_static"`` was refused for fourth-order
actions: the teleparallel sectors have a scale-free ``mu = 1/f'``,
and f(R) does not, so returning one would have been wrong rather
than approximate. :func:`theory.curvature.quasi_static_mu` supplies
the scale-dependent form,

    mu = (1/f_R) (1 + 4m)/(1 + 3m),   m = (k/a)^2 f_RR/f_R

with ``R`` read off this model's own integrated background.

What is actually at risk here is not the algebra -- the GR-05
notebook derives ``G_eff`` from the perturbed field equations and
its result is *identically* this expression, checked by subtracting
the two in Wolfram and getting zero, not by comparing summaries --
but the *units*.
``m`` is dimensionless only if ``(k/a)^2`` and ``f_RR`` are in
matching units, and getting that wrong shifts the Compton
wavelength without making anything raise: growth comes out smooth,
plausible, and wrong. So the load-bearing test here is the one
against `FRHuSawicki`, which converts the same quantities through
independently written code.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import FRHuSawicki


pytest.importorskip("sympy", reason="theory.Action needs sympy")

from CosmoFit.theory import Action  # noqa: E402
from CosmoFit.theory.curvature import quasi_static_mu  # noqa: E402


H0 = 70.0
OMEGA_M = 0.3


def lcdm_ricci(Omega_m):
    """R today in units of H0^2 for an LCDM background."""

    return 6.0 * (2.0 - 1.5 * Omega_m)


def starobinsky(alpha=1.0e-3, growth="quasi_static"):

    return Action(
        "R - 2*Lam + alpha_fr*R**2",
        params={
            "Lam": {"default": 2.1, "bounds": (0.0, 6.0)},
            "alpha_fr": {"default": alpha, "bounds": (0.0, 1.0)},
        },
        growth=growth,
    ).build("StarobinskyGrowth")


def build(model, **params):

    return model(
        model.PARAMS_CLASS(
            H0=H0, Omega_m=OMEGA_M, R_0=lcdm_ricci(OMEGA_M), **params,
        )
    )


@pytest.fixture(scope="module")
def star():

    return starobinsky()


# ============================================================
# 1. Against a second implementation of the same physics
# ============================================================


@pytest.mark.parametrize("a", [1.0, 0.7, 0.4])
@pytest.mark.parametrize("k", [0.005, 0.05, 0.5])
def test_matches_fr_hu_sawicki_given_the_same_scalaron(a, k):
    """
    `FRHuSawicki` writes the same limit a different way,

        mu = 1 + (1/3) Y^2/(Y^2 + Mhat^2),   Mhat^2 = 1/(3 f_RR)

    which is ``(1 + 4m)/(1 + 3m)`` at ``f_R = 1`` after the algebra
    is done. Feeding its own ``f_RR`` into the general formula has
    to reproduce its own answer.

    This is the test that pins the units. The two routes convert
    ``k`` [h/Mpc] into ``H0/c`` units through separately written
    code, so a factor of ``c``, of ``100``, or of ``a`` misplaced in
    either shows up here and essentially nowhere else.
    """

    model = FRHuSawicki(
        FRHuSawicki.PARAMS_CLASS(
            H0=H0, Omega_m=OMEGA_M, f_R0=-1.0e-4, n=1.0,
        )
    )

    u = model.Omega_m * a ** -3.0 + 4.0 * model.Omega_de0
    u0 = model.Omega_m + 4.0 * model.Omega_de0

    Mhat2 = -(u ** (model.n + 2.0)) / (
        (model.n + 1.0) * model.f_R0 * u0 ** (model.n + 1.0)
    )

    expected = float(model.mu(a, k))

    got = float(quasi_static_mu(1.0, 1.0 / (3.0 * Mhat2), a, k))

    assert got == pytest.approx(expected, rel=1.0e-12)


# ============================================================
# 2. The two limits, which are the physics
# ============================================================


def test_large_scales_give_one_over_f_R(star):
    """
    Far outside the scalaron's Compton wavelength there is no fifth
    force, only the rescaled Newton constant: mu -> 1/f_R.
    """

    model = build(star)

    R = float(model.ricci(0.0))

    f_R = 1.0 + 2.0 * model.params.alpha_fr * R

    # The limit is approached as k^2, so the wavenumber has to be
    # genuinely small before a 1e-9 comparison means anything: at
    # k = 1e-6 the residual m is still ~2e-8, which is a real
    # feature of the model rather than solver error.
    assert float(model.mu(1.0, 1.0e-8)) == pytest.approx(1.0 / f_R, rel=1e-9)


def test_small_scales_give_four_thirds_of_it(star):
    """
    Far inside it the scalar is massless and contributes its full
    third: mu -> 4/(3 f_R).
    """

    model = build(star)

    R = float(model.ricci(0.0))

    f_R = 1.0 + 2.0 * model.params.alpha_fr * R

    assert float(model.mu(1.0, 1.0e4)) == pytest.approx(
        4.0 / (3.0 * f_R), rel=1e-9,
    )


def test_mu_rises_monotonically_with_k(star):
    """
    Between those two limits mu is monotone in k. A sign slip in
    ``m`` keeps both endpoints and inverts the approach, which no
    endpoint test would catch.
    """

    model = build(star)

    k = np.logspace(-6.0, 4.0, 60)

    mu = np.array([float(model.mu(1.0, kk)) for kk in k])

    assert np.all(np.diff(mu) > 0.0)


# ============================================================
# 3. Collapsing onto general relativity
# ============================================================


def test_mu_approaches_one_in_proportion_to_alpha():
    """
    ``alpha -> 0`` removes the fourth-order term and with it the
    scalaron, so mu goes to 1.

    Stated as a rate rather than as a tolerance, because "mu is 1
    when alpha is small" is not well posed on its own: the
    departure is ``m = (k/a)^2 f_RR/f_R``, so for any alpha there
    is a k where it is not small. What is true at fixed k is that
    the departure falls *linearly* in alpha, and asserting that
    catches a mu which tends to the wrong constant as well as one
    that does not tend to anything.
    """

    k = 0.1

    deviations = [
        abs(float(build(starobinsky(alpha=alpha)).mu(1.0, k)) - 1.0)
        for alpha in (1.0e-8, 1.0e-10)
    ]

    assert deviations[1] < 1.0e-4

    # Two decades of alpha, two decades of departure.
    assert deviations[0] / deviations[1] == pytest.approx(100.0, rel=0.05)


# ============================================================
# 4. Shape and defaults
# ============================================================


def test_mu_is_vectorised_over_the_scale_factor(star):

    model = build(star)

    a = np.array([1.0, 0.8, 0.5, 0.25])

    mu = model.mu(a, 0.1)

    assert mu.shape == a.shape

    assert np.all(np.isfinite(mu))

    for i, ai in enumerate(a):
        assert mu[i] == pytest.approx(float(model.mu(float(ai), 0.1)))


def test_default_k_matches_the_hand_written_model(star):
    """
    Both f(R) routes answer the same question when `mu` is called
    without a wavenumber. If one drifted, growth predictions would
    differ between them for no stated reason.
    """

    model = build(star)

    assert model._DEFAULT_K == FRHuSawicki._DEFAULT_K

    assert float(model.mu(1.0)) == pytest.approx(
        float(model.mu(1.0, model._DEFAULT_K)),
    )


# ============================================================
# 5. The refusal that is still correct
# ============================================================


def test_growth_defaults_to_gr_and_leaves_mu_alone():
    """
    Without ``growth="quasi_static"`` a compiled f(R) keeps the
    inherited ``mu = 1``. Turning the scale-dependent coupling on by
    accident would silently change every growth prediction.
    """

    model = build(starobinsky(growth="gr"))

    assert float(model.mu(1.0, 0.1)) == 1.0


# ============================================================
# 6. The admissibility gate
# ============================================================
#
# Whether the theory is one worth fitting at all. A free function
# can be tuned to fit almost any data, so what separates a proposal
# from a curve is that its perturbations behave: `f_R > 0` keeps the
# graviton from being a ghost, `f_RR >= 0` keeps the scalaron from
# being tachyonic. Neither is a stylistic preference and neither is
# expensive to check.


def test_a_positive_r_squared_correction_is_viable(star):

    verdict = build(star).viability()

    assert verdict["ok"]

    assert verdict["failed"] == []


def test_a_negative_r_squared_correction_is_tachyonic(star):
    """
    ``alpha < 0`` gives ``f_RR = 2 alpha < 0``, so the scalaron's
    mass squared is negative -- the Dolgov-Kawasaki instability.
    The model still integrates and still produces a perfectly
    smooth E(z), which is exactly why this needs checking rather
    than eyeballing.
    """

    model = build(star, alpha_fr=-1.0e-3)

    assert np.all(np.isfinite(model.E(np.linspace(0.0, 3.0, 20))))

    verdict = model.viability()

    assert not verdict["ok"]

    assert verdict["failed"] == ["f_RR"]

    assert "tachyonic" in verdict["reasons"][0]


def test_mu_refuses_rather_than_returning_the_far_side_of_the_pole(star):
    """
    With ``f_RR < 0`` the denominator ``1 + 3m`` passes through
    zero, and beyond it the formula still evaluates to a finite
    number. That number is meaningless, and returning it would put
    a plausible-looking growth history into a fit.
    """

    model = build(star, alpha_fr=-1.0e-3)

    with pytest.raises(ValueError, match="not viable"):
        model.mu(1.0, 0.5)


def test_the_scalaron_is_reported_in_the_documented_units(star):
    """
    ``f_R`` dimensionless, ``f_RR`` in 1/H0^2 -- the same units the
    integrated ``R`` is in. For ``f = R - 2L + alpha R^2`` these are
    ``1 + 2 alpha R`` and ``2 alpha`` exactly, which is worth
    pinning because everything above depends on the pair being in
    the units `quasi_static_mu` expects.
    """

    model = build(star)

    R = float(model.ricci(0.0))
    alpha = float(model.params.alpha_fr)

    f_R, f_RR = model.scalaron(0.0)

    assert float(f_R) == pytest.approx(1.0 + 2.0 * alpha * R, rel=1e-12)
    assert float(f_RR) == pytest.approx(2.0 * alpha, rel=1e-12)


def test_the_gr_boundary_is_not_reported_as_a_failure():
    """
    ``f_RR = 0`` is general relativity, where the scalaron is absent
    rather than sick. Flagging it would reject the one limit every
    f(R) has to pass through.
    """

    from CosmoFit.theory.curvature import viability_failures

    assert viability_failures(1.0, 0.0) == []

    assert viability_failures(1.0, -1e-30) == ["f_RR"]

    assert viability_failures(-1.0, 1.0) == ["f_R"]

    assert sorted(viability_failures(-1.0, -1.0)) == ["f_R", "f_RR"]
