"""
Physics checks on the f(T) power-law model.

The generic closure and derivative checks in ``test_models.py``
cover this model too, once it is listed in ``ALL_MODELS``. What is
here instead is everything specific to f(T): the two limits where
the answer is known in closed form, the pole the model must refuse,
and a regression table computed by a second, independent
implementation.

The limits matter more than they look. ``n = 0`` is not
"approximately LCDM", it is LCDM identically -- ``f = T + alpha``
is teleparallel general relativity plus a cosmological constant --
so the assertion is exact equality rather than a tolerance, and it
pins the closure constant ``A = (Omega_m - 1)/(2n - 1)`` that
``E(0) = 1`` alone would leave with a free sign. ``n = 1`` is a
rescaled TEGR with no constant term, which is Einstein-de Sitter
whatever ``Omega_m`` is, and that catches the opposite error: an
``A`` that wrongly survives into a case where it must cancel.
"""

from __future__ import annotations

import numpy as np
import pytest

import CosmoFit as C
from CosmoFit.cosmology.core.errors import ModelConfigurationError


H0 = 67.4
OMEGA_M = 0.315

Z = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])


def build(n, Omega_m=OMEGA_M, **kwargs):

    return C.FTPowerLaw(
        C.FTPowerLaw.PARAMS_CLASS(
            H0=H0, Omega_m=Omega_m, n=n, **kwargs
        )
    )


def build_lcdm(Omega_m=OMEGA_M, **kwargs):

    return C.LCDM(
        C.LCDM.PARAMS_CLASS(H0=H0, Omega_m=Omega_m, **kwargs)
    )


# ============================================================
# 1. The LCDM limit, exactly
# ============================================================

@pytest.mark.parametrize("Omega_m", [0.2, 0.315, 0.4])
def test_n_zero_is_lcdm_exactly(Omega_m):
    """
    f(T) = T + alpha is TEGR plus a cosmological constant, so n = 0
    is flat LCDM identically -- not to a tolerance.
    """

    ft = build(0.0, Omega_m=Omega_m)
    lcdm = build_lcdm(Omega_m=Omega_m)

    assert np.allclose(ft.E(Z), lcdm.E(Z), rtol=0.0, atol=1e-14)
    assert np.allclose(ft.dEdz(Z), lcdm.dEdz(Z), rtol=0.0, atol=1e-13)


def test_n_zero_has_gr_growth():
    """
    The LCDM limit has to be a limit of the growth history too, not
    only of the background: mu = 1 identically at n = 0.
    """

    ft = build(0.0)

    assert np.allclose(ft.mu(1.0 / (1.0 + Z)), 1.0, rtol=0.0, atol=1e-15)


# ============================================================
# 2. The Einstein-de Sitter limit
# ============================================================

@pytest.mark.parametrize("Omega_m", [0.2, 0.315, 0.4])
def test_n_one_is_einstein_de_sitter(Omega_m):
    """
    f = (1 + alpha) T is a rescaled TEGR with no constant term, and
    the rescaling cancels out of the Friedmann equation, leaving
    E^2 = (1+z)^3 whatever Omega_m is.

    n = 1 is outside the default prior bounds -- it is a
    decelerating universe, not a candidate fit -- but the model must
    still return the right answer there, since this is the check
    that an over-eager closure constant would fail.
    """

    ft = build(1.0, Omega_m=Omega_m)

    assert np.allclose(ft.E(Z), (1.0 + Z) ** 1.5, rtol=0.0, atol=1e-12)


# ============================================================
# 3. Closure, and the pole
# ============================================================

@pytest.mark.parametrize("n", [-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.45])
def test_closure(n):
    """
    E(z=0) = 1 is what fixes A; it must hold for every n on the
    branch.
    """

    assert build(n).E(0.0) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("n", [0.5, 0.4999, 0.5001])
def test_half_is_refused_by_mu_only(n):
    """
    n = 1/2 is a pole of the growth sector, not of the background.
    The Friedmann coefficient (2n-1)A is Omega_m - 1 whatever n is,
    so E(z) is perfectly well defined there and must be returned;
    A itself diverges, so mu must refuse rather than hand back a
    number that came out of cancelling infinities.
    """

    model = build(n)

    assert model.E(0.0) == pytest.approx(1.0, abs=1e-13)
    assert np.all(np.isfinite(model.E(Z)))

    with pytest.raises(ModelConfigurationError):
        model.mu(1.0)


@pytest.mark.parametrize("n", [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0])
def test_background_is_pole_free(n):
    """
    The background coefficient is Omega_m - 1 for every n, so
    closure and finiteness hold right across the pole.
    """

    model = build(n)

    assert model._c == pytest.approx(OMEGA_M - 1.0, rel=1e-15)
    assert model.E(0.0) == pytest.approx(1.0, abs=1e-13)


def test_a_is_the_closure_value():
    """
    A = (Omega_m - 1)/(2n - 1), derived rather than fitted.
    """

    for n in (-2.0, -0.5, 0.25):
        assert build(n)._A == pytest.approx(
            (OMEGA_M - 1.0) / (2.0 * n - 1.0), rel=1e-14
        )


# ============================================================
# 4. Derivative consistency
# ============================================================

@pytest.mark.parametrize("n", [-2.0, -1.0, -0.5, 0.0, 0.25, 0.45])
def test_dEdz_matches_finite_difference(n):
    """
    dEdz is implicit differentiation of the Friedmann relation; E
    is a Newton solve of it. Two independent routes to the same
    physics, so agreement is evidence about the algebra.
    """

    ft = build(n)
    h = 1e-5
    z = Z[Z > 0]

    fd = (ft.E(z + h) - ft.E(z - h)) / (2.0 * h)

    assert np.allclose(ft.dEdz(z), fd, rtol=1e-7, atol=1e-9)


# ============================================================
# 5. GR limits of the effective coupling
# ============================================================

def test_mu_is_one_when_omega_m_is_one():
    """
    Omega_m = 1 sends A to zero at any n, which is bare TEGR: the
    coupling must be exactly Newton's.
    """

    for n in (-2.0, -0.5, 0.25, 0.45):
        ft = build(n, Omega_m=1.0)
        assert np.allclose(
            ft.mu(1.0 / (1.0 + Z)), 1.0, rtol=0.0, atol=1e-14
        )


def test_mu_direction_follows_the_sign_of_n():
    """
    mu = 1/(1 + n A E^(2n-2)) with A > 0 on this branch, so
    negative n strengthens gravity and positive n weakens it. A
    sign slip in mu would swap these and still look plausible.
    """

    a = 1.0 / (1.0 + Z)

    assert np.all(build(-1.0).mu(a) > 1.0)
    assert np.all(build(0.25).mu(a) < 1.0)


# ============================================================
# 6. Against a second implementation
# ============================================================

#: (n, z, E, mu), computed independently in Wolfram Language by
#: root-finding the Friedmann relation at 25-digit working
#: precision -- see the wljs-gr-toolkit GR-06 notebook for the
#: derivation these came from. Agreement here is the strongest
#: check available on this model, since it shares no code with the
#: implementation under test. Both sides use H0 = 67.4,
#: Omega_m = 0.315.
_REFERENCE = [
    (-2.00, 0.5, 1.186776794697002, 1.108733524897053),
    (-2.00, 1.0, 1.618581191174844, 1.015474356122953),
    (-2.00, 2.0, 2.917952847151761, 1.000444092838205),
    (-0.50, 0.5, 1.266487631788033, 1.092060752545759),
    (-0.50, 1.0, 1.709037834353426, 1.035525173675091),
    (-0.50, 2.0, 2.955799016726122, 1.006675679155192),
    (0.25, 0.5, 1.365085155684325, 0.823218745678726),
    (0.25, 1.0, 1.858447029765138, 0.880912043594461),
    (0.25, 2.0, 3.116781513892959, 0.941402843407730),
    (0.45, 0.5, 1.413704056691545, 0.321932186466781),
    (0.45, 1.0, 1.940027235400939, 0.402088931064834),
    (0.45, 2.0, 3.236735361532349, 0.541474677623002),
]


@pytest.mark.parametrize("n,z,E_ref,mu_ref", _REFERENCE)
def test_matches_independent_implementation(n, z, E_ref, mu_ref):

    ft = build(n)

    assert float(ft.E(z)) == pytest.approx(E_ref, abs=1e-12)
    assert float(ft.mu(1.0 / (1.0 + z))) == pytest.approx(
        mu_ref, abs=1e-12
    )


# ============================================================
# 7. Against the action, derived symbolically
# ============================================================

#: ``theory.Action`` reduces a gravitational Lagrangian to a
#: Friedmann constraint with sympy, and it uses the *opposite* sign
#: convention for the torsion scalar: ``T = -6H^2``, with the power
#: law written ``T + A0 (-T)^b`` so the power is taken of a positive
#: number. This class was derived in the ``T = +6H^2`` convention.
#: Agreement therefore tests the convention as well as the algebra,
#: and it shares no code with the implementation under test.
_ACTION_NS = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.45]


@pytest.mark.parametrize("n", _ACTION_NS)
def test_matches_the_action_derivation(n):

    pytest.importorskip("sympy", reason="CosmoFit.theory needs sympy")

    from CosmoFit.theory import Action

    derived = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={
            "A0": {"default": -4.2, "bounds": (-30.0, 0.0)},
            "b": {"default": 0.0, "bounds": (-2.0, 0.9)},
        },
        closure="A0",
        growth="quasi_static",
    ).build("PowerLawFT")

    from_action = derived(
        derived.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M, b=n)
    )
    by_hand = build(n)

    a = 1.0 / (1.0 + Z)

    assert np.allclose(
        np.asarray(from_action.E(Z)), by_hand.E(Z),
        rtol=0.0, atol=1e-14,
    )
    assert np.allclose(
        np.asarray(from_action.mu(a)), np.asarray(by_hand.mu(a)),
        rtol=0.0, atol=1e-14,
    )
