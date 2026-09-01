"""
Agreement with the Wolfram notebooks that derive the same physics.

The `wljs-gr-toolkit` notebooks (https://github.com/salihyesil59/wljs-gr-toolkit)
reduce a gravitational action on FLRW and solve the Friedmann
constraint symbolically in Wolfram Language. This library does the
same reduction in sympy -- :mod:`theory.minisuperspace` -- and
solves the result with its own root finder.

The two share no code. They share a convention list and the
physics, and nothing else, which is what makes comparing them worth
the trouble: an error in either has to be reproduced *exactly* by
the other to escape, and the two were written years and languages
apart.

`tests/test_ft.py` already pins Wolfram values for one model, taken
from the GR-06 notebook. This file widens that to the *engine* in
GR-02 -- `FriedmannEquations` composed with `HubbleFunction` -- and
covers both routes this library offers into a model: the
hand-written classes in `cosmology.models`, and
:class:`theory.Action`, which compiles the action itself.

Regenerating the references
---------------------------
The numbers below are printed by `cosmofit-reference.wls` in the
toolkit repository, which loads the GR-02 notebook and drives its
own functions rather than re-deriving anything::

    wolframscript -file cosmofit-reference.wls

Root-found at 30-digit working precision and reported to 25, so the
references are exact at every digit compared here and the tolerance
below is this library's error alone.

Conventions that have to line up
--------------------------------
GR-02 puts the torsion scalar at ``T = -6H^2`` on flat FLRW, which
is the sign :mod:`theory` uses for its teleparallel sector. It is
*not* the sign the GR-06 notebook uses, and it is not the one
``cosmology.models.fq`` uses for ``Q``. Those differences are
harmless where the closure absorbs them and lethal where it does
not, so this file exists partly to keep them honest.
"""

from __future__ import annotations

import pytest

from CosmoFit import FTPowerLaw, LCDM


pytest.importorskip("sympy", reason="theory.Action needs sympy")

from CosmoFit.theory import Action  # noqa: E402


H0 = 70.0
OMEGA_M = 0.3

#: The tightest tolerance the comparison supports. Both sides solve
#: the same constraint to far better than this; anything above it
#: means a derivation moved, not that a solver drifted.
RTOL = 1.0e-12


# ============================================================
# References, from GR-02
# ============================================================

#: E(z) for ``f = R - 2*Lambda``, closure fixing Lambda from
#: ``E(0) = 1``. GR-02 returns the closed form
#: ``sqrt(1 + Omega_m ((1+z)^3 - 1))``, i.e. textbook LCDM -- which
#: is the point: an action reduced by machine has to land on the
#: answer everyone already knows.
LCDM_REFERENCE = [
    (0.0, 1.0),
    (0.5, 1.30862523283024005734236883764),
    (1.0, 1.76068168616590091457692281765),
    (2.0, 2.96647939483826517948455897632),
]

#: E(z) for the f(T) power law ``f = T + alpha (-T)^n``, closure
#: fixing alpha. GR-02's constraint reads
#:
#:     E^2 - 6^(n-1) alpha (2n-1) H0^(2n-2) E^(2n) = Omega_m (1+z)^3
#:
#: and ``E(0) = 1`` sends the coefficient to ``1 - Omega_m``, which
#: is exactly the relation `FTPowerLaw` solves. That the two agree
#: as *algebra* before they agree as numbers is the stronger half of
#: this check.
FT_REFERENCE = [
    (-0.5, 0.0, 1.0),
    (-0.5, 0.5, 1.25338962590220103781424261629),
    (-0.5, 1.0, 1.67841034875758654147719923737),
    (-0.5, 2.0, 2.88831368578646360135172462738),
    (0.2, 0.0, 1.0),
    (0.2, 0.5, 1.34156820209982826060774631876),
    (0.2, 1.0, 1.81332689248254611586565583786),
    (0.2, 2.0, 3.03164322618141437237066027792),
    (0.7, 0.0, 1.0),
    (0.7, 0.5, 1.49858251246999489995529863241),
    (0.7, 1.0, 2.08882280850685585828131822190),
    (0.7, 2.0, 3.48020952245715310780422363728),
]


# ============================================================
# Compiled once: the action route is not cheap
# ============================================================


@pytest.fixture(scope="module")
def lcdm_from_action():

    return Action("R - 2*Lam", closure="Lam").build("LCDMfromAction")


@pytest.fixture(scope="module")
def ft_from_action():

    return Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={"A0": {"default": 1.0}, "b": {"default": 0.2}},
        closure="A0",
    ).build("FTfromAction")


# ============================================================
# LCDM
# ============================================================


@pytest.mark.parametrize("z,expected", LCDM_REFERENCE)
def test_lcdm_matches_the_notebook(z, expected):

    model = LCDM(LCDM.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M))

    assert float(model.E(z)) == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize("z,expected", LCDM_REFERENCE)
def test_lcdm_compiled_from_the_action_matches_the_notebook(
    lcdm_from_action, z, expected,
):
    """
    The same number, but with this library doing the variational
    calculus rather than quoting its result. This is the closest
    comparison of the two: Wolfram minisuperspace against sympy
    minisuperspace, same action in, same E(z) out.
    """

    model = lcdm_from_action(
        lcdm_from_action.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M)
    )

    assert float(model.E(z)) == pytest.approx(expected, rel=RTOL)


# ============================================================
# f(T) power law
# ============================================================


@pytest.mark.parametrize("n,z,expected", FT_REFERENCE)
def test_ft_power_law_matches_the_notebook(n, z, expected):

    model = FTPowerLaw(FTPowerLaw.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M, n=n))

    assert float(model.E(z)) == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize("n,z,expected", FT_REFERENCE)
def test_ft_compiled_from_the_action_matches_the_notebook(
    ft_from_action, n, z, expected,
):

    model = ft_from_action(
        ft_from_action.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M, b=n)
    )

    assert float(model.E(z)) == pytest.approx(expected, rel=RTOL)


# ============================================================
# What the agreement would miss
# ============================================================


def test_the_two_routes_are_not_the_same_code():
    """
    The comparison above is only worth something if the hand-written
    model and the compiled one really are independent. They are:
    one carries an `_solve_E2` written for this model, the other
    carries a constraint sympy produced from the action. Assert the
    distinction rather than trusting it, because a future
    refactor that made `FTPowerLaw` delegate to `Action` would turn
    four of the tests above into two without anything failing.
    """

    assert not issubclass(FTPowerLaw, Action)

    assert "ACTION" not in vars(FTPowerLaw)

    assert hasattr(FTPowerLaw, "_solve_E2")
