"""
Fourth-order gravity: a general f(R).

The ordinary reduction refuses these -- integrating the ``addot``
in the Einstein-Hilbert term away by parts is legitimate only while
the Lagrangian is linear in it. ``theory.curvature`` gets round that
by promoting ``R`` to an independent variable held to its geometric
value by a Lagrange multiplier, which makes the Lagrangian linear in
``addot`` again at the cost of one extra dynamical variable.

Four independent things are checked, because "it runs and E(z) is
finite" would pass on a wrong derivation:

  the equation     the derived Friedmann constraint against the
                   textbook f(R) form, symbolically

  the solution     the *third* equation of motion, which the
                   integration never uses, evaluated along the
                   solution -- it holds by the Bianchi identity, so
                   its residual measures the error

  the limit        R + alpha R^2 must approach Lambda-CDM as
                   alpha -> 0, and does so at the right rate

  the direction    this integrates backwards from today, unlike
                   ``theory.fields``. That is safe only if the
                   system is well-conditioned backwards, which is
                   measured here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import Fitter
from CosmoFit.cosmology.models import LCDM


sympy = pytest.importorskip("sympy")

from CosmoFit.theory import Action  # noqa: E402
from CosmoFit.theory.curvature import is_higher_order  # noqa: E402


Z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 3.0])

#: The Lambda-CDM value of R today, in units of H0^2:
#: R = 6 (2 H^2 + Hdot) = 6 (2 - 3 Omega_m / 2) H0^2.
def lcdm_ricci(Omega_m):
    return 6.0 * (2.0 - 1.5 * Omega_m)


def starobinsky(**overrides):
    """
    ``f(R) = R - 2 Lambda + alpha R^2`` -- a curvature-squared
    correction on top of a cosmological constant, which is the
    simplest genuinely fourth-order thing to write.
    """

    spec = dict(
        params={
            "Lam": {"default": 2.1, "bounds": (0.0, 6.0)},
            "alpha_fr": {"default": 1.0e-3, "bounds": (1.0e-6, 1.0)},
        },
    )
    spec.update(overrides)

    return Action("R - 2*Lam + alpha_fr*R**2", **spec).build("Starobinsky")


def make(model, Omega_m=0.3, **params):

    return model(
        model.PARAMS_CLASS(
            H0=70.0, Omega_m=Omega_m, R_0=lcdm_ricci(Omega_m), **params,
        )
    )


# ============================================================
# Which reduction an action needs
# ============================================================

def test_only_a_nonlinear_f_is_fourth_order():
    """
    ``R - 2 Lambda`` is linear in R and goes through the ordinary
    reduction, where the closure condition fixes ``Lambda``. Adding
    an ``R^2`` makes it fourth-order and sends it elsewhere. Getting
    that boundary wrong would either refuse Lambda-CDM or hand a
    fourth-order theory to a second-order solver.
    """

    R = sympy.Symbol("R")
    Lam, alpha = sympy.symbols("Lam alpha")

    assert not is_higher_order(R - 2 * Lam, R)
    assert is_higher_order(R + alpha * R**2, R)
    assert is_higher_order(R * sympy.exp(alpha * R), R)

    assert not Action("R - 2*Lam", closure="Lam").is_fourth_order
    assert Action(
        "R + alpha_fr*R**2", params={"alpha_fr": {"default": 0.1}},
    ).is_fourth_order


def test_the_teleparallel_sectors_are_never_fourth_order():
    """
    ``f(T)`` and ``f(Q)`` are built from first derivatives only, so
    nonlinearity in them costs nothing -- they must not be diverted
    into the multiplier reduction.
    """

    action = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={"A0": {"default": -4.2}, "b": {"default": 0.0}},
        closure="A0",
    )

    assert not action.is_fourth_order


# ============================================================
# The equation
# ============================================================

def test_the_derived_constraint_is_the_textbook_one():
    """
    The f(R) Friedmann equation is

        3 f_R H^2 = (f_R R - f)/2 - 3 H d(f_R)/dt + rho,

    and the reduction has to produce exactly that -- up to an
    overall non-vanishing factor, which is all a constraint is
    defined up to.
    """

    from CosmoFit.theory.curvature import multiplier_lagrangian
    from CosmoFit.theory.minisuperspace import (
        Minisuperspace, fluid_lagrangian, friedmann_constraint, reduce_order,
    )

    R = sympy.Symbol("R")
    alpha = sympy.Symbol("alpha")
    Omega_m = sympy.Symbol("Omega_m")

    f = R + alpha * R**2

    ms = Minisuperspace(curvature=True)

    L = multiplier_lagrangian(ms, f, R)
    L += fluid_lagrangian(ms, {0: 3 * Omega_m})

    constraint = friedmann_constraint(reduce_order(L, ms), ms)

    a, t = ms.a, ms.t
    H = sympy.diff(a, t) / a

    f_on_shell = ms.R + alpha * ms.R**2
    f_R = 1 + 2 * alpha * ms.R

    textbook = (
        3 * f_R * H**2
        - (f_R * ms.R - f_on_shell) / 2
        + 3 * H * sympy.diff(f_R, t)
        - 3 * Omega_m / a**3
    )

    ratio = sympy.simplify(
        sympy.cancel(constraint.subs(ms.k, 0) / (a**3 * textbook))
    )

    assert ratio == 1


# ============================================================
# The solution
# ============================================================

@pytest.mark.parametrize("alpha", [1.0e-1, 1.0e-2, 1.0e-3])
def test_the_unused_equation_of_motion_holds(alpha):
    """
    Two of the three equations define the right-hand side, so their
    residuals are zero by construction and measure nothing. The
    third -- from varying ``a`` -- follows from them by the Bianchi
    identity, so it holds on an exact solution and drifts on an
    approximate one.

    This is the honest accuracy measure for this system, and the
    counterpart of the constraint drift in ``theory.fields``.
    """

    model = make(starobinsky(), alpha_fr=alpha)

    assert model.history(Z).drift < 1e-10


def test_the_expansion_rate_today_is_exactly_one():
    """
    Not shot for, unlike a scalar-field action: the history is
    integrated outwards from ``a = 1`` with ``H = 1`` imposed there.
    """

    for alpha in (1.0e-1, 1.0e-3, 1.0e-5):
        model = make(starobinsky(), alpha_fr=alpha)
        assert float(model.E(0.0)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("alpha", [1.0e-1, 1.0e-3])
def test_derivative_matches_a_finite_difference(alpha):

    model = make(starobinsky(), alpha_fr=alpha)

    z = np.array([0.2, 0.7, 1.4, 2.6])
    h = 1e-4

    numerical = (
        model.E(z - 2 * h) - 8 * model.E(z - h)
        + 8 * model.E(z + h) - model.E(z + 2 * h)
    ) / (12 * h)

    assert np.allclose(model.dEdz(z), numerical, rtol=1e-6, atol=0)


# ============================================================
# The limit
# ============================================================

def test_the_correction_switches_off_smoothly():
    """
    ``alpha -> 0`` is General Relativity with a cosmological
    constant, and the model has to approach Lambda-CDM as it --
    monotonically, and by roughly a decade in the departure for
    every two decades in ``alpha``.

    A model that merely *looked* like Lambda-CDM at one value of
    ``alpha`` would pass a single-point check; this is what makes
    it a statement about the derivation.
    """

    model = starobinsky()
    reference = LCDM(LCDM.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    departures = []

    for alpha in (1.0e-1, 1.0e-3, 1.0e-5):
        derived = make(model, alpha_fr=alpha)
        departures.append(
            float(np.max(np.abs(derived.E(Z) / reference.E(Z) - 1.0)))
        )

    assert departures[0] > departures[1] > departures[2]
    assert departures[-1] < 1.0e-2

    # Lambda itself is not fit here: 2.1 is 3 * Omega_de0 at
    # Omega_m = 0.3, so the alpha -> 0 limit is *this* Lambda-CDM.
    assert departures[-1] < departures[0] / 100.0


# ============================================================
# The direction
# ============================================================

def test_integrating_backwards_is_well_conditioned():
    """
    ``theory.fields`` integrates forwards because backwards is
    where a scalar field runs away. This module integrates
    backwards, and the difference is only defensible if this system
    does not.

    So: perturb the one initial condition and see how far the
    answer moves at high redshift. An unstable system amplifies
    exponentially in e-folds; this one must not.
    """

    model = starobinsky()

    far = np.array([0.0, 1100.0])

    base = make(model, alpha_fr=1.0e-3)
    kicked = starobinsky()(
        starobinsky().PARAMS_CLASS(
            H0=70.0, Omega_m=0.3,
            R_0=lcdm_ricci(0.3) * (1.0 + 1.0e-8),
            alpha_fr=1.0e-3,
        )
    )

    amplification = abs(
        float(kicked.E(1100.0)) / float(base.E(1100.0)) - 1.0
    ) / 1.0e-8

    assert amplification < 10.0

    assert np.all(np.isfinite(base.E(far)))


# ============================================================
# Specification errors
# ============================================================

def test_a_closure_parameter_is_refused():
    """
    Nothing needs closing: ``H = 1`` at ``a = 1`` holds by
    construction, and the freedom a closure would have fixed is
    carried by ``R_0``. Accepting a ``closure=`` here would leave a
    parameter being solved for against a condition already
    satisfied.
    """

    with pytest.raises(ValueError, match="no closure condition"):
        Action(
            "R - 2*Lam + alpha_fr*R**2",
            params={"Lam": {"default": 2.1}, "alpha_fr": {"default": 1e-3}},
            closure="Lam",
        ).build("Bad")


def test_quasi_static_growth_is_scale_dependent_here():
    """
    f(R)'s mu is scale-dependent -- a Compton wavelength enters --
    so the scale-free ``1/f'`` the teleparallel sectors use would be
    wrong here rather than approximate. This used to be refused for
    that reason; it is now supplied in the correct scale-dependent
    form, and what this test guards is that it did not come back as
    the scale-free one.

    See `tests/test_theory_growth.py` for the physics; the point
    here is only that a fourth-order action accepts the option and
    that `k` reaches the answer.
    """

    model = Action(
        "R - 2*Lam + alpha_fr*R**2",
        params={"Lam": {"default": 2.1}, "alpha_fr": {"default": 1e-3}},
        growth="quasi_static",
    ).build("QuasiStatic")

    Omega_m = 0.3

    built = model(
        model.PARAMS_CLASS(
            H0=70.0, Omega_m=Omega_m, R_0=lcdm_ricci(Omega_m),
        )
    )

    assert float(built.mu(1.0, 1e-6)) != float(built.mu(1.0, 10.0))


def test_the_ricci_scalar_today_is_declared_as_a_parameter():
    """
    A fourth-order theory has an initial condition General
    Relativity does not, and it has to be visible as one.
    """

    model = starobinsky()

    assert "R_0" in model.EXTRA_PARAMS

    derived = make(model)
    assert float(derived.ricci(0.0)) == pytest.approx(lcdm_ricci(0.3), rel=1e-9)


# ============================================================
# End to end
# ============================================================

def test_a_fourth_order_model_fits_real_data():

    fit = Fitter(
        model=starobinsky(),
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "Lam", "rd"],
        initial={
            "H0": 70.0, "Omega_m": 0.3, "Lam": 2.1,
            "alpha_fr": 1.0e-3, "R_0": lcdm_ricci(0.3), "rd": 147.0,
        },
    )

    result = fit.best_fit(restarts=0, seed=0)

    assert result.success
    assert np.isfinite(result.fun)
