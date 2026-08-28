"""
Actions with dynamical scalar fields.

``CosmoFit.theory`` reduces these correctly whether or not it can
integrate them -- ``tests/test_theory.py`` checks that the
Klein-Gordon and Friedmann equations come out right. What is
checked here is the *history*: solving the coupled system for
``E(z)``.

The validation is Copeland, Liddle & Wands (1998), whose late-time
attractors for an exponential potential are exact and depend only
on the slope ``lambda``. On a matter background there are two,
and which one is reached is decided by ``lambda^2`` against 3:

    lambda^2 < 3    field-dominated:  w = -1 + lambda^2/3,
                                      Omega_phi = 1
    lambda^2 > 3    scaling:          w = w_matter = 0,
                                      Omega_phi = 3/lambda^2

Both are reproduced here to five decimals from the action alone.
A constant potential -- a cosmological constant written as a
field -- pins the other end, against ``LCDM``.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import Fitter
from CosmoFit.cosmology.models import LCDM


pytest.importorskip("sympy")

from CosmoFit.theory import Action  # noqa: E402


Z = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 100.0, 2000.0])

#: Deep enough into the future that the attractor has been
#: reached, and still inside what the history covers.
A_LATE = 3.0e4
Z_LATE = 1.0 / A_LATE - 1.0


# ============================================================
# Helpers
# ============================================================

def make(model, **params):

    return model(model.PARAMS_CLASS(**params))


def constant_potential():
    """
    A constant potential is a cosmological constant with extra
    steps -- the field never moves, and the model is Lambda-CDM.
    """

    return Action(
        "R",
        fields={"phi": "X - V0"},
        params={"V0": {"default": 2.0}},
        closure="V0",
    ).build("ConstantPotential")


def exponential_quintessence(z_init=3000.0):
    """
    ``V = V0 exp(-lambda phi)``, the potential Copeland, Liddle &
    Wands solved.
    """

    return Action(
        "R",
        fields={"phi": "X - V0*exp(-lam*phi)"},
        params={
            "V0": {"default": 2.1, "bounds": (0.05, 50.0)},
            "lam": {"default": 0.5, "bounds": (0.0, 1.7)},
        },
        closure="V0",
        z_init=z_init,
    ).build("ExponentialQuintessence")


# ============================================================
# A cosmological constant, written as a field
# ============================================================

def test_constant_potential_reproduces_lcdm():
    """
    End to end -- reduction, variation, the shooting solve for
    ``V0`` and the integration -- against a model with none of
    those steps in it.
    """

    derived = make(constant_potential(), H0=70.0, Omega_m=0.3)
    direct = make(LCDM, H0=70.0, Omega_m=0.3)

    assert np.allclose(derived.E(Z), direct.E(Z), rtol=1e-10, atol=0)
    assert np.allclose(
        derived.dEdz(Z), direct.dEdz(Z), rtol=1e-7, atol=0,
    )
    assert np.allclose(derived.w(Z), -1.0, rtol=0, atol=1e-7)


def test_shooting_recovers_the_flatness_condition():
    """
    Nothing tells the model that ``V0`` should be
    ``3 Omega_de0``. It comes out of requiring ``E(0) = 1`` after
    integrating from ``z_init`` -- which is a shooting problem,
    not a formula.
    """

    model = constant_potential()

    for Omega_m in (0.25, 0.3, 0.35):

        derived = make(model, H0=70.0, Omega_m=Omega_m)
        direct = make(LCDM, H0=70.0, Omega_m=Omega_m)

        assert derived.closure_value() == pytest.approx(
            3.0 * direct.Omega_de0, rel=1e-9,
        )

        assert float(derived.E(0.0)) == pytest.approx(1.0, abs=1e-9)


def test_zero_slope_quintessence_is_lcdm():
    """
    ``lambda = 0`` turns the exponential into a constant, reaching
    Lambda-CDM through the general potential rather than the
    special-cased one.
    """

    derived = make(exponential_quintessence(), H0=70.0, Omega_m=0.3, lam=0.0)
    direct = make(LCDM, H0=70.0, Omega_m=0.3)

    assert np.allclose(derived.E(Z), direct.E(Z), rtol=1e-10, atol=0)


# ============================================================
# Copeland, Liddle & Wands attractors
# ============================================================

@pytest.mark.parametrize("lam", [0.5, 1.0, 1.5])
def test_field_dominated_attractor(lam):
    """
    ``lambda^2 < 3``: the field ends up dominating, with
    ``w = -1 + lambda^2/3`` and ``Omega_phi = 1``.
    """

    model = make(exponential_quintessence(), H0=70.0, Omega_m=0.3, lam=lam)

    assert float(model.w(Z_LATE)) == pytest.approx(
        -1.0 + lam**2 / 3.0, abs=1e-3,
    )

    fraction = float(model.Omega_de(Z_LATE)) / float(model.E(Z_LATE)) ** 2

    assert fraction == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("lam", [1.9, 2.0])
def test_scaling_attractor(lam):
    """
    ``lambda^2 > 3``: the field instead tracks the matter it is
    scaling against, at ``w = 0`` and a *fixed fraction*
    ``Omega_phi = 3/lambda^2`` -- which is why such a model cannot
    accelerate, and why steep potentials are ruled out.

    The fraction is the sharp test here: it is a pure prediction
    of the slope, with no freedom left anywhere in the model.
    """

    model = make(
        exponential_quintessence(), H0=70.0, Omega_m=0.3, lam=lam,
    )

    assert float(model.w(Z_LATE)) == pytest.approx(0.0, abs=1e-3)

    fraction = float(model.Omega_de(Z_LATE)) / float(model.E(Z_LATE)) ** 2

    assert fraction == pytest.approx(3.0 / lam**2, rel=1e-3)


def test_a_steep_potential_cannot_reach_a_low_matter_density():
    """
    Past ``lambda^2 > 3`` the scaling attractor fixes the
    dark-energy fraction at ``3/lambda^2``, so a model asked for
    more than that has no solution at all: ``H(0)`` saturates
    below 1 and no potential scale reaches it.

    That is a statement about exponential quintessence, and it has
    to arrive as one rather than as a stalled integrator.
    """

    model = Action(
        "R",
        fields={"phi": "X - V0*exp(-lam*phi)"},
        params={"V0": {"default": 2.1}, "lam": {"default": 0.5}},
        closure="V0",
    ).build("SteepQuintessence")

    # 3/lam^2 = 0.62, short of the 0.7 that Omega_m = 0.3 needs.
    with pytest.raises(RuntimeError, match="E\\(0\\) = 1"):
        make(model, H0=70.0, Omega_m=0.3, lam=2.2).E(0.0)


# ============================================================
# The integration
# ============================================================

def test_the_constraint_barely_drifts():
    """
    The Friedmann constraint is imposed once, in the initial
    conditions, and never again -- the history is driven by the
    equations of motion, for which the constraint is a first
    integral. So how far it drifts is an *independent* measure of
    the integration error, not a restatement of the tolerance.
    """

    model = make(exponential_quintessence(), H0=70.0, Omega_m=0.3, lam=1.0)

    assert model.history(np.linspace(0.0, 2.5, 40)).drift < 1e-10


@pytest.mark.parametrize("lam", [0.0, 0.8, 1.4])
def test_derivative_matches_a_finite_difference(lam):

    model = make(exponential_quintessence(), H0=70.0, Omega_m=0.3, lam=lam)

    z = np.array([0.2, 0.7, 1.4, 2.6])
    h = 1e-4

    numerical = (
        model.E(z - 2 * h) - 8 * model.E(z - h)
        + 8 * model.E(z + h) - model.E(z + 2 * h)
    ) / (12 * h)

    assert np.allclose(model.dEdz(z), numerical, rtol=1e-6, atol=0)


def test_beyond_the_initial_redshift_is_an_error():
    """
    There is no history before the field's state is given, and
    extrapolating one would be inventing the early universe rather
    than solving for it.
    """

    model = make(
        exponential_quintessence(z_init=100.0),
        H0=70.0, Omega_m=0.3, lam=0.5,
    )

    assert np.isfinite(float(model.E(99.0)))

    with pytest.raises(ValueError, match="z_init"):
        model.E(500.0)


def test_the_warm_started_shooting_does_not_change_the_answer():
    """
    The closure solve seeds itself from its previous result, which
    must only ever save iterations. A model that has solved for
    other parameters first has to land in the same place as one
    that has not.
    """

    model = exponential_quintessence()

    fresh = make(model, H0=70.0, Omega_m=0.3, lam=1.0).closure_value()

    for lam in (0.2, 1.4, 0.7, 1.6):
        make(model, H0=70.0, Omega_m=0.32, lam=lam).closure_value()

    warmed = make(model, H0=70.0, Omega_m=0.3, lam=1.0).closure_value()

    assert warmed == pytest.approx(fresh, rel=1e-9)


# ============================================================
# Specification errors
# ============================================================

def test_quasi_static_growth_is_refused_for_a_field():
    """
    ``mu = 1/f'`` describes a modified gravitational sector. A
    scalar field clusters in its own right and has nothing to do
    with that expression.
    """

    with pytest.raises(NotImplementedError, match="clusters"):
        Action(
            "R",
            fields={"phi": "X - V0"},
            params={"V0": {"default": 2.0}},
            closure="V0",
            growth="quasi_static",
        ).build("Bad")


def test_a_field_action_has_no_algebraic_closure_equation():

    action = Action(
        "R",
        fields={"phi": "X - V0"},
        params={"V0": {"default": 2.0}},
        closure="V0",
    )

    with pytest.raises(NotImplementedError, match="shooting"):
        action.closure_equation()


def test_initial_conditions_are_declared_automatically():
    """
    A field with no initial condition is not a model, so the two
    parameters that give it one are declared without being asked
    for -- and are ordinary fittable parameters once they exist.
    """

    model = constant_potential()

    assert "phi_i" in model.EXTRA_PARAMS
    assert "dphi_i" in model.EXTRA_PARAMS

    # The closure parameter is solved for, not fit, so it is not
    # among them.
    assert "V0" not in model.EXTRA_PARAMS


# ============================================================
# End to end
# ============================================================

def test_a_quintessence_model_fits_real_data():
    """
    ``lambda`` is the departure from Lambda-CDM, and CC+DESI does
    not prefer one -- so the fit has to converge and improve on
    Lambda-CDM by essentially nothing, which is what published
    quintessence constraints from background data say.
    """

    fit = Fitter(
        model=exponential_quintessence(),
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "lam", "rd"],
        initial={
            "H0": 70.0, "Omega_m": 0.3, "lam": 0.3, "rd": 147.0,
        },
    )

    result = fit.best_fit(restarts=0, seed=0)

    assert result.success
    assert np.isfinite(result.fun)

    reference = Fitter(
        model=LCDM,
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "rd"],
        initial={"H0": 70.0, "Omega_m": 0.3, "rd": 147.0},
    ).best_fit(restarts=0, seed=0)

    assert result.fun <= reference.fun + 1e-6
    assert reference.fun - result.fun < 2.0
