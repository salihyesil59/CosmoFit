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

import warnings

import numpy as np
import pytest

from CosmoFit import Fitter
from CosmoFit.cosmology.core import ModelConfigurationError
from CosmoFit.cosmology.models import LCDM


sympy = pytest.importorskip("sympy")

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

    with pytest.raises(ModelConfigurationError, match="z_init"):
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


# ============================================================
# Gravity that couples to the field
# ============================================================

def non_minimal(xi=0.02):
    """
    ``f = (1 + xi phi^2) R`` -- scalar-tensor gravity, where the
    field sets the strength of gravity rather than sitting on top
    of it. ``xi = 0`` is General Relativity.
    """

    return Action(
        "(1 + xi*phi**2)*R",
        fields={"phi": "X - V0"},
        params={
            "xi": {"default": xi, "bounds": (-0.5, 0.5)},
            "V0": {"default": 2.1, "bounds": (0.05, 20.0)},
        },
        closure="V0",
    ).build("NonMinimal")


def test_the_gravitational_sector_can_couple_to_a_field():
    """
    ``F(phi) R`` is scalar-tensor gravity, and writing it needs the
    field's name to be in scope where the *gravity* expression is
    parsed -- which it was not, so the whole model class could not
    be expressed even though the reduction handles it and the
    documentation said so.
    """

    action = Action(
        "(1 + xi*phi**2)*R",
        fields={"phi": "X - V0"},
        params={"xi": {"default": 0.02}, "V0": {"default": 2.1}},
        closure="V0",
    )

    assert sympy.Symbol("phi") in action.gravity.free_symbols

    # Linear in R, so it stays on the ordinary reduction rather
    # than being diverted to the fourth-order one.
    assert not action.is_fourth_order


def test_the_non_minimal_constraint_is_the_textbook_one():
    """
    Scalar-tensor gravity's Friedmann equation is

        3 F H^2 + 3 H dF/dt = rho,

    with ``F`` the coefficient of ``R/2``. The ``3 H dF/dt`` term is
    the whole content of the coupling, and it is also the term
    that vanishes if the field reaches the gravitational sector as
    a plain symbol rather than a function of time -- leaving a
    constraint that still looks entirely reasonable and describes
    a rescaled General Relativity instead.
    """

    from CosmoFit.theory.minisuperspace import friedmann_constraint

    action = Action(
        "(1 + xi*phi**2)*R",
        fields={"phi": "X - V0"},
        params={"xi": {"default": 0.02}, "V0": {"default": 2.1}},
        closure="V0",
    )

    ms, L = action.lagrangian()

    constraint = friedmann_constraint(L, ms)

    phi, a, t = ms.fields["phi"], ms.a, ms.t
    H = sympy.diff(a, t) / a

    xi, V0 = sympy.Symbol("xi"), sympy.Symbol("V0")
    Omega_m = sympy.Symbol("Omega_m")

    F = 2 * (1 + xi * phi**2)

    density = sympy.diff(phi, t) ** 2 / 2 + V0 + 3 * Omega_m / a**3

    textbook = 3 * F * H**2 / 2 + 3 * H * sympy.diff(F, t) / 2 - density

    ratio = sympy.simplify(
        sympy.cancel(constraint.subs(ms.k, 0) / (a**3 * textbook))
    )

    assert ratio == 1

    # The coupling term is there at all: the constraint depends on
    # the field's velocity, which a frozen F(phi) could not
    # produce. `free_symbols` would not show this -- a derivative
    # is not a symbol -- so ask the expression directly.
    assert constraint.has(sympy.Derivative(phi, t))


def test_switching_the_coupling_off_gives_lcdm():

    derived = make(non_minimal(xi=0.0), H0=70.0, Omega_m=0.3, phi_i=1.0)
    direct = LCDM(LCDM.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    assert np.allclose(derived.E(Z), direct.E(Z), rtol=1e-9, atol=0)

    assert derived.closure_value() == pytest.approx(
        3.0 * direct.Omega_de0, rel=1e-8,
    )


@pytest.mark.parametrize("xi", [0.02, -0.02])
def test_the_coupling_changes_the_history(xi):
    """
    And by an amount that depends on its sign -- a model where the
    field never reached the gravitational sector would give the
    same answer for both.
    """

    derived = make(non_minimal(xi=xi), H0=70.0, Omega_m=0.3, phi_i=1.0)
    direct = LCDM(LCDM.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    departure = np.max(np.abs(derived.E(Z) / direct.E(Z) - 1.0))

    assert departure > 1e-4
    assert derived.history(Z).drift < 1e-10
    assert float(derived.E(0.0)) == pytest.approx(1.0, abs=1e-9)


# ============================================================
# More than one field
# ============================================================

def test_two_fields_reduce_and_integrate():
    """
    Each field gets its own equation of motion and its own pair of
    initial conditions, and the closure condition still has one
    parameter to solve for.
    """

    action = Action(
        "R",
        fields={
            "phi": "X - V1*exp(-l1*phi)",
            "psi": "X - V2*exp(-l2*psi)",
        },
        params={
            "V1": {"default": 1.0, "bounds": (0.01, 20.0)},
            "V2": {"default": 1.1, "bounds": (0.01, 20.0)},
            "l1": {"default": 0.5},
            "l2": {"default": 1.0},
        },
        closure="V1",
    )

    assert set(action.field_equations()) == {"phi", "psi"}

    model = action.build("TwoField")

    assert {"phi_i", "dphi_i", "psi_i", "dpsi_i"} <= set(model.EXTRA_PARAMS)

    derived = make(
        model, H0=70.0, Omega_m=0.3,
        V2=1.1, l1=0.5, l2=1.0,
        phi_i=0.0, dphi_i=0.0, psi_i=0.0, dpsi_i=0.0,
    )

    assert float(derived.E(0.0)) == pytest.approx(1.0, abs=1e-9)
    assert derived.history(Z).drift < 1e-10

    # Both potentials are contributing: dropping one would change
    # what the closure has to solve for.
    assert derived.closure_value() > 1.0


# ============================================================
# Growth of structure
# ============================================================

def test_a_configuration_error_is_not_swallowed_into_infinity():
    """
    ``LogPosterior.chi2`` turns exceptions into an infinite
    chi-squared, which is right for a *parameter* the model cannot
    represent: a sampler that merely proposed it should not crash.

    It is wrong for a *configuration* it cannot represent. The
    growth machinery starts its ODE at z = 9999, so a field model
    built with the default ``z_init = 3000`` fails at every
    parameter value -- and used to do so as `chi2 = inf`
    everywhere, leaving a fit that ran to completion having learned
    nothing and saying nothing.
    """

    model = exponential_quintessence(z_init=3000.0)

    fit = Fitter(
        model=model,
        datasets=["fsigma8"],
        free_params=["H0", "Omega_m", "sigma8"],
        initial={"H0": 70.0, "Omega_m": 0.3, "sigma8": 0.81, "lam": 0.5},
    )

    with pytest.raises(ModelConfigurationError, match="z = 9999"):
        fit.chi2()


def test_growth_data_works_once_the_history_reaches_far_enough():
    """
    The counterpart: the fix the message names actually fixes it.
    """

    model = exponential_quintessence(z_init=20000.0)

    fit = Fitter(
        model=model,
        datasets=["fsigma8"],
        free_params=["H0", "Omega_m", "sigma8"],
        initial={"H0": 70.0, "Omega_m": 0.3, "sigma8": 0.81, "lam": 0.5},
    )

    assert np.isfinite(fit.chi2())

    # A minimally coupled field is dark energy on top of General
    # Relativity, so mu = 1 here is not an approximation.
    cosmology = make(model, H0=70.0, Omega_m=0.3, sigma8=0.81, lam=0.5)

    assert np.allclose(cosmology.mu(1.0 / (1.0 + Z)), 1.0)


# ============================================================
# Scalar-tensor growth
# ============================================================

def scalar_tensor(xi=0.05, growth="quasi_static"):

    return Action(
        "(1 + xi*phi**2)*R",
        fields={"phi": "X - V0"},
        params={
            "xi": {"default": xi, "bounds": (-0.5, 0.5)},
            "V0": {"default": 2.1, "bounds": (0.05, 20.0)},
        },
        closure="V0",
        growth=growth,
        z_init=20000.0,
    ).build("ScalarTensorGrowth")


def test_the_scalar_tensor_coupling_is_the_published_one():
    """
    Boisseau, Esposito-Farese, Polarski & Starobinsky (2000):

        G_eff = (1 / 8 pi F) (2F + 4 F_phi^2) / (2F + 3 F_phi^2)

    Checked against the formula evaluated by hand on the model's
    own field solution -- so this tests the wiring (which ``F``,
    which field value, at which time), not the algebra, which is
    the part that could quietly be attached to the wrong thing.
    """

    xi = 0.05

    model = make(
        scalar_tensor(xi), H0=70.0, Omega_m=0.3,
        xi=xi, phi_i=1.0, dphi_i=0.0,
    )

    a = 1.0 / (1.0 + Z)

    phi = model.history(Z).state(-np.log1p(Z))[1]

    F = 1.0 + xi * phi**2
    F_phi = 2.0 * xi * phi

    expected = (2 * F + 4 * F_phi**2) / (F * (2 * F + 3 * F_phi**2))

    assert np.allclose(model.mu(a), expected, rtol=1e-12)

    # And it is neither 1 nor constant, which is the whole point:
    # the strength of gravity moves as the field rolls.
    assert abs(float(model.mu(1.0)) - 1.0) > 0.1

    assert np.ptp(model.mu(a)) > 0.1


def test_the_coupling_reduces_to_general_relativity():
    """
    ``F = 1`` with a constant coupling gives ``mu = 1`` exactly --
    the normalization the formula has to satisfy.
    """

    model = make(
        scalar_tensor(xi=0.0), H0=70.0, Omega_m=0.3,
        xi=0.0, phi_i=1.0, dphi_i=0.0,
    )

    assert np.allclose(model.mu(1.0 / (1.0 + Z)), 1.0, rtol=0, atol=1e-14)


def test_quasi_static_needs_something_to_correct():
    """
    A minimally coupled field leaves gravity alone, so ``mu = 1``
    exactly and ``quasi_static`` would be claiming a correction
    that does not exist.
    """

    with pytest.raises(NotImplementedError, match="minimally coupled"):
        Action(
            "R",
            fields={"phi": "X - V0"},
            params={"V0": {"default": 2.1}},
            closure="V0",
            growth="quasi_static",
        ).build("Bad")


def test_growth_data_warns_when_the_coupling_is_ignored():
    """
    The silent-wrong-answer guard. Scalar-tensor gravity fit
    against growth data with ``mu = 1`` gives General Relativity's
    growth on top of a modified background -- a finite
    chi-squared, a plausible posterior, and no way to tell.
    """

    with pytest.warns(UserWarning, match="scalar-tensor"):
        Fitter(
            model=scalar_tensor(growth="gr"),
            datasets=["fsigma8"],
            free_params=["H0", "Omega_m", "sigma8"],
            initial={
                "H0": 70.0, "Omega_m": 0.3, "sigma8": 0.81,
                "xi": 0.05, "phi_i": 1.0,
            },
        )


@pytest.mark.parametrize(
    "model_kwargs, datasets",
    [
        (dict(growth="quasi_static"), ["fsigma8"]),   # coupling accounted for
        (dict(growth="gr"), ["cc"]),                  # no growth data
    ],
)
def test_the_coupling_warning_stays_quiet_otherwise(model_kwargs, datasets):

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)

        Fitter(
            model=scalar_tensor(**model_kwargs),
            datasets=datasets,
            free_params=["H0", "Omega_m", "sigma8"],
            initial={
                "H0": 70.0, "Omega_m": 0.3, "sigma8": 0.81,
                "xi": 0.05, "phi_i": 1.0,
            },
        )


# ============================================================
# The coupling that decides whether mu means anything
# ============================================================


def test_the_scalar_tensor_mu_is_guarded_on_the_coupling():
    """
    ``mu = (2F + 4s)/(F(2F + 3s))`` with ``s = sum (dF/dphi)^2``,
    which is a sum of squares and so never negative. That makes
    ``F > 0`` both necessary and sufficient for the expression to be
    positive and finite: the pole is at ``2F + 3s = 0``, which needs
    ``F = -3s/2 <= 0``.

    Evaluated directly, the formula does reach those values --
    ``F = 0, s = 0.5`` gives ``inf``; ``F = -0.5, s = 0`` gives
    ``-2``; ``F = -0.3, s = 0.2`` sits on the pole and gives
    ``-6e15``. What could *not* be produced through the public API
    is a model that gets there, because the closure solve refuses
    those parameter regions first. So this guard is defence in
    depth rather than a fix for a reachable bug, and it is written
    down that way in case a later action or a change to the closure
    exposes the region.
    """

    model = scalar_tensor(xi=0.05)

    built = make(model, H0=70.0, Omega_m=0.3, phi_i=1.0)

    mu = np.asarray(built.mu(np.array([1.0, 0.6])), dtype=float)

    assert np.all(mu > 0.0)

    assert np.all(np.isfinite(mu))


def test_the_healthy_scalar_tensor_branch_is_unchanged():
    """
    The guard must cost nothing where nothing was wrong. These are
    the numbers from before it existed.
    """

    built = make(scalar_tensor(xi=0.05), H0=70.0, Omega_m=0.3, phi_i=1.0)

    mu = np.asarray(built.mu(np.array([1.0, 0.6, 0.3])), dtype=float)

    assert mu == pytest.approx([0.78478, 0.80797, 0.83023], abs=1e-5)
