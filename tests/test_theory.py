"""
Deriving a cosmological model from its action.

``CosmoFit.theory`` inverts how every other model in this library
was written: instead of an ``E(z)`` transcribed from a paper, it
takes the action and does the variational calculus. That makes it
testable in a way a hand-written model is not -- it must
reproduce, from the action alone, the models somebody already
derived by hand.

Three independent checks, in increasing order of how much they
would catch:

  GR limit        an undeformed f in each of the three geometric
                  sectors must give back the Friedmann equation,
                  symbolically and exactly

  Lambda-CDM      'R - 2*Lam' must reproduce ``LCDM`` to machine
                  precision, curvature included

  f(Q)            'Q*exp(lam*Q0/Q)' must reproduce
                  ``FQExponential`` -- a transcendental constraint
                  whose hand-written counterpart needs a Lambert W
                  to invert, and whose growth coupling mu = 1/f_Q
                  is derived here rather than typed in
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import Fitter
from CosmoFit.cosmology.models import LCDM, FQExponential


sympy = pytest.importorskip("sympy")

from CosmoFit.theory import Action  # noqa: E402
from CosmoFit.theory.minisuperspace import (  # noqa: E402
    GEOMETRIES,
    GEOMETRY_SCALAR,
    Minisuperspace,
    fluid_lagrangian,
    friedmann_constraint,
    gravity_lagrangian,
    reduce_order,
)


Z = np.array([0.0, 0.05, 0.3, 0.8, 1.5, 3.0, 7.0])


# ============================================================
# Helpers
# ============================================================

def make(model, **params):
    """Instantiate a model with the given parameters."""

    return model(model.PARAMS_CLASS(**params))


def power_law_ft(**overrides):
    """
    Power-law f(T) gravity (Bengochea & Ferraro 2009),
    ``f(T) = T + A0 (-T)^b``. ``b = 0`` collapses the extra term
    to a constant, i.e. exactly a cosmological constant.
    """

    spec = dict(
        params={
            "A0": {"default": -4.2, "bounds": (-30.0, 0.0)},
            "b": {"default": 0.0, "bounds": (-2.0, 0.9)},
        },
        closure="A0",
    )

    spec.update(overrides)

    return Action(
        "T + A0*(-T)**b", geometry="teleparallel", **spec,
    ).build("PowerLawFT")


# ============================================================
# The reduction itself
# ============================================================

@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_undeformed_action_gives_general_relativity(geometry):
    """
    ``f = R``, ``f = T`` and ``f = Q`` must each reduce to the
    Friedmann equation exactly.

    This is what pins the sign conventions of the three geometry
    scalars, which differ between papers -- get one wrong and
    every modification built on it is inverted, with nothing
    downstream complaining.
    """

    ms = Minisuperspace()

    scalar = sympy.Symbol(GEOMETRY_SCALAR[geometry])

    Omega_m = sympy.Symbol("Omega_m", positive=True)

    L = gravity_lagrangian(ms, geometry, scalar, scalar)
    L += fluid_lagrangian(ms, {0: 3 * Omega_m})

    constraint = friedmann_constraint(reduce_order(L, ms), ms)

    a, t = ms.a, ms.t
    H = sympy.diff(a, t) / a

    if geometry == "metric":
        target = 3 * (H**2 + ms.k / a**2) - 3 * Omega_m / a**3

    else:
        # Neither teleparallel sector is defined on a curved FLRW
        # background in the gauges used here.
        target = 3 * H**2 - 3 * Omega_m / a**3
        constraint = constraint.subs(ms.k, 0)

    ratio = sympy.simplify(sympy.cancel(constraint / (a**3 * target)))

    assert ratio == 1


def test_general_fr_is_refused_not_approximated():
    """
    A general ``f(R)`` gives fourth-order field equations, which
    the integration-by-parts reduction cannot handle. It has to
    refuse: silently dropping a term that is not a total
    derivative would return a different theory's ``E(z)`` with no
    sign that anything happened.
    """

    with pytest.raises(ValueError, match="nonlinear"):
        Action(
            "R + A0*R**2", params={"A0": {"default": 0.1}},
        ).build("Starobinsky")


def test_scalar_field_equations_are_correct():
    """
    The reduction handles scalar fields, and must produce both the
    scalar-field Friedmann equation and the Klein-Gordon equation.
    """

    action = Action(
        "R",
        fields={"phi": "X - V0*exp(-lam*phi)"},
        params={
            "V0": {"default": 1.0},
            "lam": {"default": 1.0},
        },
    )

    ms, L = action.lagrangian()

    phi, a, t = ms.fields["phi"], ms.a, ms.t
    H = sympy.diff(a, t) / a

    V0, lam = sympy.Symbol("V0"), sympy.Symbol("lam")
    V = V0 * sympy.exp(-lam * phi)

    # No assumptions: `Action` builds its parameter symbols plain,
    # and a symbol carrying `positive=True` is a different symbol
    # that would refuse to cancel against it.
    Omega_m = sympy.Symbol("Omega_m")

    friedmann = friedmann_constraint(L, ms)

    target = 3 * (H**2 + ms.k / a**2) - (
        sympy.diff(phi, t) ** 2 / 2 + V + 3 * Omega_m / a**3
    )

    assert sympy.simplify(
        sympy.cancel(friedmann / (a**3 * target))
    ) == 1

    klein_gordon = action.field_equations()["phi"]

    target = (
        sympy.diff(phi, t, 2)
        + 3 * H * sympy.diff(phi, t)
        + sympy.diff(V, phi)
    )

    assert sympy.simplify(
        sympy.cancel(klein_gordon / (-a**3 * target))
    ) == 1


def test_building_a_model_with_fields_is_refused():
    """
    Fields reduce correctly but are not yet integrated; the
    difference has to be visible.
    """

    action = Action(
        "R",
        fields={"phi": "X - V0*phi**2"},
        params={"V0": {"default": 1.0}},
    )

    with pytest.raises(NotImplementedError, match="scalar fields"):
        action.build("Quintessence")


# ============================================================
# Lambda-CDM, rederived
# ============================================================

def test_lcdm_from_its_action_matches_lcdm():
    """
    ``R - 2*Lam`` against the hand-written ``LCDM``: the whole
    pipeline -- reduction, lapse variation, closure, solve -- in
    one number.
    """

    model = Action("R - 2*Lam", closure="Lam").build("LCDMfromAction")

    for Omega_k in (0.0, 0.05, -0.03):

        derived = make(model, H0=67.4, Omega_m=0.315, Omega_k=Omega_k)
        direct = make(LCDM, H0=67.4, Omega_m=0.315, Omega_k=Omega_k)

        assert np.allclose(derived.E(Z), direct.E(Z), rtol=0, atol=1e-14)
        assert np.allclose(
            derived.dEdz(Z), direct.dEdz(Z), rtol=0, atol=1e-13,
        )


def test_lcdm_closure_recovers_the_flatness_condition():
    """
    The closure condition ``E(0) = 1`` is what makes
    ``Lam = 3 Omega_de0``. Nothing in the action says so -- it
    comes out of the variation.
    """

    model = Action("R - 2*Lam", closure="Lam").build("LCDMfromAction")

    for Omega_m, Omega_k in ((0.315, 0.0), (0.25, 0.05), (0.4, -0.02)):

        derived = make(model, H0=70.0, Omega_m=Omega_m, Omega_k=Omega_k)
        direct = make(LCDM, H0=70.0, Omega_m=Omega_m, Omega_k=Omega_k)

        assert derived.closure_value() == pytest.approx(
            3.0 * direct.Omega_de0, rel=1e-12,
        )


def test_unclosed_action_is_an_error():
    """
    An action nobody has normalized predicts every distance wrong
    by a constant factor and no fit would look obviously broken,
    so building one must fail rather than warn.
    """

    with pytest.raises(ValueError, match=r"E\(0\)"):
        Action(
            "R - 2*Lam", params={"Lam": {"default": 1.0}},
        ).build("Unclosed")


# ============================================================
# f(Q), rederived
# ============================================================

def build_fq():

    return Action(
        "Q * exp(lam * Q0 / Q)",
        geometry="symmetric",
        params={"lam": {"default": 0.3}},
        closure="lam",
        growth="quasi_static",
    ).build("FQfromAction")


def test_fq_constraint_matches_the_published_equation():
    """
    The derived constraint must have the same zero set as
    ``FQExponential``'s stated Friedmann equation,
    ``(E^2 - 2 lam) exp(lam/E^2) = Omega_m (1+z)^3`` -- i.e. agree
    with it up to an overall non-vanishing factor, which is all a
    constraint is defined up to.
    """

    constraint, E2, z = Action(
        "Q * exp(lam * Q0 / Q)",
        geometry="symmetric",
        params={"lam": {"default": 0.3}},
        closure="lam",
    ).constraint()

    lam = sympy.Symbol("lam")
    Omega_m = sympy.Symbol("Omega_m")

    published = (E2 - 2 * lam) * sympy.exp(lam / E2) - Omega_m * (
        1 + z
    ) ** 3

    ratio = sympy.simplify(sympy.cancel(constraint / published))

    assert ratio.free_symbols <= {z}
    assert ratio != 0


@pytest.mark.parametrize("Omega_m", [0.25, 0.30, 0.35])
def test_fq_from_its_action_matches_fq_exponential(Omega_m):
    """
    Against the hand-written model, whose ``lam`` comes from a
    Lambert ``W`` and whose ``E(z)`` is solved with it.

    The continuation solve has to land on the same branch that
    ``W_0`` picks; a root-finder let loose on this constraint need
    not.
    """

    derived = make(build_fq(), H0=68.0, Omega_m=Omega_m)
    direct = make(FQExponential, H0=68.0, Omega_m=Omega_m)

    assert derived.closure_value() == pytest.approx(direct._lam, rel=1e-10)

    assert np.allclose(derived.E(Z), direct.E(Z), rtol=0, atol=1e-13)

    a = 1.0 / (1.0 + Z)

    assert np.allclose(derived.mu(a), direct.mu(a), rtol=0, atol=1e-13)


def test_quasi_static_growth_is_off_by_default():
    """
    ``mu = 1/f'`` is a statement about perturbations that a
    background action does not by itself imply, so it has to be
    asked for -- the default stays at GR growth.
    """

    default = make(
        Action(
            "Q * exp(lam * Q0 / Q)",
            geometry="symmetric",
            params={"lam": {"default": 0.3}},
            closure="lam",
        ).build("FQnoGrowth"),
        H0=68.0, Omega_m=0.3,
    )

    assert np.allclose(default.mu(1.0 / (1.0 + Z)), 1.0)

    asked = make(build_fq(), H0=68.0, Omega_m=0.3)

    assert not np.allclose(asked.mu(1.0 / (1.0 + Z)), 1.0)


def test_quasi_static_growth_refused_in_the_metric_sector():

    with pytest.raises(NotImplementedError, match="quasi_static"):
        Action("R - 2*Lam", closure="Lam", growth="quasi_static").build("X")


# ============================================================
# Power-law f(T)
# ============================================================

def test_power_law_ft_collapses_to_lcdm_at_b_zero():
    """
    At ``b = 0`` the correction ``A0 (-T)^b`` is a constant, so
    the model is Lambda-CDM -- reached through a completely
    different sector (torsion, and a numerically solved
    constraint) than :func:`test_lcdm_from_its_action_matches_lcdm`.
    """

    derived = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=0.0)
    direct = make(LCDM, H0=70.0, Omega_m=0.3)

    assert np.allclose(derived.E(Z), direct.E(Z), rtol=0, atol=1e-13)

    assert derived.closure_value() == pytest.approx(
        -6.0 * direct.Omega_de0, rel=1e-12,
    )


@pytest.mark.parametrize("b", [-0.8, -0.3, 0.0, 0.4])
def test_derivative_is_exact_not_finite_differenced(b):
    """
    ``dEdz`` comes from implicit differentiation of the
    constraint. Checked against a high-order finite difference of
    ``E`` itself, which is the only independent handle on it.
    """

    model = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=b)

    z = np.array([0.2, 0.7, 1.4, 2.6])
    h = 1e-5

    numerical = (
        model.E(z - 2 * h) - 8 * model.E(z - h)
        + 8 * model.E(z + h) - model.E(z + 2 * h)
    ) / (12 * h)

    assert np.allclose(model.dEdz(z), numerical, rtol=1e-8, atol=0)


def test_effective_dark_energy_completes_the_budget():
    """
    ``Omega_de(z)`` is defined as whatever ``E(z)^2`` holds beyond
    the explicit fluids and curvature, so putting it back must
    return ``E(z)^2`` exactly.
    """

    model = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=-0.4)

    total = (
        model.Omega_m * (1.0 + Z) ** 3
        + model.Omega_k * (1.0 + Z) ** 2
        + model.Omega_de(Z)
    )

    assert np.allclose(total, model.E(Z) ** 2, rtol=0, atol=1e-14)


def test_power_law_ft_is_degenerate_at_b_one_half():
    """
    At ``b = 1/2`` the correction drops out of the Friedmann
    equation identically -- the combination that enters it,
    ``(1 - 2b) A0 (-T)^b``, vanishes -- so ``A0`` is not fixed by
    anything and the model has no normalization.

    That is a property of power-law f(T), not a numerical
    accident, and it has to be reported as one rather than as a
    division by zero from inside the distance integrator.
    """

    with pytest.raises(RuntimeError, match="degeneracy of the model"):
        make(power_law_ft(), H0=70.0, Omega_m=0.3, b=0.5)


# ============================================================
# The solver
# ============================================================

def test_seeded_and_cold_solves_agree():
    """
    The seed cache exists only to skip the continuation walk. It
    must never change the answer -- so a model that has been
    warmed up has to agree with a freshly built one.
    """

    fresh = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=-0.5)
    cold = fresh.E(Z)

    warm = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=-0.5)

    for _ in range(5):
        warm.E(Z)

    warm.params.Omega_m = 0.31
    warm.refresh()
    warm.E(Z)

    warm.params.Omega_m = 0.3
    warm.refresh()

    # Not bit-identical: Newton stops at its own tolerance, and the
    # path taken to get there differs. Agreeing to the last couple
    # of bits is the claim -- the seed changes the work, not the
    # answer.
    assert np.allclose(warm.E(Z), cold, rtol=0, atol=1e-14)


def test_e_of_zero_is_exactly_one():
    """
    The condition the whole closure machinery exists to enforce,
    checked on the numerically solved path rather than the
    symbolic one.
    """

    for b in (-1.0, -0.2, 0.0, 0.4):

        model = make(power_law_ft(), H0=70.0, Omega_m=0.3, b=b)

        assert float(model.E(0.0)) == pytest.approx(1.0, abs=1e-12)


# ============================================================
# Specification errors
# ============================================================

def test_action_without_a_geometry_scalar_is_refused():

    with pytest.raises(ValueError, match="none of the geometry"):
        Action("-2*Lam", closure="Lam")


def test_action_with_two_geometry_scalars_is_refused():
    """
    ``R`` and ``T`` are different formulations of gravity, not two
    terms to be added.
    """

    with pytest.raises(ValueError, match="more than one geometry"):
        Action("R + T")


def test_undeclared_parameter_is_refused():

    with pytest.raises(ValueError, match="Unknown name"):
        Action("R - 2*Lam + beta", closure="Lam")


def test_parameter_colliding_with_a_standard_one_is_refused():

    with pytest.raises(ValueError, match="collides"):
        Action(
            "R - 2*Lam + Omega_m",
            params={"Omega_m": {"default": 0.3}},
            closure="Lam",
        )


def test_closure_on_a_standard_parameter_is_declared_derived():
    """
    When the condition ``E(0) = 1`` fixes a *standard* parameter
    rather than one of the action's own, the model must say so --
    otherwise a fit is free to sample a number nothing reads.
    """

    model = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={
            "A0": {"default": -4.2},
            "b": {"default": 0.0},
        },
        closure="Omega_m",
    ).build("DerivedOmegaM")

    assert model.DERIVED_PARAMS == frozenset({"Omega_m"})


# ============================================================
# End to end
# ============================================================

def test_a_generated_model_fits_real_data():
    """
    The point of all of this: a model specified as an action goes
    through the ordinary ``Fitter`` against the ordinary datasets.

    ``b`` is the deviation from Lambda-CDM, so the fit must both
    converge and land near zero -- published f(T) constraints are
    consistent with Lambda-CDM.
    """

    fit = Fitter(
        model=power_law_ft(),
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "b", "rd"],
        initial={"H0": 70.0, "Omega_m": 0.3, "b": 0.0, "rd": 147.0},
    )

    result = fit.best_fit(restarts=2, seed=0)

    assert result.success
    assert np.isfinite(result.fun)
    assert abs(dict(zip(fit.free_params, result.x))["b"]) < 0.5
