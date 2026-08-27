"""
Holographic dark energy, checked against its own definition.

Every other background model in this library is a formula, and a
test can compare it with the formula. HDE is an ODE solution, so
there is nothing to compare against by inspection -- which is
exactly the situation where a wrong equation of motion produces a
smooth, plausible expansion history that is simply not the model.

So the central test here does not check the ODE. It checks the
*definition the ODE was derived from*: holographic dark energy is
``rho_DE = 3 c^2 M_p^2 / L^2`` with ``L`` the future event horizon,
which is equivalent to

    H(a) L(a) = c / sqrt(Omega_DE(a)),
    L(a) = a Int_a^inf da' / (a'^2 H(a')).

The event-horizon integral is computed here directly from the
solved expansion history, with the ODE used nowhere in it. If the
equation of motion were wrong -- a sign, a factor of two, the wrong
power of Omega_DE -- the two sides would not agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.integrate import cumulative_trapezoid

from CosmoFit import HDE


def build(Omega_m=0.30, c_hde=0.80, H0=68.0):

    return HDE(
        HDE.PARAMS_CLASS(H0=H0, Omega_m=Omega_m, c_hde=c_hde),
    )


# ============================================================
# The definition
# ============================================================

@pytest.mark.parametrize("c_hde", [0.6, 0.8, 1.0, 1.3])
def test_the_solution_satisfies_the_holographic_definition(c_hde):
    """
    ``H L = c / sqrt(Omega_DE)`` with ``L`` the future event
    horizon, computed by quadrature from ``E(z)`` alone.

    Agreement is limited by the quadrature, not the solution: the
    integral runs to ``a = 1e4`` on a log grid, and its residual
    falls as the grid is refined.
    """

    model = build(c_hde=c_hde)

    # In H0 = 1 units, H = E. The horizon integral needs the future,
    # so a runs well past 1.
    x = np.linspace(np.log(0.05), np.log(1.0e4), 300_000)
    a = np.exp(x)
    z = 1.0 / a - 1.0

    E = np.atleast_1d(model.E(z))
    omega = np.atleast_1d(model.omega_de_fraction(z))

    cumulative = cumulative_trapezoid(1.0 / (a ** 2 * E), a, initial=0.0)
    tail = cumulative[-1] - cumulative

    for target in (1.0, 0.7, 0.5, 0.3):

        i = int(np.argmin(np.abs(a - target)))

        horizon = a[i] * tail[i]

        assert E[i] * horizon == pytest.approx(
            c_hde / np.sqrt(omega[i]), rel=2.0e-3,
        ), f"c={c_hde}, a={target}"


def test_the_definition_check_would_notice_a_wrong_ode():
    """
    The test above is only worth having if it fails on a wrong
    equation of motion. Here the coefficient 2 is changed to 3,
    which is the kind of slip that produces a perfectly smooth and
    entirely wrong history.
    """

    model = build(c_hde=0.8)

    original = model._omega_de_spline

    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    c = 0.8

    wrong = solve_ivp(
        lambda _x, y: [
            y[0] * (1 - y[0]) * (1 + 3.0 * np.sqrt(max(y[0], 0.0)) / c)
        ],
        (0.0, np.log(1e-8)), [0.70],
        rtol=1e-10, atol=1e-14, dense_output=True,
    )

    grid = np.linspace(np.log(1e-8), 0.0, 4000)

    model._omega_de_spline = CubicSpline(
        grid, np.clip(wrong.sol(grid)[0], 1e-16, 1 - 1e-12),
    )

    try:
        x = np.linspace(np.log(0.05), np.log(1.0e4), 200_000)
        a = np.exp(x)
        E = np.atleast_1d(model.E(1.0 / a - 1.0))
        omega = np.atleast_1d(model.omega_de_fraction(1.0 / a - 1.0))

        cumulative = cumulative_trapezoid(1.0 / (a ** 2 * E), a, initial=0.0)
        tail = cumulative[-1] - cumulative

        i = int(np.argmin(np.abs(a - 1.0)))

        mismatch = abs(
            E[i] * a[i] * tail[i] / (c / np.sqrt(omega[i])) - 1.0
        )

    finally:
        model._omega_de_spline = original

    assert mismatch > 0.05, (
        f"a wrong ODE went unnoticed (mismatch {mismatch:.2e})"
    )


# ============================================================
# The background
# ============================================================

def test_flatness_closure_is_exact():

    model = build()

    assert float(np.atleast_1d(model.E(0.0))[0]) == pytest.approx(1.0, abs=1e-12)

    assert float(np.atleast_1d(model.Omega_de(0.0))[0]) == pytest.approx(0.70)


def test_dark_energy_approaches_the_proportional_to_a_limit():
    """
    For ``Omega_DE << 1`` the ODE reduces to
    ``dOmega/dln a = Omega``, so ``Omega_DE`` tends to a constant
    times ``a`` -- which is what makes the model harmless at
    recombination.

    Asserted as a *rate* rather than a tolerance, because the limit
    is asymptotic and any fixed tolerance is really a statement
    about which redshift was chosen. The leading correction to the
    ODE is the ``2 sqrt(Omega_DE) / c`` term, so the departure from
    constancy should fall as ``sqrt(Omega_DE)``, i.e. as
    ``sqrt(a)``: one factor of ``sqrt(10)`` per decade in ``1+z``.

    Measured: 1.16e-1, 3.73e-2, 1.18e-2, 3.74e-3 per decade from
    z = 1e3 -- ratios of 3.11, 3.16, 3.16 against sqrt(10) = 3.162.
    """

    model = build()

    z = np.array([1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
    a = 1.0 / (1.0 + z)

    ratio = np.atleast_1d(model.omega_de_fraction(z)) / a

    departures = np.abs(np.diff(ratio) / ratio[:-1])

    # Monotonically approaching the limit, not wandering.
    assert np.all(np.diff(departures) < 0.0)

    # And at the rate the leading correction predicts.
    for factor in departures[:-1] / departures[1:]:

        assert factor == pytest.approx(np.sqrt(10.0), rel=0.05)


def test_the_equation_of_state_moves_the_right_way_with_c():
    """
    ``w0 = -1/3 - (2/3) sqrt(Omega_DE0) / c``: smaller ``c`` means
    more negative, and ``c`` below about 0.84 puts today's value
    into phantom territory for ``Omega_m = 0.3``. That crossing is
    a prediction of the model rather than a parametrization
    choice, and it is the reason HDE is interesting.
    """

    w = [float(np.atleast_1d(build(c_hde=c).w_de(0.0))[0])
         for c in (0.6, 0.8, 1.0, 1.4)]

    assert w[0] < w[1] < w[2] < w[3]

    assert w[0] < -1.0 and w[1] < -1.0

    assert w[2] > -1.0

    # The closed form, independently.
    expected = -1.0 / 3.0 - (2.0 / 3.0) * np.sqrt(0.70) / 0.8

    assert w[1] == pytest.approx(expected, rel=1e-9)


def test_dEdz_is_analytic_and_matches_a_numerical_derivative():
    """
    ``dEdz`` is written from the ODE rather than differenced, so
    this checks the algebra rather than a step size.
    """

    model = build()

    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    h = 1.0e-5

    numeric = (
        np.atleast_1d(model.E(z + h)) - np.atleast_1d(model.E(z - h))
    ) / (2.0 * h)

    np.testing.assert_allclose(
        np.atleast_1d(model.dEdz(z)), numeric, rtol=1e-6,
    )


# ============================================================
# Behaving like a model
# ============================================================

def test_refresh_re_solves_after_parameters_change():
    """
    The background is cached, so a mutated parameter that did not
    reach it would leave every distance computed for the previous
    cosmology -- silently, and only after the first evaluation.
    """

    model = build(c_hde=0.8)

    before = float(np.atleast_1d(model.E(1.0))[0])

    model.params.c_hde = 1.5
    model.refresh()

    after = float(np.atleast_1d(model.E(1.0))[0])

    assert after != before

    assert float(np.atleast_1d(model.E(0.0))[0]) == pytest.approx(1.0, abs=1e-12)


def test_it_fits():
    """
    End to end through a Fitter, with ``c_hde`` free.
    """

    from CosmoFit import Fitter

    fit = Fitter(
        model=HDE,
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "c_hde"],
        initial={"H0": 68.0, "Omega_m": 0.30, "c_hde": 0.8, "rd": 147.1},
    )

    fit.best_fit(restarts=4, seed=0)

    assert np.isfinite(fit.best_fit_chi2)

    assert fit.best_fit_chi2 < 50.0

    assert 0.2 < fit.best_fit_params["c_hde"] < 2.0


def test_camb_refuses_it():
    """
    The density is set by the *future* event horizon, so the
    perturbations are non-local in time. CAMB would run and return
    something plausible-looking.
    """

    from CosmoFit.cosmology.boltzmann import supports_cmb_spectra

    supported, reason = supports_cmb_spectra(HDE)

    assert not supported

    assert "event horizon" in reason


# ============================================================
# The Omega_de convention
# ============================================================

def test_Omega_de_follows_the_library_convention():
    """
    Every model here returns ``rho_DE(z) / rho_crit(0)`` from
    ``Omega_de`` -- the term as it appears inside ``E(z)^2``, which
    is what the recombination module divides by ``Omega_de0`` to
    get the dark-energy evolution factor.

    HDE's ODE integrates a *different* quantity,
    ``rho_DE(z) / rho_crit(z)``, and the two agree at ``z = 0`` and
    nowhere else -- which is why returning the wrong one was quiet.
    It shipped that way in v0.30.0: ``Omega_de(0.5)`` came out
    0.59 times the right value, and 4e-10 times it at
    recombination.

    Checked here against the continuity equation, which reaches the
    same density by a route the ODE is not involved in:

        rho_DE(z) / rho_DE(0) = exp(3 Int_0^z (1 + w) / (1 + z') dz')
    """

    from scipy.integrate import quad

    model = build(c_hde=0.73)

    def continuity(z):

        integrand = lambda zz: (
            1.0 + float(np.atleast_1d(model.w_de(zz))[0])
        ) / (1.0 + zz)

        return np.exp(
            3.0 * quad(integrand, 0.0, z, epsabs=1e-12, epsrel=1e-12)[0]
        )

    for z in (0.0, 0.5, 1.0, 3.0, 10.0):

        computed = float(np.atleast_1d(model.Omega_de(z))[0])

        expected = model.Omega_de0 * continuity(z)

        assert computed == pytest.approx(expected, rel=1e-8), z


def test_the_two_density_measures_are_not_confused():
    """
    ``omega_de_fraction`` is the density *parameter*; ``Omega_de``
    is the density in units of today's critical density. They
    differ by ``E(z)^2`` -- and coincide at z = 0, which is the
    trap.
    """

    model = build()

    assert float(np.atleast_1d(model.omega_de_fraction(0.0))[0]) == (
        pytest.approx(float(np.atleast_1d(model.Omega_de(0.0))[0]))
    )

    for z in (0.5, 2.0, 10.0):

        fraction = float(np.atleast_1d(model.omega_de_fraction(z))[0])
        density = float(np.atleast_1d(model.Omega_de(z))[0])

        assert density == pytest.approx(
            fraction * float(np.atleast_1d(model.E(z))[0]) ** 2,
        )

        assert density != pytest.approx(fraction, rel=1e-3)


def test_the_density_parameter_still_sums_to_one():
    """
    Flat universe: the matter and dark-energy *parameters* add to
    one at every redshift. This is the statement that
    ``omega_de_fraction`` is the parameter and not the density.
    """

    model = build()

    z = np.array([0.0, 0.5, 2.0, 10.0, 100.0])

    E2 = np.atleast_1d(model.E(z)) ** 2

    matter = model.Omega_m * (1.0 + z) ** 3 / E2

    np.testing.assert_allclose(
        matter + np.atleast_1d(model.omega_de_fraction(z)),
        1.0,
        rtol=1e-10,
    )


# ============================================================
# Against a published constraint
# ============================================================

def test_reproduces_the_published_late_time_constraint():
    """
    Every dataset and most models in this library are checked
    against a number someone else published. HDE was checked
    against its own *definition* -- which is the right test for an
    ODE model, and not the same thing.

    So: the late-time combination of arXiv:2607.09732, which
    constrains three holographic models with cosmic chronometers,
    DESI DR2 BAO and Type Ia supernovae and reports, for the
    BAO-included combinations,

        c ~ 1,  Omega_m0 ~ 0.270-0.272,  H0 ~ 67.3-68.0

    None of those is used as an input here. No CMB and no BBN, as
    in the paper -- which is why ``r_d`` is a free parameter rather
    than computed: nothing in this combination calibrates it.
    """

    from CosmoFit import Fitter

    fit = Fitter(
        model=HDE,
        datasets=["cc", "desi", "pantheon"],
        dataset_kwargs={"desi": {"version": "desi2025"}},
        free_params=["H0", "Omega_m", "rd", "c_hde"],
        initial={"H0": 68.0, "Omega_m": 0.28, "rd": 147.1, "c_hde": 1.0},
    )

    fit.best_fit()

    parameters = fit.best_fit_params

    assert parameters["c_hde"] == pytest.approx(1.0, abs=0.05)

    assert 0.265 < parameters["Omega_m"] < 0.277

    assert 67.0 < parameters["H0"] < 68.5


def test_supernovae_and_bao_pull_c_in_opposite_directions():
    """
    The other published statement about this model, and a sharper
    one than a central value: supernovae pull ``c`` above unity
    while BAO pushes it below.

    ``c = 1`` is where the far-future equation of state reaches
    exactly -1, so which side of it the data sits on is the
    difference between a phantom and a quintessence-like fate --
    the reason the split is worth reproducing rather than
    averaging over.
    """

    from CosmoFit import Fitter

    def best_c(datasets, free, initial):

        fit = Fitter(
            model=HDE, datasets=datasets,
            dataset_kwargs={"desi": {"version": "desi2025"}},
            free_params=free, initial=initial,
        )

        fit.best_fit()

        return fit.best_fit_params["c_hde"]

    supernovae = best_c(
        ["cc", "pantheon"], ["H0", "Omega_m", "c_hde"],
        {"H0": 68.0, "Omega_m": 0.28, "c_hde": 1.0, "rd": 147.1},
    )

    bao = best_c(
        ["cc", "desi"], ["H0", "Omega_m", "rd", "c_hde"],
        {"H0": 68.0, "Omega_m": 0.28, "rd": 147.1, "c_hde": 1.0},
    )

    assert supernovae > 1.0 > bao, (supernovae, bao)
