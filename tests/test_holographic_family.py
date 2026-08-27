"""
The two holographic relatives of HDE, against their definitions and
against a published constraint.

All three put ``rho_DE = 3 c^2 M_p^2 / L^2`` and differ only in what
``L`` is: the future event horizon (HDE), the Ricci scalar (RDE), or
the conformal age (ADE). So the tests that matter are the ones that
pin down *which* length each model is actually using -- a wrong one
still produces a smooth, plausible, entirely different history.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scipy.integrate import cumulative_trapezoid, quad

from CosmoFit import ADE, RDE, Fitter


# ============================================================
# Ricci dark energy
# ============================================================

def rde(Omega_m=0.22, gamma=0.54, H0=68.0):

    return RDE(
        RDE.PARAMS_CLASS(H0=H0, Omega_m=Omega_m, rd=147.1,
                         gamma_rde=gamma),
    )


def test_rde_closes_the_friedmann_equation_for_every_gamma():
    """
    The closed form carries its own normalization, so ``E(0) = 1``
    should be exact rather than approximate -- for any ``gamma``.
    """

    for gamma in (0.2, 0.4, 0.5, 0.54, 0.9):

        value = float(np.atleast_1d(rde(gamma=gamma).E(0.0))[0])

        assert value == pytest.approx(1.0, abs=1e-14)


def test_rde_reduces_to_a_cosmological_constant_at_gamma_one_half():
    """
    The exponent ``4 - 2/gamma`` vanishes at ``gamma = 1/2``, so the
    dark sector stops evolving. That is the cleanest statement of
    what the parameter means, and it makes the model exactly LCDM
    there.
    """

    from CosmoFit import LCDM

    model = rde(Omega_m=0.30, gamma=0.5)

    assert model._exponent == pytest.approx(0.0, abs=1e-12)

    # LCDM, but with the *effective* matter density: part of the
    # Ricci density scales like matter, so the coefficient of
    # (1+z)^3 is A = 2 Omega_m / (2 - gamma) = (4/3) Omega_m.
    reference = LCDM(
        LCDM.PARAMS_CLASS(H0=68.0, Omega_m=model._amplitude, rd=147.1),
    )

    z = np.array([0.0, 0.5, 1.0, 3.0])

    np.testing.assert_allclose(model.E(z), reference.E(z), rtol=1e-12)


def test_rde_dEdz_is_analytic():

    model = rde()

    z = np.array([0.1, 0.5, 1.0, 3.0])
    h = 1.0e-6

    numeric = (
        np.atleast_1d(model.E(z + h)) - np.atleast_1d(model.E(z - h))
    ) / (2.0 * h)

    np.testing.assert_allclose(model.dEdz(z), numeric, rtol=1e-6)


def test_rde_refuses_curvature():

    with pytest.raises(ValueError, match="flat"):
        RDE(
            RDE.PARAMS_CLASS(H0=68.0, Omega_m=0.22, Omega_k=0.05,
                             rd=147.1, gamma_rde=0.54),
        ).E(0.0)


def test_rde_reproduces_the_published_constraint():
    """
    arXiv:2607.09732 constrains RDE on cosmic chronometers +
    DESI DR2 BAO + supernovae and reports, for the BAO-included
    combinations, ``gamma = 0.53-0.55`` and
    ``Omega_m0 = 0.215-0.219``. Neither is used as an input here.
    """

    fit = Fitter(
        model=RDE,
        datasets=["cc", "desi", "pantheon"],
        dataset_kwargs={"desi": {"version": "desi2025"}},
        free_params=["H0", "Omega_m", "rd", "gamma_rde"],
        initial={"H0": 68.0, "Omega_m": 0.22, "rd": 147.1,
                 "gamma_rde": 0.45},
    )

    fit.best_fit()

    assert 0.52 < fit.best_fit_params["gamma_rde"] < 0.56

    assert 0.210 < fit.best_fit_params["Omega_m"] < 0.224


# ============================================================
# Agegraphic dark energy
# ============================================================

def ade(n=2.8, H0=68.0):

    return ADE(
        ADE.PARAMS_CLASS(H0=H0, Omega_m=0.3, rd=147.1, n_ade=n),
    )


def test_ade_derives_the_matter_density_from_n():
    """
    The model's defining feature, and the reason it has one *fewer*
    parameter than LCDM: the early-time condition
    ``Omega_DE -> n^2 a^2 / 4`` fixes the whole solution from ``n``,
    so today's matter density is a prediction.

    The published constraint ``n = 2.78-2.81`` therefore predicts
    ``Omega_m = 0.278-0.283`` -- which is a real check, because
    nothing told the model what a sensible matter density is.
    """

    for n, expected in ((2.4, 0.359), (2.8, 0.280), (3.2, 0.220)):

        assert ade(n=n).Omega_m == pytest.approx(expected, abs=0.002)

    # And the published n lands where the published Omega_m is.
    assert ade(n=2.80).Omega_m == pytest.approx(0.280, abs=0.005)


def test_ade_ignores_the_matter_density_it_is_given():
    """
    `params.Omega_m` is not read. Silently ignoring an input would
    be bad, so a fit that frees it is warned about -- see
    `test_freeing_a_derived_parameter_warns`.
    """

    absurd = ADE(
        ADE.PARAMS_CLASS(H0=68.0, Omega_m=0.99, rd=147.1, n_ade=2.8),
    )

    assert absurd.Omega_m == pytest.approx(0.280, abs=0.002)


def test_ade_reaches_its_early_time_limit():
    """
    ``Omega_DE -> n^2 a^2 / 4`` is not an approximation here -- it
    is the initial condition the solution is integrated *forward*
    from, and getting that wrong is how this model was first
    written in this library: integrating backwards from today with
    a free ``Omega_m``, which silently produces a different model
    whose early-time limit is whatever the walk happens to reach.
    """

    model = ade(n=2.8)

    for z in (1.0e2, 1.0e3, 1.0e5):

        a = 1.0 / (1.0 + z)

        ratio = float(np.atleast_1d(model.omega_de_fraction(z))[0]) / a ** 2

        assert ratio == pytest.approx(2.8 ** 2 / 4.0, rel=0.01)


def test_ade_satisfies_its_conformal_age_definition():
    """
    The check that says which length scale the model is using.

    ``rho_DE = 3 n^2 M_p^2 / eta^2`` with ``eta`` the conformal age
    is equivalent to ``Omega_DE = n^2 / (eta H)^2``. The conformal
    age is computed here by quadrature from the solved ``E(z)``,
    with the equation of motion appearing nowhere in it.
    """

    n = 2.8

    model = ade(n=n)

    x = np.linspace(np.log(1.0e-8), np.log(3.0), 400_000)
    a = np.exp(x)
    z = 1.0 / a - 1.0

    E = np.atleast_1d(model.E(z))
    omega = np.atleast_1d(model.omega_de_fraction(z))

    eta = cumulative_trapezoid(1.0 / (a ** 2 * E), a, initial=0.0)

    for target in (1.0, 0.5, 0.25):

        i = int(np.argmin(np.abs(a - target)))

        assert omega[i] == pytest.approx(
            n ** 2 / (eta[i] * E[i]) ** 2, rel=2.0e-3,
        ), target


def test_ade_can_never_be_phantom():
    """
    ``w = -1 + (2/3n) sqrt(Omega_DE) / a`` has a manifestly positive
    correction, so ``w > -1`` always -- for every ``n`` and every
    redshift. That is the sharpest observational difference from
    HDE, which crosses below -1 for ``c < 1``.
    """

    z = np.array([0.0, 0.5, 2.0, 10.0])

    for n in (1.5, 2.8, 5.0):

        assert np.all(np.atleast_1d(ade(n=n).w_de(z)) > -1.0)


def test_ade_dEdz_is_analytic():

    model = ade()

    z = np.array([0.1, 0.5, 1.0, 3.0])
    h = 1.0e-6

    numeric = (
        np.atleast_1d(model.E(z + h)) - np.atleast_1d(model.E(z - h))
    ) / (2.0 * h)

    np.testing.assert_allclose(model.dEdz(z), numeric, rtol=1e-6)


def test_ade_refuses_curvature():

    with pytest.raises(ValueError, match="flat"):
        ADE(
            ADE.PARAMS_CLASS(H0=68.0, Omega_m=0.28, Omega_k=0.05,
                             rd=147.1, n_ade=2.8),
        ).E(0.0)


# ============================================================
# The derived-parameter guard
# ============================================================

def test_freeing_a_derived_parameter_warns():
    """
    Sampling `Omega_m` under ADE would produce a posterior for a
    number the model never reads -- which is to say, its prior.
    """

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=ADE,
            datasets=["cc", "desi"],
            dataset_kwargs={"desi": {"version": "desi2025"}},
            free_params=["H0", "Omega_m", "rd", "n_ade"],
            initial={"H0": 68.0, "Omega_m": 0.28, "rd": 147.1,
                     "n_ade": 2.8},
        )

    assert any(
        "derives" in str(w.message) and "Omega_m" in str(w.message)
        for w in caught
    ), [str(w.message) for w in caught]


def test_no_warning_when_the_derived_parameter_is_left_alone():

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=ADE,
            datasets=["cc", "desi"],
            dataset_kwargs={"desi": {"version": "desi2025"}},
            free_params=["H0", "rd", "n_ade"],
            initial={"H0": 68.0, "Omega_m": 0.28, "rd": 147.1,
                     "n_ade": 2.8},
        )

    assert not any("derives" in str(w.message) for w in caught)


def test_rde_dark_energy_follows_the_library_convention():
    """
    ``Omega_de(0) = 1 - Omega_m``, as for every other model -- which
    is *not* what the second term of ``E^2`` gives, because the
    Ricci density carries a matter-like piece. Getting this wrong
    is quiet: the fit is unchanged and only anything that reads
    ``Omega_de`` is off, which is how the same mistake shipped in
    HDE.
    """

    model = rde(Omega_m=0.22, gamma=0.54)

    assert float(np.atleast_1d(model.Omega_de(0.0))[0]) == pytest.approx(
        1.0 - model.Omega_m, abs=1e-12,
    )

    # The matter-like piece is real and sizeable.
    assert model._amplitude > model.Omega_m

    assert model._amplitude == pytest.approx(
        2.0 * model.Omega_m / (2.0 - 0.54), rel=1e-12,
    )
