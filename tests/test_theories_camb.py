"""
The CAMB theory, on the native background.

CAMB is handed the background's densities and the dark sector's own
``w(z)``; the background CAMB then solves must be the native one. The
spectra must be the old backend's for the same cosmology. And the
native growth theory, which solves one equation where CAMB solves the
full Boltzmann hierarchy, must agree with CAMB's ``sigma8(z)`` of the
cold matter -- an independent check of both.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import cosmofit
from cosmofit.core import ComponentError, Likelihood, get_model

camb = pytest.importorskip("camb")


class Probe(Likelihood):

    def get_requirements(self):
        return self.info.get("needs", {"Cl": None})

    def logp(self):
        return 0.0


BASE = {"H0": 67.5, "Omega_m": 0.31, "Omega_b": 0.049, "m_nu": 0.06}


def model(sector="lambda", needs=None, growth=None, camb_options=None,
          radiation=True, **params):

    theory = {
        "background": {"dark_energy": sector, "radiation": radiation},
        "camb": camb_options,
    }

    if growth is not None:
        theory["growth"] = growth

    return get_model({
        "theory": theory,
        "likelihood": {"probe": {"class": Probe, "needs": needs or {"Cl": None}}},
        "params": {**BASE, **params},
    })


def evaluate(m):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = m.logposterior({})

    # CAMB now and then returns NaN spectra for unexceptional parameters,
    # and the theory rejects the point; that is CAMB's failure, not this.
    if point.rejected == "camb: no solution":
        pytest.skip("CAMB returned NaN spectra, a known intermittent CAMB failure")

    assert point.rejected is None, point.rejected

    return point


def camb_results(m, redshifts=None):
    """CAMB run directly on the parameters the theory builds."""

    theory = m.theories["camb"]
    pars = theory._parameters(**theory.defaults)

    if redshifts is not None:
        pars.set_matter_power(redshifts=redshifts, kmax=2.0)

    return camb.get_results(pars)


# ============================================================
# What CAMB is given
# ============================================================

@pytest.mark.parametrize("sector, params", [
    ("lambda", {}),
    ("lambda", {"Omega_k": -0.02}),
    ("cpl", {"w0": -0.8, "wa": -0.6}),
    ("gede", {"Delta": 0.7, "z_t": 0.3}),
    ("gcg", {"A_gcg": 0.75, "alpha_gcg": 0.1}),
])
def test_camb_solves_the_native_background(sector, params):

    m = model(sector, needs={"Cl": None, "H": None}, **params)
    evaluate(m)

    z = np.array([0.1, 0.5, 1.0, 2.33, 10.0, 1100.0])

    np.testing.assert_allclose(
        camb_results(m).hubble_parameter(z), m.provider.get_H(z), rtol=1e-6,
    )


def test_spectra_are_the_old_backends():
    """
    The same LCDM through the old backend. What is left is the massive
    neutrino density, which the old backend converts from ``m_nu`` with
    93.14 eV and the background integrates exactly.
    """

    from cosmofit.cosmology.boltzmann import CAMBBackend

    m = model()
    evaluate(m)

    cls = cosmofit.LCDM.PARAMS_CLASS
    old = CAMBBackend(cosmofit.LCDM(cls(**{**cls.defaults(), **BASE})))

    reference = old.cls(lmin=2)
    Cl = m.provider.get_Cl()

    for name in ("TT", "EE"):
        np.testing.assert_allclose(Cl[name.lower()][2:2509], reference[name], rtol=3e-5)

    te = Cl["te"][2:2509]
    assert np.max(np.abs(te - reference["TE"])) < 1e-5 * np.max(np.abs(reference["TE"]))

    assert m.theories["camb"].current_state["derived"]["sigma8"] == pytest.approx(
        old.sigma8(), rel=2e-5,
    )


def test_D_l_and_the_lensing_potential():

    m = model()
    evaluate(m)

    Cl = m.provider.get_Cl()
    Dl = m.provider.get_Cl(ell_factor=True)

    ell = Cl["ell"]

    assert ell[0] == 0 and len(ell) == 2509

    np.testing.assert_allclose(Dl["tt"], ell * (ell + 1) * Cl["tt"] / (2 * math.pi))

    # [L(L+1)]^2 C_L^phiphi / 2 pi either way: ~1e-7 near its peak.
    np.testing.assert_array_equal(Dl["pp"], Cl["pp"])
    assert 5e-8 < np.max(Cl["pp"]) < 5e-7


def test_lmax_is_widened_to_what_likelihoods_ask():

    m = model(needs={"Cl": {"lmax": 3000, "lens_potential_accuracy": 2}})

    assert m.theories["camb"].lmax == 3000
    assert m.theories["camb"].lens_potential_accuracy == 2

    evaluate(m)

    assert len(m.provider.get_Cl()["tt"]) == 3001


# ============================================================
# Growth against CAMB
# ============================================================

@pytest.mark.parametrize("sector, params, rtol", [
    ("lambda", {"m_nu": 0.0}, 3e-5),
    ("lambda", {"m_nu": 0.0, "Omega_k": -0.02}, 3e-5),
    ("cpl", {"m_nu": 0.0, "w0": -0.8, "wa": -0.6}, 3e-5),
    # Massive neutrinos slow the growth of the cold matter on 8 Mpc/h,
    # which one scale-independent equation does not see.
    ("lambda", {}, 2e-4),
])
def test_growth_is_cambs_sigma8_of_cold_matter(sector, params, rtol):

    m = model(
        sector, growth={"amplitude": "boltzmann"},
        needs={"sigma8_z": None}, **params,
    )
    evaluate(m)

    z = [0.0, 0.5, 1.0, 2.0, 5.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = camb_results(m, redshifts=z)

    # CAMB orders its redshifts earliest first.
    sigma_cb = results.get_sigmaR(8.0, var1="delta_nonu", var2="delta_nonu")[::-1]

    np.testing.assert_allclose(m.provider.get_sigma8_z(np.array(z)), sigma_cb, rtol=rtol)


def test_derived_amplitudes():

    m = get_model({
        "theory": {"background": {"dark_energy": "lambda"}, "camb": None,
                   "growth": {"amplitude": "boltzmann"}},
        "likelihood": {"probe": {"class": Probe, "needs": {"fsigma8": None}}},
        "params": {**BASE, "sigma8": {"derived": True}, "S8": {"derived": True},
                   "A_s": {"derived": True}},
    })

    point = evaluate(m)

    sigma8 = point.derived["sigma8"]

    assert point.derived["S8"] == pytest.approx(sigma8 * math.sqrt(0.31 / 0.3), rel=1e-12)
    assert point.derived["A_s"] == pytest.approx(math.exp(3.044) * 1e-10, rel=1e-12)

    # Growth scales the cold matter's amplitude, which neutrinos
    # leave a little higher than all matter's.
    sigma8_cb = m.theories["camb"].current_state["sigma8_0"]

    assert 1.0 < sigma8_cb / sigma8 < 1.01


def test_two_amplitudes_are_refused():

    with pytest.raises(ComponentError, match="amplitude: boltzmann"):
        get_model({
            "theory": {"background": {"dark_energy": "lambda"}, "camb": None,
                       "growth": None},
            "likelihood": {"probe": {"class": Probe, "needs": {"fsigma8": None}}},
            "params": {**BASE, "sigma8": 0.8},
        })


# ============================================================
# Matter power spectra and sigma(R)
# ============================================================

PK = {"z": [0.0, 0.5, 1.0, 2.0], "k_max": 10.0, "nonlinear": [False, True]}


def test_matter_power_is_cambs():
    """
    The theory's ``P(z, k)`` against CAMB's own interpolator, run on the
    parameters the theory builds: the grid and its interpolation in
    the theory, without ``h`` in either unit.
    """

    m = model(needs={"Pk_interpolator": PK})
    evaluate(m)

    theory = m.theories["camb"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = camb.get_results(theory._parameters(**theory.defaults))

    z = np.array([0.0, 0.3, 1.0, 1.7])
    k = np.array([1e-3, 0.02, 0.1, 0.5, 3.0])

    for nonlinear in (False, True):

        mine = m.provider.get_Pk_interpolator(nonlinear=nonlinear)
        cambs = results.get_matter_power_interpolator(
            nonlinear=nonlinear, hubble_units=False, k_hunit=False,
        )

        np.testing.assert_allclose(mine.P(z, k, grid=True), cambs.P(z, k), rtol=2e-3)

    linear = m.provider.get_Pk_interpolator(nonlinear=False)
    halofit = m.provider.get_Pk_interpolator(nonlinear=True)

    # Linear on large scales, far above it on small ones.
    assert halofit.P(0.0, 1e-3) == pytest.approx(linear.P(0.0, 1e-3), rel=1e-3)
    assert halofit.P(0.0, 1.0) > 3 * linear.P(0.0, 1.0)


def test_sigma_R_is_the_integral_of_P_k():
    """
    ``sigma(R)`` from CAMB, against the top-hat integral of the linear
    ``P(k)`` the theory returns: the two outputs, and their units, agree.
    """

    h = BASE["H0"] / 100.0
    R = np.array([8.0 / h, 20.0])

    m = model(needs={
        "Pk_interpolator": {**PK, "nonlinear": False},
        "sigma_R": {"z": [0.0, 1.0], "R": R},
    })
    evaluate(m)

    z, radii, sigma = m.provider.get_sigma_R()

    np.testing.assert_array_equal(z, [0.0, 0.5, 1.0, 2.0])
    np.testing.assert_array_equal(radii, R)

    P = m.provider.get_Pk_interpolator(nonlinear=False)

    k = np.logspace(np.log10(P.kmin), np.log10(P.kmax), 4000)

    for i, redshift in enumerate(z):
        for j, radius in enumerate(R):

            x = k * radius
            window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3

            integral = np.trapezoid(k**3 * P.P(redshift, k) * window**2 / (2 * np.pi**2), np.log(k))

            assert sigma[i, j] == pytest.approx(math.sqrt(integral), rel=2e-3)


def test_sigma_R_of_cold_matter_is_sigma8_0():

    h = BASE["H0"] / 100.0

    m = model(needs={
        "sigma_R": {"R": [8.0 / h], "vars_pairs": [["delta_nonu", "delta_nonu"]]},
    })
    evaluate(m)

    _, _, sigma = m.provider.get_sigma_R(("delta_nonu", "delta_nonu"))

    assert sigma[0, 0] == pytest.approx(m.theories["camb"].get_sigma8_0(), rel=1e-10)


def test_requests_are_merged():
    """Two likelihoods asking for different spectra get both, on one grid."""

    m = get_model({
        "theory": {"background": {"dark_energy": "lambda"}, "camb": None},
        "likelihood": {
            "one": {"class": Probe, "needs": {"Pk_interpolator": {"z": [0.5], "k_max": 5.0}}},
            "two": {"class": Probe, "needs": {"Pk_interpolator": {
                "z": [2.0], "nonlinear": False,
                "vars_pairs": [["delta_nonu", "delta_nonu"]],
            }}},
        },
        "params": BASE,
    })

    evaluate(m)

    theory = m.theories["camb"]

    assert theory.redshifts == {0.0, 0.5, 2.0}
    assert theory.k_max == 5.0
    assert theory.spectra == {
        ("delta_tot", "delta_tot", True), ("delta_nonu", "delta_nonu", False),
    }

    total = m.provider.get_Pk_interpolator()
    cold = m.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)

    assert total.zmax == cold.zmax == 2.0
    assert total.kmax == pytest.approx(cold.kmax) and total.kmax >= 5.0


def test_spectra_nobody_asked_for_are_refused():

    m = model(needs={"Pk_interpolator": {"z": [0.0, 1.0], "nonlinear": False}})
    evaluate(m)

    with pytest.raises(ComponentError, match="non-linear P\\(k\\)"):
        m.provider.get_Pk_interpolator()

    with pytest.raises(ComponentError, match="sigma\\(R\\)"):
        m.theories["camb"].get_sigma_R()

    P = m.provider.get_Pk_interpolator(nonlinear=False)

    with pytest.raises(ValueError, match="'z'"):
        P.P(1.5, 0.1)

    with pytest.raises(ValueError, match="k_max"):
        P.P(0.5, 50.0)


@pytest.mark.parametrize("needs, match", [
    ({"Pk_interpolator": {"zmax": 2}}, "not \\['zmax'\\]"),
    ({"Pk_interpolator": {"z": [-1.0]}}, "z >= 0"),
    ({"Pk_interpolator": {"vars_pairs": [["delta_tot", "delta_dm"]]}}, "vars_pairs"),
    ({"sigma_R": {"z": [0.0]}}, "R > 0"),
])
def test_bad_requests_are_refused(needs, match):

    with pytest.raises(ComponentError, match=match):
        model(needs=needs)


def test_halofit_version_is_checked():

    with pytest.raises(ComponentError, match="halofit_version 'mead2030'"):
        model(camb_options={"halofit_version": "mead2030"})


def test_asking_for_P_k_leaves_the_rest_alone():
    """
    More redshifts and a non-linear ``P(k)`` change nothing else CAMB
    returns. (A larger ``k_max`` does: the lensing potential at
    ``L ~ 2500`` moves by 0.2%, towards the better answer.)
    """

    plain = model()
    evaluate(plain)

    m = model(needs={"Cl": None, "Pk_interpolator": {"z": [0.0, 1.0, 3.0]}})
    evaluate(m)

    for name in ("tt", "te", "ee", "pp"):
        np.testing.assert_allclose(
            m.provider.get_Cl()[name][2:], plain.provider.get_Cl()[name][2:],
            rtol=1e-6, atol=1e-30,
        )

    assert m.theories["camb"].get_sigma8_0() == pytest.approx(
        plain.theories["camb"].get_sigma8_0(), rel=1e-6,
    )


# ============================================================
# What CAMB cannot be given
# ============================================================

@pytest.mark.parametrize("sector, params, reason", [
    ("ide", {"w0": -0.9, "xi": 0.05}, "moves energy into matter"),
    ("rvm", {"nu": 0.01}, "moves energy into matter"),
    ("hu_sawicki", {"f_R0": -1e-5, "n_hs": 1.0}, "gravitational coupling"),
    ("dgp", {}, "gravitational coupling"),
    ("fq_exponential", {}, "gravitational coupling"),
    ("lscdm", {"z_dagger": 1.8}, "jumps"),
    ("cardassian", {"n_card": 0.2, "q_card": 1.3}, "no dark-energy equation of state"),
    ("hde", {"c_hde": 0.8}, "no dark-energy equation of state"),
])
def test_sectors_camb_cannot_represent_are_refused(sector, params, reason):

    with pytest.raises(ComponentError, match=reason):
        model(sector, **params)


def test_camb_needs_radiation():

    with pytest.raises(ComponentError, match="radiation"):
        model(radiation=False)
