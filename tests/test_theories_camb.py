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

import CosmoFit
from CosmoFit.core import ComponentError, Likelihood, get_model

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

    from CosmoFit.cosmology.boltzmann import CAMBBackend

    m = model()
    evaluate(m)

    cls = CosmoFit.LCDM.PARAMS_CLASS
    old = CAMBBackend(CosmoFit.LCDM(cls(**{**cls.defaults(), **BASE})))

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
