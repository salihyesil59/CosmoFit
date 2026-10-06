"""
The native background and early-universe theories.

Three references, none of them the code under test:

* the original model classes, with ``radiation: false`` -- the one
  setting in which the new expansion history must be the old one;
* CAMB's own background (``camb.get_background``), with radiation and
  a massive neutrino, which the old models never carried;
* the CAMB drag-epoch grid in ``tests/data/camb_drag_reference.npz``,
  which needs no CAMB installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import cosmofit
from cosmofit.core import ComponentError, Likelihood, get_model


class Probe(Likelihood):
    """Asks for what a test needs, and contributes nothing."""

    def get_requirements(self):
        return {name: None for name in self.info.get("needs", ["DM"])}

    def logp(self):
        return 0.0


def build(params, dark_energy="lambda", early=False, **options):

    theory = {"background": {"dark_energy": dark_energy, **options}}

    needs = ["DM"]

    if early:
        theory["early_universe"] = None
        needs.append("rdrag")

    return get_model({
        "theory": theory,
        "likelihood": {"probe": {"class": Probe, "needs": needs}},
        "params": params,
    })


Z = np.array([0.05, 0.3, 0.7, 1.0, 1.5, 2.33, 4.0])

#: name -> (old model class, extra parameters)
DARK_ENERGIES = {
    "lambda": ("LCDM", {}),
    "wcdm": ("WCDM", {"w0": -0.9}),
    "cpl": ("CPL", {"w0": -0.8, "wa": -0.6}),
    "jbp": ("JBP", {"w0": -0.8, "wa": 0.5}),
    "ba": ("BA", {"w0": -0.9, "wa": 0.2}),
    "logarithmic": ("LogarithmicDE", {"w0": -0.9, "wa": 0.1}),
    "pede": ("PEDE", {}),
    "gede": ("GEDE", {"Delta": 0.7, "z_t": 0.3}),
}


# ============================================================
# Without radiation, the old models exactly
# ============================================================

@pytest.mark.parametrize("Omega_k", [0.0, 0.05, -0.04])
@pytest.mark.parametrize("name", sorted(DARK_ENERGIES))
def test_radiation_free_background_is_the_old_model(name, Omega_k):

    old_name, extra = DARK_ENERGIES[name]

    params = {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049,
              "Omega_k": Omega_k, **extra}

    model = build(params, dark_energy=name, radiation=False)
    model.logposterior({})

    background = model.theories["background"]

    old_cls = getattr(cosmofit, old_name)
    values = dict(old_cls.PARAMS_CLASS.defaults())
    values.update({k: v for k, v in params.items()
                   if k in old_cls.PARAMS_CLASS.names()})
    old = old_cls(old_cls.PARAMS_CLASS(**values))

    np.testing.assert_allclose(background.get_E(Z), old.E(Z), rtol=1e-14)
    np.testing.assert_allclose(background.get_DM(Z), old.distance.DM(Z), rtol=1e-9)
    np.testing.assert_allclose(background.get_DV(Z), old.distance.DV(Z), rtol=1e-9)
    if hasattr(old, "w"):
        np.testing.assert_allclose(background.get_w(Z), old.w(Z), rtol=1e-14)


# ============================================================
# With radiation, against CAMB
# ============================================================

def _camb():
    return pytest.importorskip("camb")


@pytest.mark.parametrize("Omega_k", [0.0, -0.03])
@pytest.mark.parametrize("dark_energy", ["lambda", "cpl"])
def test_background_matches_camb(dark_energy, Omega_k):

    camb = _camb()

    H0, Om, Ob, m_nu = 67.5, 0.31, 0.049, 0.06
    w0, wa = (-0.8, -0.6) if dark_energy == "cpl" else (-1.0, 0.0)

    params = {"H0": H0, "Omega_m": Om, "Omega_b": Ob, "Omega_k": Omega_k,
              "m_nu": m_nu}

    if dark_energy == "cpl":
        params.update(w0=w0, wa=wa)

    model = build(params, dark_energy=dark_energy, early=True)
    model.logposterior({})

    background = model.theories["background"]
    early = model.theories["early_universe"].current_state

    h = H0 / 100.0
    densities = background.get_background_densities()

    results = camb.get_background(camb.set_params(
        H0=H0, ombh2=Ob * h * h,
        omch2=densities["omega_cb"] - Ob * h * h,
        omk=Omega_k, mnu=m_nu, nnu=3.044, num_massive_neutrinos=1,
        w=w0, wa=wa, dark_energy_model="ppf",
    ))

    z = np.array([0.1, 1.0, 2.33, 10.0, 1100.0])

    np.testing.assert_allclose(
        background.get_H(z), results.hubble_parameter(z), rtol=1e-6,
    )
    np.testing.assert_allclose(
        background.get_DA(z), results.angular_diameter_distance(z), rtol=1e-6,
    )

    derived = results.get_derived_params()

    assert early["rdrag"] == pytest.approx(derived["rdrag"], rel=2e-5)
    assert early["zdrag"] == pytest.approx(derived["zdrag"], rel=2e-5)
    assert early["thetastar"] == pytest.approx(derived["thetastar"], rel=1e-5)

    # Against CAMB's own H(z), integrated exactly. CAMB's reported
    # "age" sits 2e-5 below that integral -- its internal integration
    # tolerance -- so it is not the reference; its expansion history is.
    from scipy.integrate import quad

    age_from_camb_h = quad(
        lambda a: 1.0 / (a * results.hubble_parameter(1.0 / a - 1.0)),
        1e-12, 1.0, epsrel=1e-11, limit=500,
    )[0] * camb.constants.Mpc / 1e3 / camb.constants.Gyr

    age = background.get_current_derived()["age"]

    assert age == pytest.approx(age_from_camb_h, rel=1e-6)
    assert age == pytest.approx(derived["age"], rel=5e-5)


def test_rdrag_across_the_camb_calibration_grid():
    """
    The drag-epoch grid CAMB was run on for the z_drag fit, through the
    native theories: the same bound the old sound horizon is held to.
    """

    reference = np.load(Path(__file__).parent / "data" / "camb_drag_reference.npz")

    model = build({
        "H0": 67.36,
        "omega_b": {"prior": {"min": 0.0, "max": 1.0}},
        "omega_cdm": {"prior": {"min": 0.0, "max": 1.0}},
        "N_eff": {"prior": {"min": 1.0, "max": 6.0}},
        "m_nu": {"prior": {"min": 0.0, "max": 1.0}},
    }, early=True, parameterization="physical")

    early = model.theories["early_universe"]

    errors = []

    for i in range(0, len(reference["r_drag"]), 7):

        result = model.logposterior({
            "omega_b": reference["omega_b"][i],
            "omega_cdm": reference["omega_cb"][i] - reference["omega_b"][i],
            "N_eff": reference["N_eff"][i],
            "m_nu": reference["m_nu"][i],
        })

        assert result.rejected is None

        errors.append(abs(early.current_state["rdrag"] / reference["r_drag"][i] - 1.0))

    assert max(errors) < 1e-4, max(errors)


def test_rdrag_matches_the_old_sound_horizon():
    """
    The old calculator and the native theory integrate the same
    physics; only the cosmological constant's ~1e-9 share of H at the
    drag epoch separates them.
    """

    params = {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "m_nu": 0.06}

    model = build(params, early=True)
    model.logposterior({})

    old = cosmofit.LCDM(cosmofit.CosmologyParameters(**params))

    assert model.theories["early_universe"].current_state["rdrag"] == pytest.approx(
        old.sound_horizon.rd_computed(), rel=1e-8,
    )


# ============================================================
# Parameterizations, derived values, refusals
# ============================================================

def test_physical_and_fractional_parameterizations_agree():

    h = 0.68
    omega_b, omega_cdm = 0.0224, 0.120

    physical = build({"H0": 68.0, "omega_b": omega_b, "omega_cdm": omega_cdm},
                     parameterization="physical")
    physical.logposterior({})

    Omega_m = physical.theories["background"].get_current_derived()["Omega_m"]

    fractional = build({"H0": 68.0, "Omega_m": Omega_m, "Omega_b": omega_b / h ** 2})
    fractional.logposterior({})

    np.testing.assert_allclose(
        physical.theories["background"].get_DM(Z),
        fractional.theories["background"].get_DM(Z),
        rtol=1e-12,
    )

    derived = fractional.theories["background"].get_current_derived()

    assert derived["omega_cdm"] == pytest.approx(omega_cdm, rel=1e-12)


def test_radiation_closes_the_budget():

    model = build({"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049})
    model.logposterior({})

    background = model.theories["background"]
    derived = background.get_current_derived()

    assert float(background.get_E(0.0)) == pytest.approx(1.0, abs=1e-14)
    assert derived["Omega_m"] + derived["Omega_r"] + derived["Omega_de"] == pytest.approx(1.0, abs=1e-14)
    # Photons plus the massless species only: one of the 3.044 is
    # massive and counted in Omega_m today, which is why this is 7.8e-5
    # rather than the all-massless ~9.2e-5.
    assert derived["Omega_r"] == pytest.approx(7.81e-5, rel=2e-3)


def test_dark_energy_parameters_are_required():

    with pytest.raises(ComponentError, match=r"needs the parameter\(s\) \['wa'\]"):
        build({"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "w0": -0.9},
              dark_energy="cpl")


def test_early_universe_refuses_a_radiation_free_background():

    with pytest.raises(ComponentError, match="radiation"):
        build({"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049},
              early=True, radiation=False)


def test_a_universe_with_no_expanding_solution_is_rejected():
    """Strongly closed: E^2 turns negative inside the table."""

    model = build({
        "H0": 68.0, "Omega_b": 0.049, "Omega_m": 0.31,
        "Omega_k": {"prior": {"min": -3.0, "max": 0.5}},
    })

    result = model.logposterior({"Omega_k": -2.5})

    assert result.logpost == -math.inf
    assert result.rejected == "background: no solution"


def test_distance_table_extends_on_request():

    model = build({"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049})
    model.logposterior({})

    background = model.theories["background"]

    far = float(background.get_comoving_distance(1100.0))

    assert math.isfinite(far)
    assert 13_000 < far < 15_000
