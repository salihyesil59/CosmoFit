"""
Which models may be handed to CAMB.

The gate used to be a list of class names plus "does it define
``w(z)``". That let through every model that has a ``w`` but whose
physics is not a smooth dark-energy fluid on top of GR:

* ``IDE`` -- matter and dark energy exchange energy. CAMB received
  plain wCDM with an uncoupled CDM fluid, so the C_l and sigma8 came
  out independent of the coupling ``xi`` while ``xi`` still moved the
  background. No error, and no visible symptom.
* every modified-gravity model built from an action with
  ``growth="quasi_static"`` (``f(T)``, ``f(Q)``, ``f(R)``,
  scalar-tensor ``F(phi) R``) -- each carries both an effective ``w``
  and its own ``mu(a, k)``. CAMB solved GR's perturbations for them.
* any ``define_model`` model given both ``w`` and ``mu``.

The gate now asks what a model *does* rather than what it is called:
a model that overrides ``mu`` modifies the perturbation equations,
and one that overrides ``Omega_matter`` does not conserve matter.
None of this needs CAMB installed -- ``supports_cmb_spectra`` is
answered without importing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import CPL, IDE, LCDM, WCDM
from CosmoFit.cosmology.boltzmann import supports_cmb_spectra
from CosmoFit.cosmology.custom import define_model
from CosmoFit.cosmology.models.fr import FRHuSawicki
from CosmoFit.cosmology.models.rvm import RunningVacuum


def _flat_lcdm(p, z):

    return np.sqrt(p["Omega_m"] * (1.0 + z) ** 3 + 1.0 - p["Omega_m"])


@pytest.mark.parametrize("model", [LCDM, WCDM, CPL])
def test_smooth_dark_energy_is_supported(model):

    supported, reason = supports_cmb_spectra(model)

    assert supported, reason


def test_interacting_dark_energy_is_refused():

    supported, reason = supports_cmb_spectra(IDE)

    assert not supported
    assert "matter" in reason


def test_running_vacuum_is_refused_for_its_matter():

    supported, reason = supports_cmb_spectra(RunningVacuum)

    assert not supported
    assert "matter" in reason


def test_lcdm_subclass_with_its_own_mu_is_refused():
    """
    Hu-Sawicki is LCDM in the background; only its ``mu`` differs.
    It used to be caught by its name alone.
    """

    supported, _ = supports_cmb_spectra(FRHuSawicki)

    assert not supported

    class Renamed(FRHuSawicki):
        pass

    supported, reason = supports_cmb_spectra(Renamed)

    assert not supported
    assert "f_R0" in reason


def test_custom_model_with_w_and_mu_is_refused():

    model = define_model(
        "WithGrowth",
        E=_flat_lcdm,
        w=lambda p, z: -1.0 + 0.0 * z,
        mu=lambda p, a, k: 1.1 + 0.0 * a,
    )

    supported, reason = supports_cmb_spectra(model)

    assert not supported
    assert "mu" in reason


def test_custom_model_with_only_w_is_still_supported():

    model = define_model(
        "PlainW",
        E=_flat_lcdm,
        w=lambda p, z: -1.0 + 0.0 * z,
    )

    supported, reason = supports_cmb_spectra(model)

    assert supported, reason


# ============================================================
# Models built from an action
# ============================================================

def _sympy():

    return pytest.importorskip("sympy")


def test_teleparallel_action_model_with_its_own_growth_is_refused():
    """
    ``growth="quasi_static"`` gives the model ``mu = 1/f'``, and its
    action-derived ``w`` used to carry it through the gate anyway.

    With the default ``growth="gr"`` the model declares that its
    extra degrees of freedom do not reach the clustering -- an
    effective dark energy -- and a smooth PPF fluid is that same
    assumption, so it stays supported (next test).
    """

    _sympy()

    from CosmoFit.theory import Action

    model = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={
            "A0": {"default": -4.2, "bounds": (-30.0, 0.0)},
            "b": {"default": 0.1, "bounds": (-2.0, 0.9)},
        },
        closure="A0",
        growth="quasi_static",
    ).build("PowerLawFTGate")

    assert hasattr(model, "w")

    supported, reason = supports_cmb_spectra(model)

    assert not supported
    assert "mu" in reason


def test_teleparallel_action_model_as_effective_dark_energy_is_supported():

    _sympy()

    from CosmoFit.theory import Action

    model = Action(
        "T + A0*(-T)**b",
        geometry="teleparallel",
        params={
            "A0": {"default": -4.2, "bounds": (-30.0, 0.0)},
            "b": {"default": 0.1, "bounds": (-2.0, 0.9)},
        },
        closure="A0",
    ).build("PowerLawFTGateGR")

    supported, reason = supports_cmb_spectra(model)

    assert supported, reason


def test_minimally_coupled_quintessence_is_supported():
    """
    A canonical scalar field with no coupling to curvature is a
    smooth dark-energy fluid -- exactly what CAMB's PPF module is
    for -- and must stay supported.
    """

    _sympy()

    from CosmoFit.theory import Action

    model = Action(
        "R",
        fields={"phi": "X - V0"},
        params={"V0": {"default": 2.0}},
        closure="V0",
    ).build("ConstantPotentialGate")

    supported, reason = supports_cmb_spectra(model)

    assert supported, reason
