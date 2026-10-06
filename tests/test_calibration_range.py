"""
Fitting formulas used outside their calibration, and parameters the
compressed CMB cannot see.

The z_star and z_drag fits are calibrated against CAMB over ranges
the code recorded and never checked; the default priors reach far
beyond them (omega_m h^2 down to 0.0125, N_eff up to 8). Points out
there are not rejected -- that would truncate the prior with no trace
-- but the user is told, once.
"""

from __future__ import annotations

import warnings

import pytest

from cosmofit import LCDM, Fitter
from cosmofit.cosmology.core import utils


def _build(**values):

    params = dict(H0=67.4, Omega_m=0.315, Omega_b=0.0493)
    params.update(values)

    return LCDM(LCDM.PARAMS_CLASS(**params))


@pytest.fixture(autouse=True)
def fresh_warning_registry():

    saved = set(utils._EXTRAPOLATION_WARNED)
    utils._EXTRAPOLATION_WARNED.clear()

    yield

    utils._EXTRAPOLATION_WARNED.clear()
    utils._EXTRAPOLATION_WARNED.update(saved)


def test_inside_the_range_is_quiet():

    model = _build()

    with warnings.catch_warnings():

        warnings.simplefilter("error")

        model.recombination.z_star()
        model.sound_horizon.z_drag()


def test_z_star_outside_its_range_warns_once():

    model = _build(Omega_m=0.9)

    with pytest.warns(UserWarning, match="z_star fitting formula"):
        model.recombination.z_star()

    with warnings.catch_warnings():

        warnings.simplefilter("error")

        model.recombination.z_star()


def test_z_drag_warns_for_n_eff():

    model = _build(N_eff=7.0)

    with pytest.warns(UserWarning, match="N_eff = 7"):
        model.sound_horizon.z_drag()


def test_free_n_eff_with_compressed_cmb_only_warns():

    with pytest.warns(UserWarning, match="cannot respond to it"):

        Fitter(
            model=LCDM, datasets=["planck", "desi"],
            free_params=["H0", "Omega_m", "N_eff"],
            initial={"H0": 67.4, "Omega_m": 0.315}, compute_rd=True,
        )
