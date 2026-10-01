"""
Which tau goes next to plik_lite.

The "tau" dataset shipped one number, 0.0544 +- 0.0073, described as
Planck's low-l polarization constraint and recommended as the
companion to "planck_lite". It is not the low-l constraint: it is the
TT,TE,EE+lowE posterior, and ``test_planck_lowe`` already shows as
much -- profiling plik_lite and the low-l EE table together reproduces
exactly 0.0544 +- 0.0073. Adding it to a fit that already has
plik_lite counts the high-l spectra twice. Low-l EE alone gives
0.0506 +- 0.0086, which is now the default.
"""

from __future__ import annotations

import warnings

import pytest

from CosmoFit import LCDM, Fitter
from CosmoFit.cosmology.core.parameters import CosmologyParameters
from CosmoFit.data.loader import load_gaussian_prior
from CosmoFit.likelihoods.priors import TauLikelihood


def test_default_is_the_low_l_constraint():

    data = load_gaussian_prior("tau")

    assert data.value == pytest.approx(0.0506)
    assert data.sigma == pytest.approx(0.0086)

    likelihood = TauLikelihood(LCDM(CosmologyParameters(H0=67.4, Omega_m=0.315)))

    assert likelihood.version == "planck2018_lowe"


def test_the_full_posterior_is_still_available():

    data = load_gaussian_prior("tau", "planck2018")

    assert data.value == pytest.approx(0.0544)
    assert data.sigma == pytest.approx(0.0073)


def _fit(version):

    pytest.importorskip("camb")

    return Fitter(
        model=LCDM,
        datasets=["planck_lite", "tau"],
        free_params=["H0", "Omega_m", "ln1e10As", "tau_reio"],
        initial={"H0": 67.4, "Omega_m": 0.315},
        dataset_kwargs={"tau": {"version": version}},
    )


def test_full_posterior_next_to_plik_lite_warns():

    with pytest.warns(UserWarning, match="counts them twice"):

        _fit("planck2018")


def test_default_next_to_plik_lite_does_not_warn():

    with warnings.catch_warnings():

        warnings.filterwarnings("error", message=".*counts them twice.*")

        _fit("planck2018_lowe")
