"""
Pantheon+ sample selection, and the Cepheid-calibrated variant.

Until this was written the loader kept every non-calibrator light
curve, 44 of them below the z = 0.01 Hubble-flow cut that every
published Pantheon+ fit applies, and the Cepheid option put the 77
calibrators on the Hubble diagram at their redshifts instead of at
their Cepheid distances -- so the one configuration meant to
measure H0 could not. These tests pin both down, the second by
measuring H0 with it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from cosmofit import LCDM, Fitter
from cosmofit.cosmology.core.parameters import CosmologyParameters
from cosmofit.data.loader import load_pantheon
from cosmofit.likelihoods.pantheon import PantheonLikelihood


def _cosmology(**overrides):

    values = dict(H0=73.0, Omega_m=0.334, MB=-19.25)
    values.update(overrides)

    return LCDM(CosmologyParameters(**values))


# ============================================================
# Sample selection
# ============================================================

def test_hubble_flow_sample_is_the_published_one():
    """
    1590 light curves, every one above z = 0.01: the sample of
    Brout et al. (2022), and the count every other Pantheon+
    implementation reproduces.
    """

    data = load_pantheon()

    assert data.size == 1590
    assert np.all(data.z_hd > 0.01)

    # Calibrators above the cut are ordinary Hubble-flow supernovae
    # here, not dropped for being calibrators.
    assert int(np.sum(data.cepheid == 1)) == 10


def test_cepheid_sample_adds_every_calibrator():

    data = load_pantheon(include_cepheid=True)

    assert data.size == 1657
    assert int(np.sum(data.cepheid == 1)) == 77

    calibrators = data.cepheid == 1

    # The release fills CEPH_DIST with -9 off the calibrators; on
    # them it must be a real distance modulus (29 to 34 for hosts
    # at 7 to 60 Mpc).
    assert np.all(
        (data.ceph_dist[calibrators] > 28.0)
        & (data.ceph_dist[calibrators] < 35.0)
    )

    # Everything that is not a calibrator still obeys the cut.
    assert np.all(data.z_hd[~calibrators] > 0.01)


def test_covariance_is_cut_with_the_data():

    full = load_pantheon(include_cepheid=True)
    flow = load_pantheon()

    assert full.covariance.shape == (1657, 1657)
    assert flow.covariance.shape == (1590, 1590)


# ============================================================
# The Cepheid variant
# ============================================================

def test_calibrators_are_predicted_from_their_cepheids():
    """
    A calibrator's predicted magnitude is ``CEPH_DIST + M_B``, so it
    must not move when the expansion history does; every other
    supernova must.
    """

    cosmology = _cosmology()

    likelihood = PantheonLikelihood(cosmology, include_cepheid=True)

    calibrators = likelihood.data.cepheid == 1

    before = likelihood.model()

    cosmology.params.update(H0=67.0)
    cosmology.refresh()

    after = likelihood.model()

    np.testing.assert_allclose(
        before[calibrators],
        likelihood.data.ceph_dist[calibrators] + cosmology.MB,
    )
    np.testing.assert_array_equal(after[calibrators], before[calibrators])
    assert np.all(after[~calibrators] > before[~calibrators])


def test_cepheids_default_to_a_free_absolute_magnitude():

    likelihood = PantheonLikelihood(_cosmology(), include_cepheid=True)

    assert likelihood.marginalize_MB is False


def test_marginalizing_away_the_calibration_is_refused():

    with pytest.raises(ValueError, match="calibration"):

        PantheonLikelihood(
            _cosmology(), include_cepheid=True, marginalize_MB=True,
        )


def test_cepheid_variant_measures_h0():
    """
    The point of the calibrators: Pantheon+SH0ES on its own fixes
    H0. Brout et al. (2022) find 73.5 +- 1.1 for flat LCDM; profiling
    (H0, M_B) at their Omega_m = 0.334 has to land there.

    Before the fix the calibrators sat on the Hubble diagram, the
    H0 - M_B degeneracy was unbroken, and this minimum was anywhere
    along a line.
    """

    cosmology = _cosmology()

    likelihood = PantheonLikelihood(cosmology, include_cepheid=True)

    def chi2(theta):

        cosmology.params.update(H0=theta[0], MB=theta[1])
        cosmology.refresh()

        return likelihood.chi2()

    result = minimize(chi2, x0=[70.0, -19.3], method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6})

    h0, mb = result.x

    assert h0 == pytest.approx(73.5, abs=0.5)
    assert mb == pytest.approx(-19.25, abs=0.05)


# ============================================================
# Double counting with an H0 prior
# ============================================================

def _fit(h0_version):

    return Fitter(
        model=LCDM,
        datasets=["pantheon", "h0"],
        free_params=["H0", "Omega_m", "MB"],
        initial={"H0": 73.0, "Omega_m": 0.33, "MB": -19.25},
        dataset_kwargs={
            "pantheon": {"include_cepheid": True},
            "h0": {"version": h0_version},
        },
    )


def test_sh0es_prior_on_top_of_the_cepheids_warns():

    with pytest.warns(UserWarning, match="Cepheid hosts twice"):

        _fit("sh0es2022")


def test_an_independent_h0_does_not_warn():

    with warnings.catch_warnings():

        warnings.filterwarnings("error", message=".*Cepheid hosts twice.*")

        _fit("tdcosmo2025")
