"""
The SDSS BAO + full-shape consensus, against its published table.

This is the same BOSS/eBOSS galaxies as ``sdss_bao``, analysed for
their full anisotropic clustering rather than the BAO peak alone. It
adds the growth rate, and -- the part that actually matters -- the
covariance *between* growth and geometry.

That covariance is why the dataset exists here. Using the BAO-only
dataset alongside the separate ``fsigma8`` compilation covers the
same galaxies while treating those two as independent, and the
released covariance says they are correlated at 0.19 to 0.64
within a redshift bin, depending on the tracer. That combination had been reachable without a warning
for as long as both datasets have existed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoFit import LCDM, Fitter
from CosmoFit.data.loader import load_sdss_fsbao


FIDUCIAL = dict(
    H0=67.8, Omega_m=0.31, Omega_b=0.049, rd=147.2, sigma8=0.811,
)


#: Alam et al. (2021), Phys. Rev. D 103, 083533, Table 3 --
#: the consensus BAO + full-shape measurements, transcribed from
#: the paper rather than from the files this test reads.
PUBLISHED = {
    (0.38, "DM_over_rs"): (10.27, 0.15),
    (0.38, "DH_over_rs"): (24.89, 0.58),
    (0.38, "fsigma8"): (0.497, 0.045),
    (0.51, "DM_over_rs"): (13.38, 0.18),
    (0.51, "DH_over_rs"): (22.43, 0.48),
    (0.51, "fsigma8"): (0.459, 0.038),
    (0.698, "DM_over_rs"): (17.65, 0.30),
    (0.698, "DH_over_rs"): (19.77, 0.47),
    (0.698, "fsigma8"): (0.473, 0.044),
    (1.48, "DM_over_rs"): (30.21, 0.79),
    (1.48, "DH_over_rs"): (13.23, 0.47),
    (1.48, "fsigma8"): (0.462, 0.045),
}


@pytest.fixture(scope="module")
def dataset():

    return load_sdss_fsbao()


# ============================================================
# Against the paper
# ============================================================

def test_every_measurement_matches_the_published_table(dataset):
    """
    Values *and* error bars. The errors come from the covariance
    file, which is a separate product from the data file, so this
    also checks the two were paired correctly.
    """

    assert len(dataset.z) == len(PUBLISHED)

    seen = set()

    for z, value, observable, sigma in zip(
        dataset.z, dataset.value, dataset.observable,
        dataset.covariance.sigma,
    ):

        key = (round(float(z), 3), str(observable))

        assert key in PUBLISHED, key

        seen.add(key)

        expected, expected_sigma = PUBLISHED[key]

        assert value == pytest.approx(expected, abs=0.006)

        assert sigma == pytest.approx(expected_sigma, abs=0.006)

    assert seen == set(PUBLISHED)


def test_growth_and_geometry_are_correlated_in_the_covariance(dataset):
    """
    The reason to prefer this over BAO-only plus a growth
    compilation. A block-diagonal read, or a covariance paired with
    the wrong data file, would leave these at zero.
    """

    covariance = dataset.covariance.matrix

    sigma = dataset.covariance.sigma

    correlation = covariance / np.outer(sigma, sigma)

    index = {
        (round(float(z), 3), str(o)): i
        for i, (z, o) in enumerate(zip(dataset.z, dataset.observable))
    }

    # Measured, per bin: +0.388, +0.389, +0.185, +0.636. The QSO
    # bin is much the strongest, which is worth stating rather than
    # covering with a loose bound.
    expected = {0.38: 0.388, 0.51: 0.389, 0.698: 0.185, 1.48: 0.636}

    for z, value in expected.items():

        i = index[(z, "DM_over_rs")]
        j = index[(z, "fsigma8")]

        assert correlation[i, j] == pytest.approx(value, abs=0.01), (
            f"z={z}: corr(DM/rd, fsigma8) = {correlation[i, j]:.3f}"
        )

        # And the growth rate anticorrelates with D_H throughout.
        k = index[(z, "DH_over_rs")]

        assert -0.40 < correlation[k, j] < -0.15


def test_the_covariance_is_block_diagonal_across_samples(dataset):
    """
    The LRG and QSO files are separate samples with no published
    cross-correlation, so their blocks must not be coupled -- while
    each block's own structure survives.
    """

    covariance = dataset.covariance.matrix

    # First nine rows are the BOSS+eBOSS LRG file, last three the QSO.
    assert np.all(covariance[:9, 9:] == 0.0)
    assert np.all(covariance[9:, :9] == 0.0)

    assert np.any(covariance[9:, 9:] != np.diag(np.diag(covariance[9:, 9:])))


# ============================================================
# What it constrains
# ============================================================

def test_it_is_the_only_bao_dataset_that_responds_to_sigma8():
    """
    Every other BAO dataset here measures distances only. This one
    has an f*sigma8 axis, so `sigma8` has to reach it.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["sdss_fsbao"],
        free_params=["H0", "Omega_m", "sigma8"],
        initial=FIDUCIAL,
    )

    baseline = fit.logpost.chi2(fit.theta0)

    theta = fit.theta0.copy()
    theta[fit.free_params.index("sigma8")] = 0.65

    assert fit.logpost.chi2(theta) != baseline

    # And the BAO-only sibling does not.
    bao_only = Fitter(
        model=LCDM,
        datasets=["sdss_bao"],
        free_params=["H0", "Omega_m", "sigma8"],
        initial=FIDUCIAL,
    )

    theta = bao_only.theta0.copy()
    theta[bao_only.free_params.index("sigma8")] = 0.65

    assert bao_only.logpost.chi2(theta) == bao_only.logpost.chi2(
        bao_only.theta0,
    )


def test_fsigma8_is_predicted_without_an_AP_correction():
    """
    A full-shape fit varies the geometry alongside the growth rate,
    so the fiducial it was measured against is already fitted
    rather than something to correct back to. The AP rescaling that
    `likelihoods/fsigma8.py` applies would count it twice.
    """

    from CosmoFit.stats.fitter import DATASET_REGISTRY

    cosmology = LCDM(LCDM.PARAMS_CLASS(**FIDUCIAL))

    likelihood = DATASET_REGISTRY["sdss_fsbao"](cosmology)

    prediction = likelihood.model()

    growth_rows = likelihood.data.observable == "fsigma8"

    expected = cosmology.background.fsigma8(
        likelihood.data.z[growth_rows],
    )

    np.testing.assert_allclose(prediction[growth_rows], expected)


# ============================================================
# The double-counting that was reachable without a warning
# ============================================================

@pytest.mark.parametrize(
    "datasets",
    [
        ["sdss_bao", "fsigma8"],
        ["sdss_fsbao", "fsigma8"],
        ["sdss_bao", "sdss_fsbao"],
    ],
)
def test_overlapping_sdss_combinations_warn(datasets):
    """
    All three cover the same BOSS/eBOSS galaxies. The first was
    reachable in silence until this dataset was added, which is how
    it was noticed.
    """

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=LCDM,
            datasets=datasets,
            free_params=["H0", "Omega_m"],
            initial=FIDUCIAL,
        )

    assert any(
        "sdss" in str(w.message).lower() or "eboss" in str(w.message).lower()
        for w in caught
    ), [str(w.message) for w in caught]
