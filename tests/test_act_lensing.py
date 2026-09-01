"""
Validation of the ACT DR6 lensing likelihood.

The failure mode this guards against is a *smooth rescaling* of the
prediction. ACT's products are built on the lensing convergence,
Planck's on the potential, and the two differ by ``2 pi / 4``. Get
that wrong and nothing crashes: the theory comes out uniformly too
big or too small, and a fit absorbs it into the amplitude, leaving a
perfectly ordinary posterior centred somewhere else.

So the decisive test is the lensing amplitude. Fitting a single
scaling of the theory to the data must return ACT's published
``A_lens = 1.013 +- 0.023`` -- a number that depends on the
convergence/potential conversion, the binning matrix, the bin
slicing and the Hartlap correction all being right at once, and that
is used nowhere as an input.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM, CosmologyParameters
from CosmoFit.data.loader import load_act_lensing


def _has_camb() -> bool:

    try:

        import camb  # noqa: F401

    except ImportError:

        return False

    return True


requires_camb = pytest.mark.skipif(

    not _has_camb(),

    reason="CAMB not installed (optional 'cmb' extra)",

)


PLANCK_BEST_FIT = dict(
    H0=67.36,
    Omega_m=0.3153,
    Omega_b=0.02237 / 0.6736 ** 2,
    ln1e10As=3.044,
    n_s=0.9649,
    tau_reio=0.0544,
)


@pytest.fixture(scope="module")
def likelihood():

    if not _has_camb():
        pytest.skip("CAMB not installed")

    from CosmoFit.likelihoods.act_lensing import ACTDR6LensingLikelihood

    return ACTDR6LensingLikelihood(

        LCDM(CosmologyParameters(**PLANCK_BEST_FIT)),

    )


def amplitude(lk):
    """
    Best-fit single scaling of the theory and its uncertainty:

        A = t^T C^-1 d / t^T C^-1 t,   sigma_A = (t^T C^-1 t)^-1/2
    """

    theory = lk.model()

    weighted = lk.covariance.solve(theory)

    fisher = float(theory @ weighted)

    return float(lk.data.value @ weighted) / fisher, 1.0 / np.sqrt(fisher)


# ============================================================
# The data
# ============================================================

@pytest.mark.parametrize(
    "version,n_bin,high",
    [("act_baseline", 10, 763), ("act_extended", 13, 1250)],
)
def test_variants_have_the_published_shape(version, n_bin, high):

    data = load_act_lensing(version)

    assert data.size == n_bin
    assert data.ell_range[1] == high

    assert data.spectrum == "KK"

    # ACT's covariance is CMB-marginalized, so there is no separate
    # linear correction to apply -- and claiming one would mean
    # correcting twice.
    assert not data.has_linear_correction

    assert data.windows.shape == (n_bin, data.lmax + 1)


def test_bin_centres_come_from_the_binning_matrix():
    """
    The effective multipoles are computed as ``W @ ell`` rather than
    read from a column, so they cannot disagree with the windows
    that produce the prediction. Check they land where ACT says.
    """

    data = load_act_lensing("act_baseline")

    assert data.ell[0] == pytest.approx(53.0, abs=1.0)
    assert data.ell[-1] == pytest.approx(700.5, abs=1.0)

    assert np.all(np.diff(data.ell) > 0)


def test_extended_contains_the_baseline():
    """
    Both variants slice the same released vector from the same
    start, so the extended one must begin with exactly the baseline
    bins. A different `start` in one of them would show here.
    """

    baseline = load_act_lensing("act_baseline")
    extended = load_act_lensing("act_extended")

    np.testing.assert_allclose(

        extended.value[: baseline.size], baseline.value,

    )

    np.testing.assert_allclose(

        extended.ell[: baseline.size], baseline.ell,

    )


def test_hartlap_correction_widens_the_covariance():
    """
    The covariance is estimated from 796 simulations, so its inverse
    is biased and the released matrix must be *inflated* before use.
    Checked against the released file directly.
    """

    from pathlib import Path

    import CosmoFit

    released = np.loadtxt(

        Path(CosmoFit.__file__).parent
        / "data" / "cmb" / "act_dr6_lensing" / "covmat_act_cmbmarg.txt",

    )

    data = load_act_lensing("act_baseline")

    keep = np.arange(len(released))[2:-6]

    expected_hartlap = (796 - data.size - 2.0) / (796 - 1.0)

    ratio = (

        data.covariance.matrix

        / released[np.ix_(keep, keep)]

    )

    np.testing.assert_allclose(ratio, 1.0 / expected_hartlap, rtol=1e-12)

    # Inflated, not shrunk.
    assert expected_hartlap < 1.0


# ============================================================
# Against ACT's published result
# ============================================================

@requires_camb
def test_lensing_amplitude_matches_the_published_value(likelihood):
    """
    The decisive check: ACT's headline is a 2.3% measurement of the
    lensing amplitude, ``A_lens = 1.013 +- 0.023`` relative to
    Planck's LCDM. Evaluating the theory at exactly that cosmology
    must return it.

    A wrong convergence/potential conversion is a factor of
    ``2 pi / 4 = 1.57``, which would put ``A_lens`` at 0.65 or 1.6
    rather than near 1 -- so this is also the test that pins the
    single conversion the module turns on.
    """

    A, sigma = amplitude(likelihood)

    assert A == pytest.approx(1.013, abs=0.03), A

    assert sigma == pytest.approx(0.023, rel=0.20), sigma


@requires_camb
def test_chi2_is_reasonable_at_planck_best_fit(likelihood):

    chi2 = likelihood.chi2()

    assert 3.0 < chi2 < 22.0, chi2

    pulls = likelihood.residuals() / likelihood.covariance.sigma

    assert np.abs(pulls).max() < 3.5
    assert (pulls > 0).any() and (pulls < 0).any()


@requires_camb
def test_agrees_with_planck_lensing_on_the_same_cosmology(likelihood):
    """
    Two independent reconstructions evaluated at the same cosmology
    must both give an amplitude near 1. They are different
    telescopes and different conventions, so agreement here is a
    real cross-check of both implementations rather than a
    tautology.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    planck = PlanckLensingLikelihood(likelihood.cosmology)

    act_amplitude, act_sigma = amplitude(likelihood)
    planck_amplitude, planck_sigma = amplitude(planck)

    assert planck_amplitude == pytest.approx(1.0, abs=0.06)

    difference = abs(act_amplitude - planck_amplitude)

    assert difference < 3.0 * np.hypot(act_sigma, planck_sigma)


# ============================================================
# Wiring
# ============================================================

@requires_camb
def test_amplitude_responds_to_the_primordial_amplitude(likelihood):

    model = likelihood.cosmology

    baseline = likelihood.model().copy()

    model.params.ln1e10As += 0.10
    model.refresh()

    assert np.all(likelihood.model() > baseline)

    model.params.ln1e10As -= 0.10
    model.refresh()

    # Same reason as the matching check in test_planck_lensing.py:
    # CAMB's output is not bit-reproducible between calls, so an
    # rtol of 1e-12 on a recomputed spectrum fails intermittently.
    # See the comment there for the measurements.
    np.testing.assert_allclose(likelihood.model(), baseline, rtol=1e-6)


@requires_camb
def test_shares_the_camb_backend_with_planck_lensing():

    from CosmoFit.likelihoods.act_lensing import ACTDR6LensingLikelihood
    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    model = LCDM(CosmologyParameters(**PLANCK_BEST_FIT))

    act = ACTDR6LensingLikelihood(model)
    planck = PlanckLensingLikelihood(model)

    assert act.backend is planck.backend


@requires_camb
def test_fitter_warns_about_combining_the_two_reconstructions():
    """
    ACT's map overlaps Planck's on the sky, so the two are
    correlated and the joint constraint from treating them as
    independent is overstated.
    """

    import warnings

    from CosmoFit import Fitter

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(

            model=LCDM,

            datasets=["planck_lensing", "act_lensing"],

            free_params=["H0", "Omega_m"],

            initial=PLANCK_BEST_FIT,

        )

    assert any(

        "overlaps" in str(w.message) for w in caught

    ), [str(w.message) for w in caught]
