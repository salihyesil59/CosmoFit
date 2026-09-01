"""
Validation of the Planck 2018 CMB lensing likelihood.

The lensing likelihood has one feature that makes it easy to get
wrong and hard to notice: the reconstruction's normalization depends
on the CMB spectra it was measured from, and the correction for that
is *designed to vanish at the fiducial cosmology*. Testing only at
Planck's best fit therefore cannot tell you whether the correction
was implemented at all -- which is exactly why several checks here
deliberately move away from it.

The other trap is the ``PP`` scaling. Planck's bandpowers are
``[L(L+1)]^2 C_L^{phiphi} / 2 pi``; the raw spectrum and the
``L(L+1)C_L/2pi`` convention differ from it by orders of magnitude
at the relevant multipoles, so a wrong choice is caught by a
sanity check, but only if one is written.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM, CPL, CosmologyParameters
from CosmoFit.data.loader import load_planck_lensing


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


#: Planck 2018 best-fit LCDM.
PLANCK_BEST_FIT = dict(
    H0=67.36,
    Omega_m=0.3153,
    Omega_b=0.02237 / 0.6736 ** 2,
    ln1e10As=3.044,
    n_s=0.9649,
    tau_reio=0.0544,
)


@pytest.fixture(scope="module")
def dataset():

    return load_planck_lensing()


@pytest.fixture(scope="module")
def likelihood():

    if not _has_camb():
        pytest.skip("CAMB not installed")

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    return PlanckLensingLikelihood(

        LCDM(CosmologyParameters(**PLANCK_BEST_FIT)),

    )


# ============================================================
# The data loads as released
# ============================================================

def test_bandpowers_and_windows_have_consistent_shapes(dataset):

    assert dataset.size == 9

    assert dataset.windows.shape == (9, dataset.lmax + 1)
    assert dataset.delta_windows.shape == (9, 4, dataset.lmax + 1)

    np.testing.assert_allclose(

        np.diag(dataset.covariance.matrix),

        dataset.sigma ** 2,

        rtol=5e-3,

    )


def test_windows_cover_the_stated_multipole_range(dataset):
    """
    The conservative baseline is L = 8-400. Each bin's window must
    be zero outside its own range and non-zero inside it, and the
    union must be exactly that range -- which catches a window file
    read with the wrong column or scattered to the wrong indices.
    """

    low, high = dataset.ell_range

    support = np.any(dataset.windows != 0.0, axis=0)

    ells = np.nonzero(support)[0]

    assert ells.min() == low
    assert ells.max() == high

    # Contiguous: no gaps between the bins.
    assert np.array_equal(ells, np.arange(low, high + 1))


def test_effective_multipoles_are_inside_their_windows(dataset):
    """
    Each bandpower's quoted L_av must sit within the multipole range
    its own window covers -- a cheap check that bandpowers and
    windows were not paired up in the wrong order.
    """

    for b in range(dataset.size):

        support = np.nonzero(dataset.windows[b])[0]

        assert support.min() <= dataset.ell[b] <= support.max()


# ============================================================
# Against Planck
# ============================================================

@requires_camb
def test_chi2_at_planck_best_fit(likelihood):
    """
    At Planck's own best-fit LCDM the lensing likelihood must return
    chi2 near the number of bandpowers. Planck 2018 reports ~9 for
    these 9 points.

    This is the same bar the compressed distance priors are held to:
    a likelihood that cannot reproduce the cosmology its own data
    came from is broken, whatever else it does.
    """

    chi2 = likelihood.chi2()

    assert 5.0 < chi2 < 14.0, chi2


@requires_camb
def test_residuals_are_not_systematically_offset(likelihood):
    """
    A wrong overall normalization -- the most likely consequence of
    a scaling mistake -- would show as every pull having the same
    sign, which a chi2 bound alone would not necessarily catch.
    """

    pulls = likelihood.residuals() / likelihood.covariance.sigma

    assert np.abs(pulls).max() < 3.0

    assert abs(pulls.mean()) < 1.0

    # Both signs present.
    assert (pulls > 0).any() and (pulls < 0).any()


@requires_camb
def test_lensing_amplitude_responds_the_right_way(likelihood):
    """
    Raising the primordial amplitude must raise the predicted
    lensing power. Obvious, and the check that the model is
    connected to the parameters at all rather than returning a
    fixed array.
    """

    model = likelihood.cosmology

    baseline = likelihood.model().copy()

    model.params.ln1e10As += 0.10
    model.refresh()

    louder = likelihood.model()

    assert np.all(louder > baseline)

    model.params.ln1e10As -= 0.10
    model.refresh()

    # CAMB is not bit-reproducible from one call to the next. Its
    # OpenMP reductions depend on how many threads the runtime
    # actually uses, and that varies with what else is running.
    # Measured on this very spectrum: the same parameters computed
    # with one thread and with eight differ by up to 1e-10
    # relative, and a full-suite run was seen to land 1.4e-12 away
    # from the quiet-machine answer. An rtol of 1e-12 here was
    # therefore not a tight check, it was a coin toss -- and one
    # that came up tails more often the more loaded the machine,
    # which is exactly the pattern that makes a flaky test hard to
    # read.
    #
    # The tolerance has to clear that noise and still sit far below
    # the effect it is checking: restoring the amplitude has to
    # undo a change worth ~40% at the smallest bandpower. 1e-6 is
    # four orders above the noise and five below the signal, so
    # every way this can really fail -- refresh() not recomputing,
    # the parameter not actually restored, a half-updated state --
    # still fails it.
    np.testing.assert_allclose(likelihood.model(), baseline, rtol=1e-6)


# ============================================================
# The linear correction
# ============================================================

@requires_camb
def test_linear_correction_is_actually_applied(likelihood):
    """
    The correction vanishes at the fiducial cosmology *by
    construction*, so a chi2 check there passes whether or not it
    was implemented. Move the CMB spectra away from the fiducial and
    it must not.

    Compared here against the same prediction with the correction
    forced off, so what is measured is the correction's own size.
    """

    data = likelihood.data

    model = likelihood.cosmology

    def with_and_without():

        spectra = likelihood.backend.lensing_spectra(data.lmax)

        binned = data.windows @ spectra["PP"]

        correction = sum(

            data.delta_windows[:, i, :] @ spectra[name]

            for i, name in enumerate(data.CORRECTION_SPECTRA)

        )

        return binned, binned + correction - data.fiducial_correction

    # Well away from the fiducial: a 20% shift in the amplitude and
    # a visibly different tilt.
    model.params.ln1e10As += 0.20
    model.params.n_s -= 0.03
    model.refresh()

    plain, corrected = with_and_without()

    relative = np.abs(corrected / plain - 1.0)

    assert relative.max() > 1e-3, (

        f"the linear correction changes the prediction by at most "

        f"{relative.max():.2e} far from the fiducial cosmology -- it "

        f"is probably not being applied"

    )

    model.params.ln1e10As -= 0.20
    model.params.n_s += 0.03
    model.refresh()


@requires_camb
def test_correction_is_small_at_the_fiducial_cosmology(likelihood):
    """
    The flip side: near Planck's best fit the correction should be a
    small perturbation, not a large one. A sign error or a wrong
    column order would show up here as a big shift.
    """

    data = likelihood.data

    spectra = likelihood.backend.lensing_spectra(data.lmax)

    correction = sum(

        data.delta_windows[:, i, :] @ spectra[name]

        for i, name in enumerate(data.CORRECTION_SPECTRA)

    ) - data.fiducial_correction

    binned = data.windows @ spectra["PP"]

    assert np.abs(correction / binned).max() < 0.05


# ============================================================
# Wiring
# ============================================================

@requires_camb
def test_camb_derived_sigma8_matches_planck(likelihood):
    """
    CAMB's own sigma8 at Planck's best fit, which is a published
    number (0.8111) and a good check that the amplitude reaches the
    transfer function correctly.
    """

    assert likelihood.backend.sigma8() == pytest.approx(0.8111, abs=0.002)


@requires_camb
def test_free_sigma8_and_camb_sigma8_are_different_things(likelihood):
    """
    The free ``sigma8`` parameter and CAMB's derived one are
    unrelated by default. This pins that, because the fit-level
    warning about it only makes sense if it is true.
    """

    model = likelihood.cosmology

    model.params.sigma8 = 0.60
    model.refresh()

    assert likelihood.backend.sigma8() == pytest.approx(0.8111, abs=0.002)

    model.params.sigma8 = 0.811
    model.refresh()


@requires_camb
def test_fitter_warns_about_the_double_amplitude():

    import warnings

    from CosmoFit import Fitter

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(

            model=LCDM,

            datasets=["planck_lensing", "s8"],

            free_params=["H0", "Omega_m", "sigma8"],

            initial={**PLANCK_BEST_FIT, "sigma8": 0.81},

        )

    assert any("two independent" in str(w.message) for w in caught), (

        [str(w.message) for w in caught]

    )


@requires_camb
def test_modified_gravity_is_refused():

    from CosmoFit import FRHuSawicki
    from CosmoFit.cosmology.boltzmann import BoltzmannError
    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    model = FRHuSawicki(

        FRHuSawicki.PARAMS_CLASS(H0=67.4, Omega_m=0.315),

    )

    with pytest.raises(BoltzmannError):

        PlanckLensingLikelihood(model)


@requires_camb
def test_works_for_a_w_of_z_model():
    """
    Anything with a w(z) goes through CAMB's PPF module, lensing
    included -- so CPL must work, and at the LCDM limit must give
    LCDM's answer.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    lcdm = PlanckLensingLikelihood(

        LCDM(CosmologyParameters(**PLANCK_BEST_FIT)),

    )

    cpl = PlanckLensingLikelihood(

        CPL(CosmologyParameters(w0=-1.0, wa=0.0, **PLANCK_BEST_FIT)),

    )

    np.testing.assert_allclose(cpl.model(), lcdm.model(), rtol=1e-8)

    evolving = PlanckLensingLikelihood(

        CPL(CosmologyParameters(w0=-0.8, wa=-0.5, **PLANCK_BEST_FIT)),

    )

    assert np.max(np.abs(evolving.model() / lcdm.model() - 1.0)) > 1e-3
