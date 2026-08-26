"""
Validation of the from-scratch Planck likelihood.

The point of this file is to separate two failure modes that a
single end-to-end chi2 would blend together:

1. **The binning and covariance algebra.** Planck's bandpower
   windows are stored with Fortran-inclusive ranges and the
   covariance ships as a Fortran unformatted record with only one
   triangle populated. Every reimplementation of this likelihood
   has an off-by-one or a transposition waiting in it, and neither
   produces an error -- they produce a chi2 that is merely wrong.

   Tested here against a *published* log-likelihood for a *fixed*
   input spectrum, so no Boltzmann code is involved and any
   disagreement is unambiguously in this library's algebra.

2. **The CAMB parameter translation.** Tested separately, against
   the physics: known limits (a CPL with w0=-1, wa=0 must
   reproduce LCDM's spectrum bit for bit) and known numbers (the
   first acoustic peak sits where Planck measured it).

Tests needing CAMB skip cleanly when it is not installed; the
algebra tests never need it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CosmoFit.data.loader import load_plik_lite
from CosmoFit.likelihoods.covariance import make_covariance


DATA_DIR = Path(__file__).parent / "data"

REFERENCE_SPECTRUM = DATA_DIR / "Dl_planck2015fit.dat"


#: Published values from ``planck_lite_py.py``'s own ``test()``.
#: Cobaya's plik_lite gives -291.33481235418003 and
#: -101.58123068722568 for the same inputs -- the ~2e-13 spread
#: between the two is matrix-inversion roundoff, which sets the
#: tolerance used below.
EXPECTED_LOGLIKE = {
    "TTTEEE": -291.33481235418026,
    "TT": -101.58123068722583,
}


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


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def dataset():

    return load_plik_lite()


@pytest.fixture(scope="module")
def reference_cl():
    """
    The reference spectrum as C_l from l = 30 upward, matching what
    the likelihood's Boltzmann backend hands it.
    """

    ell, dl_tt, dl_te, dl_ee = np.genfromtxt(

        REFERENCE_SPECTRUM,

        unpack=True,

    )

    factor = ell * (ell + 1.0) / (2.0 * np.pi)

    offset = 30 - int(ell[0])

    return {

        "TT": (dl_tt / factor)[offset:],

        "TE": (dl_te / factor)[offset:],

        "EE": (dl_ee / factor)[offset:],

    }


# ============================================================
# 1. Data files load as released
# ============================================================

def test_covariance_diagonal_matches_released_errors(dataset):
    """
    The 613x613 covariance is read out of a Fortran unformatted
    record with a 4-byte offset, and only its lower triangle is
    populated in the file.

    Its diagonal must reproduce the per-bandpower sigmas that ship
    as a plain text column in a *different* file. Two independent
    products of the Planck pipeline agreeing to float precision is
    strong evidence the record was read correctly -- a wrong
    offset or a bad reshape would scramble the diagonal completely.
    """

    matrix = dataset.covariance.matrix

    np.testing.assert_allclose(

        np.diag(matrix),

        dataset.sigma ** 2,

        rtol=1e-10,

    )

    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)


def test_bandpower_counts(dataset):

    assert dataset.size == 613

    assert dataset.n_bin == (215, 199, 199)


# ============================================================
# 2. Binning + covariance algebra, against a published number
# ============================================================

def _binned_model(dataset, cl):
    """
    Apply Planck's window functions -- the same operation
    :meth:`PlanckLiteLikelihood._bin` performs, written out here
    independently so the test is not just the implementation
    calling itself.
    """

    counts = dict(zip(("TT", "TE", "EE"), dataset.n_bin))

    pieces = []

    for name in ("TT", "TE", "EE"):

        binned = np.empty(counts[name])

        for i in range(counts[name]):

            lo = dataset.blmin[i]
            hi = dataset.blmax[i] + 1

            binned[i] = np.dot(

                cl[name][lo:hi],

                dataset.weights[lo:hi],

            )

        pieces.append(binned)

    return np.concatenate(pieces)


def test_loglike_matches_planck_lite_py_TTTEEE(dataset, reference_cl):
    """
    All 613 bandpowers, against ``planck-lite-py``'s published
    value for this exact spectrum.
    """

    residual = dataset.value - _binned_model(dataset, reference_cl)

    loglike = -0.5 * dataset.covariance.chi2(residual)

    assert loglike == pytest.approx(

        EXPECTED_LOGLIKE["TTTEEE"],

        abs=1e-9,

    )


def test_loglike_matches_planck_lite_py_TT(dataset, reference_cl):
    """
    The TT-only selection, which exercises the covariance
    sub-blocking: taking the TT block of the joint covariance and
    inverting *that*, rather than inverting the joint matrix and
    slicing the result (which would be the marginal, not the
    conditional, and is a real and easy mistake).
    """

    residual = dataset.value - _binned_model(dataset, reference_cl)

    n_tt = dataset.n_bin[0]

    keep = np.arange(n_tt)

    block = make_covariance(

        cov=dataset.covariance.matrix[np.ix_(keep, keep)],

    )

    loglike = -0.5 * block.chi2(residual[keep])

    assert loglike == pytest.approx(

        EXPECTED_LOGLIKE["TT"],

        abs=1e-9,

    )


# ============================================================
# 3. The CAMB translation
# ============================================================

@requires_camb
def test_cpl_at_lcdm_limit_reproduces_lcdm_spectrum():
    """
    CPL with ``w0 = -1, wa = 0`` is LCDM, so its C_l must be
    LCDM's -- to numerical noise, not merely to a plotting eye.

    This is the check that catches the specific way CAMB's
    dark-energy API fails silently: ``pars.DarkEnergy = obj``
    copies into Fortran state, so setting the ``w(a)`` table on the
    Python object *after* assigning it is lost, and CAMB then
    happily returns a perfectly valid cosmological-constant
    spectrum for a w0-wa model.
    """

    from CosmoFit import LCDM, CPL, CosmologyParameters
    from CosmoFit.cosmology.boltzmann import CAMBBackend

    kwargs = dict(H0=67.36, Omega_m=0.3153, Omega_b=0.0493)

    lcdm = CAMBBackend(LCDM(CosmologyParameters(**kwargs))).cls()

    cpl = CAMBBackend(

        CPL(CosmologyParameters(w0=-1.0, wa=0.0, **kwargs)),

    ).cls()

    np.testing.assert_allclose(cpl["TT"], lcdm["TT"], rtol=1e-8)

    # ...and a genuinely different w0-wa must genuinely differ, or
    # the test above would pass for a backend that ignores w(a)
    # entirely.
    evolving = CAMBBackend(

        CPL(CosmologyParameters(w0=-0.9, wa=-0.4, **kwargs)),

    ).cls()

    assert np.max(np.abs(evolving["TT"] / lcdm["TT"] - 1.0)) > 1e-3


@requires_camb
def test_first_acoustic_peak_is_where_planck_measured_it():
    """
    A blunt but decisive check on the parameter translation: at
    Planck's best-fit LCDM the first TT peak must sit at
    ``l ~ 220`` with ``D_l ~ 5700 muK^2``. Getting ``omch2`` wrong
    by adding the neutrino density instead of subtracting it, or
    mixing up ``A_s`` with the Chaplygin-gas parameter of the same
    name, moves this visibly.
    """

    from CosmoFit import LCDM, CosmologyParameters
    from CosmoFit.cosmology.boltzmann import CAMBBackend

    model = LCDM(

        CosmologyParameters(

            H0=67.36,

            Omega_m=0.3153,

            Omega_b=0.02237 / 0.6736 ** 2,

            ln1e10As=3.045,

            n_s=0.9649,

            tau_reio=0.0544,

        ),

    )

    spectra = CAMBBackend(model).cls()

    ell = spectra["ell"]

    dl = ell * (ell + 1.0) / (2.0 * np.pi) * spectra["TT"]

    window = (ell > 150) & (ell < 300)

    peak_ell = ell[window][np.argmax(dl[window])]
    peak_dl = dl[window].max()

    assert 210 <= peak_ell <= 230, peak_ell
    assert 5500 <= peak_dl <= 6000, peak_dl


@requires_camb
def test_chi2_at_planck_best_fit_is_reasonable():
    """
    End to end: at Planck's own best-fit LCDM the likelihood must
    return a chi2 near the number of data points.

    This is the same class of check that caught the compressed
    distance priors returning chi2 ~ 100 for 3 points -- a
    likelihood that cannot reproduce the cosmology its own data
    came from is broken, whatever else it does.
    """

    from CosmoFit import LCDM, CosmologyParameters
    from CosmoFit.likelihoods.planck_lite import PlanckLiteLikelihood

    model = LCDM(

        CosmologyParameters(

            H0=67.36,

            Omega_m=0.3153,

            Omega_b=0.02237 / 0.6736 ** 2,

            ln1e10As=3.045,

            n_s=0.9649,

            tau_reio=0.0544,

        ),

    )

    likelihood = PlanckLiteLikelihood(model)

    chi2 = likelihood.chi2()

    # 613 bandpowers; Planck's published best fit sits near 586.
    assert 500 < chi2 < 700, chi2


@requires_camb
def test_modified_gravity_models_are_refused():
    """
    A model whose whole content is that gravity is not GR must not
    be quietly handed to a solver that assumes GR.
    """

    from CosmoFit import FRHuSawicki, FQExponential
    from CosmoFit.cosmology.boltzmann import CAMBBackend, BoltzmannError

    for model_cls in (FRHuSawicki, FQExponential):

        model = model_cls(

            model_cls.PARAMS_CLASS(H0=67.4, Omega_m=0.315),

        )

        with pytest.raises(BoltzmannError):

            CAMBBackend(model)
