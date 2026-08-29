"""
Validation of Planck's low-multipole EE likelihood.

This is the one likelihood in the library that is not Gaussian, and
the checks reflect that: there is no residual vector to compare and
no covariance to invert, only a lookup into a released probability
table. What can go wrong is indexing -- the wrong axis, the wrong
step, an off-by-two in the multipole offset -- and every one of those
produces a finite, plausible-looking log-likelihood.

So the decisive test is not that a number comes out, but that the
*published constraint* comes back: profiling ``tau`` with the
primordial amplitude re-optimized must reproduce Planck's
``tau = 0.0544 +- 0.0073``. Nothing about that number is used as an
input anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM, CosmologyParameters
from CosmoFit.data.loader import load_planck_lowe


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
def dataset():

    return load_planck_lowe()


@pytest.fixture(scope="module")
def likelihood():

    if not _has_camb():
        pytest.skip("CAMB not installed")

    from CosmoFit.likelihoods.planck_lowe import PlanckLowEELikelihood

    return PlanckLowEELikelihood(

        LCDM(CosmologyParameters(**PLANCK_BEST_FIT)),

    )


# ============================================================
# The table
# ============================================================

def test_table_shape_and_range(dataset):

    assert dataset.table.shape == (3000, 28)

    assert dataset.lmin == 2
    assert dataset.lmax == 29
    assert dataset.size == 28

    # Log-probabilities: negative everywhere, and finite.
    assert np.all(dataset.table <= 0.0)
    assert np.all(np.isfinite(dataset.table))


def test_table_peaks_at_plausible_power(dataset):
    """
    Each multipole's most probable ``D_l^EE`` must be small and
    positive -- the reionization bump is ~0.04 muK^2 -- which
    catches a table read transposed or with the wrong step.
    """

    peaks = dataset.table.argmax(axis=0) * dataset.step

    assert np.all(peaks >= 0.0)
    assert np.all(peaks < 0.15)


# ============================================================
# The lookup
# ============================================================

@requires_camb
def test_log_likelihood_is_a_sum_over_the_table(likelihood, dataset):
    """
    Recompute the lookup independently and compare, so the test is
    not the implementation calling itself.
    """

    predicted = likelihood.model()

    index = (predicted / dataset.step).astype(int)

    expected = sum(

        dataset.table[index[j], j] for j in range(dataset.size)

    )

    assert likelihood.log_likelihood() == pytest.approx(expected)

    assert likelihood.chi2() == pytest.approx(-2.0 * expected)


@requires_camb
def test_out_of_table_predictions_are_rejected_not_clamped(likelihood):
    """
    A cosmology predicting more EE power than the table covers is
    one the data excludes. It must return ``-inf`` rather than
    silently using the last row, which would put a finite likelihood
    on an arbitrarily bad model.
    """

    model = likelihood.cosmology

    model.params.ln1e10As = 3.60
    model.params.tau_reio = 0.19
    model.refresh()

    assert likelihood.log_likelihood() == -np.inf
    assert likelihood.chi2() == np.inf

    model.params.ln1e10As = PLANCK_BEST_FIT["ln1e10As"]
    model.params.tau_reio = PLANCK_BEST_FIT["tau_reio"]
    model.refresh()

    assert np.isfinite(likelihood.log_likelihood())


@requires_camb
def test_it_has_no_covariance(likelihood):
    """
    The base class must not have invented one. If it had, something
    downstream would eventually use it and get a Gaussian answer to
    a non-Gaussian question.
    """

    assert likelihood.covariance is None


# ============================================================
# The published constraint
# ============================================================

@requires_camb
def test_profiling_tau_recovers_the_published_constraint():
    r"""
    The decisive check.

    Scan ``tau``; at each value re-optimize ``ln1e10As`` against
    low-l EE *and* the high-l bandpowers together, since the two
    are degenerate through ``A_s e^{-2 tau}`` and neither pins
    ``tau`` alone. The resulting profile must reproduce Planck's
    published ``tau = 0.0544 +- 0.0073``.

    Nothing here uses that number as an input: it comes out of the
    released probability table, the released bandpowers, and CAMB.
    """

    from scipy.optimize import minimize_scalar

    from CosmoFit.likelihoods.planck_lite import PlanckLiteLikelihood
    from CosmoFit.likelihoods.planck_lowe import PlanckLowEELikelihood

    model = LCDM(CosmologyParameters(**PLANCK_BEST_FIT))

    low_ee = PlanckLowEELikelihood(model)
    high_l = PlanckLiteLikelihood(model, use_low_ell=True)

    def profiled_chi2(tau):

        model.params.tau_reio = float(tau)

        def at_amplitude(ln_amplitude):

            model.params.ln1e10As = float(ln_amplitude)
            model.refresh()

            return high_l.chi2() + low_ee.chi2()

        return minimize_scalar(

            at_amplitude,

            bounds=(2.9, 3.2),

            method="bounded",

            options={"xatol": 5e-4},

        )

    taus = np.array([0.040, 0.046, 0.052, 0.058, 0.064])

    results = [profiled_chi2(t) for t in taus]

    chi2 = np.array([r.fun for r in results])
    amplitude = np.array([r.x for r in results])

    chi2 -= chi2.min()

    quadratic = np.polyfit(taus, chi2, 2)

    centre = -quadratic[1] / (2.0 * quadratic[0])
    sigma = 1.0 / np.sqrt(quadratic[0])

    assert centre == pytest.approx(0.0544, abs=0.004), centre
    assert sigma == pytest.approx(0.0073, rel=0.25), sigma

    # The amplitude must track tau along the A_s exp(-2 tau)
    # degeneracy: d ln(10^10 A_s) / d tau = 2. A slope near 2 is a
    # strong sign the two likelihoods are being combined coherently
    # rather than one dominating.
    slope = np.polyfit(taus, amplitude, 1)[0]

    assert slope == pytest.approx(2.0, rel=0.05), slope


# ============================================================
# Wiring
# ============================================================

@requires_camb
def test_fitter_refuses_lowe_together_with_the_tau_prior():
    """
    The Gaussian ``"tau"`` dataset is a compression of exactly this
    likelihood. Using both is Planck's low-l polarization twice.
    """

    import warnings

    from CosmoFit import Fitter

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(

            model=LCDM,

            datasets=["planck_lowe", "tau"],

            free_params=["H0", "Omega_m"],

            initial=PLANCK_BEST_FIT,

        )

    assert any(

        "planck_lowe" in str(w.message) and "tau" in str(w.message)

        for w in caught

    ), [str(w.message) for w in caught]


@requires_camb
def test_shares_one_camb_backend_with_the_bandpower_likelihood():
    """
    Three CMB likelihoods on one cosmology must still mean one CAMB
    call per step.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood
    from CosmoFit.likelihoods.planck_lite import PlanckLiteLikelihood
    from CosmoFit.likelihoods.planck_lowe import PlanckLowEELikelihood

    model = LCDM(CosmologyParameters(**PLANCK_BEST_FIT))

    a = PlanckLiteLikelihood(model)
    b = PlanckLensingLikelihood(model)
    c = PlanckLowEELikelihood(model)

    assert a.backend is b.backend is c.backend

    # ...and widened to satisfy the most demanding of them.
    assert a.backend.lmax >= 2508
    assert a.backend.lens_potential_accuracy >= 4
