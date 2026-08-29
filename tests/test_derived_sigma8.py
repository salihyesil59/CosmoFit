"""
sigma8 as a derived quantity rather than a sampled one.

The problem this closes: ``sigma8`` is the free parameter the growth
machinery normalizes with, and a CAMB-computed CMB likelihood fixes
an amplitude of its own through ``ln1e10As``. Both in one fit and
nothing makes them agree -- the sampler settles one on the growth
data and the other on the CMB, and reports a posterior for each. Two
numbers, one physical quantity, no error anywhere.

What makes it worth testing rather than just documenting is the
consequence. With ``sigma8`` free, the S8 measurement is *absorbed*:
the parameter slides to whatever the lensing survey wants and the
chi2 goes to zero, which looks like agreement. Derived, the CMB's
own prediction meets the measurement and the tension appears. Those
two behaviours are what these tests pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoFit import LCDM, CosmologyParameters


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


# ============================================================
# The property itself
# ============================================================

def test_free_by_default():

    model = LCDM(CosmologyParameters(sigma8=0.60, **PLANCK_BEST_FIT))

    assert model.derive_sigma8 is False
    assert model.sigma8 == 0.60


def test_deriving_without_a_backend_says_why():
    """
    There is nothing to derive from unless a CMB likelihood has
    attached a Boltzmann backend. Silently falling back to the free
    parameter would be the worst outcome -- the flag would appear to
    work and change nothing.
    """

    model = LCDM(CosmologyParameters(sigma8=0.60, **PLANCK_BEST_FIT))

    model.derive_sigma8 = True

    with pytest.raises(ValueError, match="no Boltzmann backend"):

        model.sigma8  # noqa: B018  (the access is the assertion)


@requires_camb
def test_derived_value_matches_planck():

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    model = LCDM(CosmologyParameters(sigma8=0.60, **PLANCK_BEST_FIT))

    PlanckLensingLikelihood(model)

    model.derive_sigma8 = True

    assert model.sigma8 == pytest.approx(0.8111, abs=0.002)

    # ...and the free parameter is genuinely ignored, not merely
    # coincidentally close.
    model.params.sigma8 = 0.40

    assert model.sigma8 == pytest.approx(0.8111, abs=0.002)


@requires_camb
def test_the_growth_evolution_stays_the_models_own():
    """
    Only the z = 0 normalization changes hands. The redshift
    dependence still comes from the model's own growth ODE, which is
    where ``mu(a, k)`` lives -- so switching the source must rescale
    ``fsigma8(z)`` by a constant, not reshape it.
    """

    from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood

    model = LCDM(CosmologyParameters(sigma8=0.60, **PLANCK_BEST_FIT))

    PlanckLensingLikelihood(model)

    z = np.array([0.0, 0.3, 0.7, 1.2])

    free = model.growth.fsigma8(z)

    model.derive_sigma8 = True

    derived = model.growth.fsigma8(z)

    ratio = derived / free

    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)

    assert ratio[0] == pytest.approx(model.sigma8 / 0.60, rel=1e-10)


# ============================================================
# What it does to a fit
# ============================================================

@requires_camb
def test_free_sigma8_absorbs_the_s8_measurement():
    """
    The behaviour that makes this worth fixing.

    With ``sigma8`` free, nothing in a growth-only fit stops it
    sliding onto the lensing measurement -- the S8 chi2 goes to
    essentially zero and the fit reports agreement it did not
    predict.
    """

    from CosmoFit import Fitter

    fit = Fitter(

        model=LCDM,

        datasets=["s8"],

        free_params=["Omega_m", "sigma8"],

        initial={**PLANCK_BEST_FIT, "sigma8": 0.811},

    )

    fit.best_fit()

    assert fit.best_fit_chi2 < 0.01


@requires_camb
def test_derived_sigma8_exposes_the_s8_tension():
    """
    Derived, the CMB's prediction meets the lensing measurement and
    the tension is visible. KiDS-1000 measures S8 = 0.759 +- 0.023
    while Planck's LCDM implies ~0.83, so the chi2 must land near
    the square of that ~3 sigma gap.
    """

    from CosmoFit import Fitter

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        fit = Fitter(

            model=LCDM,

            datasets=["planck_lensing", "s8"],

            free_params=["H0", "Omega_m", "ln1e10As"],

            initial=PLANCK_BEST_FIT,

            derive_sigma8=True,

        )

    s8_likelihood = next(

        lk for lk in fit.likelihoods if lk.name == "S8"

    )

    predicted = float(np.atleast_1d(s8_likelihood.model())[0])

    assert predicted == pytest.approx(0.83, abs=0.03), predicted

    assert 4.0 < s8_likelihood.chi2() < 20.0, s8_likelihood.chi2()


# ============================================================
# Guards and plumbing
# ============================================================

@requires_camb
def test_refuses_sigma8_as_a_free_parameter():

    from CosmoFit import Fitter

    with pytest.raises(ValueError, match="cannot also be a free parameter"):

        Fitter(

            model=LCDM,

            datasets=["planck_lensing", "s8"],

            free_params=["H0", "sigma8"],

            initial=PLANCK_BEST_FIT,

            derive_sigma8=True,

        )


def test_refuses_without_a_from_scratch_cmb_dataset():

    from CosmoFit import Fitter

    with pytest.raises(ValueError, match="needs a CMB likelihood"):

        Fitter(

            model=LCDM,

            datasets=["fsigma8", "s8"],

            free_params=["H0", "Omega_m"],

            initial=PLANCK_BEST_FIT,

            derive_sigma8=True,

        )


@requires_camb
def test_the_warning_names_the_fix():
    """
    The double-amplitude warning existed before the switch did, and
    said only that the combination was wrong. It must now say what
    to do about it, or it sends people to fix it by hand.
    """

    from CosmoFit import Fitter

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(

            model=LCDM,

            datasets=["planck_lensing", "s8"],

            free_params=["H0", "Omega_m", "sigma8"],

            initial={**PLANCK_BEST_FIT, "sigma8": 0.811},

        )

    messages = " ".join(str(w.message) for w in caught)

    assert "derive_sigma8=True" in messages


@requires_camb
def test_is_part_of_the_chain_signature():
    """
    Two fits differing only in where sigma8 comes from sample
    different posteriors, so one's chain must not be resumable as
    the other's.
    """

    from CosmoFit import Fitter
    from CosmoFit.stats.chains import compare_signatures

    common = dict(

        model=LCDM,

        datasets=["planck_lensing", "s8"],

        free_params=["H0", "Omega_m", "ln1e10As"],

        initial=PLANCK_BEST_FIT,

    )

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        free = Fitter(**common)
        derived = Fitter(**common, derive_sigma8=True)

    assert free.chain_id() != derived.chain_id()

    differences = compare_signatures(

        free._chain_signature(), derived._chain_signature(),

    )

    assert any("derive_sigma8" in d for d in differences), differences
