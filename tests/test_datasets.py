"""
Checks on every bundled dataset and its likelihood.

Datasets fail in a particular way: a column read in the wrong
order, a covariance that does not match its data vector, a
measurement transcribed with a typo. None of those raise; they
produce a chi2 that is merely wrong, and a fit that is merely
biased.

So these tests check the two things that catch that class of bug:
that every dataset's own internal shapes agree, and that every
likelihood returns a *reasonable* chi2 at a fiducial cosmology
close to the concordance model. A dataset whose chi2 per data point
is 5 at Planck's best fit has something wrong with it, whatever the
code does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cosmofit import LCDM, CosmologyParameters
from cosmofit.data.loader import available_datasets, available_versions
from cosmofit.stats.fitter import DATASET_REGISTRY, CONFLICTING_DATASETS


#: A concordance cosmology. Every likelihood here should be
#: comfortable at this point -- that is what makes the chi2 bounds
#: below meaningful rather than arbitrary.
FIDUCIAL = dict(
    H0=67.8,
    Omega_m=0.310,
    Omega_b=0.0490,
    rd=147.2,
    sigma8=0.811,
    ln1e10As=3.045,
    n_s=0.9649,
    tau_reio=0.0544,
)


#: Datasets whose likelihood needs an optional dependency.
NEEDS_CAMB = {"planck_lite", "planck_lensing", "planck_lowe", "act_lensing"}


def _has_camb() -> bool:

    try:

        import camb  # noqa: F401

    except ImportError:

        return False

    return True


@pytest.fixture(scope="module")
def cosmology():

    return LCDM(CosmologyParameters(**FIDUCIAL))


# ============================================================
# The registry is self-consistent
# ============================================================

def test_every_registered_dataset_is_loadable():
    """
    Every entry in the loader's registry must actually load --
    catching a registry entry pointing at a file that was never
    added to the package.
    """

    catalogue = available_datasets()

    assert catalogue, "no datasets registered"

    for name, versions in catalogue.items():

        assert versions, f"{name} has no versions"

        assert available_versions(name) == versions


def test_every_dataset_has_a_likelihood():
    """
    The loader registry and the fitter registry must agree on what
    exists -- otherwise a dataset ships that no fit can use, or a
    fit names a dataset that cannot load.
    """

    loadable = set(available_datasets())

    fittable = set(DATASET_REGISTRY)

    assert loadable == fittable, (

        f"only loadable: {sorted(loadable - fittable)}; "

        f"only fittable: {sorted(fittable - loadable)}"

    )


def test_conflict_rules_name_real_datasets():

    for first, second in CONFLICTING_DATASETS:

        assert first in DATASET_REGISTRY
        assert second in DATASET_REGISTRY


# ============================================================
# Every likelihood is sane at a concordance cosmology
# ============================================================

#: ``(dataset, max chi2 per data point)``. The bounds are loose on
#: purpose -- this catches an order-of-magnitude error (a unit
#: mix-up, an inverted ratio, a covariance that does not correspond
#: to its data), not a mild preference of one dataset for a
#: slightly different cosmology.
CHI2_BOUNDS = {
    "cc": 2.0,
    "desi": 2.0,
    "sdss_bao": 2.0,
    "sdss_fsbao": 2.0,
    "eboss_elg": 2.0,
    # -2 log L from a 100x100x100 grid, three quantities. The
    # released grid is normalized to a maximum probability of 1,
    # so this does reach ~0 at the best-fitting point, unlike
    # "planck_lowe" below.
    "eboss_elg_fs": 2.0,
    # eBOSS DR16's Lyman-alpha BAO sits ~2 sigma from a Planck-like
    # LCDM -- a real and much-discussed feature of that measurement,
    # not a defect here, and chi2/N ~ 2.4 is what it looks like.
    # Bounded just above it so a genuine error is still caught. Note
    # this is -2 log L from a table, so it does not reach zero at a
    # perfect fit either (see "planck_lowe" below).
    "eboss_lya": 4.0,
    "bao_lowz": 3.0,
    "pantheon": 2.0,
    "des_sn5yr": 2.0,
    "union3": 2.0,
    "planck": 3.0,
    "planck_lite": 2.0,
    "planck_lensing": 2.0,
    "act_lensing": 2.5,
    # planck_lowe reports -2 log L from a tabulated probability, not
    # a sum of squared pulls -- it does not go to zero at a perfect
    # fit, so its "chi2 per point" is ~14 by construction. Bounded
    # only to catch an order-of-magnitude indexing error.
    "planck_lowe": 20.0,
    "fsigma8": 2.0,
    # KiDS-1000's S8 = 0.759 sits ~2.9 sigma below what the
    # Planck-like fiducial above implies (S8 = 0.824), so chi2 ~ 8.5
    # for one data point is the S8 tension, not a defect. Bounded
    # just above it, so a genuine error would still be caught.
    "s8": 12.0,
    "h0": 30.0,
    "omega_b": 6.0,
    "tau": 3.0,
}


@pytest.mark.parametrize("name", sorted(DATASET_REGISTRY))
def test_chi2_is_reasonable_at_concordance(name, cosmology):
    """
    Every dataset's chi2 per data point must be O(1) at a
    concordance cosmology.

    Two entries are deliberately allowed to be large:

    - ``"h0"``, because the SH0ES measurement genuinely disagrees
      with a Planck-like ``H0 = 67.8`` at ~5 sigma. That is the
      Hubble tension, not a bug, and a bound tight enough to
      "pass" here would be a bound that hid it. The cap is set
      just above what the real tension gives, so a *further*
      order-of-magnitude error would still be caught.

    - ``"s8"``, for the same reason at smaller scale: KiDS-1000
      measures ``S8 = 0.759`` where the fiducial implies 0.824,
      a ~2.9 sigma gap and a chi2 of ~8.5 for a single point.
    """

    if name in NEEDS_CAMB and not _has_camb():

        pytest.skip("CAMB not installed (optional 'cmb' extra)")

    likelihood = DATASET_REGISTRY[name](cosmology)

    chi2 = likelihood.chi2()

    assert np.isfinite(chi2), f"{name}: chi2 is not finite"

    assert chi2 >= 0.0, f"{name}: negative chi2"

    per_point = chi2 / likelihood.n_data

    assert per_point < CHI2_BOUNDS[name], (

        f"{name}: chi2/N = {per_point:.2f} "

        f"({chi2:.2f} over {likelihood.n_data} points) at a "

        f"concordance cosmology -- too large to be a preference, "

        f"likely a data or convention error"

    )


@pytest.mark.parametrize("name", sorted(DATASET_REGISTRY))
def test_covariance_matches_data_length(name, cosmology):
    """
    The residual vector and the covariance must have the same
    dimension, checked through the likelihood rather than by
    reading the files -- so a loader that silently truncated
    something is caught.
    """

    if name in NEEDS_CAMB and not _has_camb():

        pytest.skip("CAMB not installed (optional 'cmb' extra)")

    likelihood = DATASET_REGISTRY[name](cosmology)

    residual = np.atleast_1d(likelihood.residuals())

    assert len(residual) == likelihood.n_data

    if likelihood.covariance is None:

        # A non-Gaussian likelihood -- Planck's low-l EE is a
        # tabulated probability with no covariance at all. Its
        # `residuals()` is a diagnostic against the table's most
        # probable value, not the quantity its chi2 is built from,
        # so there is nothing further to check here.
        return

    assert likelihood.covariance.shape == (

        likelihood.n_data,

        likelihood.n_data,

    )


# ============================================================
# The new datasets, specifically
# ============================================================

def test_desi_dr2_differs_from_dr1_but_has_the_same_structure():
    """
    DR2 is a re-measurement of the same observables, not a new kind
    of data -- so it must load through the identical code path,
    with the same observable types, and different numbers.
    """

    from cosmofit.data.loader import load_desi

    dr1 = load_desi("desi2024")
    dr2 = load_desi("desi2025")

    # DR1 ships 12 entries, DR2 ships 13: the QSO tracer at
    # z ~ 1.49 was only good enough for a single D_V in DR1 and is
    # split into D_M and D_H in DR2. So the *shapes* legitimately
    # differ -- what must not differ is the vocabulary.
    assert dr1.size == 12
    assert dr2.size == 13

    assert set(dr2.observable) <= set(dr1.observable)

    assert dr1.covariance.shape == (12, 12)
    assert dr2.covariance.shape == (13, 13)

    # The bins they do share must have moved -- identical numbers
    # would mean the loader is reading one file for both versions.
    shared = dr1.observable == "DM_over_rs"

    assert not np.allclose(

        dr1.value[shared][:3],

        dr2.value[dr2.observable == "DM_over_rs"][:3],

    )


def test_sixdfgs_rescale_is_applied():
    """
    6dFGS's ``rs_rescale`` must actually reach the prediction.

    Without it the model's ``r_d/D_V`` at z = 0.106 is ~2.7% low,
    which is most of a sigma on a 4.5% measurement. The check is
    that the two predictions differ by exactly the rescale factor.
    """

    from cosmofit.likelihoods.bao_lowz import BAOLowZLikelihood

    model = LCDM(CosmologyParameters(**FIDUCIAL))

    likelihood = BAOLowZLikelihood(model)

    rescale = likelihood.data.rs_rescale

    assert rescale is not None

    # 6dFGS carries the rescale; the MGS point does not.
    np.testing.assert_allclose(

        rescale,

        [153.9 / 149.8, 1.0],

    )

    prediction = likelihood.model()

    # The 6dFGS entry is r_d/D_V, so the rescale multiplies it.
    unscaled = model.rd / model.distance.DV(0.106)

    assert prediction[0] == pytest.approx(

        unscaled * rescale[0],

        rel=1e-12,

    )


def test_union3_is_binned_and_offset_marginalized():

    from cosmofit.likelihoods.union3 import Union3Likelihood

    model = LCDM(CosmologyParameters(**FIDUCIAL))

    likelihood = Union3Likelihood(model)

    assert likelihood.n_data == 22

    # The analytic offset absorbs the H0 degeneracy, so shifting H0
    # must leave the chi2 essentially unchanged -- the defining
    # property of a marginalized zero point, and the thing that
    # breaks if the marginalization is wired up wrong.
    chi2_a = likelihood.chi2()

    model.params.H0 = 73.0
    model.refresh()

    chi2_b = likelihood.chi2()

    assert chi2_b == pytest.approx(chi2_a, rel=1e-6)


@pytest.mark.parametrize(
    "dataset,version,value,sigma",
    [
        ("h0", "sh0es2022", 73.04, 1.04),
        ("omega_b", "bbn2024", 0.02218, 0.00055),
        ("tau", "planck2018", 0.0544, 0.0073),
        # 1.22 (stat), 1.33 (sys) and 0.70 (SN) in quadrature.
        ("h0", "cchp2024", 70.39, 1.94),
    ],
)
def test_prior_values_are_as_published(dataset, version, value, sigma):
    """
    These are single numbers typed into a file from a paper, which
    is the easiest thing in the whole library to get wrong and the
    hardest to notice. Pin them.
    """

    from cosmofit.data.loader import load_gaussian_prior

    data = load_gaussian_prior(dataset, version)

    assert data.value == pytest.approx(value)
    assert data.sigma == pytest.approx(sigma)


@pytest.mark.parametrize(
    "version,value,sigma",
    [
        ("kids1000", 0.759, 0.0225),
        ("des_y3", 0.776, 0.017),
        # Asymmetric errors, symmetrized to the mean of the two sides.
        ("kids_legacy", 0.815, 0.0185),
        ("hsc_y3", 0.769, 0.0325),
        ("desy3_kids1000", 0.790, 0.016),
    ],
)
def test_s8_values_are_as_published(version, value, sigma):

    from cosmofit.data.loader import load_s8

    data = load_s8(version)

    assert data.value == pytest.approx(value)
    assert data.sigma == pytest.approx(sigma)


def test_prior_likelihood_recovers_its_own_measurement():
    """
    A Gaussian prior evaluated at exactly its own central value
    must give chi2 = 0 -- which checks that the quantity mapping
    (``omega_b h^2``, not ``Omega_b``) is right.
    """

    from cosmofit.likelihoods.priors import OmegaBLikelihood

    target = 0.02218

    h = 0.678

    model = LCDM(

        CosmologyParameters(

            H0=100.0 * h,

            Omega_m=0.31,

            Omega_b=target / h ** 2,

        ),

    )

    assert OmegaBLikelihood(model).chi2() == pytest.approx(0.0, abs=1e-12)


# ============================================================
# Overlapping datasets warn
# ============================================================

@pytest.mark.parametrize(
    "pair",
    sorted(CONFLICTING_DATASETS),
    ids=[f"{a}+{b}" for a, b in sorted(CONFLICTING_DATASETS)],
)
def test_conflicting_datasets_warn(pair):
    """
    Combining two non-independent datasets must say so. The failure
    it guards against -- double-counted data, understated error
    bars -- has no other symptom.
    """

    from cosmofit import Fitter

    if set(pair) & NEEDS_CAMB and not _has_camb():

        pytest.skip("CAMB not installed (optional 'cmb' extra)")

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(

            model=LCDM,

            datasets=list(pair),

            free_params=["H0", "Omega_m"],

            initial=dict(FIDUCIAL),

        )

    messages = [str(w.message) for w in caught]

    assert any(

        pair[0] in message and pair[1] in message

        for message in messages

    ), messages


# ============================================================
# Cosmic chronometers: the systematics correlate Moresco's measurements
# ============================================================

# The 15 measurements of Moresco et al. (2012, 2015, 2016): the rows of
# HzTable_MM_BC03.dat in gitlab.com/mmoresco/CCcovariance.
CC_MORESCO_REDSHIFTS = (
    0.1791, 0.1993, 0.3519, 0.3802, 0.4004, 0.4247, 0.4497, 0.4783,
    0.5929, 0.6797, 0.7812, 0.8754, 1.037, 1.363, 1.965,
)


def _cc_favale_covariance(data):
    """
    The covariance as the script of Favale et al. (2023) builds it,
    written the way that script is: element by element for the Moresco
    points, then placed in the full matrix, whose remaining diagonal is
    the tabulated error squared.

    The script's ``C_diag`` for the Moresco points holds their
    statistical errors, which with the systematic give the tabulated
    one; here it is that difference, so the sum's diagonal is
    ``sigma^2`` by construction.
    """

    from cosmofit.data.loader import DATA_DIR

    zmod, imf, _, _, spsooo = np.loadtxt(
        DATA_DIR / "cc" / "favale2023" / "data_MM20.dat", unpack=True,
    )

    index = [int(np.argmin(np.abs(data.z - z))) for z in CC_MORESCO_REDSHIFTS]

    z, H = data.z[index], data.H[index]

    imf_intp = np.interp(z, zmod, imf) / 100
    spsooo_intp = np.interp(z, zmod, spsooo) / 100

    cov_tot = np.diag(data.sigma ** 2)

    for i in range(len(z)):
        for j in range(len(z)):

            if i == j:
                continue

            cov_tot[index[i], index[j]] = (
                H[i] * imf_intp[i] * H[j] * imf_intp[j]
                + H[i] * spsooo_intp[i] * H[j] * spsooo_intp[j]
            )

    return cov_tot


def test_cc_diagonal_is_the_tabulated_errors():
    """
    The tabulated errors are the total ones: for the Moresco points they
    are the statistical and the IMF and SPS errors in quadrature, so
    what is left of 6.2 at z = 0.1791 once the systematic is taken out
    is the statistical 4.5 -- the mean of the 4 and 5 of Moresco et al.
    (2012)'s two stellar-population models.
    """

    from cosmofit.data.loader import DATA_DIR, load_cc

    data = load_cc()

    np.testing.assert_allclose(
        np.diag(data.covariance.matrix), data.sigma ** 2, rtol=1e-14,
    )

    table = np.loadtxt(DATA_DIR / "cc" / "favale2023" / "data_MM20.dat")

    i = int(np.argmin(np.abs(data.z - 0.1791)))

    systematic = data.H[i] * np.hypot(
        np.interp(data.z[i], table[:, 0], table[:, 1]),
        np.interp(data.z[i], table[:, 0], table[:, 4]),
    ) / 100.0

    assert np.sqrt(data.sigma[i] ** 2 - systematic ** 2) == pytest.approx(4.5, abs=0.05)


@pytest.mark.parametrize("version", ["favale2023", "favale2023_extended"])
def test_cc_covariance_is_the_one_of_favale(version):
    """
    The covariance correlated all 32 measurements by the Moresco et al.
    (2020) budget. The authors' own script correlates the 15 Moresco
    ones and leaves the other 17 on the diagonal: their errors are
    their own, from other analyses.
    """

    from cosmofit.data.loader import load_cc

    data = load_cc(version)

    covariance = data.covariance.matrix

    np.testing.assert_allclose(
        covariance, _cc_favale_covariance(data), rtol=1e-12, atol=0,
    )

    moresco = np.isin(data.z, CC_MORESCO_REDSHIFTS)

    assert moresco.sum() == 15

    off = ~np.eye(len(data.z), dtype=bool)

    assert np.all(covariance[np.ix_(moresco, moresco)] > 0)
    assert not np.any(covariance[~moresco][off[~moresco]])
    assert np.count_nonzero(covariance[off]) == 15 * 14

    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_cc_h0_error_with_correlated_systematics():
    """
    Flat LCDM to CC alone, Fisher error on H0 at the best fit: 3.70 with
    the errors alone, 4.24 with the Moresco points correlated (and 5.61
    when all 32 were, which no source supports).
    """

    from scipy.optimize import minimize

    from cosmofit.data.loader import load_cc

    data = load_cc()

    inverse = np.linalg.inv(data.covariance.matrix)

    def model(p):

        return p[0] * np.sqrt(p[1] * (1.0 + data.z) ** 3 + 1.0 - p[1])

    def chi2(p):

        r = data.H - model(p)

        return r @ inverse @ r

    best = minimize(chi2, [70.0, 0.3], method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-10}).x

    jacobian = np.empty((len(data.z), 2))

    for k, step in enumerate((1e-3, 1e-5)):

        dp = np.zeros(2)
        dp[k] = step

        jacobian[:, k] = (model(best + dp) - model(best - dp)) / (2 * step)

    error = np.sqrt(np.linalg.inv(jacobian.T @ inverse @ jacobian)[0, 0])

    assert best[0] == pytest.approx(71.9, abs=0.1)
    assert error == pytest.approx(4.24, abs=0.02)


def test_cc_extended_is_the_32_and_two_more():
    """
    ``favale2023_extended`` adds Jiao et al. (2023) and Tomasetti et al.
    (2023) and changes nothing about the 32: their rows and columns of
    the covariance are those of ``favale2023``, and the two new points
    are on the diagonal alone, like every other that is not Moresco's.
    """

    from cosmofit.data.loader import load_cc

    base = load_cc("favale2023")
    extended = load_cc("favale2023_extended")

    old = np.isin(extended.z, base.z)

    assert old.sum() == 32 and len(extended.z) == 34

    np.testing.assert_array_equal(extended.z[old], base.z)
    np.testing.assert_array_equal(extended.H[old], base.H)

    full = extended.covariance.matrix
    np.testing.assert_allclose(full[np.ix_(old, old)], base.covariance.matrix)

    new = {round(z, 2): (H, s) for z, H, s in zip(extended.z, extended.H, extended.sigma) if z not in base.z}

    # Jiao: 15.1 (stat) and the systematic +29.1/-11.3 symmetrized, in quadrature.
    assert new == {0.8: (113.1, 25.2), 1.26: (135.0, 65.0)}

    assert not np.any(full[~old][:, old])
