"""
eBOSS DR16's tabulated BAO likelihoods, against their papers.

These two measurements are not distributed as a mean and a
covariance, and the reason is the thing worth testing. Both are
released as likelihood *surfaces* because a Gaussian would
misrepresent them, so a test that only checked "the code runs and
chi2 is finite" would pass equally well on an implementation that
threw the shape away.

So the surfaces are checked against the published constraints
instead, and in each case the published number is used nowhere as an
input:

  ELG          D_V/r_d = 18.33 (+0.57/-0.62)   arXiv:2007.09008
  Lyman-alpha  D_M/r_d = 37.5 +- 1.1           arXiv:2007.08995
               D_H/r_d = 8.99 +- 0.19

The Lyman-alpha check does double duty. eBOSS quote that combined
constraint from a joint fit to the forest auto-correlation and its
cross-correlation with quasars, but ship only the two surfaces
separately -- so this library multiplies them, which assumes they
are independent. Recovering the published errors from the product is
what makes that assumption defensible. Had the neglected correlation
mattered, the recovered errors would have come out too tight, which
is the direction that silently overstates a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM
from CosmoFit.data.loader import load_eboss_table
from CosmoFit.likelihoods.eboss_dr16 import (
    EBOSSELGLikelihood,
    EBOSSLyaLikelihood,
)


PLANCK = dict(H0=67.36, Omega_m=0.3153, Omega_b=0.04930, rd=147.09)


@pytest.fixture(scope="module")
def cosmology():

    return LCDM(LCDM.PARAMS_CLASS(**PLANCK))


@pytest.fixture(scope="module")
def elg(cosmology):

    return EBOSSELGLikelihood(cosmology)


@pytest.fixture(scope="module")
def lya(cosmology):

    return EBOSSLyaLikelihood(cosmology)


# ============================================================
# Helpers -- these read the likelihood's own interpolator, so
# what they measure is the object a fit would use.
# ============================================================

def scan_1d(likelihood, n=20000):
    """Fine scan of the surface over its full support."""

    low, high = likelihood.data.bounds[0]

    x = np.linspace(low, high, n)

    return x, likelihood.log_likelihood_at(x[:, None])


def scan_2d(likelihood, n=1200):
    """Fine scan of a two-dimensional surface, as a probability."""

    (xlo, xhi), (ylo, yhi) = likelihood.data.bounds

    x = np.linspace(xlo, xhi, n)
    y = np.linspace(ylo, yhi, n)

    points = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)

    probability = np.exp(likelihood.log_likelihood_at(points))

    return x, y, probability / probability.sum()


def moments(axis, weights):
    """Mean and standard deviation of a normalized marginal."""

    weights = weights / weights.sum()

    mean = float((weights * axis).sum())

    return mean, float(np.sqrt((weights * (axis - mean) ** 2).sum()))


# ============================================================
# ELG: the published asymmetric error bar
# ============================================================

def test_elg_peak_is_the_published_value(elg):

    x, log_like = scan_1d(elg)

    assert x[log_like.argmax()] == pytest.approx(18.33, abs=0.01)


def test_elg_reproduces_the_published_asymmetric_errors(elg):
    """
    de Mattia et al. quote ``18.33 (+0.57/-0.62)``, and those are
    the ``delta chi2 = 1`` crossings of this surface -- not a
    standard deviation, which for this table is 1.05, and not a 68%
    credible interval, which is +0.52/-1.21. Recovering the
    published pair to two decimals says the grid is being read on
    its own terms.
    """

    x, log_like = scan_1d(elg)

    chi2 = -2.0 * (log_like - log_like.max())

    peak = x[chi2.argmin()]

    below = x < peak
    above = x > peak

    low = np.interp(1.0, chi2[below][::-1], x[below][::-1])
    high = np.interp(1.0, chi2[above], x[above])

    assert peak - low == pytest.approx(0.62, abs=0.02)
    assert high - peak == pytest.approx(0.57, abs=0.02)


def test_elg_is_not_the_gaussian_it_would_be_summarized_as(elg):
    """
    The justification for carrying a 399-point table instead of two
    numbers.

    eBOSS detect the ELG BAO feature at only 1.4 sigma, so the
    likelihood keeps a long low-``D_V`` shoulder. Roughly a tenth of
    the probability sits below ``D_V/r_d = 16.5``; a Gaussian at
    ``18.33 +- 0.6`` puts about a thousandth there. That is two
    orders of magnitude, in the tail that decides whether a model
    with a low expansion rate at ``z ~ 0.85`` is excluded.
    """

    from scipy.stats import norm

    x, log_like = scan_1d(elg)

    probability = np.exp(log_like)
    probability /= probability.sum()

    tabulated = probability[x < 16.5].sum()

    gaussian = norm.cdf(16.5, 18.33, 0.6)

    assert tabulated == pytest.approx(0.106, abs=0.01)

    assert tabulated / gaussian > 50.0


def test_elg_table_stops_while_the_likelihood_is_still_alive(elg):
    """
    Truncating this table is a real prior, and the dataset says so
    rather than leaving it to be discovered.

    The surface reaches the low edge of its support at
    ``delta chi2 ~ 3.3`` -- inside 2 sigma. The high edge has
    decayed and costs nothing.
    """

    (low_edge, high_edge), = elg.data.edge_delta_chi2

    assert low_edge == pytest.approx(3.3, abs=0.3)

    assert high_edge > 12.0


# ============================================================
# Lyman-alpha: the published combined constraint
# ============================================================

def test_lya_product_reproduces_the_published_combined_errors(lya):
    """
    The check that multiplying the auto and cross surfaces is
    allowed. Published: 37.5 +- 1.1 and 8.99 +- 0.19.
    """

    dm, dh, probability = scan_2d(lya)

    mean_dm, sigma_dm = moments(dm, probability.sum(axis=1))
    mean_dh, sigma_dh = moments(dh, probability.sum(axis=0))

    assert mean_dm == pytest.approx(37.5, abs=0.1)
    assert sigma_dm == pytest.approx(1.1, rel=0.06)

    assert mean_dh == pytest.approx(8.99, abs=0.02)
    assert sigma_dh == pytest.approx(0.19, rel=0.06)


def test_each_lya_half_is_weaker_than_the_combination(lya, cosmology):
    """
    Sanity on the direction of the combination: neither half alone
    can be tighter than the two together.
    """

    dm, dh, combined = scan_2d(lya)

    _, sigma_combined = moments(dh, combined.sum(axis=0))

    for version in ("dr16_auto", "dr16_cross"):

        half = EBOSSLyaLikelihood(cosmology, version=version)

        _, _, probability = scan_2d(half)

        _, sigma = moments(dh, probability.sum(axis=0))

        assert sigma > sigma_combined, version


def test_combining_the_halves_is_additive_in_chi2(lya, cosmology):
    """
    Multiplying likelihoods is adding log-likelihoods, and the
    combination is done once at load rather than twice at
    evaluation. If that ever stops holding, the default version is
    silently not the product of its halves.
    """

    total = sum(
        EBOSSLyaLikelihood(cosmology, version=v).chi2()
        for v in ("dr16_auto", "dr16_cross")
    )

    assert lya.chi2() == pytest.approx(total, rel=1e-10)


def test_lya_grids_have_decayed_by_their_edges(lya):
    """
    Unlike ELG, truncating these costs nothing -- so any failure of
    the edge handling would have to show up somewhere else.
    """

    for low_edge, high_edge in lya.data.edge_delta_chi2:

        assert low_edge > 20.0
        assert high_edge > 20.0


# ============================================================
# The interpolation choice
# ============================================================

def test_interpolating_the_probability_would_not_reproduce_the_paper(lya):
    """
    Why the log is splined and not the probability.

    The released values span thirty orders of magnitude. A cubic
    fitted to them is dominated by the peak and undershoots to
    *negative* probabilities between nodes in the tails -- which
    have no logarithm, and which a normalized marginal cannot
    survive. Pinned so the choice is not "tidied up" later.
    """

    from scipy.interpolate import RectBivariateSpline

    data = lya.data

    direct = RectBivariateSpline(

        data.axes[0],

        data.axes[1],

        np.exp(data.log_prob),

        kx=3,

        ky=3,

    )

    dm = np.linspace(*data.bounds[0], 1200)
    dh = np.linspace(*data.bounds[1], 1200)

    surface = direct(dm, dh)

    assert surface.min() < 0.0, (
        "expected the probability-space spline to undershoot; if it "
        "no longer does, re-derive whether the log is still needed"
    )


def test_the_surface_interpolates_rather_than_snapping_to_nodes(lya):
    """
    A lookup that rounded to the nearest grid point would pass most
    of the tests above and quantize every chi2 in a fit -- which is
    exactly the failure mode that defeated `best_fit()` on the CMB.
    """

    dm_axis = lya.data.axes[0]

    midpoint = 0.5 * (dm_axis[24] + dm_axis[25])

    peak_dh = lya.data.peak[1]

    values = [
        lya.log_likelihood_at([dm_axis[24], peak_dh]),
        lya.log_likelihood_at([midpoint, peak_dh]),
        lya.log_likelihood_at([dm_axis[25], peak_dh]),
    ]

    assert values[0] != values[1] != values[2]

    # And it is a genuine interpolation, not a step.
    assert min(values[0], values[2]) <= values[1] <= max(values[0], values[2])


# ============================================================
# Outside the grid
# ============================================================

@pytest.mark.parametrize(
    "values",
    [
        [10.0],          # below
        [30.0],          # above
    ],
)
def test_elg_excludes_predictions_off_the_table(elg, values):

    assert elg.log_likelihood_at(values) == -np.inf


def test_lya_excludes_if_either_coordinate_is_off(lya):

    inside = lya.data.peak

    assert np.isfinite(lya.log_likelihood_at(inside))

    assert lya.log_likelihood_at([inside[0], 100.0]) == -np.inf
    assert lya.log_likelihood_at([100.0, inside[1]]) == -np.inf


def test_an_excluded_cosmology_gives_infinite_chi2():
    """
    ``chi2`` has to stay ``+inf`` rather than NaN, so a sampler
    rejects the step instead of propagating a NaN into the chain.
    """

    absurd = LCDM(LCDM.PARAMS_CLASS(**{**PLANCK, "H0": 20.0}))

    likelihood = EBOSSLyaLikelihood(absurd)

    assert likelihood.chi2() == np.inf

    assert not np.isnan(likelihood.chi2())


# ============================================================
# The dataset object
# ============================================================

def test_effective_redshifts():

    assert load_eboss_table("eboss_elg").z_eff == pytest.approx(0.845)
    assert load_eboss_table("eboss_lya").z_eff == pytest.approx(2.334)


def test_size_counts_observables_not_grid_points(elg, lya):
    """
    ``n_data`` feeds AIC/BIC and every "chi2 per point" report. A
    50x50 grid is one measurement of two quantities, not 2500 of
    them.
    """

    assert elg.n_data == 1
    assert lya.n_data == 2


def test_the_halves_share_a_grid_with_the_combination():

    combined = load_eboss_table("eboss_lya", "dr16")

    for version in ("dr16_auto", "dr16_cross"):

        half = load_eboss_table("eboss_lya", version)

        for a, b in zip(combined.axes, half.axes):

            np.testing.assert_array_equal(a, b)


def test_grid_shape_must_match_the_axes():

    with pytest.raises(ValueError, match="does not match"):

        from CosmoFit.data.dataset import TabulatedBAODataset

        TabulatedBAODataset(

            z_eff=1.0,

            observable=("DV_over_rs",),

            axes=(np.linspace(1.0, 2.0, 5),),

            log_prob=np.zeros(4),

        )


def test_observables_and_axes_must_agree_in_number():

    from CosmoFit.data.dataset import TabulatedBAODataset

    with pytest.raises(ValueError, match="grid axes"):

        TabulatedBAODataset(

            z_eff=1.0,

            observable=("DM_over_rs", "DH_over_rs"),

            axes=(np.linspace(1.0, 2.0, 5),),

            log_prob=np.zeros(5),

        )


def test_a_minus_infinity_in_the_grid_is_refused():
    """
    An exact zero in a released probability becomes ``-inf`` here,
    and a spline through ``-inf`` is meaningless over its whole
    support rather than just at that node -- so it is caught at
    construction instead of poisoning every evaluation.
    """

    from CosmoFit.data.dataset import TabulatedBAODataset

    log_prob = np.zeros(5)
    log_prob[2] = -np.inf

    with pytest.raises(ValueError, match="finite"):

        TabulatedBAODataset(

            z_eff=1.0,

            observable=("DV_over_rs",),

            axes=(np.linspace(1.0, 2.0, 5),),

            log_prob=log_prob,

        )


# ============================================================
# How non-Gaussian each surface actually is
# ============================================================

def gaussian_comparison(likelihood):
    """
    Compare a 2-D surface with the Gaussian that has its mean and
    covariance: returns the correlation coefficient and the largest
    ``|delta chi2|`` between the two inside 3 sigma.
    """

    x, y, probability = scan_2d(likelihood, n=500)

    mx, _ = moments(x, probability.sum(axis=1))
    my, _ = moments(y, probability.sum(axis=0))

    X, Y = np.meshgrid(x, y, indexing="ij")

    cxx = float((probability * (X - mx) ** 2).sum())
    cyy = float((probability * (Y - my) ** 2).sum())
    cxy = float((probability * (X - mx) * (Y - my)).sum())

    covariance = np.array([[cxx, cxy], [cxy, cyy]])

    delta = np.stack([X - mx, Y - my], axis=-1)

    gaussian_chi2 = np.einsum(
        "...i,ij,...j->...", delta, np.linalg.inv(covariance), delta,
    )

    log_like = likelihood.log_likelihood_at(
        np.stack([X, Y], axis=-1),
    )

    table_chi2 = -2.0 * (log_like - likelihood.data.log_prob.max())

    inside = table_chi2 < 11.83

    return (
        cxy / np.sqrt(cxx * cyy),
        float(np.abs(table_chi2 - gaussian_chi2)[inside].max()),
    )


def test_lya_carries_a_correlation_two_error_bars_would_not(lya):
    """
    The measured reason to keep the surface.

    ``D_M/r_d`` and ``D_H/r_d`` are anticorrelated at -0.46 -- a
    tilt no pair of independent error bars reproduces, and the
    thing most obviously lost by summarizing the grid.
    """

    correlation, _ = gaussian_comparison(lya)

    assert correlation == pytest.approx(-0.46, abs=0.03)


def test_lya_is_only_moderately_non_gaussian(lya, cosmology):
    """
    Stated as a measurement rather than a slogan, because the
    contours *look* elliptical and it would be easy to claim
    otherwise.

    Against a Gaussian with the same mean and covariance, the
    combined surface departs by ~1.7 in chi2 out at 3 sigma. The
    cross-correlation half alone departs by ~5.0 -- the combination
    is more Gaussian than either piece, which is what multiplying
    two similar surfaces does.
    """

    _, combined = gaussian_comparison(lya)

    _, cross = gaussian_comparison(
        EBOSSLyaLikelihood(cosmology, version="dr16_cross"),
    )

    assert combined == pytest.approx(1.7, abs=0.4)

    assert cross == pytest.approx(5.0, abs=0.8)

    assert cross > combined


# ============================================================
# The full-shape grid: three dimensions, and a growth rate
# ============================================================

@pytest.fixture(scope="module")
def elg_fs(cosmology):

    from CosmoFit.likelihoods.eboss_dr16 import EBOSSELGFullShapeLikelihood

    return EBOSSELGFullShapeLikelihood(cosmology)


def marginal(likelihood, axis, n=120):
    """
    One marginal of a 3-D surface, evaluated through the
    likelihood's own interpolator, as (grid, normalized weights).
    """

    axes = [
        np.linspace(low, high, n)
        for low, high in likelihood.data.bounds
    ]

    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    probability = np.exp(likelihood.log_likelihood_at(points))

    other = tuple(i for i in range(3) if i != axis)

    weights = probability.sum(axis=other)

    return axes[axis], weights / weights.sum()


def percentiles(grid, weights, levels=(0.1587, 0.5, 0.8413)):

    return np.interp(levels, np.cumsum(weights), grid)


def test_full_shape_grid_reproduces_the_published_constraints(elg_fs):
    """
    de Mattia et al. quote, from the consensus of the Fourier- and
    configuration-space analyses at z_eff = 0.85:

        D_M/r_d   = 19.5 +- 1.0
        D_H/r_d   = 19.6 (-2.1/+2.2)
        f sigma_8 = 0.315 +- 0.095

    Recovered here by marginalizing the released grid, with none of
    those numbers used as an input.

    The comparison has to be made with *percentile* intervals rather
    than standard deviations. The released grid keeps a long tail --
    it runs out to D_M/r_d = 35 -- and the standard deviation of the
    D_M marginal is 1.40 against the published 1.0, while its
    16th-to-84th percentile half-width is 1.05. The tail is real and
    belongs in the likelihood; it just is not what an error bar
    quoted from an MCMC summarizes.
    """

    published = [(19.5, 1.0), (19.6, 2.15), (0.315, 0.095)]

    for axis, (centre, sigma) in enumerate(published):

        grid, weights = marginal(elg_fs, axis)

        low, median, high = percentiles(grid, weights)

        assert median == pytest.approx(centre, abs=0.15 * max(sigma, 0.1)), (
            f"axis {axis}: median {median:.4f} vs published {centre}"
        )

        half_width = 0.5 * (high - low)

        assert half_width == pytest.approx(sigma, rel=0.15), (
            f"axis {axis}: half-width {half_width:.4f} vs {sigma}"
        )


def test_the_growth_rate_is_predicted_without_an_AP_correction(elg_fs):
    """
    A full-shape grid varies ``D_M/r_d`` and ``D_H/r_d`` alongside
    ``f sigma_8``, so the geometry it was measured against is a
    *coordinate* of the grid rather than a fiducial to correct back
    to. Applying the Alcock-Paczynski rescaling that
    ``likelihoods/fsigma8.py`` needs would count it twice.
    """

    prediction = elg_fs.model()

    assert prediction[2] == pytest.approx(
        elg_fs.cosmology.background.fsigma8(elg_fs.data.z_eff),
    )


def test_the_grid_is_three_dimensional_not_flattened(elg_fs):
    """
    A 100x100x100 grid read as 2-D would still interpolate, still
    return finite numbers, and be wrong everywhere.
    """

    assert len(elg_fs.data.axes) == 3

    assert elg_fs.data.log_prob.shape == (100, 100, 100)

    assert elg_fs.n_data == 3


def test_the_shipped_grid_is_the_converted_one(elg_fs):
    """
    This dataset is the only one in the package that is *not* the
    released file: 60 MB of ASCII, 10.3% of it underflowed to exact
    zeros that have no logarithm.
    ``tools/convert_eboss_elg_fs_grid.py` floors the log 200 below
    its peak and stores float32.

    Both properties are checked here rather than trusted, since a
    regenerated file that lost either would still load.
    """

    log_prob = elg_fs.data.log_prob

    assert np.all(np.isfinite(log_prob))

    depth = log_prob.max() - log_prob.min()

    assert depth == pytest.approx(200.0, abs=0.5)

    # A floor at exp(-200) = 1e-87 cannot matter: the floored points
    # carry no probability at all.
    floored = log_prob <= log_prob.max() - 199.9

    assert floored.sum() > 0

    assert np.exp(log_prob[floored] - log_prob.max()).sum() < 1e-60


def test_off_grid_predictions_are_excluded_in_three_dimensions(elg_fs):

    peak = list(elg_fs.data.peak)

    assert np.isfinite(elg_fs.log_likelihood_at(peak))

    for axis in range(3):

        off = list(peak)
        off[axis] = elg_fs.data.bounds[axis][1] * 2.0

        assert elg_fs.log_likelihood_at(off) == -np.inf, axis


def test_the_two_elg_analyses_are_registered_as_conflicting():
    """
    ``eboss_elg`` and ``eboss_elg_fs`` are the same galaxies twice.
    """

    from CosmoFit.stats.fitter import CONFLICTING_DATASETS

    pairs = {frozenset(pair) for pair in CONFLICTING_DATASETS}

    assert frozenset({"eboss_elg", "eboss_elg_fs"}) in pairs
    assert frozenset({"fsigma8", "eboss_elg_fs"}) in pairs


def test_the_3d_interpolator_cannot_overshoot_its_nodes(elg_fs):
    """
    The bug that the published-value check above caught, pinned.

    The 3-D grid ships with its log floored 200 below the peak,
    because a tenth of the released probabilities underflowed to
    exact zero. A cubic interpolator rings at the step where that
    plateau begins: measured on a 120^3 mesh it reached a log of
    **+146** against a node maximum of 0, and exp(146) = 1e63 swamps
    every normalization -- the D_M marginal's median came out at
    35.4, the top of the grid, instead of 19.4.

    Linear interpolation is bounded by the surrounding nodes. This
    asserts that property directly rather than re-checking the
    symptom, so any future change of interpolator has to justify
    itself here.
    """

    data = elg_fs.data

    axes = [np.linspace(low, high, 60) for low, high in data.bounds]

    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    interpolated = elg_fs.log_likelihood_at(points)

    assert interpolated.max() <= data.log_prob.max() + 1e-9

    assert interpolated.min() >= data.log_prob.min() - 1e-9


def test_a_full_shape_fit_runs_end_to_end():
    """
    The dataset has to reach a Fitter, not just a likelihood --
    which means `sigma8` is a parameter the grid actually responds
    to, unlike every other BAO dataset in the library.
    """

    from CosmoFit import Fitter

    fit = Fitter(
        model=LCDM,
        datasets=["cc", "eboss_elg_fs"],
        free_params=["H0", "Omega_m", "sigma8"],
        initial={**PLANCK, "sigma8": 0.811},
    )

    baseline = fit.logpost.chi2(fit.theta0)

    assert np.isfinite(baseline)

    # sigma8 must move it: the f*sigma8 axis is the whole point.
    theta = fit.theta0.copy()
    theta[fit.free_params.index("sigma8")] = 0.60

    assert fit.logpost.chi2(theta) != baseline
