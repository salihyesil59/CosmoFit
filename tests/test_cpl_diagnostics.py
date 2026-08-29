"""
`stats.cpl_diagnostics`.

The module that turns a (w0, wa) posterior into the statements a
dark-energy paper makes: where w(z) crosses -1, in which direction,
which region of the plane the posterior sits in, and how far LCDM is
from it. 37% of it had ever run, and what ran was only what
`plots.w0_wa_plane(show_fractions=True)` happens to touch.

Almost everything here has a closed form to check against, so these
are not round-trip tests. The crossing redshift is asserted by
putting it back into w(z) and getting -1; the regions by their
definitions at the two ends of the CPL evolution; and the Mahalanobis
conversion against the three enclosed probabilities the module's own
docstring quotes.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit.stats.cpl_diagnostics import (
    REGIONS,
    classify_region,
    crossing_direction,
    crossing_redshift,
    mahalanobis_from_lcdm,
    region_fractions,
    wz,
    wz_posterior_bands,
)


# ============================================================
# w(z)
# ============================================================


def test_wz_is_w0_today_and_w0_plus_wa_in_the_far_past():
    """
    The two ends of the CPL evolution, which every classification
    below is built on.
    """

    w0 = np.array([-0.9, -1.1])
    wa = np.array([-0.5, 0.3])

    values = wz(w0, wa, [0.0, 1e8])

    np.testing.assert_allclose(values[:, 0], w0)

    np.testing.assert_allclose(values[:, 1], w0 + wa, rtol=1e-7)


def test_wz_shape_is_samples_by_redshifts():

    values = wz(np.zeros(7), np.zeros(7), [0.0, 0.5, 1.0])

    assert values.shape == (7, 3)


def test_posterior_bands_are_ordered_and_the_right_length():

    rng = np.random.default_rng(0)

    w0 = rng.normal(-1.0, 0.1, size=2000)
    wa = rng.normal(0.0, 0.3, size=2000)

    z = np.linspace(0.0, 2.0, 11)

    bands = wz_posterior_bands(w0, wa, z)

    assert set(bands) == {2.5, 16, 50, 84, 97.5}

    for band in bands.values():
        assert band.shape == z.shape

    # A percentile band cannot cross a higher one.
    for lower, upper in zip((2.5, 16, 50, 84), (16, 50, 84, 97.5)):
        assert np.all(bands[lower] <= bands[upper])


# ============================================================
# The w(z) = -1 crossing
# ============================================================


def test_the_crossing_redshift_really_is_where_w_equals_minus_one():
    """
    The strong form of this test: rather than re-deriving
    `-(1+w0)/(1+w0+wa)`, put the answer back into w(z) and require
    -1 out. A sign error in the formula could not survive it.
    """

    rng = np.random.default_rng(1)

    w0 = rng.uniform(-1.4, -0.6, size=4000)
    wa = rng.uniform(-2.0, 2.0, size=4000)

    z_cross, fraction = crossing_redshift(w0, wa, z_min=0.0, z_max=2.5)

    assert 0.0 < fraction < 1.0

    assert z_cross.size == int(round(fraction * w0.size))

    # Recover which samples they were, and evaluate w there.
    denominator = 1.0 + w0 + wa
    all_z = -(1.0 + w0) / denominator
    valid = np.isfinite(all_z) & (all_z > 0.0) & (all_z < 2.5)

    w_at_crossing = (
        w0[valid] + wa[valid] * z_cross / (1.0 + z_cross)
    )

    np.testing.assert_allclose(w_at_crossing, -1.0, atol=1e-10)


def test_a_sample_with_no_crossing_in_range_is_dropped():
    """
    LCDM itself -- (w0, wa) = (-1, 0) -- gives 0/0. It must not
    appear as a crossing at some arbitrary redshift.
    """

    z_cross, fraction = crossing_redshift(
        np.array([-1.0, -0.9]), np.array([0.0, 0.0]),
    )

    assert fraction == 0.0

    assert z_cross.size == 0


def test_crossing_direction_separates_the_two_senses():
    """
    One sample built to cross each way, and nothing else. The
    fractions are then exactly one and zero.
    """

    # w0 = -0.9 (quintessence today), w0 + wa = -1.6 (phantom early)
    q_to_p = crossing_direction(np.array([-0.9]), np.array([-0.7]))

    assert q_to_p["n_crossing"] == 1
    assert q_to_p["quintessence_to_phantom"] == 1.0
    assert q_to_p["phantom_to_quintessence"] == 0.0

    # The reverse: phantom today, quintessence-like early.
    p_to_q = crossing_direction(np.array([-1.1]), np.array([0.7]))

    assert p_to_q["n_crossing"] == 1
    assert p_to_q["phantom_to_quintessence"] == 1.0
    assert p_to_q["quintessence_to_phantom"] == 0.0


# ============================================================
# The four regions
# ============================================================


@pytest.mark.parametrize(
    "w0, wa, expected",
    [
        (-1.2, -0.3, "phantom"),        # below -1 at both ends
        (-0.8, +0.1, "quintessence"),   # above -1 at both ends
        (-1.2, +0.5, "quintom-a"),      # phantom today, quintessence early
        (-0.8, -0.5, "quintom-b"),      # quintessence today, phantom early
    ],
)
def test_classify_region_from_the_two_ends(w0, wa, expected):

    assert classify_region(w0, wa) == expected


def test_classify_region_is_scalar_in_scalar_out_and_array_in_array_out():

    assert isinstance(classify_region(-1.2, -0.3), str)

    labels = classify_region(np.array([-1.2, -0.8]), np.array([-0.3, 0.1]))

    assert isinstance(labels, np.ndarray)

    assert labels.shape == (2,)

    assert list(labels) == ["phantom", "quintessence"]


def test_the_regions_tile_the_plane():
    """
    Every point gets exactly one of the four labels, so the
    fractions are a partition and sum to 1. This is what makes
    `region_fractions` quotable as a probability.
    """

    rng = np.random.default_rng(2)

    w0 = rng.normal(-1.0, 0.4, size=5000)
    wa = rng.normal(0.0, 1.0, size=5000)

    labels = classify_region(w0, wa)

    assert set(np.unique(labels)) <= set(REGIONS)

    fractions = region_fractions(w0, wa)

    assert list(fractions) == list(REGIONS)

    assert sum(fractions.values()) == pytest.approx(1.0)

    assert all(0.0 <= value <= 1.0 for value in fractions.values())


def test_a_posterior_entirely_in_one_region():

    fractions = region_fractions(
        np.full(100, -1.3), np.full(100, -0.2),
    )

    assert fractions["phantom"] == 1.0

    assert sum(fractions[r] for r in REGIONS if r != "phantom") == 0.0


# ============================================================
# LCDM's distance from the posterior
# ============================================================


def test_a_posterior_centred_on_lcdm_is_no_distance_away():

    rng = np.random.default_rng(3)

    result = mahalanobis_from_lcdm(
        rng.normal(-1.0, 0.05, size=20000),
        rng.normal(0.0, 0.15, size=20000),
    )

    assert result["distance"] < 0.1

    assert result["sigma"] < 0.2

    assert result["p_value"] > 0.9


def test_the_distance_is_not_the_significance():
    """
    The trap the module's docstring is written around, and the
    reason it returns both numbers rather than one.

    `D^2` follows chi-square with **two** degrees of freedom here,
    not one, so a D of 2.20 is not 2.2 sigma -- it is 1.70. Reporting
    `distance` as a number of sigma overstates the tension with LCDM,
    which is exactly the direction a dark-energy paper would like it
    to be wrong in.
    """

    from scipy import stats

    rng = np.random.default_rng(4)

    # An offset posterior, so D is comfortably non-zero.
    result = mahalanobis_from_lcdm(
        rng.normal(-0.85, 0.06, size=40000),
        rng.normal(-0.40, 0.20, size=40000),
    )

    assert result["sigma"] < result["distance"]

    # And by exactly the chi2_2 -> two-tailed-normal conversion.
    expected_p = stats.chi2.sf(result["distance_squared"], df=2)

    assert result["p_value"] == pytest.approx(expected_p)

    assert result["sigma"] == pytest.approx(stats.norm.isf(expected_p / 2.0))

    assert result["confidence_level"] == pytest.approx(1.0 - result["p_value"])


@pytest.mark.parametrize(
    "distance, enclosed",
    [
        (1.515, 0.6827),
        (2.486, 0.9545),
        (3.439, 0.9973),
    ],
)
def test_the_two_dimensional_thresholds_the_docstring_quotes(distance, enclosed):
    """
    The three numbers the docstring gives as the 2-D equivalents of
    "1, 2, 3 sigma". They are properties of chi-square with two
    degrees of freedom, so they can be checked directly -- and if
    they were the 1-D thresholds by mistake, they would read
    1.000, 2.000, 3.000.
    """

    from scipy import stats

    assert stats.chi2.cdf(distance ** 2, df=2) == pytest.approx(
        enclosed, abs=5e-4,
    )


def test_the_lcdm_point_can_be_moved():

    rng = np.random.default_rng(5)

    w0 = rng.normal(-1.0, 0.05, size=5000)
    wa = rng.normal(0.0, 0.15, size=5000)

    at_lcdm = mahalanobis_from_lcdm(w0, wa)

    elsewhere = mahalanobis_from_lcdm(w0, wa, lcdm_point=(-0.7, 0.9))

    assert elsewhere["distance"] > at_lcdm["distance"]

    np.testing.assert_allclose(at_lcdm["mean"], [w0.mean(), wa.mean()])

    assert at_lcdm["covariance"].shape == (2, 2)
