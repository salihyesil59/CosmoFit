"""
Model-comparison statistics, and the impossible result they used to
report without comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit.stats.model_comparison import likelihood_ratio_test

# ============================================================
# An impossible nested comparison
# ============================================================

def test_a_negative_delta_chi2_is_warned_about():
    """
    Between nested models the general one contains the simple one,
    so its minimum is at worst equal and delta_chi2 cannot be
    negative. One that is means an optimizer stopped somewhere that
    is not the minimum -- and `best_fit`'s stall detection cannot
    see it, because the run converged and reported success.

    Measured case: LsCDM against LCDM on CC + eBOSS BAO + SN +
    Planck priors returned 1654.51 against 1653.88. With
    `restarts=` it finds 1653.40.
    """

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:

        _warnings.simplefilter("always")

        result = likelihood_ratio_test(
            chi2_null=1653.88, k_null=3,
            chi2_alt=1654.51, k_alt=4,
        )

    assert any("impossible" in str(w.message) for w in caught), (
        [str(w.message) for w in caught]
    )

    # Reported as measured, so the caller can see how bad it was.
    assert result["delta_chi2"] < 0.0


def test_a_negative_delta_chi2_does_not_become_minus_infinity():
    """
    The formula absorbs it silently otherwise: chi2.sf of a negative
    number is 1.0, and norm.isf(1.0) is -inf. That reads as "no
    evidence" rather than "this number cannot happen", and prints in
    a summary table as -inf sigma.
    """

    import warnings as _warnings

    with _warnings.catch_warnings():

        _warnings.simplefilter("ignore")

        result = likelihood_ratio_test(
            chi2_null=100.0, k_null=2, chi2_alt=101.0, k_alt=3,
        )

    assert result["sigma"] == 0.0

    assert np.isfinite(result["sigma"])


def test_a_positive_comparison_is_untouched():
    """
    The guard must not perturb the normal path.
    """

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:

        _warnings.simplefilter("always")

        result = likelihood_ratio_test(
            chi2_null=1665.74, k_null=3, chi2_alt=1660.35, k_alt=4,
        )

    assert not caught

    assert result["delta_chi2"] == pytest.approx(5.39)
    assert result["sigma"] == pytest.approx(2.0486, abs=1e-3)


def test_convergence_noise_at_the_nested_limit_is_not_warned_about():
    """
    When the general model's best fit *is* the nested limit -- LsCDM
    running off to z_dagger ~ 98, CPL sitting at w0 = -1, wa = 0 --
    the two optimizers reach the same minimum by different routes
    and disagree at their own convergence tolerance.

    Measured across a 32-fit scan: -8e-06 to -5e-04, every one of
    them at the limit and none of them a failure. Warning about
    those would train the reader to ignore the warning, which is
    worse than not having it.
    """

    import warnings as _warnings

    for shortfall in (8.0e-6, 4.7e-4):

        with _warnings.catch_warnings(record=True) as caught:

            _warnings.simplefilter("always")

            result = likelihood_ratio_test(
                chi2_null=100.0, k_null=3,
                chi2_alt=100.0 + shortfall, k_alt=4,
            )

        assert not caught, f"warned about {shortfall:.1e}"

        # Still not reported as evidence.
        assert result["sigma"] == 0.0


def test_the_tolerance_is_adjustable():

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:

        _warnings.simplefilter("always")

        likelihood_ratio_test(
            chi2_null=100.0, k_null=3, chi2_alt=100.01, k_alt=4,
            tolerance=1e-9,
        )

    assert any("impossible" in str(w.message) for w in caught)
