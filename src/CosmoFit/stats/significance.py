"""
One conversion from a p-value to "n sigma", for the whole library.

"n sigma" means the two-tailed Gaussian equivalent: the p-value of a
standard normal falling more than n from zero in *either* direction,
so 1, 2, 3 sigma are p = 0.317, 0.0455, 0.0027. It is the convention
cosmology papers quote, and for a chi-square with one degree of
freedom it is exactly ``sqrt(delta_chi2)``.

This module exists because the library had two answers.
:mod:`stats.tension` and :mod:`stats.cpl_diagnostics` used the
two-tailed form, while :func:`stats.model_comparison.likelihood_ratio_test`
used the one-tailed ``norm.isf(p)``. That reported delta_chi2 = 4 for
one extra parameter as 1.69 sigma instead of 2.0, so the same evidence
came out weaker in a model comparison than in a tension.
"""

from __future__ import annotations

import numpy as np

from scipy import stats


def p_to_sigma(p_value: float) -> float:
    """
    Two-tailed p-value to an equivalent number of Gaussian sigmas.

    Clipped at zero: a p-value above 0.5 means agreement better than
    chance, which is not negative evidence. A p-value of exactly zero
    is floored at 1e-300 (about 37 sigma) rather than returned as
    infinity.
    """

    p_value = float(np.clip(p_value, 1.0e-300, 1.0))

    return float(max(stats.norm.isf(0.5 * p_value), 0.0))
