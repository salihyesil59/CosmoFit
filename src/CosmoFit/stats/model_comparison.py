"""
Model comparison statistics.

Implements the standard chi2-based model selection tools used
throughout cosmology: AIC, BIC, and the likelihood-ratio test
for nested models (e.g. CPL vs its LCDM limit w0=-1, wa=0).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import norm


# ============================================================
# Information criteria
# ============================================================

def aic(chi2: float, k: int) -> float:
    """Akaike Information Criterion: chi2 + 2k."""

    return chi2 + 2.0 * k


def bic(chi2: float, k: int, n_data: int) -> float:
    """Bayesian Information Criterion: chi2 + k * ln(n_data)."""

    return chi2 + k * np.log(n_data)


# ============================================================
# Likelihood-ratio test
# ============================================================

#: How negative a nested ``delta_chi2`` may be before it is treated
#: as a real failure rather than convergence noise.
#:
#: Not zero, and the reason is measured. When the general model's
#: best fit *is* the nested limit -- LsCDM running off to
#: ``z_dagger ~ 98``, CPL sitting at ``w0 = -1, wa = 0`` -- the two
#: optimizers stop at the same minimum by different routes and
#: disagree at their own convergence tolerance. Observed across a
#: 32-fit scan: -8e-06 to -5e-04, every one of them at the limit.
#: Warning about those would train the reader to ignore the warning.
NESTED_CHI2_TOLERANCE = 1.0e-3


def likelihood_ratio_test(
    chi2_null: float,
    k_null: int,
    chi2_alt: float,
    k_alt: int,
    tolerance: float = NESTED_CHI2_TOLERANCE,
) -> dict:
    """
    Likelihood-ratio test between two *nested* models.

    "null" is the simpler model (e.g. LCDM), "alt" is the more
    general model that reduces to it for a special parameter
    choice (e.g. CPL with w0=-1, wa=0).

    Returns
    -------
    dict with keys:
        delta_chi2, delta_k, p_value, sigma

    Notes
    -----
    A **negative** ``delta_chi2`` is reported and warned about
    rather than passed through. Between nested models it cannot
    happen: the general model contains the simple one, so its
    minimum is at worst equal. Seeing one means an optimizer
    stopped somewhere that is not the minimum -- typically in a
    second basin, which is a failure `best_fit`'s stall detection
    cannot see, since the run converged and reported success.

    The formula would otherwise absorb it silently:
    ``chi2.sf(negative) == 1.0`` and ``norm.isf(1.0) == -inf``, so
    the result reads as "no evidence" instead of "this number is
    impossible". ``sigma`` is therefore clamped to zero -- a nested
    model cannot be evidence *against* the general one -- while
    ``delta_chi2`` is returned as measured, so the caller can see
    how bad it was. Refit with ``best_fit(restarts=...)``.

    Only a shortfall larger than ``tolerance`` is warned about.
    A general model whose best fit *is* the nested limit reaches the
    same minimum by a different route and disagrees at the
    optimizer's own convergence level -- see
    :data:`NESTED_CHI2_TOLERANCE`. ``sigma`` is clamped for any
    negative value regardless, since none of them is evidence.
    """

    if k_alt <= k_null:
        raise ValueError(
            "The 'alt' model must have more free parameters "
            "than the 'null' model for a likelihood-ratio test."
        )

    delta_chi2 = chi2_null - chi2_alt
    delta_k = k_alt - k_null

    if delta_chi2 < -abs(tolerance):

        warnings.warn(

            f"Nested model comparison gave delta_chi2 = "
            f"{delta_chi2:.4g}, which is impossible: the more "
            f"general model contains the simpler one, so it cannot "
            f"fit worse. One of the two minima was not found -- "
            f"most likely the general model's optimizer settled in "
            f"a second basin, converging and reporting success from "
            f"the wrong place. Refit with "
            f"`best_fit(restarts=...)`. `sigma` is reported as 0.",

            UserWarning,

            stacklevel=2,

        )

    p_value = stats.chi2.sf(max(delta_chi2, 0.0), df=delta_k)
    sigma = norm.isf(p_value)

    return {
        "delta_chi2": float(delta_chi2),
        "delta_k": int(delta_k),
        "p_value": float(p_value),
        "sigma": float(max(sigma, 0.0)),
    }


# ============================================================
# Model comparison summary
# ============================================================

def compare_models(
    *,
    name_null: str,
    chi2_null: float,
    k_null: int,
    name_alt: str,
    chi2_alt: float,
    k_alt: int,
    n_data: int,
) -> dict:
    """
    Full AIC/BIC/LRT comparison between two nested models,
    matching the "CPL vs LCDM" analysis block of the CPL_MCMC
    notebook.
    """

    lrt = likelihood_ratio_test(chi2_null, k_null, chi2_alt, k_alt)

    return {
        name_null: {
            "chi2": float(chi2_null),
            "k": int(k_null),
            "AIC": aic(chi2_null, k_null),
            "BIC": bic(chi2_null, k_null, n_data),
        },
        name_alt: {
            "chi2": float(chi2_alt),
            "k": int(k_alt),
            "AIC": aic(chi2_alt, k_alt),
            "BIC": bic(chi2_alt, k_alt, n_data),
        },
        "delta_AIC": aic(chi2_null, k_null) - aic(chi2_alt, k_alt),
        "delta_BIC": bic(chi2_null, k_null, n_data) - bic(chi2_alt, k_alt, n_data),
        "likelihood_ratio_test": lrt,
    }
