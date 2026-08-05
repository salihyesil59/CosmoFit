"""
Model comparison statistics.

Implements the standard chi2-based model selection tools used
throughout cosmology: AIC, BIC, and the likelihood-ratio test
for nested models (e.g. CPL vs its LCDM limit w0=-1, wa=0).
"""

from __future__ import annotations

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

def likelihood_ratio_test(
    chi2_null: float,
    k_null: int,
    chi2_alt: float,
    k_alt: int,
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
    """

    if k_alt <= k_null:
        raise ValueError(
            "The 'alt' model must have more free parameters "
            "than the 'null' model for a likelihood-ratio test."
        )

    delta_chi2 = chi2_null - chi2_alt
    delta_k = k_alt - k_null

    p_value = stats.chi2.sf(delta_chi2, df=delta_k)
    sigma = norm.isf(p_value)

    return {
        "delta_chi2": float(delta_chi2),
        "delta_k": int(delta_k),
        "p_value": float(p_value),
        "sigma": float(sigma),
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
