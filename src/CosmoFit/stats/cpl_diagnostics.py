"""
CPL-specific posterior diagnostics.

Utility functions that turn MCMC posterior samples of the CPL
equation-of-state parameters (w0, wa) into the derived
quantities used to characterize deviations from LCDM:

    - w(z) posterior bands
    - the "w(z) = -1 crossing" redshift distribution
    - the crossing direction (quintessence -> phantom or the
      reverse)
    - the Mahalanobis distance of the LCDM point (w0, wa) =
      (-1, 0) from the 2D posterior

These reproduce the final analysis cells of the CPL_MCMC
notebook.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# w(z)
# ============================================================

def wz(w0_samples, wa_samples, z):
    """
    CPL equation of state w(z) = w0 + wa * z / (1+z), evaluated
    for every posterior sample at every redshift in ``z``.

    Returns
    -------
    ndarray, shape (n_samples, len(z))
    """

    w0_samples = np.asarray(w0_samples, dtype=float)
    wa_samples = np.asarray(wa_samples, dtype=float)
    z = np.asarray(z, dtype=float)

    factor = z / (1.0 + z)

    return w0_samples[:, None] + wa_samples[:, None] * factor[None, :]


def wz_posterior_bands(w0_samples, wa_samples, z, quantiles=(2.5, 16, 50, 84, 97.5)):
    """
    Percentile bands of w(z) over the posterior, at each
    redshift in ``z``.

    Returns
    -------
    dict mapping each quantile (as given, e.g. 16) to an
    ndarray of length len(z).
    """

    samples = wz(w0_samples, wa_samples, z)

    return {
        q: np.percentile(samples, q, axis=0)
        for q in quantiles
    }


# ============================================================
# w(z) = -1 crossing
# ============================================================

def crossing_redshift(w0_samples, wa_samples, z_min=0.0, z_max=2.5):
    """
    Posterior distribution of the redshift at which w(z) = -1,

        z_cross = -(1 + w0) / (1 + w0 + wa)

    Only samples with a *physical* crossing (finite, and inside
    [z_min, z_max]) are kept.

    Returns
    -------
    z_cross : ndarray
        Crossing redshifts for the physically-valid samples.
    fraction : float
        Fraction of all posterior samples with a physical
        crossing in range.
    """

    w0_samples = np.asarray(w0_samples, dtype=float)
    wa_samples = np.asarray(wa_samples, dtype=float)

    denominator = 1.0 + w0_samples + wa_samples

    z_cross_all = -(1.0 + w0_samples) / denominator

    valid = (
        np.isfinite(z_cross_all)
        & (z_cross_all > z_min)
        & (z_cross_all < z_max)
    )

    z_cross = z_cross_all[valid]
    fraction = z_cross.size / w0_samples.size

    return z_cross, fraction


def crossing_direction(w0_samples, wa_samples, z_min=0.0, z_max=2.5, z_ref=2.5):
    """
    Among the samples with a physical w(z)=-1 crossing, what
    fraction cross from quintessence-like (w>-1) at z=0 to
    phantom-like (w<-1) at high z, versus the reverse?

    Returns
    -------
    dict with keys 'quintessence_to_phantom',
    'phantom_to_quintessence' (fractions of the crossing
    sub-sample), and 'n_crossing'.
    """

    w0_samples = np.asarray(w0_samples, dtype=float)
    wa_samples = np.asarray(wa_samples, dtype=float)

    denominator = 1.0 + w0_samples + wa_samples
    z_cross_all = -(1.0 + w0_samples) / denominator

    valid = (
        np.isfinite(z_cross_all)
        & (z_cross_all > z_min)
        & (z_cross_all < z_max)
    )

    w0_valid = w0_samples[valid]
    wa_valid = wa_samples[valid]
    n_crossing = int(valid.sum())

    w_at_zero = w0_valid
    w_at_zref = w0_valid + wa_valid * z_ref / (1.0 + z_ref)

    q_to_p = (w_at_zero > -1.0) & (w_at_zref < -1.0)
    p_to_q = (w_at_zero < -1.0) & (w_at_zref > -1.0)

    return {
        "n_crossing": n_crossing,
        "quintessence_to_phantom": float(np.sum(q_to_p)) / n_crossing,
        "phantom_to_quintessence": float(np.sum(p_to_q)) / n_crossing,
    }


# ============================================================
# Mahalanobis distance from LCDM
# ============================================================

def mahalanobis_from_lcdm(w0_samples, wa_samples, lcdm_point=(-1.0, 0.0)):
    """
    How far the LCDM point (w0, wa) = (-1, 0) sits from the 2D
    (w0, wa) posterior.

    Returns both the raw Mahalanobis distance and -- separately --
    the significance, because **the two are not the same number in
    2D and conflating them overstates the tension with LCDM**.

    ``distance`` (D) is the Mahalanobis distance,
    ``sqrt((x-mu)^T C^-1 (x-mu))``. It is *not* a number of sigma.
    For a 2D Gaussian, ``D^2`` follows a chi-square distribution with
    **2** degrees of freedom, not 1, so the probability enclosed at a
    given D is smaller than the familiar 1D intuition suggests:

        D = 1.515  encloses 68.27%  (the "1 sigma" probability)
        D = 2.486  encloses 95.45%  (the "2 sigma" probability)
        D = 3.439  encloses 99.73%  (the "3 sigma" probability)

    So a D of 2.20 is *not* 2.2 sigma -- it corresponds to 91.1%
    exclusion, i.e. 1.70 sigma in the usual one-dimensional
    two-tailed sense. ``sigma`` below applies that conversion; quote
    it, not ``distance``, when reporting a deviation from LCDM.

    Returns
    -------
    dict with keys:
        mean, covariance : the posterior mean and covariance used.
        distance_squared : D^2.
        distance : D, the Mahalanobis distance (NOT sigma).
        p_value : P(chi2_2 > D^2), the probability of being at least
            this far out under the posterior.
        confidence_level : 1 - p_value, i.e. the confidence at which
            LCDM is excluded.
        sigma : the equivalent one-dimensional two-tailed
            significance -- the number to report.

    Note this is a Gaussian approximation to the posterior: it uses
    only the mean and covariance, so a strongly non-Gaussian (e.g.
    banana-shaped) w0-wa contour is not fully captured. Compare
    against the fraction of samples further out than the LCDM point
    if that matters.
    """

    from scipy import stats as _stats

    samples = np.column_stack([
        np.asarray(w0_samples, dtype=float),
        np.asarray(wa_samples, dtype=float),
    ])

    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)

    delta = np.asarray(lcdm_point, dtype=float) - mean
    cov_inv = np.linalg.inv(cov)

    d2 = float(delta @ cov_inv @ delta)

    # 2 degrees of freedom: the posterior is 2D (w0, wa).
    p_value = float(_stats.chi2.sf(d2, df=2))

    # Two-tailed 1D equivalent, the convention "n sigma" refers to.
    sigma = float(_stats.norm.isf(p_value / 2.0))

    return {
        "mean": mean,
        "covariance": cov,
        "distance_squared": d2,
        "distance": float(np.sqrt(d2)),
        "p_value": p_value,
        "confidence_level": 1.0 - p_value,
        "sigma": sigma,
    }
