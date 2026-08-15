"""
Posteriors for quantities derived from the expansion history.

:mod:`stats.cpl_diagnostics` covers quantities that are closed-form
functions of CPL's ``w0``/``wa`` (the w(z)=-1 crossing, the
Mahalanobis distance from LCDM). The quantities here instead need the
model's actual ``E(z)``/``dE/dz``, so they are computed by pushing
posterior samples back through the cosmology -- which means they work
for *every* model in the library, not just CPL.

    z_t   the acceleration transition redshift, where the
          deceleration parameter q(z) changes sign
    q0    the present-day deceleration parameter, q(z=0)

Both are standard reported numbers in dark-energy papers, and both are
what the ``fitter.plots.deceleration()`` figure shows graphically.

Why not just evaluate at the best fit
-------------------------------------
Because q(z) is nonlinear in the parameters, ``z_t`` evaluated at the
posterior-median parameters is *not* the median of the ``z_t``
posterior, and it carries no uncertainty. Quoting a derived quantity
with an error bar means mapping every posterior sample through the
transformation and taking percentiles of the result, which is what
these functions do.

Example
-------
>>> from CosmoFit.stats import derived
>>> fit.run_mcmc(...)
>>> z_t = derived.transition_redshift(fit)
>>> derived.summarize(z_t)
{'median': 0.73..., 'plus': 0.04..., 'minus': 0.04..., ...}
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "q_of_z",
    "transition_redshift",
    "deceleration_today",
    "summarize",
]


#: Default cap on how many posterior samples are pushed through the
#: cosmology. A long chain can hold hundreds of thousands of samples,
#: and evaluating q(z) on a grid for each is pure Python-level work;
#: a few thousand is already far more than enough to pin down
#: percentiles of a smooth 1D derived quantity (the 68% interval of a
#: 5000-sample estimate is itself stable to well under a percent).
_MAX_SAMPLES = 5000


def _sample_block(fit, burnin, max_samples):
    """
    Thinned posterior samples, plus the parameter names they
    correspond to.

    Thinning happens along the *step* axis of the ``(nsteps, nwalkers,
    ndim)`` chain, keeping every walker -- not by striding the
    flattened array. That distinction matters: ``get_chain(flat=True)``
    lays samples out step-major, so a flat stride that shares a factor
    with ``nwalkers`` keeps landing on the same few walkers. A stride
    of 128 across 64 walkers, which is exactly what a 640k-sample
    chain thinned to 5000 produces, would return walker 0 at every
    second step and nothing else -- consecutive steps of a single
    walker, correlated over tau ~ 79 steps, so a nominal 5000 samples
    would carry barely ~100 independent ones. Measured, that shifted
    the z_t median by 0.006, about 13% of its uncertainty.

    Thinning along steps instead keeps all ``nwalkers`` chains and
    spaces the retained steps far enough apart to decorrelate them.

    The stride is fixed rather than a random draw, so the result is
    deterministic: calling this twice on the same chain gives the same
    answer, which matters when the number ends up in a paper.
    """

    if fit.sampler is None:
        raise RuntimeError("Call run_mcmc() first.")

    if burnin is None:
        burnin = fit.burnin

    chain = fit.sampler.get_chain(discard=burnin)

    n_steps, n_walkers, ndim = chain.shape

    if max_samples is not None and n_steps * n_walkers > max_samples:
        keep_steps = max(1, max_samples // n_walkers)
        stride = int(np.ceil(n_steps / keep_steps))
        chain = chain[::stride]

    flat = chain.reshape(-1, ndim)

    return flat, list(fit.free_params)


def q_of_z(fit, z, burnin=None, max_samples=_MAX_SAMPLES):
    """
    Deceleration parameter q(z) for every (thinned) posterior sample.

    Parameters
    ----------
    fit : stats.fitter.Fitter
        A fitter with a completed :meth:`~stats.fitter.Fitter.run_mcmc`.

    z : array_like
        Redshifts to evaluate q at.

    burnin : int, optional
        Steps to discard. Defaults to the fitter's own ``burnin``.

    max_samples : int or None, optional
        Cap on the number of posterior samples used (see
        :data:`_MAX_SAMPLES`). ``None`` uses every sample.

    Returns
    -------
    ndarray, shape (n_samples, len(z))
    """

    z = np.atleast_1d(np.asarray(z, dtype=float))

    flat, names = _sample_block(fit, burnin, max_samples)

    cosmology = fit.cosmology

    # q(z) = -1 + (1+z) E'(z)/E(z) needs only E and dE/dz, both of
    # which are analytic in the parameters for every model here -- no
    # `refresh()` (which would rebuild the chi(z) table) is required.
    #
    # The parameter object is shared and mutated in place, so snapshot
    # it and put it back afterwards: leaving the cosmology sitting on
    # the last posterior sample would silently change whatever the
    # caller does next (a plot, a chi2, another diagnostic).
    saved = cosmology.params.as_dict()

    out = np.empty((len(flat), z.size), dtype=float)

    try:
        for i, theta in enumerate(flat):
            cosmology.params.update(**dict(zip(names, theta)))
            out[i] = cosmology.background.q(z)
    finally:
        cosmology.params.update(**saved)
        cosmology.refresh()

    return out


def transition_redshift(
    fit,
    burnin=None,
    max_samples=_MAX_SAMPLES,
    z_max: float = 3.0,
    n_grid: int = 601,
):
    """
    Posterior of the acceleration transition redshift ``z_t``, where
    the deceleration parameter changes sign (q(z_t) = 0) -- the epoch
    the universe switched from decelerating to accelerating.

    The root is bracketed on a uniform grid over ``[0, z_max]`` and
    then refined by linear interpolation across the bracketing
    interval. With the default 601-point grid this agrees with the
    closed-form flat-LCDM result
    ``z_t = (2 Omega_Lambda / Omega_m)^(1/3) - 1`` to 4e-6 in z_t
    across a real posterior -- four orders of magnitude below the
    ~0.03 width of the posterior itself.

    Parameters
    ----------
    fit, burnin, max_samples
        See :func:`q_of_z`.

    z_max : float, optional
        Upper end of the search range.

    n_grid : int, optional
        Grid resolution used to bracket the sign change.

    Returns
    -------
    ndarray
        One ``z_t`` per posterior sample. Samples with no sign change
        in ``[0, z_max]`` -- an expansion history that never
        transitions in range -- are ``nan``; :func:`summarize` drops
        them, and their count is reported as ``n_undefined``.
    """

    z = np.linspace(0.0, z_max, n_grid)

    q = q_of_z(fit, z, burnin=burnin, max_samples=max_samples)

    z_t = np.full(q.shape[0], np.nan)

    for i, row in enumerate(q):

        crossings = np.nonzero(np.diff(np.signbit(row)))[0]

        if crossings.size == 0:
            continue

        k = crossings[0]
        q_lo, q_hi = row[k], row[k + 1]

        # q_hi != q_lo is guaranteed: they straddle zero with
        # opposite signbits, so their difference cannot vanish.
        z_t[i] = z[k] + (z[k + 1] - z[k]) * (0.0 - q_lo) / (q_hi - q_lo)

    return z_t


def deceleration_today(fit, burnin=None, max_samples=_MAX_SAMPLES):
    """
    Posterior of the present-day deceleration parameter, q(z=0).

    Returns
    -------
    ndarray, one value per posterior sample.
    """

    return q_of_z(fit, 0.0, burnin=burnin, max_samples=max_samples)[:, 0]


def summarize(values, percentiles=(2.5, 16, 50, 84, 97.5)) -> dict:
    """
    Median and credible intervals of a derived-quantity posterior, in
    the same ``median``/``plus``/``minus`` shape
    :meth:`~stats.fitter.Fitter.summary` uses.

    Non-finite entries (e.g. samples with no transition redshift, see
    :func:`transition_redshift`) are dropped and counted separately
    rather than silently poisoning the percentiles.

    Returns
    -------
    dict with keys ``median``, ``plus``, ``minus``, ``mean``, ``std``,
    ``lower95``, ``upper95``, ``n``, ``n_undefined``.
    """

    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]
    n_undefined = int(values.size - finite.size)

    if finite.size == 0:
        raise ValueError(
            "No finite samples to summarize -- every sample was nan "
            "(for transition_redshift, that means no sample had a "
            "q(z)=0 crossing inside the search range; try a larger "
            "z_max)."
        )

    lo95, lo68, med, hi68, hi95 = np.percentile(finite, percentiles)

    return {
        "median": float(med),
        "plus": float(hi68 - med),
        "minus": float(med - lo68),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "lower95": float(lo95),
        "upper95": float(hi95),
        "n": int(finite.size),
        "n_undefined": n_undefined,
    }
