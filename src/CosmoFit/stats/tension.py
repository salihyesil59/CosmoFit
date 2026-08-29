r"""
How much two measurements disagree.

This library quotes tensions constantly -- ``4.1 sigma`` in the
Hubble constant, ``2.9 sigma`` in ``S8`` -- and until now computed
every one of them by hand as

    sigma = ``|a - b|`` / sqrt(sigma_a^2 + sigma_b^2)

That formula is right, and it assumes three things that are worth
being asked about each time rather than once: that both posteriors
are Gaussian, that they are one-dimensional, and that they are
independent. The functions here are the same arithmetic where those
hold, and the alternatives where they do not.

Four of them, in increasing order of what they need:

:func:`gaussian_tension`
    Two numbers with error bars. The formula above, named, with its
    assumptions written down.

:func:`sample_tension`
    Two sets of posterior samples of the same parameter. Makes no
    Gaussian assumption -- it builds the distribution of the
    *difference* and asks how much of it lies further from zero
    than zero itself. The right tool for the skewed posteriors this
    library keeps producing.

:func:`gaussian_tension_nd`
    Two multi-dimensional posteriors, summarized by mean and
    covariance. A tension in a plane is not the larger of its two
    projections, and can be much bigger than either.

:func:`suspiciousness`
    Two *evidences* plus their joint. This is the one that does not
    care about the prior -- which matters, because
    :mod:`stats.evidence` has to warn that a Bayes factor does.

References
----------
Handley & Lemos (2019), Phys. Rev. D 100, 043504,
`arXiv:1902.04029 <https://arxiv.org/abs/1902.04029>`_
(suspiciousness).

Raveri & Hu (2019), Phys. Rev. D 99, 043506,
`arXiv:1806.04649 <https://arxiv.org/abs/1806.04649>`_
(the parameter-difference family this borrows from).
"""

from __future__ import annotations

import numpy as np

from scipy import stats


def _sigma_from_p(p_value: float) -> float:
    """
    Two-tailed p-value to an equivalent number of Gaussian sigmas.

    Clipped at zero: a p-value above 0.5 means the two agree better
    than chance, which is not a negative tension.
    """

    p_value = float(np.clip(p_value, 1.0e-300, 1.0))

    return float(max(stats.norm.isf(0.5 * p_value), 0.0))


def gaussian_tension(a, sigma_a, b, sigma_b) -> dict:
    """
    Tension between two independent Gaussian measurements of one
    parameter.

    Parameters
    ----------
    a, sigma_a, b, sigma_b : float

    Returns
    -------
    dict
        ``difference``, ``combined_sigma``, ``n_sigma``, ``p_value``.

    Notes
    -----
    Assumes both posteriors are Gaussian, one-dimensional and
    independent. The first of those is the one that usually fails
    here -- see :func:`sample_tension`, which does not need it.
    """

    difference = float(a) - float(b)

    combined = float(np.hypot(sigma_a, sigma_b))

    if combined <= 0.0:

        raise ValueError(
            "Both uncertainties are zero; there is no scale to "
            "measure a difference against."
        )

    n_sigma = abs(difference) / combined

    return {
        "difference": difference,
        "combined_sigma": combined,
        "n_sigma": float(n_sigma),
        "p_value": float(2.0 * stats.norm.sf(n_sigma)),
    }


def sample_tension(samples_a, samples_b, n_pairs=200_000, seed=0,
                   bins=200) -> dict:
    """
    Tension between two posteriors of the same parameter, from
    samples, with no Gaussian assumption.

    The parameter-difference construction: if the two are
    independent, the distribution of ``a - b`` is what you get by
    pairing their samples at random. Perfect agreement puts that
    distribution's peak at zero; a tension pushes zero into its
    tail. The quoted probability is the fraction of the difference
    distribution at a *higher density* than zero -- so it works for
    a skewed or double-peaked difference, where "how many standard
    deviations from zero" would not.

    Parameters
    ----------
    samples_a, samples_b : array_like
        Posterior samples, equally weighted. Need not be the same
        length.
    n_pairs : int, optional
        Random pairs drawn to build the difference distribution.
    seed : int, optional
    bins : int, optional
        Histogram resolution for the density comparison.

    Returns
    -------
    dict
        ``n_sigma``, ``p_value``, ``median_difference``, and the
        difference samples themselves as ``difference``.
    """

    a = np.asarray(samples_a, dtype=float).ravel()
    b = np.asarray(samples_b, dtype=float).ravel()

    if a.size < 2 or b.size < 2:

        raise ValueError(
            "Need at least two samples from each posterior."
        )

    rng = np.random.default_rng(seed)

    difference = (
        rng.choice(a, size=n_pairs, replace=True)
        - rng.choice(b, size=n_pairs, replace=True)
    )

    density, edges = np.histogram(difference, bins=bins, density=True)

    centres = 0.5 * (edges[:-1] + edges[1:])

    density_at_zero = float(np.interp(0.0, centres, density, left=0.0, right=0.0))

    widths = np.diff(edges)

    # Probability mass at a higher density than zero: the fraction
    # of the difference posterior that is "more likely" than
    # agreement.
    excluded = float((density[density > density_at_zero]
                      * widths[density > density_at_zero]).sum())

    p_value = 1.0 - excluded

    return {
        "n_sigma": _sigma_from_p(p_value),
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
        "median_difference": float(np.median(difference)),
        "difference": difference,
    }


def gaussian_tension_nd(mean_a, cov_a, mean_b, cov_b) -> dict:
    """
    Tension between two multi-dimensional Gaussian posteriors.

        chi2 = (mu_a - mu_b)^T (C_a + C_b)^-1 (mu_a - mu_b)

    read against a chi-square with ``n`` degrees of freedom.

    Worth having separately because a tension in a plane is *not*
    the larger of its two one-dimensional projections: two
    posteriors can overlap in every parameter separately and still
    be far apart jointly, if their degeneracy directions differ.

    Returns
    -------
    dict
        ``chi2``, ``dof``, ``p_value``, ``n_sigma``.
    """

    mean_a = np.atleast_1d(np.asarray(mean_a, dtype=float))
    mean_b = np.atleast_1d(np.asarray(mean_b, dtype=float))

    if mean_a.shape != mean_b.shape:

        raise ValueError(
            f"Means have different shapes, {mean_a.shape} and "
            f"{mean_b.shape}; they must describe the same parameters."
        )

    difference = mean_a - mean_b

    total = np.atleast_2d(cov_a) + np.atleast_2d(cov_b)

    chi2 = float(difference @ np.linalg.solve(total, difference))

    dof = int(len(difference))

    p_value = float(stats.chi2.sf(chi2, df=dof))

    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "n_sigma": _sigma_from_p(p_value),
    }


def suspiciousness(joint, first, second) -> dict:
    r"""
    Tension from evidences, with the prior dependence divided out.

    The evidence ratio

        ln R = ln Z_AB - ln Z_A - ln Z_B

    measures whether two datasets prefer to be described together,
    but it moves with the prior volume -- widen a prior and ``R``
    rises, with nothing about the data changed.
    :mod:`stats.evidence` documents that at length, and measures it.

    Handley & Lemos' suspiciousness removes it by subtracting the
    information the data gained,

        ln S = ln R - ln I,      ln I = D_A + D_B - D_AB

    where ``D`` is each run's Kullback-Leibler divergence from prior
    to posterior -- which carries the same prior dependence and
    cancels it. What is left is a statement about the data.

    Parameters
    ----------
    joint, first, second : stats.nested.NestedResult
        Nested-sampling runs of the two datasets together and
        separately, over the **same** parameters and priors.

    Returns
    -------
    dict
        ``ln_S``, ``ln_R``, ``ln_I``, ``d`` (the effective number of
        constrained parameters), ``chi2``, ``p_value``, ``n_sigma``.

    Notes
    -----
    The degrees of freedom use ``d = d_A + d_B - d_AB`` with
    ``d = 2 * (<ln L> - ln L_max)``-style Bayesian model
    dimensionality; here it is approximated by the number of
    parameters, which is exact when every one of them is
    constrained by both datasets and an overestimate otherwise.
    Pass runs over the same parameter set and this is the standard
    result.
    """

    ln_r = float(
        joint.log_evidence - first.log_evidence - second.log_evidence
    )

    ln_i = float(
        first.information + second.information - joint.information
    )

    ln_s = ln_r - ln_i

    dof = len(joint.free_params)

    chi2 = float(dof - 2.0 * ln_s)

    p_value = float(stats.chi2.sf(max(chi2, 0.0), df=dof))

    return {
        "ln_S": ln_s,
        "ln_R": ln_r,
        "ln_I": ln_i,
        "d": dof,
        "chi2": chi2,
        "p_value": p_value,
        "n_sigma": _sigma_from_p(p_value),
    }
