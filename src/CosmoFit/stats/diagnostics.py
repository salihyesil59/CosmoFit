"""
Chain diagnostics and posterior intervals.

What a fit should be asked before its numbers are quoted, gathered in
one place:

``split_rhat``
    The Gelman-Rubin statistic, split form (Gelman et al., *Bayesian
    Data Analysis*, 3rd ed., Sec. 11.4): each chain cut in half, so a
    single chain that is still drifting fails the test against itself.
    ``R - 1 < 0.01`` is the usual standard for a published posterior,
    and ``0.05`` a loose one.

``walker_group_rhat``
    The same statistic for one ``emcee`` run, between disjoint groups of
    walkers. The walkers of an ensemble sampler are not independent
    chains -- each proposal uses the others' positions -- so this is a
    weaker test than ``split_rhat`` across independent runs. It still
    catches the commonest failure: walkers that have not mixed, with
    some groups stuck where they started.

``hpd_interval``, ``one_sided_limit``
    The equal-tailed 16/50/84 percentiles every summary in this library
    used to report are the right interval for a near-Gaussian
    posterior and the wrong one otherwise. A posterior piled against a
    prior bound -- a neutrino mass, an ``f_R0``, a coupling that the data
    want to be zero -- needs the shortest interval or a one-sided
    limit, not a median with error bars that put a third of the
    probability below a physical boundary.

``auto_burnin``
    A burn-in measured in autocorrelation times rather than chosen in
    advance.
"""

from __future__ import annotations

import numpy as np


#: Burn-in, in units of the largest autocorrelation time, for
#: :func:`auto_burnin`. emcee's documentation suggests "a few" tau;
#: three is the usual reading of that.
BURNIN_TAU = 3.0


def split_rhat(chains) -> np.ndarray:
    r"""
    Split Gelman-Rubin ``R`` for each parameter.

    Parameters
    ----------
    chains : array_like, shape (n_chains, n_steps, ndim)
        Independent chains (two or more), or groups of samples to be
        compared as if they were.

    Returns
    -------
    ndarray, shape (ndim,)
        ``R`` per parameter; 1 for chains drawn from the same
        distribution.
    """

    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 3:
        raise ValueError(
            "Expected chains of shape (n_chains, n_steps, ndim), got "
            f"{chains.shape}."
        )

    n_chains, n_steps, _ = chains.shape

    half = n_steps // 2

    if half < 2:
        raise ValueError("Each chain needs at least four steps.")

    # Split every chain in two, so drift within a chain counts as
    # disagreement between its halves.
    split = np.concatenate(
        [chains[:, :half], chains[:, n_steps - half:]], axis=0,
    )

    n = half

    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)

    between = n * means.var(axis=0, ddof=1)
    within = variances.mean(axis=0)

    pooled = (n - 1) / n * within + between / n

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(pooled / within)

    # A parameter that does not move at all (fixed by construction)
    # is as converged as it can be.
    rhat = np.where(within > 0, rhat, 1.0)


    return rhat


def walker_group_rhat(chain, n_groups: int = 4) -> np.ndarray:
    """
    Split ``R`` between disjoint groups of an ensemble's walkers.

    Parameters
    ----------
    chain : array_like, shape (n_steps, n_walkers, ndim)
        Post-burn-in chain, as ``sampler.get_chain(discard=...)``
        returns it.

    n_groups : int
        How many groups to split the walkers into. Each group's walkers
        are pooled into one "chain" step by step.
    """

    chain = np.asarray(chain, dtype=float)

    n_steps, n_walkers, ndim = chain.shape

    n_groups = int(min(n_groups, n_walkers))

    if n_groups < 2:
        raise ValueError("Need at least two walkers to compare.")

    groups = np.array_split(np.arange(n_walkers), n_groups)

    # Each group: its walkers laid end to end in step order, so the
    # group is a single long "chain" with the same number of samples.
    size = min(len(g) for g in groups)

    pooled = np.stack(
        [
            chain[:, g[:size], :].reshape(n_steps * size, ndim)
            for g in groups
        ]
    )

    return split_rhat(pooled)


def hpd_interval(samples, level: float = 0.68) -> tuple[float, float]:
    """
    Shortest interval containing ``level`` of the samples.

    For a unimodal posterior this is the highest-posterior-density
    interval. Unlike percentiles it does not have to straddle the
    median, so a posterior against a bound gets an interval that
    reaches the bound.
    """

    x = np.sort(np.asarray(samples, dtype=float).ravel())

    n = x.size

    if n < 2:
        raise ValueError("Need at least two samples.")

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1.")

    k = max(1, int(np.ceil(level * n)) - 1)

    widths = x[k:] - x[: n - k]

    start = int(np.argmin(widths))

    return float(x[start]), float(x[start + k])


def one_sided_limit(samples, level: float = 0.95, side: str = "upper") -> float:
    """
    The value below (``side="upper"``) or above (``"lower"``) which
    ``level`` of the samples lie -- the way to quote a parameter the
    data only bound from one side.
    """

    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'.")

    q = level if side == "upper" else 1.0 - level

    return float(np.quantile(np.asarray(samples, dtype=float), q))


def auto_burnin(sampler, factor: float = BURNIN_TAU) -> int:
    """
    A burn-in of ``factor`` times the largest autocorrelation time,
    measured on the second half of the chain (where the walkers have
    the best chance of having forgotten their start), and never more
    than half the chain.
    """

    n_steps = int(sampler.iteration)

    if n_steps < 4:
        return 0

    tau = np.asarray(
        sampler.get_autocorr_time(discard=n_steps // 2, quiet=True),
        dtype=float,
    )

    tau = tau[np.isfinite(tau)]

    if tau.size == 0:
        return n_steps // 2

    return int(min(np.ceil(factor * tau.max()), n_steps // 2))


def posterior_summary(flat, names, interval: str = "equal-tailed") -> dict:
    """
    Median and a 68% interval per parameter.

    Parameters
    ----------
    flat : ndarray, shape (n_samples, ndim)

    names : list of str

    interval : {"equal-tailed", "hpd"}
        ``"equal-tailed"`` (default): the 16th and 84th percentiles.
        ``"hpd"``: the shortest 68% interval.

    Returns
    -------
    dict
        ``{name: {"median", "plus", "minus", "low", "high"}}``, with
        ``plus``/``minus`` the distances from the median to ``high``/
        ``low``.
    """

    if interval not in ("equal-tailed", "hpd"):
        raise ValueError("interval must be 'equal-tailed' or 'hpd'.")

    flat = np.asarray(flat, dtype=float)

    if flat.shape[0] == 0:
        raise ValueError(
            "No samples left after burn-in: the burn-in is at least "
            "as long as the chain."
        )

    result = {}

    for i, name in enumerate(names):

        column = flat[:, i]

        median = float(np.median(column))

        if interval == "hpd":
            low, high = hpd_interval(column, 0.68)
        else:
            low, high = (float(v) for v in np.percentile(column, [16, 84]))

        result[name] = {
            "median": median,
            "plus": high - median,
            "minus": median - low,
            "low": low,
            "high": high,
        }

    return result
