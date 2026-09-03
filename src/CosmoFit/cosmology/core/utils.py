"""
Small shared helpers used across the ``cosmology`` package.
"""

from __future__ import annotations

import math

import numpy as np


def require_positive(value: float, name: str) -> float:
    """
    Return ``value`` unchanged if it is strictly positive and
    finite, otherwise raise a ``ValueError`` naming the offending
    parameter.

    Used by calculators (e.g. the recombination fitting formulas)
    that take a logarithm or a square root of a physical density
    and would otherwise fail with an opaque NumPy warning/error
    when handed a non-physical (e.g. negative or zero) parameter
    combination.
    """

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"'{name}' must be a positive, finite number; "
            f"got {value!r}."
        )

    return value


def coupling_from_derivative(f_prime, model="this model"):
    r"""
    ``mu = G_eff/G_N = 1 / f'`` for the teleparallel and
    symmetric-teleparallel sectors, with the two ways that
    expression stops meaning anything refused rather than returned.

    In ``f(T)`` and ``f(Q)`` gravity the sub-horizon quasi-static
    coupling is the reciprocal of the derivative of ``f`` with
    respect to the geometry scalar. Two things can go wrong, and
    neither announces itself:

    * ``f' = 0`` is a pole. The effective gravitational constant
      diverges, and either side of it ``mu`` is finite -- so a
      sampler stepping across it gets a perfectly ordinary-looking
      number computed on the far side of a singularity.
    * ``f' < 0`` makes ``mu`` negative, which is *repulsive*
      gravity. The growth equation happily integrates it and
      returns a smooth, plausible ``fsigma8`` that means nothing.

    Measured before this existed: ``FTPowerLaw`` at ``n = 0.6``
    returned ``mu = -0.91, -1.79, -4.88, +5.08`` across
    ``z = 0, 0.5, 1, 2`` -- negative, then through the pole, and
    silent throughout.

    ``f' > 0`` is the standard viability condition for these
    theories, the counterpart of ``f_R > 0`` in the metric sector
    (see :func:`theory.curvature.viability_failures`). Raising is
    the right response rather than returning ``nan``: a fitter
    treats the exception as a rejected point, which is exactly what
    an unphysical region of parameter space deserves.
    """

    f_prime = np.asarray(f_prime, dtype=float)

    finite = np.isfinite(f_prime)

    # Reported separately, because they are different faults and the
    # numbers do not describe each other. Quoting `nanmin` when the
    # real problem is a NaN names a perfectly good value as the
    # culprit -- which an early version of this did.
    if not np.all(finite):

        n_bad = int(np.count_nonzero(~finite))

        raise ValueError(
            f"{model} gives a non-finite f' at {n_bad} of "
            f"{f_prime.size} points in the requested range, so "
            f"G_eff/G_N = 1/f' is not defined there. This usually "
            f"means the background itself has left the branch the "
            f"model is valid on."
        )

    if np.any(f_prime <= 0.0):

        worst = float(np.min(f_prime))

        raise ValueError(
            f"{model} has f' = {worst:.6g} somewhere in the "
            f"requested range, so G_eff/G_N = 1/f' is "
            + ("singular" if worst == 0.0 else "negative -- "
               "repulsive gravity")
            + ". Computed anyway it gives finite, plausible growth, "
            "so it is not returned. f' > 0 is the viability "
            "condition for this sector."
        )

    return 1.0 / f_prime


#: Why a teleparallel or symmetric-teleparallel model is rejected.
#: One condition, because there is only one: ``mu = 1/f'``, so
#: everything about whether the coupling means anything is decided
#: by the sign of ``f'``.
TELEPARALLEL_CONDITIONS = {
    "f_prime": (
        "f' > 0 -- the effective gravitational coupling is 1/f'. "
        "Where f' passes through zero the coupling diverges; where "
        "it is negative gravity is repulsive, and the growth "
        "equation integrates that into a perfectly smooth fsigma8. "
        "This is the counterpart of f_R > 0 in the metric sector."
    ),
}


def teleparallel_failures(f_prime):
    """
    Which of :data:`TELEPARALLEL_CONDITIONS` a model violates.

    Returns the failing keys -- empty when the model is admissible
    over the range sampled. The companion of
    :func:`theory.curvature.viability_failures`, which does the same
    job for the metric sector.

    Reporting and refusing are kept apart on purpose.
    :func:`coupling_from_derivative` raises, because a caller asking
    for ``mu`` wants a number and must not be given a meaningless
    one. This returns a verdict instead, because a caller asking
    whether a model is viable wants an answer rather than an
    exception -- and wants it for the whole range at once.
    """

    f_prime = np.asarray(f_prime, dtype=float)

    bad = np.any(~np.isfinite(f_prime)) or np.any(f_prime <= 0.0)

    return ["f_prime"] if bad else []


#: The scalaron amplitude today that Solar System tests allow, from
#: the thin-shell condition with the Galactic Newtonian potential
#: (Hu & Sawicki 2007, arXiv:0705.1158). Bounds quoted in the
#: literature for weaker environments are looser -- around 1e-5 from
#: galaxies and 1e-4 from clusters -- so this is the strictest of
#: the family and the one a model has to survive to be viable
#: everywhere.
SOLAR_SYSTEM_BOUND = 1.0e-6


def screening_margin(f_R_today, bound=SOLAR_SYSTEM_BOUND):
    """
    How the scalaron's amplitude today compares with what local
    tests allow.

    ``|f_R - 1|`` is the fractional departure of the gravitational
    coupling, and unscreened it would show up as a fifth force. The
    standard cosmological proxy for "the Sun and the Earth are
    screened" is that this number is smaller than the Galactic
    potential, ~1e-6.

    **This is an observational exclusion, not a sickness.** A model
    failing here is internally consistent -- no ghost, no tachyon --
    and simply already ruled out, which is a different statement
    from the conditions in :func:`viability_failures` and is
    reported separately for that reason.

    **And it is the linear estimate.** Chameleon screening is
    non-linear: a model can suppress the fifth force locally far
    better than ``|f_R - 1|`` suggests, and settling that needs the
    thin-shell calculation in the actual environment rather than one
    cosmological number. So a failure here means "excluded unless
    screening rescues it", which is how the literature treats the
    same quantity -- not a proof.

    Returns ``(deviation, ok)``.
    """

    deviation = float(np.max(np.abs(np.asarray(f_R_today, dtype=float) - 1.0)))

    return deviation, deviation <= bound
