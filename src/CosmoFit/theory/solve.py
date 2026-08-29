"""
Solving a Friedmann constraint for ``E(z)``.

:mod:`~theory.action` produces an expression ``C(E2, z; params)``
that vanishes on-shell. Turning that into the ``E(z)`` every
distance, likelihood and plot in CosmoFit calls means solving it,
and the constraint of an interesting model is rarely a polynomial
-- the exponential ``f(Q)`` model's is transcendental, and its
hand-written counterpart in this library needs a Lambert ``W`` to
invert. So there are two paths here:

* a **closed form**, when sympy can solve the constraint for
  ``E2`` and returns exactly one branch. Vectorized and exact.

* a **continuation solve** otherwise. Newton's method on the whole
  redshift array at once, walked out from ``z = 0`` -- where the
  closure condition guarantees ``E2 = 1`` -- along a ladder of
  intermediate redshifts ``s*z``, each step seeded by the last.

The continuation is what makes branch selection well-posed. A
transcendental constraint generally has several roots, and a
root-finder handed the whole redshift range at once can land on a
different branch at different ``z`` and return a discontinuous,
physically meaningless ``E(z)`` without failing. Starting from the
known root at ``z = 0`` and never letting the solution jump
follows one branch by construction -- the same reason
``FQExponential`` has to pick a specific Lambert ``W`` branch
rather than any root of its equation.

The derivative ``dE/dz`` is never finite-differenced: implicit
differentiation of the constraint gives

    dE2/dz = -(dC/dz) / (dC/dE2)

exactly, from expressions sympy already has.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


#: Newton iterations per continuation step.
_NEWTON_STEPS = 12

#: Steps on the continuation ladder from z = 0 to the target z.
_CONTINUATION_STEPS = 6

#: Residual (relative to the constraint's own scale) below which
#: a Newton solve counts as converged.
_TOLERANCE = 1.0e-12


# ============================================================
# Compiling
# ============================================================

def compile_constraint(C: sp.Expr, E2: sp.Symbol, z: sp.Symbol, args):
    """
    Lambdify a constraint and its two partial derivatives.

    Only the *numerator* of the constraint is used. A constraint
    assembled from a Lagrangian routinely carries a denominator in
    ``a`` (equivalently ``1 + z``) which never vanishes on the
    physical domain, and clearing it keeps Newton's method away
    from spurious poles.

    Parameters
    ----------
    args : sequence of Symbol
        Parameter symbols, in the order the returned callables
        expect them after ``(E2, z)``.

    Returns
    -------
    (value_and_slope, df_dz) : tuple of callables
        ``value_and_slope(E2, z, *params)`` returns the pair
        ``[C, dC/dE2]``; ``df_dz(E2, z, *params)`` returns
        ``dC/dz``.
    """

    numerator = sp.numer(sp.together(sp.cancel(C)))

    signature = (E2, z, *args)

    # The constraint and its E2-derivative share almost all of
    # their work (a transcendental f leaves the same exponential or
    # power in both), so they are compiled as one callable with
    # common subexpressions factored out rather than as two that
    # each recompute it.
    value_and_slope = sp.lambdify(
        signature,
        [numerator, sp.diff(numerator, E2)],
        "numpy",
        cse=True,
    )

    return (
        value_and_slope,
        sp.lambdify(signature, sp.diff(numerator, z), "numpy"),
    )


# ------------------------------------------------------------

def closed_form(C: sp.Expr, E2: sp.Symbol, z: sp.Symbol, args):
    """
    Try to solve the constraint for ``E2`` symbolically.

    Returns a lambdified ``E2(z, *params)`` if sympy finds
    exactly one solution, else ``None`` -- more than one branch is
    treated as "no closed form" rather than resolved by guesswork,
    and left to the continuation solve, which picks the branch
    physically.
    """

    try:
        solutions = sp.solve(sp.numer(sp.together(C)), E2, dict=False)

    except (NotImplementedError, sp.PolynomialError, TypeError):
        return None

    solutions = [s for s in solutions if not s.has(sp.I)]

    if len(solutions) != 1:
        return None

    return sp.lambdify((z, *args), sp.simplify(solutions[0]), "numpy")


# ============================================================
# Numerical solve
# ============================================================

def solve_E2(
    z,
    functions,
    values,
    *,
    seed=None,
):
    """
    Solve ``C(E2, z) = 0`` for ``E2`` on a whole redshift array.

    Walks out from ``z = 0``, where ``E2 = 1`` by the closure
    condition, along ``_CONTINUATION_STEPS`` intermediate
    redshifts, running Newton's method at each. Every point of the
    array is advanced together, so this costs a fixed number of
    vectorized passes rather than a root-find per redshift.

    Parameters
    ----------
    z : ndarray
        Redshifts. May be in any order, and may include negative
        values (the future), which the continuation handles the
        same way.

    functions : tuple
        ``(value_and_slope, df_dz)`` from
        :func:`compile_constraint`.

    values : tuple
        Parameter values, matching the order given to
        :func:`compile_constraint`.

    seed : ndarray, optional
        Starting guess. When given, the continuation ladder is
        skipped and Newton runs directly, on the understanding
        that the seed is already near the right branch -- the
        residual is checked either way, and a failed direct solve
        falls back to the full ladder. Defaults to ``None``, i.e.
        walking out from ``E2 = 1`` at ``z = 0``.

    Returns
    -------
    ndarray
        ``E2`` at each redshift.

    Raises
    ------
    RuntimeError
        If Newton fails to converge, rather than returning a
        number that merely looks like a solution.
    """

    value_and_slope, _ = functions

    z = np.asarray(z, dtype=float)

    if seed is not None:

        E2 = _newton(value_and_slope, np.array(seed, dtype=float), z, values)

        if _converged(value_and_slope, E2, z, values):
            return E2

    E2 = np.ones_like(z)

    for step in range(1, _CONTINUATION_STEPS + 1):

        E2 = _newton(
            value_and_slope, E2, z * (step / _CONTINUATION_STEPS), values,
        )

    if not _converged(value_and_slope, E2, z, values):

        bad = ~np.isfinite(E2) | (E2 <= 0.0)

        raise RuntimeError(
            f"Could not solve the Friedmann constraint over "
            f"z = [{z.min():.4g}, {z.max():.4g}]"
            + (f" ({int(np.sum(bad))} non-physical root(s))" if np.any(bad) else "")
            + ". The parameter values may be outside this model's "
            "physical domain -- an f(T) or f(Q) model can have no "
            "real E(z) at all for some parameters, which is a "
            "statement about the model rather than a bug. Narrow "
            "the prior bounds."
        )

    return E2


# ------------------------------------------------------------

def _newton(value_and_slope, E2, z, values):
    """
    Newton's method on ``E2``, all redshifts advanced together,
    stopping as soon as every point has stopped moving.
    """

    for _ in range(_NEWTON_STEPS):

        residual, slope = value_and_slope(E2, z, *values)

        residual = np.asarray(residual, dtype=float)
        slope = np.asarray(slope, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(slope != 0.0, residual / slope, 0.0)

        # A step that would drive E2 non-positive has left the
        # physical branch; halve back onto it rather than let a
        # negative E2 turn into a NaN in sqrt() much later.
        new = E2 - delta
        new = np.where(np.isfinite(new) & (new > 0.0), new, 0.5 * E2)

        moved = np.abs(new - E2) > _TOLERANCE * np.abs(E2)

        E2 = new

        if not np.any(moved):
            break

    return E2


# ------------------------------------------------------------

def _converged(value_and_slope, E2, z, values) -> bool:
    """
    Whether ``E2`` really solves the constraint -- physical
    (finite and positive) and with a residual small against the
    constraint's own local scale, so the test means the same thing
    however the action happens to be normalized.
    """

    if not np.all(np.isfinite(E2)) or np.any(E2 <= 0.0):
        return False

    residual, slope = value_and_slope(E2, z, *values)

    scale = np.maximum(
        np.abs(np.asarray(slope, dtype=float)) * np.abs(E2), 1.0e-300,
    )

    return bool(
        np.all(np.abs(np.asarray(residual, dtype=float)) <= 1.0e-9 * scale)
    )


    return E2


# ------------------------------------------------------------

def dE2_dz(z, E2, functions, values):
    """
    ``dE2/dz`` by implicit differentiation of the constraint --
    exact, given ``E2`` already solved, and free of the truncation
    error a finite difference would carry into
    ``background.q()``.
    """

    value_and_slope, df_dz = functions

    z = np.asarray(z, dtype=float)

    _, slope = value_and_slope(E2, z, *values)

    return (
        -np.asarray(df_dz(E2, z, *values), dtype=float)
        / np.asarray(slope, dtype=float)
    )
