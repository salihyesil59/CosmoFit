"""
Small shared helpers used across the ``cosmology`` package.
"""

from __future__ import annotations

import math


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
