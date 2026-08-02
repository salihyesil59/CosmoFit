"""
Numerical methods used throughout CosmoFit.

This subpackage contains numerical algorithms such as
integration routines and interpolation methods used by
the cosmological calculators.
"""

from .integrals import DistanceIntegrator

__all__ = [
    "DistanceIntegrator",
]