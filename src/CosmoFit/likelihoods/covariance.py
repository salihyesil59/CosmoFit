"""
Covariance matrix utilities.

Re-exported from :mod:`data.covariance`. The implementation lives
in the ``data`` package (rather than here) so that ``data.loader``
-- which needs :func:`make_covariance` to build dataset covariance
objects -- does not have to import the ``likelihoods`` package to
get it. ``likelihoods/__init__.py`` eagerly imports the concrete
likelihood classes (``CCLikelihood``, ``PantheonLikelihood``, ...),
which themselves import ``data.loader``; if ``data.loader`` also
imported from ``likelihoods``, that would be a circular import
(reproducible any time ``data.loader`` -- or a module that imports
it before ``likelihoods`` does -- is the first thing imported).

This module is kept so existing ``from .covariance import ...`` /
``from CosmoFit.likelihoods.covariance import ...`` imports keep working
unchanged.
"""

from __future__ import annotations

from CosmoFit.data.covariance import (
    CovarianceBase,
    DenseCovariance,
    DiagonalCovariance,
    PrecisionCovariance,
    make_covariance,
)

__all__ = [
    "CovarianceBase",
    "DenseCovariance",
    "DiagonalCovariance",
    "PrecisionCovariance",
    "make_covariance",
]
