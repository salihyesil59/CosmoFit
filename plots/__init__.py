"""
Plotting utilities for CosmoFit.

See :class:`~plots.plotter.FitPlotter`, attached to every
:class:`~stats.fitter.Fitter` as ``fitter.plots``.
"""

from .plotter import FitPlotter

__all__ = [
    "FitPlotter",
]
