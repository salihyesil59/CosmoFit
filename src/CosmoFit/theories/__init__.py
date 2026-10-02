"""
Native theories of the 2.0 core.

``background``
    :class:`~theories.background.Background` -- the expansion history,
    radiation and massive neutrinos included, with a dark energy from
    :mod:`theories.dark_energy`.
``early_universe``
    :class:`~theories.early.EarlyUniverse` -- the sound horizon at the
    drag epoch and at recombination, from the background's own
    expansion.

Named in an input's ``theory`` block by these names.
"""

from .background import Background, neutrino_density
from .dark_energy import DARK_ENERGY, DarkEnergy, get_dark_energy
from .early import EarlyUniverse

__all__ = [
    "Background",
    "DARK_ENERGY",
    "DarkEnergy",
    "EarlyUniverse",
    "get_dark_energy",
    "neutrino_density",
]
