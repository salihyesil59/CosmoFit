"""
Native theories of the 2.0 core.

``background``
    :class:`~theories.background.Background` -- the expansion history,
    radiation and massive neutrinos included, with any dark sector from
    :data:`~theories.sectors.DARK_SECTORS`.
``early_universe``
    :class:`~theories.early.EarlyUniverse` -- the sound horizon at the
    drag epoch and at recombination, from the background's own
    expansion.

Named in an input's ``theory`` block by these names.

Dark sectors, by family:

* dark energy on top of GR (:mod:`theories.dark_energy`): ``lambda``,
  ``wcdm``, ``cpl``, ``jbp``, ``ba``, ``logarithmic``, ``pede``,
  ``gede``, ``lscdm``, ``gcg``, ``ide``;
* modified Friedmann equations (:mod:`theories.modified_gravity`):
  ``dgp``, ``cardassian``, ``fq_exponential``, ``ft_power_law``,
  ``frt_linear``;
* the holographic family (:mod:`theories.holographic`): ``hde``,
  ``ade``, ``rde``;
* the running vacuum (:mod:`theories.running_vacuum`): ``rvm``;
* any old-style model, without radiation
  (:mod:`theories.legacy_expansion`): ``legacy``.
"""

from .background import Background, neutrino_density, neutrino_pressure
from .dark_energy import DARK_ENERGY, DarkEnergy, get_dark_energy
from .dark_sector import DarkSector, ExpansionContext
from .early import EarlyUniverse
from .sectors import DARK_SECTORS, get_dark_sector

__all__ = [
    "Background",
    "DARK_ENERGY",
    "DARK_SECTORS",
    "DarkEnergy",
    "DarkSector",
    "EarlyUniverse",
    "ExpansionContext",
    "get_dark_energy",
    "get_dark_sector",
    "neutrino_density",
    "neutrino_pressure",
]
