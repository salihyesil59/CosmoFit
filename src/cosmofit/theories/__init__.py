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

``growth``
    :class:`~theories.growth.Growth` -- the linear growth factor, growth
    rate and ``f sigma8``, on the background's expansion and through the
    dark sector's growth hooks.

``camb``
    :class:`~theories.boltzmann.CAMB` -- CMB spectra and ``sigma8`` from
    CAMB, for dark sectors it can represent
    (:func:`~theories.boltzmann.cmb_support`).

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
from .boltzmann import CAMB, cmb_support
from .dark_energy import DARK_ENERGY, DarkEnergy, HuSawicki, get_dark_energy
from .dark_sector import DarkSector, ExpansionContext, GrowthContext
from .early import EarlyUniverse
from .growth import Growth
from .sectors import DARK_SECTORS, get_dark_sector

__all__ = [
    "Background",
    "CAMB",
    "DARK_ENERGY",
    "DARK_SECTORS",
    "DarkEnergy",
    "DarkSector",
    "EarlyUniverse",
    "ExpansionContext",
    "Growth",
    "GrowthContext",
    "HuSawicki",
    "cmb_support",
    "get_dark_energy",
    "get_dark_sector",
    "neutrino_density",
    "neutrino_pressure",
]
