"""
Cosmological models.

This subpackage contains concrete implementations of the
:class:`~cosmology.core.base.Cosmology` base class.
"""

from .lcdm import LCDM
from .cpl import CPL

__all__ = [
    "LCDM",
    "CPL",
]
