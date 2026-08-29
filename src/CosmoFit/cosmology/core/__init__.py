"""
Core cosmology classes and definitions.

This subpackage contains the fundamental building blocks used
throughout CosmoFit, including the base cosmology class,
parameter containers, and physical constants.
"""

from .base import Cosmology
from .errors import ModelConfigurationError
from .parameters import CosmologyParameters
from .constants import constants

__all__ = [
    "Cosmology",
    "ModelConfigurationError",
    "CosmologyParameters",
    "constants",
]
