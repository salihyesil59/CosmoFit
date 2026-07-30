"""
Dataset containers.

This module defines lightweight dataclasses used throughout the
project to store observational datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# Cosmic Chronometers
# ============================================================

@dataclass(slots=True)
class CCDataset:
    """
    Cosmic Chronometer measurements.
    """

    z: np.ndarray

    H: np.ndarray

    sigma: np.ndarray

    covariance: np.ndarray | None = None

    reference: str = ""

    
    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z,

        )


# ============================================================
# DESI BAO
# ============================================================

@dataclass(slots=True)
class DESIDataset:
    """
    DESI BAO measurements.
    """

    z: np.ndarray

    observable: np.ndarray

    observable_type: np.ndarray

    covariance: np.ndarray

    reference: str = ""


    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z,

        )


# ============================================================
# Pantheon+
# ============================================================

@dataclass(slots=True)
class PantheonDataset:
    """
    Pantheon+SH0ES Supernova sample.
    """

    z_cmb: np.ndarray

    z_hel: np.ndarray

    mu: np.ndarray

    covariance: np.ndarray

    cepheid: Optional[np.ndarray] = None

    reference: str = ""


    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z_cmb,

        )


# ============================================================
# Planck Distance Prior
# ============================================================

@dataclass(slots=True)
class PlanckDataset:
    """
    Planck distance-prior measurements.
    """

    values: np.ndarray

    covariance: np.ndarray

    labels: tuple[str, ...] = (
        "R",
        "lA",
        "omega_b_h2",
    )

    reference: str = ""


    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.values,

        )


# ============================================================
# Generic container (optional)
# ============================================================

@dataclass(slots=True)
class GenericDataset:
    """
    Generic dataset container.
    """

    name: str

    data: dict