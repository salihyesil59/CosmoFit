"""
Dataset containers.

This module defines lightweight dataclasses used throughout the
project to store observational datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.covariance import CovarianceBase


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

    covariance: "CovarianceBase | None" = None

    reference: str = ""


    def __post_init__(
        self,
    ):

        n = len(

            self.z,

        )

        if not (

            len(self.H)

            == n

            == len(self.sigma)

        ):

            raise ValueError(

                "CC arrays have inconsistent lengths.",

            )

        if (

            self.covariance is not None

            and self.covariance.shape != (

                n,

                n,

            )

        ):

            raise ValueError(

                "CC covariance has wrong shape.",

            )

# -----------------------------------------------------------
    
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

# -----------------------------------------------------------

    @property
    def has_covariance(
        self,
    ) -> bool:
        """
        Whether the dataset has a covariance matrix.
        """

        return self.covariance is not None


# ============================================================
# DESI BAO
# ============================================================

@dataclass(slots=True)
class DESIDataset:
    """
    DESI BAO measurements.
    """

    z: np.ndarray

    value: np.ndarray

    observable: np.ndarray

    covariance: "CovarianceBase"

    reference: str = ""

# -----------------------------------------------------------

    def __post_init__(
        self,
    ):

        n = len(

            self.z,

        )

        if not (

            len(self.observable)

            == n

            == len(self.value)

        ):

            raise ValueError(

                "DESI arrays have inconsistent lengths.",

            )

        if self.covariance.shape != (

            n,

            n,

        ):

            raise ValueError(

                "DESI covariance has wrong shape.",

            )

# -----------------------------------------------------------

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

    Three redshifts are kept per supernova, matching the official
    Pantheon+SH0ES data release columns:

    z_hd
        Hubble-diagram redshift (CMB-frame, corrected for the
        local peculiar-velocity/bulk-flow model). This is the
        redshift the *cosmological* comoving/transverse distance
        should be evaluated at.
    z_cmb
        CMB-frame redshift, without the peculiar-velocity
        correction applied to get ``z_hd``. Kept for reference;
        not used by :class:`likelihoods.pantheon.PantheonLikelihood`.
    z_hel
        Heliocentric redshift. This is the redshift that sets the
        observed time/flux dilation of the light curve, so it is
        what the ``(1 + z)`` factor in the luminosity distance
        D_L = (1 + z_hel) * D_M(z_hd) should use -- *not* z_hd or
        z_cmb (Brout et al. 2022, Pantheon+SH0ES; see also the
        official analysis code's ``dl_at_zhel_zhd`` convention).
    """

    z_hd: np.ndarray

    z_cmb: np.ndarray

    z_hel: np.ndarray

    m_b_corr: np.ndarray

    covariance: "CovarianceBase"

    cepheid: Optional[np.ndarray] = None

    reference: str = ""

# -----------------------------------------------------------

    def __post_init__(
    self,
    ):

        n = len(

            self.z_cmb,

        )

        if not (

            len(self.z_hd)

            == n

            == len(self.z_hel)

            == len(self.m_b_corr)

        ):

            raise ValueError(

                "Pantheon arrays have inconsistent lengths.",

            )

        if self.covariance.shape != (

            n,

            n,

        ):

            raise ValueError(

                "Pantheon covariance has wrong shape.",

            )

        if (

            self.cepheid is not None

            and len(self.cepheid) != n

        ):

            raise ValueError(

                "Cepheid mask has wrong length.",

            )

    # -----------------------------------------------------------

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

    # -----------------------------------------------------------

    @property
    def has_cepheid(
        self,
    ) -> bool:
        """
        Whether the dataset contains Cepheid calibration flags.
        """

        return self.cepheid is not None


# ============================================================
# Planck Distance Prior
# ============================================================

@dataclass(slots=True)
class PlanckDataset:
    """
    Planck distance-prior measurements.
    """

    values: np.ndarray

    covariance: "CovarianceBase"

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