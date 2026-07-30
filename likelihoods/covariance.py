"""
Covariance matrix utilities.

Provides a common interface for both full covariance matrices
and diagonal uncertainties.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

from scipy.linalg import cho_factor
from scipy.linalg import cho_solve


# ============================================================
# Base class
# ============================================================

class CovarianceBase(ABC):
    """
    Abstract covariance class.
    """

    def __init__(self, n):

        self.n = int(n)

    # ---------------------------------------------------------

    @abstractmethod
    def chi2(self, residual):

        pass

    # ---------------------------------------------------------

    @property
    def size(
        self,
    ) -> int:
        """
        Matrix dimension.
        """

        return self.n

    # --------------------------------------------------------

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(n={self.n})"
        )

    # --------------------------------------------------------

    def __str__(
        self,
    ) -> str:
        """
        Human-readable representation.
        """

        return (
            f"{self.__class__.__name__}"
            f"(n={self.size})"
        )

    __repr__ = __str__


# ============================================================
# Full covariance
# ============================================================

class DenseCovariance(CovarianceBase):
    """
    Full covariance matrix using a Cholesky decomposition.
    """

    def __init__(self, matrix):

        self.matrix = np.asarray(
            matrix,
            dtype=float,
        )

        super().__init__(
            self.matrix.shape[0],
        )

        self._factor = cho_factor(
            self.matrix,
            lower=True,
            check_finite=False,
        )

        sign, logdet = np.linalg.slogdet(
            self.matrix,
        )

        if sign <= 0:

            raise ValueError(
                "Covariance matrix must be positive definite."
            )

        self._logdet = logdet

    # ---------------------------------------------------------

    @property
    def logdet(self):

        return self._logdet

    # ---------------------------------------------------------

    @property
    def condition_number(
        self,
    ) -> float:
        """
        Matrix condition number.
        """

        return np.linalg.cond(

            self.matrix,

        )

    # --------------------------------------------------------

    @property
    def is_positive_definite(
        self,
    ) -> bool:
        """
        Check whether the covariance matrix is positive definite.
        """

        try:

            np.linalg.cholesky(

                self.matrix,

            )

            return True

        except np.linalg.LinAlgError:

            return False
            
    # ---------------------------------------------------------

    @property
    def correlation(self):

        sigma = np.sqrt(

            np.diag(

                self.matrix,

            )

        )

        return self.matrix / np.outer(

            sigma,

            sigma,

        )

    # ---------------------------------------------------------

    @property
    def shape(self):

        return self.matrix.shape

    # ---------------------------------------------------------

    def solve(self, vector):

        return cho_solve(
            self._factor,
            vector,
            check_finite=False,
        )

    # ---------------------------------------------------------

    def chi2(self, residual):

        residual = np.asarray(
            residual,
            dtype=float,
        )

        return float(

            residual

            @

            self.solve(residual)

        )

    # ---------------------------------------------------------

    
# ============================================================
# Diagonal covariance
# ============================================================

class DiagonalCovariance(CovarianceBase):
    """
    Diagonal covariance defined from measurement uncertainties.
    """

    def __init__(self, sigma):

        self.sigma = np.asarray(
            sigma,
            dtype=float,
        )

        super().__init__(
            len(self.sigma),
        )

        self.inv_var = 1.0 / self.sigma**2

        self._logdet = np.sum(
            np.log(
                self.sigma**2,
            )
        )

    # ---------------------------------------------------------

    @property
    def logdet(self):

        return self._logdet

    # ---------------------------------------------------------

    @property
    def is_positive_definite(
        self,
    ) -> bool:
        """
        Diagonal covariance matrices are always positive definite.
        """

        return True

    # ---------------------------------------------------------

    @property
    def condition_number(
        self,
    ) -> float:
        """
        Condition number of the diagonal covariance matrix.
        """

        variance = self.sigma**2

        return float(

            np.max(

                variance,

            )

            /

            np.min(

                variance,

            )

        )

    # ---------------------------------------------------------

    @property
    def shape(
        self,
    ):
        """
        Shape of the equivalent covariance matrix.
        """

        return (

            self.n,

            self.n,

        )

    # ---------------------------------------------------------

    def chi2(self, residual):

        residual = np.asarray(
            residual,
            dtype=float,
        )

        return float(

            np.sum(

                residual**2

                *

                self.inv_var

            )

        )

    
# ============================================================
# Factory
# ============================================================

def make_covariance(
    *,
    cov=None,
    sigma=None,
):
    """
    Construct the appropriate covariance object.

    Parameters
    ----------
    cov : ndarray, optional
        Full covariance matrix.

    sigma : ndarray, optional
        Measurement uncertainties.

    Returns
    -------
    CovarianceBase
        Covariance handler.
    """

    if (cov is None) == (sigma is None):

        raise ValueError(
            "Specify exactly one of "
            "'cov' or 'sigma'."
        )

    if cov is not None:

        return DenseCovariance(cov)

    return DiagonalCovariance(sigma)