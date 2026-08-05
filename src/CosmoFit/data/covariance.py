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

    @abstractmethod
    def solve(self, vector):
        """
        Return C^-1 @ vector.

        Needed (in addition to chi2) whenever a likelihood
        wants to analytically marginalize over a linear
        nuisance parameter (e.g. the SN absolute magnitude).
        """

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
    def sigma(self):
        """
        Per-point uncertainty, i.e. sqrt of the diagonal.

        Off-diagonal (correlation) information is of course lost,
        so this is only a marginal 1D uncertainty -- but it is
        the standard error-bar convention for plotting individual
        data points from a dataset whose likelihood otherwise
        uses the full covariance matrix (e.g. Hubble-diagram or
        H(z) figures for a dataset with correlated systematics).
        """

        return np.sqrt(np.diag(self.matrix))

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
# Precision (inverse covariance) matrix
# ============================================================

class PrecisionCovariance(CovarianceBase):
    """
    Covariance defined by a precomputed *precision* (inverse
    covariance) matrix.

    Some public data releases (e.g. DES-SN5YR) ship the precision
    matrix directly rather than the covariance itself -- using it
    as-is avoids inverting it (to get a covariance for
    ``DenseCovariance``) only to have every ``solve()`` call
    effectively invert it back via a Cholesky solve; a single
    matrix-vector product against the precision matrix is both
    cheaper and avoids that redundant, numerically lossy round
    trip.

    The covariance matrix itself (``.matrix``, ``.sigma``, ...) is
    only computed lazily, on first access -- likelihood evaluation
    (``chi2``/``solve``) never needs it.
    """

    def __init__(self, precision):

        self.precision = np.asarray(
            precision,
            dtype=float,
        )

        super().__init__(
            self.precision.shape[0],
        )

        sign, logdet_precision = np.linalg.slogdet(
            self.precision,
        )

        if sign <= 0:

            raise ValueError(
                "Precision matrix must be positive definite."
            )

        # ln|C| = -ln|C^-1|
        self._logdet = -logdet_precision

        self._matrix = None

    # ---------------------------------------------------------

    @property
    def matrix(self):
        """
        The covariance matrix itself, computed lazily (and cached)
        by inverting the precision matrix. Only needed for
        diagnostics/plotting (``sigma``, ``correlation``, ...),
        never for ``solve()``/``chi2()``.
        """

        if self._matrix is None:

            self._matrix = np.linalg.inv(self.precision)

        return self._matrix

    # ---------------------------------------------------------

    @property
    def logdet(self):

        return self._logdet

    # ---------------------------------------------------------

    @property
    def sigma(self):
        """
        Per-point uncertainty, i.e. sqrt of the diagonal of the
        covariance matrix. See :attr:`DenseCovariance.sigma`.
        """

        return np.sqrt(np.diag(self.matrix))

    # ---------------------------------------------------------

    @property
    def is_positive_definite(self) -> bool:

        try:

            np.linalg.cholesky(self.precision)

            return True

        except np.linalg.LinAlgError:

            return False

    # ---------------------------------------------------------

    @property
    def shape(self):

        return self.precision.shape

    # ---------------------------------------------------------

    def solve(self, vector):
        """
        Return C^-1 @ vector -- just the stored precision matrix
        applied directly, no factorization/inversion needed.
        """

        vector = np.asarray(
            vector,
            dtype=float,
        )

        return self.precision @ vector

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

    def solve(self, vector):

        vector = np.asarray(
            vector,
            dtype=float,
        )

        return vector * self.inv_var

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
    precision=None,
):
    """
    Construct the appropriate covariance object.

    Parameters
    ----------
    cov : ndarray, optional
        Full covariance matrix.

    sigma : ndarray, optional
        Measurement uncertainties.

    precision : ndarray, optional
        Precision (inverse covariance) matrix, for datasets that
        ship it directly (e.g. DES-SN5YR) -- see
        :class:`PrecisionCovariance`.

    Returns
    -------
    CovarianceBase
        Covariance handler.
    """

    given = [x is not None for x in (cov, sigma, precision)]

    if sum(given) != 1:

        raise ValueError(
            "Specify exactly one of "
            "'cov', 'sigma', or 'precision'."
        )

    if cov is not None:

        return DenseCovariance(cov)

    if precision is not None:

        return PrecisionCovariance(precision)

    return DiagonalCovariance(sigma)