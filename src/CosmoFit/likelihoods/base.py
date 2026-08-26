"""
Base likelihood class.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

from .covariance import make_covariance


class BaseLikelihood(ABC):

    def __init__(
        self,
        name,
        dataset,
        cosmology,
    ):

        self.name = name

        self.data = dataset

        self.cosmology = cosmology

        covariance = getattr(dataset, "covariance", None)

        if covariance is not None:

            self.covariance = covariance

        elif getattr(dataset, "sigma", None) is not None:

            self.covariance = make_covariance(

                sigma=dataset.sigma,

            )

        else:

            # Not every likelihood is Gaussian. Planck's low-l EE
            # is a tabulated probability with no mean and no
            # covariance to speak of (see
            # :mod:`likelihoods.planck_lowe`), and forcing one on it
            # would mean inventing a summary of exactly the
            # distribution whose shape is the reason the table
            # exists. Such a likelihood must override `chi2` and
            # `log_likelihood` itself.
            self.covariance = None

    # ========================================================
    # Properties
    # ========================================================

    @property
    def n_data(
        self,
    ) -> int:
        """
        Number of data points.
        """

        return self.data.size

    @property
    def name_and_size(
        self,
    ) -> str:
        """
        Name together with the number of data points.
        """

        return f"{self.name} ({self.n_data})"

    # ========================================================
    # Abstract interface
    # ========================================================

    @abstractmethod
    def model(
        self,
    ):
        """
        Return theoretical predictions.
        """

        pass

    @abstractmethod
    def chi2(
        self,
    ) -> float:
        """
        Return chi-square.
        """

        pass

    # ========================================================
    # Common methods
    # ========================================================

    def predictions(
        self,
    ):
        """
        Alias for model().
        """

        return self.model()

    def log_likelihood(
        self,
    ) -> float:
        """
        Return log-likelihood.
        """

        return -0.5 * self.chi2()

    def summary(
        self,
    ) -> dict:
        """
        Return a summary of the likelihood evaluation.
        """

        chi2 = self.chi2()

        return {

            "name": self.name,

            "n_data": self.n_data,

            "chi2": chi2,

            "loglike": self.log_likelihood(),

        }

    # ========================================================
    # Representation
    # ========================================================

    def __str__(
        self,
    ) -> str:
        """
        Human-readable representation.
        """

        return (
            f"{self.__class__.__name__}"
            f"(name='{self.name}', "
            f"n_data={self.n_data})"
        )

    __repr__ = __str__


# ============================================================
# Analytic offset marginalization
# ============================================================

class AnalyticOffsetMixin:
    """
    Mixin for likelihoods that analytically marginalize over a
    single additive nuisance offset -- an SN absolute magnitude
    (:class:`~likelihoods.pantheon.PantheonLikelihood`) or an
    already-standardized distance-modulus zero point
    (:class:`~likelihoods.des_sn5yr.DESSN5YRLikelihood`), both of
    which are fully degenerate with H0 and so cannot be
    constrained by SN data alone:

        chi2 = A - B^2 / C

        A = delta^T C^-1 delta
        B = 1^T C^-1 delta
        C = 1^T C^-1 1

    where ``delta`` is data-minus-model *without* the offset
    applied (see e.g. Conley et al. 2011, arXiv:1104.1443,
    Appendix; this is also exactly what the DES-SN5YR data
    release's own reference likelihood implementation does).

    ``C`` depends only on the (fixed) covariance matrix, never on
    the cosmology, so it -- and the covariance solve it requires
    -- is precomputed once by :meth:`_setup_offset_marginalization`
    rather than on every call to :meth:`chi2`. For a dataset the
    size of Pantheon+ or DES-SN5YR (order-1000x1000 dense
    covariance), that solve is the dominant per-step cost in an
    MCMC that includes it, so precomputing it roughly halves that
    cost (one covariance solve per step instead of two).

    A using class must:

    - Call ``self._setup_offset_marginalization()`` once, after
      ``BaseLikelihood.__init__`` (i.e. once ``self.covariance``
      and ``self.n_data`` exist).
    - Implement ``residuals()`` returning data-minus-model without
      the offset applied.
    """

    def _setup_offset_marginalization(self) -> None:

        ones = np.ones(self.n_data)

        self._Cinv_ones = self.covariance.solve(ones)
        self._C_offset = float(ones @ self._Cinv_ones)

    # ---------------------------------------------------------

    def _marginalized_terms(self):
        """
        Compute the A, B, C terms of the analytic marginalization,
        plus the resulting chi2 and best-fit offset. Only ``A``
        and ``B`` (which depend on ``delta``, i.e. on the current
        cosmology) are recomputed here; ``C`` was cached in
        ``_setup_offset_marginalization``.
        """

        delta = self.residuals()

        Cinv_delta = self.covariance.solve(delta)

        A = float(delta @ Cinv_delta)
        B = float(self._Cinv_ones @ delta)
        C = self._C_offset

        chi2 = A - (B ** 2) / C
        best_offset = B / C

        return A, B, C, chi2, best_offset

    # ---------------------------------------------------------

    def best_fit_offset(self) -> float:
        """
        Best-fit additive offset that analytic marginalization
        would assign, given the current cosmology.
        """

        _, _, _, _, best_offset = self._marginalized_terms()

        return best_offset

    # ---------------------------------------------------------

    def marginalized_chi2(self) -> float:
        """
        Chi-square with the offset analytically marginalized out.
        """

        _, _, _, chi2, _ = self._marginalized_terms()

        return float(chi2)