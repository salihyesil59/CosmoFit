"""
Pantheon+ / Pantheon+SH0ES Supernova likelihood.
"""

from __future__ import annotations

import numpy as np

from data.loader import load_pantheon

from .base import BaseLikelihood


class PantheonLikelihood(BaseLikelihood):
    """
    Pantheon+ / Pantheon+SH0ES Supernova likelihood.

    Parameters
    ----------
    cosmology
        Cosmology model instance (LCDM, CPL, ...).

    version
        Pantheon dataset version.

    include_cepheid
        If True, include Cepheid calibrator supernovae.

    marginalize_MB : bool, optional
        If True (default), the SN absolute magnitude (and,
        equivalently, the H0 - M_B degeneracy) is marginalized
        over analytically instead of being fit as an explicit
        nuisance parameter:

            chi2 = A - B^2 / C

            A = delta^T C^-1 delta
            B = 1^T C^-1 delta
            C = 1^T C^-1 1

        where delta = m_b_corr - mu_model(z). This is the
        standard approach for SN-only / SN+BAO+CC analyses that
        do not use a Cepheid host-distance calibration to break
        the H0-M_B degeneracy, and it is what the CPL_MCMC
        notebook this library reproduces uses.

        If False, ``cosmology.MB`` is added to the model as an
        explicit free/fixed nuisance parameter instead
        (m_B = mu(z) + M_B), useful when calibrating H0 with
        Cepheid distances.
    """

    def __init__(
        self,
        cosmology,
        version: str = "pantheon+sh0es",
        include_cepheid: bool = False,
        marginalize_MB: bool = True,
    ):

        dataset = load_pantheon(
            version=version,
            include_cepheid=include_cepheid,
        )

        self.marginalize_MB = marginalize_MB

        super().__init__(
            name="Pantheon+",
            dataset=dataset,
            cosmology=cosmology,
        )

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted apparent magnitude.

        If ``marginalize_MB`` is True, this returns the
        distance-modulus-only prediction (the best-fit additive
        offset is added back in by :meth:`best_fit_offset`, not
        here) -- this keeps ``residuals()`` consistent with what
        ``chi2()`` actually marginalizes over.

        m_B = mu(z)                 [marginalize_MB=True]
        m_B = mu(z) + M_B           [marginalize_MB=False]
        """

        mu_model = self.cosmology.distance.mu(
            self.data.z_cmb,
        )

        if self.marginalize_MB:
            return mu_model

        return mu_model + self.cosmology.MB

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Data minus model residuals.
        """

        return (

            self.data.m_b_corr

            -

            self.model()

        )

    # ---------------------------------------------------------

    def _marginalized_terms(self):
        """
        Compute the A, B, C terms of the analytic marginalization
        over a single additive nuisance parameter, plus the
        resulting chi2 and best-fit offset.
        """

        delta = self.residuals()
        ones = np.ones_like(delta)

        Cinv_delta = self.covariance.solve(delta)
        Cinv_ones = self.covariance.solve(ones)

        A = float(delta @ Cinv_delta)
        B = float(ones @ Cinv_delta)
        C = float(ones @ Cinv_ones)

        chi2 = A - (B**2) / C
        best_offset = B / C

        return A, B, C, chi2, best_offset

    # ---------------------------------------------------------

    def best_fit_offset(self) -> float:
        """
        Best-fit additive offset (effectively M_B, up to the
        H0 normalization baked into ``mu()``) that analytic
        marginalization would assign, given the current
        cosmology.

        Only meaningful when ``marginalize_MB`` is True.
        """

        _, _, _, _, best_offset = self._marginalized_terms()

        return best_offset

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        if self.marginalize_MB:

            _, _, _, chi2, _ = self._marginalized_terms()

            return float(chi2)

        return self.covariance.chi2(

            self.residuals(),

        )
