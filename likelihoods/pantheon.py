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

        if self.marginalize_MB:

            # C^-1 @ 1 and 1^T C^-1 1 depend only on the (fixed)
            # covariance matrix, never on the cosmology -- so they
            # are the same at every MCMC step. Solving against the
            # ~1600x1600 Pantheon+ covariance is the single most
            # expensive operation in a CC+DESI+Pantheon MCMC run
            # (a Cholesky *solve* is O(n^2) and dominates once
            # Pantheon is in the mix); computing it once here
            # instead of on every call to `chi2()` roughly halves
            # that cost (one covariance solve per step instead of
            # two).
            ones = np.ones(self.n_data)
            self._Cinv_ones = self.covariance.solve(ones)
            self._C = float(ones @ self._Cinv_ones)

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

        The distance modulus uses two *different* redshifts, per
        the official Pantheon+SH0ES convention (Brout et al.
        2022): the transverse comoving distance D_M is evaluated
        at the Hubble-diagram redshift ``z_hd`` (CMB-frame,
        peculiar-velocity corrected -- the cosmologically
        meaningful redshift), while the ``(1 + z)`` source-frame
        dilation factor uses the heliocentric redshift ``z_hel``
        (what the light curve itself was actually stretched by):

            D_L = (1 + z_hel) * D_M(z_hd)

        Using a single redshift for both (e.g. z_cmb throughout)
        is a common simplification but introduces a systematic
        bias of order the z_hel/z_hd offset (~1e-3, i.e.
        comparable to or larger than the per-SN distance-modulus
        precision for the best-measured low-z SNe).
        """

        dm_model = self.cosmology.distance.DM(
            self.data.z_hd,
        )

        dl_model = dm_model * (1.0 + self.data.z_hel)

        mu_model = 5.0 * np.log10(dl_model) + 25.0

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

        ``C`` (and the covariance solve it requires) is
        independent of the cosmology and is precomputed once in
        ``__init__`` -- see the comment there. Only ``A`` and
        ``B`` (which depend on ``delta``, i.e. on the current
        cosmology) are recomputed here.
        """

        delta = self.residuals()

        Cinv_delta = self.covariance.solve(delta)

        A = float(delta @ Cinv_delta)
        B = float(self._Cinv_ones @ delta)
        C = self._C

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
