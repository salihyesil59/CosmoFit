"""
Pantheon+ / Pantheon+SH0ES Supernova likelihood.
"""

from __future__ import annotations

import numpy as np

from cosmofit.data.loader import PANTHEON_Z_MIN, load_pantheon

from .base import BaseLikelihood, AnalyticOffsetMixin


class PantheonLikelihood(BaseLikelihood, AnalyticOffsetMixin):
    """
    Pantheon+ / Pantheon+SH0ES Supernova likelihood.

    Parameters
    ----------
    cosmology
        Cosmology model instance (LCDM, CPL, ...).

    version
        Pantheon dataset version.

    include_cepheid
        If False (default), the Hubble-flow sample: the 1590 light
        curves with ``z_HD > 0.01``, which constrain the shape of
        the distance-redshift relation but not its absolute scale.

        If True, the Pantheon+SH0ES sample: the 77 Cepheid-calibrator
        light curves are added, and for those the predicted distance
        modulus is the host galaxy's Cepheid distance (``CEPH_DIST``)
        rather than the cosmological ``mu(z)`` -- their redshifts are
        too small to say anything about the expansion, and their
        Cepheids are what fix ``M_B``. That is what lets this one
        dataset measure H0 (Brout et al. 2022, Eqs. 14-15).

    z_min
        Hubble-flow cut on ``z_HD``. Default 0.01, as in every
        published Pantheon+ fit.

    marginalize_MB : bool, optional
        If True (the default without Cepheids), the SN absolute magnitude (and,
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
        (m_B = mu(z) + M_B).

        Defaults to ``not include_cepheid``. Marginalizing with the
        calibrators in is refused: the analytic marginalization
        removes exactly the absolute scale the Cepheids exist to
        supply, so the calibrators would be carried along and
        contribute nothing.
    """

    def __init__(
        self,
        cosmology,
        version: str = "pantheon+sh0es",
        include_cepheid: bool = False,
        marginalize_MB: bool | None = None,
        z_min: float = PANTHEON_Z_MIN,
    ):

        if marginalize_MB is None:
            marginalize_MB = not include_cepheid

        if include_cepheid and marginalize_MB:

            raise ValueError(

                "`include_cepheid=True` with `marginalize_MB=True` "

                "throws the calibration away: the analytic "

                "marginalization removes the absolute magnitude M_B, "

                "and fixing M_B is the only thing the Cepheid hosts "

                "do. Leave `marginalize_MB` at its default (False with "

                "Cepheids) and free `MB`.",

            )

        dataset = load_pantheon(
            version=version,
            include_cepheid=include_cepheid,
            z_min=z_min,
        )

        self.include_cepheid = bool(include_cepheid)
        self.marginalize_MB = marginalize_MB

        super().__init__(
            name="Pantheon+",
            dataset=dataset,
            cosmology=cosmology,
        )

        if self.marginalize_MB:

            self._setup_offset_marginalization()

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

        if self.include_cepheid:

            # The calibrators' distances are measured, not
            # predicted: their Cepheids give mu directly, and their
            # redshifts (most below 0.01) are peculiar-velocity
            # noise. Brout et al. (2022) Eq. 14.
            mu_model = np.where(
                self.data.cepheid == 1,
                self.data.ceph_dist,
                mu_model,
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
    # best_fit_offset() is provided by AnalyticOffsetMixin. It is
    # only meaningful here when marginalize_MB is True.
    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        if self.marginalize_MB:

            return self.marginalized_chi2()

        return self.covariance.chi2(

            self.residuals(),

        )
