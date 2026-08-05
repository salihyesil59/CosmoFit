"""
DES-SN5YR (Dark Energy Survey 5-year) Supernova likelihood.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import load_des_sn5yr

from .base import BaseLikelihood, AnalyticOffsetMixin


class DESSN5YRLikelihood(BaseLikelihood, AnalyticOffsetMixin):
    """
    DES-SN5YR (Dark Energy Survey 5-year) Supernova likelihood.

    Unlike :class:`~likelihoods.pantheon.PantheonLikelihood` (which
    compares an apparent-magnitude-like observable, ``m_b_corr``,
    to ``mu(z) + M_B``), DES-SN5YR distributes the SN distance
    modulus directly (``MU``, computed assuming a fiducial H0=70).
    Comparing that to a model ``mu(z)`` computed at a different H0
    leaves a constant offset -- the same H0/absolute-calibration
    degeneracy Pantheon+ has, just expressed directly in distance-
    modulus space rather than magnitude space -- which is
    marginalized over analytically the same way (see
    :class:`~likelihoods.base.AnalyticOffsetMixin`); this exactly
    matches the DES-SN5YR data release's own reference likelihood
    (``DES-Dovekie-SN_Likelihood.py``, ``cov_log_likelihood``).

    Parameters
    ----------
    cosmology
        Cosmology model instance (LCDM, CPL, ...).

    version
        DES-SN5YR dataset version.

    marginalize_offset : bool, optional
        If True (default), the constant distance-modulus offset
        (fully degenerate with H0) is marginalized over
        analytically. If False, ``cosmology.MB`` is added to the
        model as an explicit free/fixed nuisance parameter instead
        (mu_model = mu(z) + M_B).

    Warning
    -------
    Do not combine ``"pantheon"`` and ``"des_sn5yr"`` in the same
    ``Fitter`` fit. DES-SN5YR's low-redshift "anchor" sample
    includes CfA3, CfA4, and Foundation supernovae (~11% of its
    1820 SNe, IDSURVEY != 10) that are *also* compiled into
    Pantheon+ -- fitting both at once double-counts those
    supernovae (correlated, non-independent data treated as
    independent), understating uncertainties and biasing any joint
    result. Use one SN Ia compilation per fit, not both.

    References
    ----------
    Sanchez et al. (2024), arXiv:2406.05046 (data release).
    DES Collaboration (2024), arXiv:2401.02929 (cosmology results).
    """

    def __init__(
        self,
        cosmology,
        version: str = "des-sn5yr",
        marginalize_offset: bool = True,
    ):

        dataset = load_des_sn5yr(
            version=version,
        )

        self.marginalize_offset = marginalize_offset

        super().__init__(
            name="DES-SN5YR",
            dataset=dataset,
            cosmology=cosmology,
        )

        if self.marginalize_offset:

            self._setup_offset_marginalization()

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted distance modulus.

        m_B = mu(z)                    [marginalize_offset=True]
        m_B = mu(z) + M_B              [marginalize_offset=False]

        As with Pantheon+, D_M is evaluated at the Hubble-diagram
        redshift ``z_hd`` and the ``(1 + z)`` source-frame dilation
        factor uses the heliocentric redshift ``z_hel``:

            D_L = (1 + z_hel) * D_M(z_hd)

        This matches the DES-SN5YR reference likelihood's
        ``extract_theory_points``, which computes
        ``5 log10[(1+zcmb)(1+zhel) D_A(zcmb)] + 25`` --
        algebraically identical to the above, since
        D_A(z) = D_M(z)/(1+z).
        """

        dm_model = self.cosmology.distance.DM(
            self.data.z_hd,
        )

        dl_model = dm_model * (1.0 + self.data.z_hel)

        mu_model = 5.0 * np.log10(dl_model) + 25.0

        if self.marginalize_offset:
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

            self.data.mu

            -

            self.model()

        )

    # ---------------------------------------------------------
    # best_fit_offset() is provided by AnalyticOffsetMixin. It is
    # only meaningful here when marginalize_offset is True.
    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        if self.marginalize_offset:

            return self.marginalized_chi2()

        return self.covariance.chi2(

            self.residuals(),

        )
