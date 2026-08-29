"""
Union3 binned supernova likelihood.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import load_union3

from .base import BaseLikelihood, AnalyticOffsetMixin


class Union3Likelihood(BaseLikelihood, AnalyticOffsetMixin):
    """
    Union3 (Rubin et al. 2023): 2087 supernovae compressed into 22
    binned distance moduli with a full 22x22 covariance.

    The third of the three modern SN Ia compilations, alongside
    :class:`~likelihoods.pantheon.PantheonLikelihood` and
    :class:`~likelihoods.des_sn5yr.DESSN5YRLikelihood`. It matters
    here because the DESI dark-energy results are argued with all
    three, and they do not agree with each other on how far the
    data sits from a cosmological constant -- DES-SN5YR pulls
    hardest, Pantheon+ least, Union3 in between. A library that can
    only fit one of them cannot reproduce that comparison, which is
    the actual state of the evidence.

    What makes Union3 different is not the sample so much as the
    analysis: UNITY1.5 is a Bayesian hierarchical model that fits
    light-curve standardization, host-galaxy mass dependence,
    selection effects and outliers *jointly with* cosmology, and
    marginalizes them internally. The released product is therefore
    already a binned distance-modulus vector -- there is no
    per-supernova catalogue to re-standardize, and no stretch/colour
    nuisance parameters left for a downstream fit to vary.

    As with both other SN samples, the overall zero point is
    degenerate with H0 and is marginalized analytically (see
    :class:`~likelihoods.base.AnalyticOffsetMixin`) rather than
    fit -- which, for a 22-point dataset, matters more than it does
    for a 1600-point one.

    Parameters
    ----------
    cosmology
        Cosmology model instance (LCDM, CPL, ...).

    version : str, optional
        Dataset version.

    marginalize_offset : bool, optional
        If True (default), the constant distance-modulus offset is
        marginalized over analytically. If False,
        ``cosmology.MB`` is added to the model as an explicit
        nuisance parameter instead.

    Warning
    -------
    Do not combine ``"union3"`` with ``"pantheon"`` or
    ``"des_sn5yr"`` in the same fit. Union3 shares a large majority
    of its supernovae with Pantheon+ (both compile essentially the
    same literature samples), and its high-redshift half overlaps
    the DES sample; treating any two of the three as independent
    double-counts most of the data. This is the same rule the
    library already applies between Pantheon+ and DES-SN5YR --
    one SN Ia compilation per fit.

    References
    ----------
    Rubin et al. (2023), "Union Through UNITY: Cosmology with
    2,000 SNe Using a Unified Bayesian Framework",
    arXiv:2311.12098 (ApJ, accepted).
    """

    def __init__(
        self,
        cosmology,
        version: str = "union3",
        marginalize_offset: bool = True,
    ):

        dataset = load_union3(

            version=version,

        )

        self.marginalize_offset = marginalize_offset

        super().__init__(

            name="Union3",

            dataset=dataset,

            cosmology=cosmology,

        )

        if self.marginalize_offset:

            self._setup_offset_marginalization()

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "mu(z)"

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted binned distance modulus.

            mu = 5 log10[(1 + z_hel) D_M(z_cmb)] + 25

        The heliocentric/CMB-frame redshift split is the same one
        :class:`~likelihoods.pantheon.PantheonLikelihood` and
        :class:`~likelihoods.des_sn5yr.DESSN5YRLikelihood` apply:
        the comoving distance is a function of the CMB-frame
        redshift, while the ``(1 + z)`` source-frame dilation
        factor in the luminosity distance is heliocentric.

        Union3's release sets ``zhel = zcmb`` for every bin (the
        bins average over many lines of sight, so there is no
        single heliocentric correction left to apply), so the two
        columns are numerically identical here -- the distinction
        is kept anyway so the formula reads the same as the other
        two SN likelihoods rather than silently relying on that.
        """

        dm_model = self.cosmology.distance.DM(

            self.data.z_cmb,

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
