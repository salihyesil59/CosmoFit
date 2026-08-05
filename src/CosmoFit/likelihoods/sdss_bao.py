"""
SDSS BAO likelihood (BOSS DR12 + eBOSS DR16 LRG + eBOSS DR16 QSO).
"""

from __future__ import annotations

from CosmoFit.data.loader import load_sdss_bao

from .desi import BAODistanceLikelihood


class SDSSBAOLikelihood(BAODistanceLikelihood):
    """
    Combined SDSS BAO-only likelihood: BOSS DR12 (z=0.38, 0.51) +
    eBOSS DR16 LRG (z=0.698) + eBOSS DR16 QSO (z=1.48), each an
    independent, non-overlapping-redshift measurement -- see
    :func:`~data.loader.load_sdss_bao` for why BOSS DR12's usual
    third bin (z=0.61) is omitted.

    Same (z, value, observable-type) structure as
    :class:`~likelihoods.desi.DESILikelihood`, so it shares the
    latter's ``model()``/``residuals()``/``chi2()`` implementation
    via :class:`~likelihoods.desi.BAODistanceLikelihood`.

    Warning
    -------
    Do not combine ``"desi"`` and ``"sdss_bao"`` in the same
    ``Fitter`` fit -- DESI's own footprint and redshift ranges
    overlap the BOSS/eBOSS samples this dataset is built from (DESI
    is, among other things, an SDSS successor survey targeting much
    of the same sky), so treating them as independent would
    double-count structure.

    References
    ----------
    Alam et al. (2017), arXiv:1607.03155 (BOSS DR12).
    eBOSS Collaboration / Alam et al. (2021), arXiv:2007.08991
    (eBOSS DR16 summary).
    Data as distributed by CobayaSampler/bao_data (originally with
    CosmoMC).
    """

    def __init__(
        self,
        cosmology,
        version="dr12+dr16",
    ):

        dataset = load_sdss_bao(

            version,

        )

        super().__init__(

            name="SDSS-BAO",

            dataset=dataset,

            cosmology=cosmology,

        )
