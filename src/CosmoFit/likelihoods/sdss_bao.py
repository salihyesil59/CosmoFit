"""
SDSS BAO likelihood (BOSS DR12 + eBOSS DR16 LRG + eBOSS DR16 QSO).
"""

from __future__ import annotations

from CosmoFit.data.loader import load_sdss_bao
from CosmoFit.data.loader import load_sdss_fsbao

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


class SDSSFullShapeLikelihood(BAODistanceLikelihood):
    """
    SDSS **BAO + full-shape** consensus: ``D_M/r_d``, ``D_H/r_d``
    and ``f sigma_8`` at z = 0.38, 0.51, 0.698 and 1.48, twelve
    numbers with the covariance between them.

    The same galaxies as :class:`SDSSBAOLikelihood`, analysed for
    their full anisotropic clustering rather than the BAO peak
    alone. Two things follow. It measures the **growth rate**, so
    unlike every other BAO dataset here it constrains ``sigma8``.
    And it carries the **correlation between growth and geometry**
    -- 0.19 to 0.64 between ``D_M/r_d`` and ``f sigma_8`` within a
    bin, strongest for the quasars -- which is exactly what is thrown away by using the
    BAO-only dataset together with a separate ``f sigma_8``
    compilation drawn from the same surveys.

    ``f sigma_8`` is compared with the model's own ``fsigma8(z)``
    with **no Alcock-Paczynski rescaling**, unlike
    :class:`~likelihoods.fsigma8.FSigma8Likelihood`: a full-shape
    fit varies the geometry alongside the growth rate, so the
    fiducial is already fitted rather than something to correct
    back to.

    Warning
    -------
    Mutually exclusive with ``"sdss_bao"`` (the same BAO
    measurements) and with ``"fsigma8"`` (whose compilation
    includes these surveys' growth measurements). Overlaps
    ``"desi"``, as its BAO-only sibling does.

    References
    ----------
    eBOSS Collaboration / Alam et al. (2021), Phys. Rev. D 103,
    083533, `arXiv:2007.08991 <https://arxiv.org/abs/2007.08991>`_.
    """

    def __init__(
        self,
        cosmology,
        version="dr16",
    ):

        dataset = load_sdss_fsbao(

            version,

        )

        super().__init__(

            name="SDSS-BAO+FS",

            dataset=dataset,

            cosmology=cosmology,

        )
