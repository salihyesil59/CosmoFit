"""
Low-redshift BAO likelihood (6dFGS + SDSS DR7 MGS).
"""

from __future__ import annotations

from CosmoFit.data.loader import load_bao_lowz

from .desi import BAODistanceLikelihood


class BAOLowZLikelihood(BAODistanceLikelihood):
    """
    The two BAO measurements below z = 0.2: 6dFGS at z = 0.106
    (Beutler et al. 2011) and the SDSS DR7 Main Galaxy Sample at
    z = 0.15 (Ross et al. 2015).

    Every other BAO dataset in this library starts at z = 0.295
    (DESI) or z = 0.38 (BOSS DR12), so these two points are the
    only BAO leverage available in the regime where the expansion
    history is closest to today's -- exactly where a dark-energy
    equation of state that evolves has to show up. Two points with
    ~4-5% errors will not, on their own, move a fit much; what they
    do is extend the lever arm of the BAO-only distance ladder
    downward, which matters for the ``H0 * r_d`` degeneracy that
    BAO alone cannot break.

    Both surveys are independent of DESI and of SDSS BOSS/eBOSS
    (6dFGS is a different hemisphere and instrument; the DR7 MGS
    is a brighter, lower-redshift sample than the BOSS LRGs), so
    unlike ``"desi"`` and ``"sdss_bao"`` -- which must not be
    combined -- this dataset *can* be added to either.

    Two details are handled rather than papered over:

    - 6dFGS reports ``r_s/D_V``, not ``D_V/r_s``. That is kept as
      its own observable type (``"rs_over_DV"`` in
      :data:`~likelihoods.desi.MODEL_MAP`) rather than inverted in
      the data file, because inverting a Gaussian gives something
      that is neither Gaussian nor centred where the inversion of
      the mean is.

    - 6dFGS's measurement is calibrated against an Eisenstein & Hu
      (1998) fitting-formula sound horizon (153.9 Mpc for their
      fiducial) where a Boltzmann code gives 149.8 Mpc for the same
      cosmology, so the theory ``r_d`` is multiplied by 153.9/149.8
      before the comparison. See
      :attr:`~data.dataset.DESIDataset.rs_rescale`. At 2.7% on a
      4.5%-precision point this is not optional.

    Parameters
    ----------
    cosmology
        Cosmology model instance (LCDM, CPL, ...).

    version : str, optional
        Dataset version.

    References
    ----------
    Beutler et al. (2011), MNRAS 416, 3017, arXiv:1106.3366.
    Ross et al. (2015), MNRAS 449, 835, arXiv:1409.3242.
    """

    def __init__(
        self,
        cosmology,
        version: str = "6dfgs+mgs",
    ):

        dataset = load_bao_lowz(

            version,

        )

        super().__init__(

            name="BAO low-z",

            dataset=dataset,

            cosmology=cosmology,

        )

    # --------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "BAO distance ratios (z < 0.2)"
