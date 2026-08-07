"""
fsigma8 (RSD growth-rate) likelihood.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLikelihood

from CosmoFit.data.loader import load_fsigma8


class FSigma8Likelihood(BaseLikelihood):
    """
    Growth-rate (fsigma8) likelihood, e.g. the "Gold-2018" RSD
    compilation (:func:`~data.loader.load_fsigma8`).

    Applies the same Alcock-Paczynski correction the reference
    likelihood (snesseris/RSD-growth) applies: each survey's raw
    RSD measurement was converted into fsigma8 assuming some
    fiducial cosmology's H(z)*D_A(z) product (``data.HdAz``); a
    test cosmology with a different H(z)*D_A(z) needs its predicted
    fsigma8 rescaled by that ratio before comparing to the data,
    rather than compared directly.
    """

    def __init__(
        self,
        cosmology,
        version="gold2018",
    ):

        dataset = load_fsigma8(version)

        super().__init__(

            name="fsigma8",

            dataset=dataset,

            cosmology=cosmology,

        )

    # --------------------------------------------------------

    @property
    def observable(self):

        return "fsigma8(z)"

    # --------------------------------------------------------

    def model(self) -> np.ndarray:
        """
        AP-corrected fsigma8(z) prediction.
        """

        z = self.data.z

        theory = self.cosmology.background.fsigma8(z)

        H = self.cosmology.H(z)
        DA = self.cosmology.distance.DA(z)

        ratio = (H * DA) / self.data.HdAz

        return theory / ratio

    # --------------------------------------------------------

    def residuals(self) -> np.ndarray:

        return self.data.fsigma8 - self.model()

    # --------------------------------------------------------

    def chi2(self) -> float:

        return self.covariance.chi2(self.residuals())
