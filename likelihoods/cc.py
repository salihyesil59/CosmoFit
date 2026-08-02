"""
Cosmic Chronometer likelihood.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLikelihood

from data.loader import load_cc


class CCLikelihood(BaseLikelihood):

    def __init__(
        self,
        cosmology,
        version="favale2023",
    ):

        dataset = load_cc(

            version,

        )

        super().__init__(

            name="CC",

            dataset=dataset,

            cosmology=cosmology,

        )

    # --------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "H(z)"

    # --------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute model predictions.
        """

        return self.cosmology.background.H(

            self.data.z,

        )

    # --------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Compute residuals.
        """

        return (

            self.data.H

            - self.model()

        )

    # --------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Compute chi-square.
        """

        return self.covariance.chi2(

            self.residuals(),

        )