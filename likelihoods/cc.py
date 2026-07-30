"""
Cosmic Chronometer likelihood.
"""

from __future__ import annotations

import numpy as np

from .base import Likelihood


class CCLikelihood(Likelihood):
    """
    Cosmic Chronometer likelihood.
    """

    def __init__(
        self,
        dataset,
        calculator,
    ):
        super().__init__(

            name="CC",

            dataset=dataset,

            calculator=calculator,

        )

    # --------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute model predictions.
        """

        return self.calculator.H(

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

            self.model()

            - self.data.H

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