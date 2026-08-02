"""
DESI BAO likelihood.
"""

from __future__ import annotations

import numpy as np

from data.loader import load_desi

from .base import BaseLikelihood


MODEL_MAP = {

    "DM_over_rs": lambda m, z:
        m.distance.DM(z) / m.rd,

    "DH_over_rs": lambda m, z:
        m.distance.DH(z) / m.rd,

    "DV_over_rs": lambda m, z:
        m.distance.DV(z) / m.rd,

}


class DESILikelihood(BaseLikelihood):

    def __init__(
        self,
        cosmology,
        version="desi2024",
    ):

        dataset = load_desi(

            version,

        )

        super().__init__(

            name="DESI",

            dataset=dataset,

            cosmology=cosmology,

        )

    # -----------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute the theoretical DESI observables.
        """

        prediction: list[float] = []

        for observable, z in zip(

            self.data.observable,

            self.data.z,

        ):

            try:

                model = MODEL_MAP[

                    observable

                ]

            except KeyError as exc:

                raise ValueError(

                    f"Unsupported DESI observable: {observable}",

                ) from exc

            prediction.append(

                model(

                    self.cosmology,

                    z,

                ),

            )

        return np.asarray(

            prediction,

            dtype=float,

        )

    # -----------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Compute the DESI residual vector.
        """

        return (

            self.data.value

            - self.model()

        )

    # -----------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Compute the DESI chi-square.
        """

        return self.covariance.chi2(

            self.residuals(),

        )