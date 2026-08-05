"""
DESI BAO likelihood.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import load_desi

from .base import BaseLikelihood


MODEL_MAP = {

    "DM_over_rs": lambda m, z:
        m.distance.DM(z) / m.rd,

    "DH_over_rs": lambda m, z:
        m.distance.DH(z) / m.rd,

    "DV_over_rs": lambda m, z:
        m.distance.DV(z) / m.rd,

}


class BAODistanceLikelihood(BaseLikelihood):
    """
    Shared implementation for BAO distance-ratio likelihoods whose
    dataset is a vector of (z, value, observable-type) triplets
    against a single covariance -- currently
    :class:`DESILikelihood` and
    :class:`~likelihoods.sdss_bao.SDSSBAOLikelihood`. ``observable``
    selects which entry of :data:`MODEL_MAP` (``"DM_over_rs"``,
    ``"DH_over_rs"``, ``"DV_over_rs"``) each data point is compared
    against.
    """

    # -----------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute the theoretical BAO observables.
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

                    f"Unsupported BAO observable: {observable}",

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
        Compute the BAO residual vector.
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
        Compute the BAO chi-square.
        """

        return self.covariance.chi2(

            self.residuals(),

        )


class DESILikelihood(BAODistanceLikelihood):

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
