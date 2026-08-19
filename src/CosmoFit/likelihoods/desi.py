"""
DESI BAO likelihood.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import load_desi

from .base import BaseLikelihood


#: Maps each observable name appearing in a BAO dataset's
#: ``observable`` column to the model prediction for it.
#:
#: The keys are DESI's own spelling, taken verbatim from the
#: released data file, and they say ``rs`` where the predictions
#: below divide by ``m.rd``. That is not an inconsistency: both
#: denote the sound horizon *at the drag epoch*, ``r_s(z_d)``,
#: which the BAO literature writes as either ``r_s`` or ``r_d``.
#: CosmoFit calls it ``rd`` throughout (the parameter, the
#: `SoundHorizon` calculator, and :data:`OBSERVABLE_LABELS` below),
#: so that figures and code agree on one symbol; only these keys
#: keep the data file's wording, so a dataset can be loaded
#: without renaming its own columns.
MODEL_MAP = {

    "DM_over_rs": lambda m, z:
        m.distance.DM(z) / m.rd,

    "DH_over_rs": lambda m, z:
        m.distance.DH(z) / m.rd,

    "DV_over_rs": lambda m, z:
        m.distance.DV(z) / m.rd,

}


#: LaTeX label for each observable in :data:`MODEL_MAP`, for figure
#: titles and axes (:meth:`~plots.FitPlotter.bao_distances`). Written
#: with ``r_d``, matching what the predictions above actually
#: compute, rather than transliterating the key.
OBSERVABLE_LABELS = {
    "DM_over_rs": r"$D_M(z)\,/\,r_d$",
    "DH_over_rs": r"$D_H(z)\,/\,r_d$",
    "DV_over_rs": r"$D_V(z)\,/\,r_d$",
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
