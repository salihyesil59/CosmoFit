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
#: Every entry takes ``(cosmology, z)`` plus an optional ``rd``
#: overriding the cosmology's own sound horizon -- the hook
#: :attr:`~data.dataset.DESIDataset.rs_rescale` needs for a survey
#: that calibrated its measurement against a differently-defined
#: ``r_d`` (see :class:`~likelihoods.desi.BAODistanceLikelihood`).
#: Callers that do not care (every plot, every dataset without a
#: rescale) can keep calling them with two arguments.
MODEL_MAP = {

    "DM_over_rs": lambda m, z, rd=None:
        m.distance.DM(z) / (m.rd if rd is None else rd),

    "DH_over_rs": lambda m, z, rd=None:
        m.distance.DH(z) / (m.rd if rd is None else rd),

    "DV_over_rs": lambda m, z, rd=None:
        m.distance.DV(z) / (m.rd if rd is None else rd),

    # 6dFGS reports the reciprocal (Beutler et al. 2011). Kept as
    # its own observable rather than inverted in the data file:
    # 0.336 +- 0.015 is Gaussian in r_s/D_V, and the reciprocal of
    # a Gaussian is neither Gaussian nor centred on 1/mean.
    "rs_over_DV": lambda m, z, rd=None:
        (m.rd if rd is None else rd) / m.distance.DV(z),

    # Growth, for the tabulated full-shape likelihoods. No
    # Alcock-Paczynski rescaling here, unlike
    # `likelihoods/fsigma8.py`: a full-shape grid varies D_M/r_d and
    # D_H/r_d alongside f*sigma8, so the geometry it was measured
    # against is a coordinate of the grid rather than a fiducial to
    # correct back to. Applying the correction as well would count
    # it twice.
    "fsigma8": lambda m, z, rd=None: m.background.fsigma8(z),

}


#: LaTeX label for each observable in :data:`MODEL_MAP`, for figure
#: titles and axes (:meth:`~plots.FitPlotter.bao_distances`). Written
#: with ``r_d``, matching what the predictions above actually
#: compute, rather than transliterating the key.
OBSERVABLE_LABELS = {
    "DM_over_rs": r"$D_M(z)\,/\,r_d$",
    "DH_over_rs": r"$D_H(z)\,/\,r_d$",
    "DV_over_rs": r"$D_V(z)\,/\,r_d$",
    "rs_over_DV": r"$r_d\,/\,D_V(z)$",
}


class BAODistanceLikelihood(BaseLikelihood):
    """
    Shared implementation for BAO distance-ratio likelihoods whose
    dataset is a vector of (z, value, observable-type) triplets
    against a single covariance -- currently
    :class:`DESILikelihood` and
    :class:`~likelihoods.sdss_bao.SDSSBAOLikelihood` and
    :class:`~likelihoods.bao_lowz.BAOLowZLikelihood`.
    ``observable`` selects which entry of :data:`MODEL_MAP`
    (``"DM_over_rs"``, ``"DH_over_rs"``, ``"DV_over_rs"``,
    ``"rs_over_DV"``) each data point is compared against.

    If the dataset carries an
    :attr:`~data.dataset.DESIDataset.rs_rescale`, each point's
    prediction uses ``rd * rescale`` in place of the cosmology's
    own ``rd`` -- the units conversion a survey needs when it
    quoted its BAO ratio against a fitting-formula sound horizon
    rather than an integrated one.
    """

    # -----------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute the theoretical BAO observables.
        """

        prediction: list[float] = []

        rescale = self.data.rs_rescale

        for index, (observable, z) in enumerate(

            zip(

                self.data.observable,

                self.data.z,

            ),

        ):

            try:

                model = MODEL_MAP[

                    observable

                ]

            except KeyError as exc:

                raise ValueError(

                    f"Unsupported BAO observable: {observable}",

                ) from exc

            rd = (

                None

                if rescale is None

                else self.cosmology.rd * rescale[index]

            )

            prediction.append(

                model(

                    self.cosmology,

                    z,

                    rd,

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
