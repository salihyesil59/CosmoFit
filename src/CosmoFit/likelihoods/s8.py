"""
S8 weak-lensing prior likelihood.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLikelihood

from CosmoFit.data.loader import load_s8


class S8Likelihood(BaseLikelihood):
    """
    A single Gaussian S8 = sigma8 * sqrt(Omega_m / 0.3) constraint
    from a weak-lensing survey (default: KiDS-1000; also DES Y3 --
    see :func:`~data.loader.load_s8`).

    Don't combine ``"s8"`` with a different ``version`` in the same
    fit -- KiDS-1000 and DES Y3 are independent, not a single joint
    constraint; pass ``dataset_kwargs={"s8": {"version": "des_y3"}}``
    to use one or the other, not both at once.
    """

    def __init__(
        self,
        cosmology,
        version="kids1000",
    ):

        dataset = load_s8(version)

        super().__init__(

            name="S8",

            dataset=dataset,

            cosmology=cosmology,

        )

    # --------------------------------------------------------

    @property
    def observable(self):

        return "S8"

    # --------------------------------------------------------

    def model(self) -> float:
        """
        S8 = sigma8 * sqrt(Omega_m / 0.3).
        """

        return self.cosmology.sigma8 * np.sqrt(
            self.cosmology.Omega_m / 0.3
        )

    # --------------------------------------------------------

    def residuals(self) -> np.ndarray:

        return np.array([self.data.value - self.model()])

    # --------------------------------------------------------

    def chi2(self) -> float:

        return self.covariance.chi2(self.residuals())
