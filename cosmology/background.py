"""
Background cosmology calculations.
"""

from __future__ import annotations

import numpy as np


class BackgroundCalculator:

    def __init__(self, cosmology):

        self.cosmo = cosmology

    # ---------------------------------------------------------

    def E(self, z):

        return self.cosmo.E(z)

    # ---------------------------------------------------------

    def H(self, z):

        return self.cosmo.H(z)

    # ---------------------------------------------------------

    def invH(self, z):

        return 1.0 / self.H(z)

    # ---------------------------------------------------------

    def dEdz(self, z):

        return self.cosmo.dEdz(z)

    # ---------------------------------------------------------

    def Omega_m(self, z):

        z = np.asarray(z)

        return (

            self.cosmo.Omega_m

            *

            (1.0 + z) ** 3

            /

            self.E(z) ** 2

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):

        return (

            self.cosmo.Omega_de(z)

            /

            self.E(z) ** 2

        )

    # ---------------------------------------------------------

    def q(self, z):
        """
        Deceleration parameter.

        q(z) = -1 + (1+z)/E * dE/dz
        """

        z = np.asarray(z)

        return (

            -1.0

            +

            (1.0 + z)

            *

            self.dEdz(z)

            /

            self.E(z)

        )

    # --------------------------------------------------------

    def set_cosmology(
        self,
        cosmology,
    ) -> None:
        """
        Update the cosmological model.
        """

        self.cosmology = cosmology


    # --------------------------------------------------------

    @property
    def model(
        self,
    ):
        """
        Return the current cosmological model.
        """

        return self.cosmology