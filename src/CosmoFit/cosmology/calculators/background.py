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
        """
        Matter density parameter at redshift ``z``.

        Delegates the numerator to the model's own
        :meth:`~cosmology.core.base.Cosmology.Omega_matter`, so a
        model in which matter does not dilute as ``(1+z)^3`` (a
        running vacuum, an interacting dark sector) reports its
        real matter fraction here -- and, through it, in the
        linear growth equation.
        """

        return (

            self.cosmo.Omega_matter(z)

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

    # ---------------------------------------------------------

    def growth_rate(self, z):

        return self.cosmo.growth.growth_rate(z)

    # ---------------------------------------------------------

    def sigma8(self, z):

        return self.cosmo.growth.sigma8(z)

    # ---------------------------------------------------------

    def fsigma8(self, z):

        return self.cosmo.growth.fsigma8(z)

    # --------------------------------------------------------

    def set_cosmology(
        self,
        cosmology,
    ) -> None:
        """
        Update the cosmological model.
        """

        self.cosmo = cosmology


    # --------------------------------------------------------

    @property
    def model(
        self,
    ):
        """
        Return the current cosmological model.
        """

        return self.cosmo