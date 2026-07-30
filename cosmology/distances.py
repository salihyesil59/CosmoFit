"""
Distance calculations for cosmological models.

This module converts the dimensionless comoving distance

    χ(z)

into the commonly used cosmological distance measures.
"""

from __future__ import annotations

import numpy as np

from .constants import c


class DistanceCalculator:
    """
    Cosmological distance calculator.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmological model.
    """

    def __init__(self, cosmology):

        self.cosmo = cosmology

    # ---------------------------------------------------------

    def chi(self, z):
        """
        Dimensionless comoving distance.
        """

        return self.cosmo.integrator.chi(z)

    # ---------------------------------------------------------

    def Dc(self, z):
        """
        Line-of-sight comoving distance [Mpc].
        """

        return (c / self.cosmo.H0) * self.chi(z)

    # ---------------------------------------------------------

    def Dm(self, z):
        """
        Transverse comoving distance.

        Flat universe:

            D_M = D_C

        Curved geometry will be added later.
        """

        return self.Dc(z)

    # ---------------------------------------------------------

    def Da(self, z):
        """
        Angular diameter distance.
        """

        z = np.asarray(z)

        return self.Dm(z) / (1.0 + z)

    # ---------------------------------------------------------

    def Dl(self, z):
        """
        Luminosity distance.
        """

        z = np.asarray(z)

        return self.Dm(z) * (1.0 + z)

    # ---------------------------------------------------------

    def mu(self, z):
        """
        Distance modulus.
        """

        dl = self.Dl(z)

        return 5.0 * np.log10(dl) + 25.0

    # ---------------------------------------------------------

    def DV(self, z):
        """
        BAO volume-averaged distance.

        D_V(z) =
            [ D_M(z)^2 * cz/H(z) ]^(1/3)
        """

        z = np.asarray(z)

        return (

            self.Dm(z) ** 2

            *

            (c * z / self.cosmo.H(z))

        ) ** (1.0 / 3.0)

    # ---------------------------------------------------------

    def F_AP(self, z):
        """
        Alcock-Paczynski parameter.

        F_AP = D_M H / c
        """

        return (

            self.Dm(z)

            *

            self.cosmo.H(z)

            / c)