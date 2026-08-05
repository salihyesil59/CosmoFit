"""
Distance calculations for cosmological models.

This module converts the dimensionless comoving distance

    χ(z)

into the commonly used cosmological distance measures.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import constants


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

    def DC(self, z):
        """
        Line-of-sight comoving distance [Mpc].
        """

        return (constants.c / self.cosmo.H0) * self.chi(z)

    # ---------------------------------------------------------

    def DM_from_chi(self, chi):
        """
        Transverse comoving distance [Mpc] from a raw
        dimensionless comoving distance ``chi``.

        This is the curvature branch used by :meth:`DM`, split
        out so that callers needing distances far outside the
        cached low-z interpolation grid (e.g. the CMB decoupling
        redshift z* ~ 1090, used by the Planck distance-prior
        likelihood) can supply a directly-integrated ``chi``
        without duplicating the curvature formula or disturbing
        the low-z grid's resolution.
        """

        Ok = self.cosmo.Omega_k
        hubble_distance = constants.c / self.cosmo.H0

        if Ok == 0.0:
            return hubble_distance * chi

        sqrt_Ok = np.sqrt(np.abs(Ok))

        if Ok > 0.0:
            return hubble_distance / sqrt_Ok * np.sinh(sqrt_Ok * chi)

        return hubble_distance / sqrt_Ok * np.sin(sqrt_Ok * chi)

    # ---------------------------------------------------------

    def DM(self, z):
        """
        Transverse comoving distance [Mpc].

        Generalizes to non-flat geometries via the curvature
        parameter Omega_k:

            Omega_k == 0 (flat):    D_M = D_C
            Omega_k  > 0 (open):    D_M = (c/H0) / sqrt(Omega_k)
                                            * sinh(sqrt(Omega_k) * chi)
            Omega_k  < 0 (closed):  D_M = (c/H0) / sqrt(|Omega_k|)
                                            * sin(sqrt(|Omega_k|) * chi)

        where chi(z) = H0/c * D_C(z) is the dimensionless
        comoving distance. All three branches agree in the
        Omega_k -> 0 limit.

        Note: ``z`` must lie within the cached distance-integrator
        grid (``cosmology.integrator.zmax``, 5.0 by default). For
        redshifts beyond that range (e.g. the CMB), integrate
        ``chi`` directly and use :meth:`DM_from_chi` instead.
        """

        if self.cosmo.Omega_k == 0.0:
            return self.DC(z)

        return self.DM_from_chi(self.chi(z))

    # ---------------------------------------------------------

    def DA(self, z):
        """
        Angular diameter distance.
        """

        z = np.asarray(z)

        return self.DM(z) / (1.0 + z)

    # ---------------------------------------------------------

    def DL(self, z):
        """
        Luminosity distance.
        """

        z = np.asarray(z)

        return self.DM(z) * (1.0 + z)

    # ---------------------------------------------------------

    def mu(self, z):
        """
        Distance modulus.
        """

        dl = self.DL(z)

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

            self.DM(z) ** 2

            *

            (constants.c * z / self.cosmo.H(z))

        ) ** (1.0 / 3.0)

    # ---------------------------------------------------------

    def DH(self, z):
        """
        Hubble distance [Mpc].

        D_H(z) = c / H(z)
        """

        return constants.c / self.cosmo.H(z)

    # ---------------------------------------------------------

    def F_AP(self, z):
        """
        Alcock-Paczynski parameter.

        F_AP = D_M H / c
        """

        return (

            self.DM(z)

            *

            self.cosmo.H(z)

            / constants.c)