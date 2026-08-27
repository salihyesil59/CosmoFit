"""
Jassal-Bagla-Padmanabhan (JBP) dark-energy model.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


class JBP(Cosmology):
    """
    Jassal-Bagla-Padmanabhan (JBP) dark-energy model.

    Equation of state

        w(z) = w0 + wa * z / (1 + z)^2

    Unlike CPL's w(z) = w0 + wa*z/(1+z) (monotonic, saturating to
    w0+wa as z -> infinity), JBP's w(z) is non-monotonic in z: the
    wa term peaks around z=1 and decays back toward w0 at both low
    and high redshift, so JBP and CPL can prefer noticeably
    different high-z behavior for similar low-z data. Reuses the
    CPL/wCDM ``w0``/``wa`` parameters -- no new free parameters.

    Reference
    ---------
    Jassal, Bagla & Padmanabhan (2005), MNRAS 356, L11,
    arXiv:astro-ph/0404378.
    """

    MODEL_NAME = "JBP"

    # ---------------------------------------------------------

    def w(self, z):
        """
        Dark-energy equation of state.
        """

        z = np.asarray(z, dtype=float)

        return self.w0 + self.wa * z / (1.0 + z) ** 2

    # ---------------------------------------------------------

    def fde(self, z):
        """
        Dark-energy density evolution factor.

        Closed-form solution of the continuity equation
        d ln(rho_DE)/dz = 3(1+w(z))/(1+z) for the JBP w(z):

            rho_DE(z) / rho_DE(0)
                = (1+z)^(3(1+w0)) * exp[ (3/2) wa (z/(1+z))^2 ]

        Returns
        -------
        ndarray
            ρ_DE(z) / ρ_DE(0)
        """

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + self.w0)) * np.exp(
            1.5 * self.wa * (z / (1.0 + z)) ** 2
        )

    # ---------------------------------------------------------

    def E(self, z):
        """
        Dimensionless Hubble parameter.
        """

        z = np.asarray(z, dtype=float)

        return np.sqrt(
            self.Omega_m * cube(1.0 + z)
            + self.Omega_k * (1.0 + z) ** 2
            + self.Omega_de0 * self.fde(z)
        )

    # ---------------------------------------------------------

    def dEdz(self, z):
        """
        Derivative of E(z).
        """

        z = np.asarray(z, dtype=float)

        fde = self.fde(z)

        # d ln(fde)/dz = 3(1+w(z))/(1+z), by construction of fde
        # above -- equivalently 3(1+w0)/(1+z) + 3 wa z/(1+z)^3.
        dlnf_dz = (
            3.0 * (1.0 + self.w0) / (1.0 + z)
            + 3.0 * self.wa * z / cube(1.0 + z)
        )

        dE2_dz = (
            3.0 * self.Omega_m * (1.0 + z) ** 2
            + 2.0 * self.Omega_k * (1.0 + z)
            + self.Omega_de0 * fde * dlnf_dz
        )

        return dE2_dz / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z):
        """
        Dark-energy density parameter.
        """

        z = np.asarray(z, dtype=float)

        return self.Omega_de0 * self.fde(z)
