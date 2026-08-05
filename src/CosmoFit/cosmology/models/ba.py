"""
Barboza-Alcaniz (BA) dark-energy model.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class BA(Cosmology):
    """
    Barboza-Alcaniz (BA) dark-energy model.

    Equation of state

        w(z) = w0 + wa * z * (1 + z) / (1 + z^2)

    Designed to stay finite (-> w0 - wa as z -> infinity, rather
    than diverging or saturating) at arbitrarily high redshift
    unlike CPL, which several papers cite as a theoretical
    advantage when extrapolating w(z) into the matter/radiation
    era. Reuses the CPL/wCDM ``w0``/``wa`` parameters -- no new
    free parameters.

    Reference
    ---------
    Barboza & Alcaniz (2008), Phys. Lett. B 666, 415,
    arXiv:0805.1713.
    """

    MODEL_NAME = "BA"

    # ---------------------------------------------------------

    def w(self, z):
        """
        Dark-energy equation of state.
        """

        z = np.asarray(z, dtype=float)

        return self.w0 + self.wa * z * (1.0 + z) / (1.0 + z ** 2)

    # ---------------------------------------------------------

    def fde(self, z):
        """
        Dark-energy density evolution factor.

        Closed-form solution of the continuity equation
        d ln(rho_DE)/dz = 3(1+w(z))/(1+z) for the BA w(z):

            rho_DE(z) / rho_DE(0)
                = (1+z)^(3(1+w0)) * (1+z^2)^(3 wa / 2)

        Returns
        -------
        ndarray
            ρ_DE(z) / ρ_DE(0)
        """

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + self.w0)) * (
            1.0 + z ** 2
        ) ** (1.5 * self.wa)

    # ---------------------------------------------------------

    def E(self, z):
        """
        Dimensionless Hubble parameter.
        """

        z = np.asarray(z, dtype=float)

        return np.sqrt(
            self.Omega_m * (1.0 + z) ** 3
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
        # above -- equivalently 3(1+w0)/(1+z) + 3 wa z/(1+z^2).
        dlnf_dz = (
            3.0 * (1.0 + self.w0) / (1.0 + z)
            + 3.0 * self.wa * z / (1.0 + z ** 2)
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
