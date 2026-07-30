import numpy as np

from .base import Cosmology


class CPL(Cosmology):

    def w(self, z):

        z = np.asarray(z)

        return self.w0 + self.wa * z / (1.0 + z)

    # ---------------------------------------------------------

    def fde(self, z):
        """
        Dark energy evolution factor.
        """

        z = np.asarray(z)

        return (

            (1.0 + z) ** (3.0 * (1.0 + self.w0 + self.wa))

            *

            np.exp(-3.0 * self.wa * z / (1.0 + z))

        )

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z)

        return np.sqrt(

            self.Omega_m * (1.0 + z) ** 3

            +

            self.Omega_de0 * self.fde(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z)

        fde = self.fde(z)

        dlnf_dz = (

            3.0 * (1.0 + self.w0 + self.wa) / (1.0 + z)

            -

            3.0 * self.wa / (1.0 + z) ** 2

        )

        dE2_dz = (

            3.0 * self.Omega_m * (1.0 + z) ** 2

            +

            self.Omega_de0 * fde * dlnf_dz

        )

        return dE2_dz / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z):

        z = np.asarray(z)

        return self.Omega_de0 * self.fde(z)