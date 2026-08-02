import numpy as np

from cosmology.core import Cosmology


class LCDM(Cosmology):

    def E(self, z):

        return np.sqrt(

            self.Omega_m * (1.0 + z) ** 3

            +

            self.Omega_de0

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z)

        return (

            3.0 * self.Omega_m * (1.0 + z) ** 2

            /

            (2.0 * self.E(z))

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):

        return np.full_like(
            np.asarray(z),
            self.Omega_de0,
            dtype=float,
        )