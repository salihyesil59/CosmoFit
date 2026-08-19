import numpy as np

from CosmoFit.cosmology.core import Cosmology


class LCDM(Cosmology):
    """
    Flat or curved Lambda-CDM.

    E(z)^2 = Omega_m (1+z)^3 + Omega_k (1+z)^2 + Omega_de0

    The curvature term vanishes identically for the default
    ``Omega_k = 0`` (flat) case, so this is a strict
    generalization of the flat-LCDM expansion history.
    """

    MODEL_NAME = "LCDM"
    MODEL_LABEL = r"$\Lambda$CDM"

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * (1.0 + z) ** 3

            +

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_de0

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        return (

            (
                3.0 * self.Omega_m * (1.0 + z) ** 2

                +

                2.0 * self.Omega_k * (1.0 + z)
            )

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