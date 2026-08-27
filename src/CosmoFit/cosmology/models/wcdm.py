"""
wCDM: constant dark-energy equation of state.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


class WCDM(Cosmology):
    """
    Flat or curved wCDM: dark energy with a constant equation of
    state w0 (does not evolve with redshift; ``wa`` is ignored,
    same as ``LCDM`` ignoring ``w0``/``wa`` and ``CPL`` being the
    two-parameter, ``wa``-evolving generalization of this model).

    E(z)^2 = Omega_m (1+z)^3 + Omega_k (1+z)^2
             + Omega_de0 (1+z)^(3(1+w0))

    Reduces to flat/curved LCDM for w0 = -1.
    """

    MODEL_NAME = "wCDM"
    MODEL_LABEL = r"$w$CDM"

    # ---------------------------------------------------------

    def w(self, z):
        """
        Dark-energy equation of state. Constant by construction
        (``w(z) = w0`` at every redshift) -- provided mainly so
        that generic code written against ``CPL.w(z)`` (e.g. the
        w(z) evolution plot) also works for ``WCDM``.
        """

        z = np.asarray(z, dtype=float)

        return np.full_like(z, self.w0, dtype=float)

    # ---------------------------------------------------------

    def fde(self, z):
        """
        Dark-energy density evolution factor.

        Returns
        -------
        ndarray
            ρ_DE(z) / ρ_DE(0) = (1+z)^(3(1+w0))
        """

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + self.w0))

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * cube(1.0 + z)

            +

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_de0 * self.fde(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        dE2_dz = (

            3.0 * self.Omega_m * (1.0 + z) ** 2

            +

            2.0 * self.Omega_k * (1.0 + z)

            +

            self.Omega_de0
            * 3.0 * (1.0 + self.w0)
            * (1.0 + z) ** (3.0 * (1.0 + self.w0) - 1.0)

        )

        return dE2_dz / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z):

        return self.Omega_de0 * self.fde(z)
