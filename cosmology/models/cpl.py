"""
Chevallier-Polarski-Linder (CPL) dark-energy model.
"""

from __future__ import annotations

import numpy as np

from cosmology.core import Cosmology


class CPL(Cosmology):
    """
    Chevallier-Polarski-Linder (CPL) dark-energy model.

    Equation of state

        w(z) = w0 + wa * z / (1 + z)
    """

    MODEL_NAME = "CPL"

    # ---------------------------------------------------------

    def w(
        self,
        z,
    ):
        """
        Dark-energy equation of state.
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        return (
            self.w0
            +
            self.wa
            * z
            /
            (1.0 + z)
        )

    # ---------------------------------------------------------

    def fde(
        self,
        z,
    ):
        """
        Dark-energy density evolution factor.

        Returns
        -------
        ndarray
            ρ_DE(z) / ρ_DE(0)
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        return (
            (1.0 + z)
            ** (
                3.0
                * (
                    1.0
                    + self.w0
                    + self.wa
                )
            )
            *
            np.exp(
                -3.0
                * self.wa
                * z
                / (1.0 + z)
            )
        )

    # ---------------------------------------------------------

    def E(
        self,
        z,
    ):
        """
        Dimensionless Hubble parameter.
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        return np.sqrt(
            self.Omega_m
            * (1.0 + z) ** 3
            +
            self.Omega_k
            * (1.0 + z) ** 2
            +
            self.Omega_de0
            * self.fde(z)
        )

    # ---------------------------------------------------------

    def dEdz(
        self,
        z,
    ):
        """
        Derivative of E(z).
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        fde = self.fde(z)

        dlnf_dz = (
            3.0
            * (
                1.0
                + self.w0
                + self.wa
            )
            / (1.0 + z)
            -
            3.0
            * self.wa
            / (1.0 + z) ** 2
        )

        dE2_dz = (
            3.0
            * self.Omega_m
            * (1.0 + z) ** 2
            +
            2.0
            * self.Omega_k
            * (1.0 + z)
            +
            self.Omega_de0
            * fde
            * dlnf_dz
        )

        return (
            dE2_dz
            /
            (
                2.0
                * self.E(z)
            )
        )

    # ---------------------------------------------------------

    def Omega_de(
        self,
        z,
    ):
        """
        Dark-energy density parameter.
        """

        z = np.asarray(
            z,
            dtype=float,
        )

        return (
            self.Omega_de0
            * self.fde(z)
        )