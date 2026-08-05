"""
Generalized Chaplygin Gas (GCG) model.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class GCG(Cosmology):
    """
    Generalized Chaplygin Gas (GCG).

    Unlike LCDM/wCDM/CPL/JBP/BA -- which all just reparametrize
    dark energy's equation of state on top of an independent
    matter component -- GCG replaces dark energy with a single
    exotic fluid with equation of state

        p = -A / rho^alpha

    that unifies dark matter and dark energy: it behaves like
    pressureless dust at early times (large rho, small |p/rho|)
    and like a cosmological constant at late times (rho -> const),
    interpolating between the two as the universe expands. The
    ``Omega_m`` parameter here is therefore ordinary *baryonic +
    any additional* matter on top of the GCG fluid, not the
    GCG fluid's own dust-like contribution.

    Density evolution (closed-form solution of the GCG continuity
    equation):

        rho_GCG(z) / rho_GCG(0)
            = [ A_s + (1 - A_s) (1+z)^(3(1+alpha)) ]^(1/(1+alpha))

    Effective equation of state:

        w(z) = -A_s / [ A_s + (1-A_s)(1+z)^(3(1+alpha)) ]
               * (1+z)^(3(1+alpha)) ... equivalently
        w(z) = -1 + (1-A_s)(1+z)^(3(1+alpha))
                     / [A_s + (1-A_s)(1+z)^(3(1+alpha))]

    so w(0) = -A_s and w(z) -> 0 (dust-like) as z -> infinity.
    A_s = 1 reduces GCG to flat/curved LCDM (rho_GCG(z) = rho_GCG(0)
    for any alpha); alpha = 1 is the original ("pure") Chaplygin
    gas of Kamenshchik, Moschella & Pasquier (2001).

    Parameters
    ----------
    Uses the shared ``A_s`` and ``alpha`` fields of
    :class:`~cosmology.core.parameters.CosmologyParameters`
    (ignored by every other model, the same way LCDM ignores
    ``w0``/``wa``).

    References
    ----------
    Kamenshchik, Moschella & Pasquier (2001), Phys. Lett. B 511, 265,
    arXiv:gr-qc/0103004.
    Bento, Bertolami & Sen (2002), Phys. Rev. D 66, 043507,
    arXiv:gr-qc/0202064.
    """

    MODEL_NAME = "GCG"

    # ---------------------------------------------------------

    def _g(self, z):
        """
        A_s + (1-A_s)(1+z)^(3(1+alpha)) -- the base of the fde(z)
        and dlnfde/dz expressions below, computed once and shared
        between them.
        """

        z = np.asarray(z, dtype=float)

        return self.A_s + (1.0 - self.A_s) * (1.0 + z) ** (
            3.0 * (1.0 + self.alpha)
        )

    # ---------------------------------------------------------

    def w(self, z):
        """
        Effective dark-energy equation of state.
        """

        z = np.asarray(z, dtype=float)

        g = self._g(z)

        return -1.0 + (1.0 - self.A_s) * (1.0 + z) ** (
            3.0 * (1.0 + self.alpha)
        ) / g

    # ---------------------------------------------------------

    def fde(self, z):
        """
        GCG density evolution factor.

        Returns
        -------
        ndarray
            ρ_GCG(z) / ρ_GCG(0)
        """

        z = np.asarray(z, dtype=float)

        return self._g(z) ** (1.0 / (1.0 + self.alpha))

    # ---------------------------------------------------------

    def E(self, z):
        """
        Dimensionless Hubble parameter.

        Note: ``Omega_m`` here is *additional* matter on top of
        the GCG fluid (which supplies its own effective matter
        and dark-energy behavior) -- see the class docstring.
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

        g = self._g(z)
        fde = g ** (1.0 / (1.0 + self.alpha))

        # d ln(fde)/dz = 3(1-A_s)(1+z)^(3(1+alpha)-1) / g(z)
        # (== 3(1+w(z))/(1+z), by construction of fde above).
        dlnf_dz = (
            3.0
            * (1.0 - self.A_s)
            * (1.0 + z) ** (3.0 * (1.0 + self.alpha) - 1.0)
            / g
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
        GCG effective density parameter.
        """

        z = np.asarray(z, dtype=float)

        return self.Omega_de0 * self.fde(z)
