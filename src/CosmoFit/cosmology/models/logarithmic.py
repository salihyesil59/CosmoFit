"""
Logarithmic dark-energy equation of state.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class LogarithmicDE(Cosmology):
    r"""
    Logarithmic (Efstathiou) dark-energy parametrization,

        w(z) = w0 + wa ln(1 + z)

    with the resulting density evolution obtained by integrating
    the continuity equation exactly:

        rho_de(z)/rho_de0 = (1+z)^{3(1+w0)}
                            exp[(3/2) wa ln^2(1+z)]

    The fourth ``w0``-``wa`` parametrization in the library,
    alongside CPL, JBP and BA -- and it is here because it fails
    differently from all three. CPL, JBP and BA all *saturate*:
    each has a finite ``w(z -> infinity)``, so none can express a
    dark energy whose equation of state keeps drifting. The
    logarithmic form does not saturate, which makes it the natural
    control case for asking whether a detected ``wa`` is telling
    you about the data or about the shape you assumed.

    Efstathiou proposed it specifically as a better fit to
    quintessence models' actual ``w(z)`` over ``0 < z < 4`` than a
    linear-in-``z`` form, and it reuses the existing ``w0``/``wa``
    parameters, so it drops into any ``w0``-``wa`` comparison --
    including the w0-wa plane figure -- without new machinery.

    Caveat
    ------
    The ``ln^2(1+z)`` in the exponent grows without bound, so for
    ``wa > 0`` the dark-energy density diverges faster than any
    power law at high redshift, and for ``wa < 0`` it is driven to
    zero. Neither is pathological over the redshift range this
    library's datasets cover (``z < 2.5`` for BAO/SNe), but it does
    mean the model should not be extrapolated to recombination --
    which is exactly why fitting it against the *compressed*
    ``"planck"`` distance priors (which integrate ``E(z)`` out to
    ``z ~ 1090``) needs a sanity check on the resulting
    ``Omega_de(z*)``, not blind trust.

    References
    ----------
    Efstathiou (1999), "Constraining the equation of state of the
    Universe from distant Type Ia supernovae and cosmic microwave
    background anisotropies", MNRAS 310, 842,
    arXiv:astro-ph/9904356.
    """

    MODEL_NAME = "LogarithmicDE"
    MODEL_LABEL = r"Logarithmic $w(z)$"

    # ---------------------------------------------------------

    def w(self, z):

        z = np.asarray(z, dtype=float)

        return self.w0 + self.wa * np.log1p(z)

    # ---------------------------------------------------------

    def _density_factor(self, z):
        r"""
        ``rho_de(z) / rho_de0``.

        From ``dln rho_de / dln(1+z) = 3(1 + w)``:

            int_0^z 3[1 + w0 + wa ln(1+z')] dln(1+z')
                = 3(1+w0) L + (3/2) wa L^2,   L = ln(1+z)

        so the factor is ``exp`` of that. Written with ``log1p``
        for accuracy at small ``z``.
        """

        L = np.log1p(np.asarray(z, dtype=float))

        return np.exp(

            3.0 * (1.0 + self.w0) * L

            + 1.5 * self.wa * L ** 2

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):

        return self.Omega_de0 * self._density_factor(z)

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * (1.0 + z) ** 3

            +

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_de(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        # d(rho_de)/dz = rho_de * 3(1 + w(z)) / (1 + z), the
        # continuity equation itself -- no need to differentiate
        # the exponential by hand.
        d_omega_de = (

            self.Omega_de(z)

            * 3.0

            * (1.0 + self.w(z))

            / (1.0 + z)

        )

        return (

            (

                3.0 * self.Omega_m * (1.0 + z) ** 2

                +

                2.0 * self.Omega_k * (1.0 + z)

                +

                d_omega_de

            )

            /

            (2.0 * self.E(z))

        )
