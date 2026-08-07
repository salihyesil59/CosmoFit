"""
f(R,T) modified gravity, linear matter-geometry coupling.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class FRTLinear(Cosmology):
    r"""
    f(R,T) gravity, linear model: f(R,T) = R + 2*lambda*T.

    f(R,T) gravity (Harko, Lobo, Nojiri & Odintsov 2011) makes the
    gravitational Lagrangian depend on the trace T of the matter
    stress-energy tensor as well as the Ricci scalar R, coupling
    gravity directly to the matter content rather than adding a
    separate dark-energy fluid on top of standard GR. This is the
    original, simplest member of that family: f(T) = lambda*T.

    Reference derivation (their eq. 26, for dust p=0, T=rho):
    3H^2 = (8*pi + 3*lambda)*rho. Extending their eq. 23 (general
    perfect fluid, not just dust) from a single dust fluid to a
    two-component universe -- pressureless matter (rho_m, p_m=0) and
    a Lambda-like component (rho_L, p_L=-rho_L) -- gives (cross-
    checked: this reduces to their eq. 26 exactly as rho_L -> 0):

        3H^2 = (8*pi + 3*lambda)*rho_m + (8*pi + 4*lambda)*rho_L

    Non-dimensionalizing with beta = lambda / (8*pi) (beta=0 is
    exactly GR):

        E(z)^2 = Omega_k*(1+z)^2
                 + (1 + 3*beta) * Omega_m * (1+z)^3
                 + (1 + 4*beta) * Omega_L

    Unlike LCDM, ``Omega_m`` and ``Omega_L`` are independent free
    parameters here, *not* tied by a flatness closure
    (Omega_L = 1 - Omega_m - Omega_k) -- at beta=0 that closure falls
    out on its own, but for beta != 0 it generally doesn't, which
    matches how f(R,T) papers actually fit this model (independent
    Omega_m/Omega_L posteriors, not summing to 1). Only the linear
    (f(T) proportional to T) case is implemented; the general
    f(R,T) = R + alpha*T^n form is not, since the units/normalization
    convention for alpha in the papers surveyed for this couldn't be
    pinned down with confidence -- see the project's dev notes.

    Parameters
    ----------
    Adds ``Omega_L`` (the Lambda-like component's density parameter,
    default 0.7, independent of ``Omega_m``) and ``beta`` (the
    dimensionless matter-geometry coupling, default 0.0 = GR) via
    ``EXTRA_PARAMS``.

    References
    ----------
    Harko, Lobo, Nojiri & Odintsov (2011), "f(R,T) gravity", Phys.
    Rev. D 84, 024020, arXiv:1104.2669.
    """

    MODEL_NAME = "FRTLinear"

    #: ``beta``'s bounds are deliberately narrow: this is a weak-
    #: coupling perturbation around GR (beta=0), and literature fits
    #: of this model find it consistent with |beta| of a few percent
    #: -- values of order 1 aren't just "strongly coupled", they can
    #: make E(z)^2 negative (unphysical) at moderate-to-high z for
    #: otherwise-ordinary Omega_m/Omega_L, since (1+3*beta) and
    #: (1+4*beta) can go negative and dominate. `LogPosterior`
    #: treats that region as chi2=+inf / log-posterior=-inf rather
    #: than crashing, but bounding the prior to the physically
    #: sensible regime is the right fix, not just a safety net.
    EXTRA_PARAMS = {
        "Omega_L": {
            "default": 0.7, "bounds": (0.0, 1.2), "label": r"$\Omega_L$",
        },
        "beta": {
            "default": 0.0, "bounds": (-0.2, 0.2), "label": r"$\beta$",
        },
    }

    # ---------------------------------------------------------

    def E(self, z):
        """
        Dimensionless Hubble parameter.
        """

        z = np.asarray(z, dtype=float)

        return np.sqrt(
            self.Omega_k * (1.0 + z) ** 2
            + (1.0 + 3.0 * self.beta) * self.Omega_m * (1.0 + z) ** 3
            + (1.0 + 4.0 * self.beta) * self.Omega_L
        )

    # ---------------------------------------------------------

    def dEdz(self, z):
        """
        Derivative of E(z).
        """

        z = np.asarray(z, dtype=float)

        dE2_dz = (
            2.0 * self.Omega_k * (1.0 + z)
            + 3.0 * (1.0 + 3.0 * self.beta) * self.Omega_m * (1.0 + z) ** 2
        )

        return dE2_dz / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z):
        """
        Effective dark-energy density (the Lambda-like component,
        rescaled by the matter-geometry coupling -- constant in z,
        as in LCDM).
        """

        z = np.asarray(z, dtype=float)

        return np.full_like(z, (1.0 + 4.0 * self.beta) * self.Omega_L)
