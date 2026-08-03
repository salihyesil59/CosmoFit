"""
Recombination-epoch quantities.

This module turns a background cosmology (H0, Omega_m, Omega_b,
Omega_k, ...) into the two numbers the CMB "distance prior"
compression needs:

    z_star   redshift of photon decoupling
    r_s(z*)  comoving sound horizon at z*

z_star uses the standard Hu & Sugiyama (1996) fitting formula.
r_s(z*) is computed by directly integrating the sound-horizon
integral

    r_s(z) = integral_z^inf  c_s(z') / H(z')  dz'

    c_s(z) = c / sqrt(3 (1 + R_b(z)))
    R_b(z) = 3 omega_b_h2 / (4 Omega_gamma_h2 (1+z))

using the cosmology's own E(z) *plus* a radiation term
Omega_r (1+z)^4 added on top of it. Radiation is deliberately
excluded from the background E(z) used by LCDM/CPL themselves
(it is irrelevant for the late-time CC/BAO/SN likelihoods those
models target, and adding it there would needlessly complicate
every other calculator), but at z* ~ 1090 it contributes ~20% of
the total energy budget and cannot be dropped without biasing
r_s (and hence l_A) by several percent -- so it is added in
locally, only for this integral.

References
----------
z_star : Hu & Sugiyama (1996), ApJ 471, 542, Eq. (E-1).
r_s integral / R_b(z) : Hu & Sugiyama (1996); Eisenstein & Hu
    (1998), ApJ 496, 605.
Omega_gamma_h2, N_eff : see cosmology.core.constants.

A simpler (but less accurate -- see :meth:`sound_horizon_eh98`)
alternative is the Eisenstein & Hu (1998) Eq. (26) closed-form
fit, kept here for comparison/testing; it is a fit to the sound
horizon at the *drag* epoch, not z_star, and does not account for
radiation domination, so it overestimates r_s(z*) by several
Mpc for typical LCDM parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from cosmology.core.constants import c as SPEED_OF_LIGHT
from cosmology.core.constants import Omega_gamma_h2, N_eff
from cosmology.core.utils import require_positive


class RecombinationCalculator:
    """
    Recombination-epoch quantities for a given cosmology.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmological model. Uses ``H0``, ``Omega_m``, ``Omega_k``,
        ``Omega_b``, and the model's own ``E(z)``.
    """

    def __init__(self, cosmology):

        self.cosmo = cosmology

    # ---------------------------------------------------------
    # Derived densities
    # ---------------------------------------------------------

    @property
    def h(self) -> float:
        """Dimensionless Hubble parameter, h = H0 / 100."""

        return self.cosmo.H0 / 100.0

    @property
    def omega_m_h2(self) -> float:
        """Physical matter density, Omega_m * h^2."""

        return self.cosmo.Omega_m * self.h**2

    @property
    def omega_b_h2(self) -> float:
        """Physical baryon density, Omega_b * h^2."""

        return self.cosmo.Omega_b * self.h**2

    @property
    def Omega_r(self) -> float:
        """
        Present-day radiation density parameter (photons + the
        standard-model effective number of neutrino species).
        """

        omega_r_h2 = Omega_gamma_h2 * (1.0 + 0.2271 * N_eff)

        return omega_r_h2 / self.h**2

    # ---------------------------------------------------------
    # z_star -- Hu & Sugiyama (1996)
    # ---------------------------------------------------------

    def z_star(self) -> float:
        """
        Redshift of photon decoupling (Hu & Sugiyama 1996).
        """

        wb = require_positive(self.omega_b_h2, "omega_b_h2")
        wm = require_positive(self.omega_m_h2, "omega_m_h2")

        g1 = (
            0.0783 * wb ** (-0.238)
            / (1.0 + 39.5 * wb**0.763)
        )

        g2 = (
            0.560
            / (1.0 + 21.1 * wb**1.81)
        )

        return (
            1048.0
            * (1.0 + 0.00124 * wb ** (-0.738))
            * (1.0 + g1 * wm**g2)
        )

    # ---------------------------------------------------------
    # r_s -- direct, radiation-aware integral (primary method)
    # ---------------------------------------------------------

    def E_recomb(self, z):
        """
        E(z) including the radiation term -- for use in any
        early-universe (z ~ z_star) integral, since the
        cosmology's own ``E(z)`` deliberately omits radiation
        (see module docstring).
        """

        return np.sqrt(
            self.cosmo.E(z) ** 2
            + self.Omega_r * (1.0 + z) ** 4
        )

    def _sound_speed(self, z):
        """Baryon-photon sound speed c_s(z) [km/s]."""

        Rb = (
            3.0 * self.omega_b_h2
            / (4.0 * Omega_gamma_h2 * (1.0 + z))
        )

        return SPEED_OF_LIGHT / np.sqrt(3.0 * (1.0 + Rb))

    def sound_horizon(self, z: float | None = None) -> float:
        """
        Comoving sound horizon [Mpc] at redshift ``z``, computed
        by directly integrating the sound speed against a
        radiation-aware H(z):

            r_s(z) = integral_z^inf c_s(z') / H(z') dz'

        Defaults to ``z = z_star()`` (the photon-decoupling
        epoch relevant for the Planck distance-prior likelihood).
        """

        if z is None:
            z = self.z_star()

        integral, _ = quad(
            lambda zp: self._sound_speed(zp) / (self.cosmo.H0 * self.E_recomb(zp)),
            z,
            np.inf,
            limit=200,
        )

        return integral

    # ---------------------------------------------------------
    # r_s -- Eisenstein & Hu (1998) Eq. (26) fit (comparison only)
    # ---------------------------------------------------------

    def sound_horizon_eh98(self) -> float:
        """
        Comoving sound horizon [Mpc] via the closed-form
        Eisenstein & Hu (1998) Eq. (26) fitting formula:

            r_s = 44.5 * ln(9.83 / omega_m_h2)
                  / sqrt(1 + 10 * omega_b_h2^(3/4))    [Mpc]

        This is a fit to the sound horizon at the *drag* epoch
        (z_drag ~ 1060), not z_star (~1090), and does not account
        for radiation domination explicitly (it is baked into the
        fit at the parameter values EH98 calibrated against).
        Provided for comparison/testing; :meth:`sound_horizon` is
        the more accurate, radiation-aware calculation and is
        what :class:`likelihoods.planck.PlanckLikelihood` uses.
        """

        wb = require_positive(self.omega_b_h2, "omega_b_h2")
        wm = require_positive(self.omega_m_h2, "omega_m_h2")

        return (
            44.5 * np.log(9.83 / wm)
            / np.sqrt(1.0 + 10.0 * wb**0.75)
        )

    # ---------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------

    def summary(self) -> dict:
        """
        Return z_star, r_s(z_star), and the densities they were
        computed from, in a single dict.
        """

        z_star = self.z_star()

        return {
            "z_star": z_star,
            "r_s": self.sound_horizon(z_star),
            "r_s_eh98": self.sound_horizon_eh98(),
            "omega_m_h2": self.omega_m_h2,
            "omega_b_h2": self.omega_b_h2,
            "Omega_r": self.Omega_r,
        }
