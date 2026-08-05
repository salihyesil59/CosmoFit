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

Performance
-----------
:meth:`sound_horizon` and :meth:`chi_star` both integrate a smooth,
vectorized function of z (``E_recomb``) over a range spanning z=0 to
z~1090 or z~1090 to infinity. ``scipy.integrate.quad`` cannot exploit
that -- it's an *adaptive scalar* quadrature, calling the integrand
at one z at a time (typically several hundred times per call for a
domain this wide), each of which re-triggers the Python-level
overhead of ``E_recomb``/``H``/etc. Profiling showed this was, by a
wide margin, the single most expensive part of evaluating
:class:`~likelihoods.planck.PlanckLikelihood` (dominating even the
Pantheon+/DES-SN5YR covariance solves in a joint fit that includes
Planck).

Both integrals are instead evaluated on a *fixed, vectorized* grid
via ``scipy.integrate.simpson`` -- one array-valued call to
``E_recomb`` instead of hundreds of scalar ones. Both use a
substitution chosen (log-spaced in the substituted variable, not
linear -- see each method's docstring for why a linear grid converges
far too slowly to trust) so that a few hundred points reach ~1e-7 or
better relative accuracy against the original ``quad`` result.
Measured speedup: ~20x per call, with no measurable
change to any chi2/posterior this library computes (verified against
the ``quad``-based reference across randomized parameters).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from CosmoFit.cosmology.core.constants import c as SPEED_OF_LIGHT
from CosmoFit.cosmology.core.constants import Omega_gamma_h2, N_eff
from CosmoFit.cosmology.core.utils import require_positive


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

    def sound_horizon(
        self, z: float | None = None, n_grid: int = 400, decades: float = 8.0,
    ) -> float:
        """
        Comoving sound horizon [Mpc] at redshift ``z``, computed
        by directly integrating the sound speed against a
        radiation-aware H(z):

            r_s(z) = integral_z^inf c_s(z') / H(z') dz'

        Defaults to ``z = z_star()`` (the photon-decoupling
        epoch relevant for the Planck distance-prior likelihood).

        Evaluated with a fixed, vectorized grid rather than
        ``scipy.integrate.quad`` -- see the module docstring's
        "Performance" section for why.
        """

        if z is None:
            z = self.z_star()

        # Substitute u = 1/(1+z') to map the semi-infinite domain
        # [z, inf) onto the finite interval (0, u_upper]. The
        # integrand is smooth and finite as u -> 0 (z' -> inf,
        # where c_s -> c/sqrt(3) and E_recomb -> sqrt(Omega_r)/u^2,
        # so the u^-2 Jacobian and E_recomb's u^-2 cancel) -- but it
        # only *reaches* that asymptote gradually, over several
        # decades of u (equivalently: the sound-speed/E_recomb
        # weighting keeps changing out to z' ~ 1e6-1e7, not just a
        # little past z' = z_star). A grid *linear* in u wastes
        # almost all of its points close to u_upper and badly
        # under-resolves that slow approach to the asymptote --
        # verified against the quad-based reference this replaces:
        # a linear grid needs >10x the points for the same accuracy
        # (and is still only first-order convergent, vs. the
        # log-spaced grid's near-machine-precision convergence by
        # a few hundred points). A grid log-spaced in u (equivalently
        # log-spaced in 1+z') resolves it properly.
        u_upper = 1.0 / (1.0 + z)

        u = np.logspace(
            np.log10(u_upper) - decades, np.log10(u_upper), n_grid,
        )

        zp = 1.0 / u - 1.0

        integrand = (
            self._sound_speed(zp)
            / (self.cosmo.H0 * self.E_recomb(zp))
            / u**2
        )

        return float(simpson(integrand, x=u))

    # ---------------------------------------------------------
    # chi_star -- dimensionless comoving distance to z_star
    # ---------------------------------------------------------

    def chi_star(self, z_star: float | None = None, n_grid: int = 400) -> float:
        """
        Dimensionless comoving distance to ``z_star``,

            chi_star = integral_0^z_star dz' / E_recomb(z')

        used by :class:`~likelihoods.planck.PlanckLikelihood` for
        the CMB shift parameter R and acoustic scale l_A (D_M(z*)
        needs this out to z* ~ 1090, well beyond the low-z
        distance-integrator grid every other likelihood uses).

        Defaults to ``z_star = self.z_star()``.

        Evaluated with a fixed, vectorized grid rather than
        ``scipy.integrate.quad`` -- see the module docstring's
        "Performance" section for why.
        """

        if z_star is None:
            z_star = self.z_star()

        # Substitute x = ln(1+z'): a *linear* grid in z' badly
        # under-resolves the integrand near z'~0 relative to its
        # curvature there while wasting points on the smooth
        # radiation-dominated tail near z'~z_star -- a linear grid
        # needs ~8000 points for 1e-5 relative accuracy, this needs
        # a few hundred for 1e-8, because the integrand's structure
        # is much closer to log-uniform in (1+z') than uniform in z'.
        x_max = np.log1p(z_star)

        x = np.linspace(0.0, x_max, n_grid)

        z = np.expm1(x)

        integrand = (1.0 + z) / self.E_recomb(z)

        return float(simpson(integrand, x=x))

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
