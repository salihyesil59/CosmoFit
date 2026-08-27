"""
Recombination-epoch quantities for the CMB distance priors.

This module turns a background cosmology (H0, Omega_m, Omega_b,
Omega_k, ...) into the numbers the CMB "distance prior" compression
needs:

    z_star   redshift of photon decoupling
    r_s(z*)  comoving sound horizon at z*
    chi_star dimensionless comoving distance to z*

Why these follow the distance-prior conventions exactly
-------------------------------------------------------
A compressed CMB likelihood is not a measurement of the sky -- it is a
*summary of someone else's fit*, computed under a specific set of
definitions. The theory prediction must therefore be built with the
same definitions the compression was, or the comparison is
apples-to-oranges no matter how good the physics in either half is.

This bit us concretely. An earlier version of this module used a more
detailed radiation model than the priors were derived with (photons
plus 3.046 massless neutrinos, omega_r = 4.18e-5) and took z_star from
the Hu & Sugiyama (1996) fitting formula. Evaluated at *Planck's own
best-fit LCDM*, where a correct implementation must return chi2 ~ 0,
that produced:

    l_A = 302.34  vs  301.471 +- 0.090  ->  -8.9 sigma,  chi2 ~ 100

for 3 data points -- a systematic large enough to visibly drag H0,
Omega_m and r_d in any joint fit including "planck". R and omega_b
were fine (0.13 and 0.07 sigma), which is what localized it to
r_s(z*).

Both halves of that are now matched to the source of the priors, Chen,
Huang & Wang (2019), arXiv:1808.05724, Eqs. (1)-(6):

* **Radiation.** CHW19 define Omega_r = Omega_m / (1 + z_eq) with
  z_eq = 2.5e4 Omega_m h^2 (T_CMB/2.7 K)^-4 -- 0.8% below the
  photon+neutrino sum, and with massive neutrinos left inside
  Omega_m at every redshift. Less radiation means later
  matter-radiation equality and a larger sound horizon.

* **z_star.** CHW19 state they take z_star from the Planck 2018
  chains, i.e. from CAMB's recombination -- not from a fitting
  formula. Hu & Sugiyama (1996) was calibrated against 1990s
  recombination calculations and runs ~0.22% high for Planck-like
  parameters (1091.91 vs CAMB's 1089.94), and l_A is sensitive enough
  to z_star (dl_A/dz_star ~ 0.18) that this alone is a ~4 sigma
  shift. :meth:`z_star` therefore uses a fit calibrated directly
  against CAMB -- see :data:`_ZSTAR_COEF`. The Hu & Sugiyama value
  remains available as :meth:`z_star_hs96` for comparison.

Together these bring the same fiducial check to

    l_A = 301.42,  R = 1.7503   ->  +0.55 and -0.03 sigma,  chi2 ~ 0.3

i.e. the likelihood now reproduces the cosmology its own data came
from, which is the minimum bar for using it in a fit.

References
----------
Distance-prior definitions, Omega_r, R_b, and the data vector:
    Chen, Huang & Wang (2019), JCAP 02 (2019) 028, arXiv:1808.05724.
z_star (comparison only): Hu & Sugiyama (1996), ApJ 471, 542, Eq. (E-1).
r_s / R_b: Hu & Sugiyama (1996); Eisenstein & Hu (1998), ApJ 496, 605.

Performance
-----------
:meth:`sound_horizon` and :meth:`chi_star` both integrate a smooth,
vectorized function of z over a range spanning z=0 to z~1090 or
z~1090 to infinity. ``scipy.integrate.quad`` cannot exploit that --
it's an *adaptive scalar* quadrature, calling the integrand one z at a
time (typically several hundred times per call for a domain this
wide), each re-triggering the Python-level overhead of ``E_cmb``.
Profiling showed this was, by a wide margin, the single most expensive
part of evaluating :class:`~likelihoods.planck.PlanckLikelihood`.

Both integrals are instead evaluated on a *fixed, vectorized* grid via
``scipy.integrate.simpson`` -- one array-valued call to ``E_cmb``
instead of hundreds of scalar ones -- using a substitution chosen
(log-spaced in the substituted variable, not linear) so that a few
hundred points reach ~1e-7 or better relative accuracy. Measured
speedup ~20x per call, with no measurable change to any chi2.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.powers import cube
from scipy.integrate import simpson

from CosmoFit.cosmology.core.constants import c as SPEED_OF_LIGHT
from CosmoFit.cosmology.core.constants import Tcmb
from CosmoFit.cosmology.core.utils import require_positive


#: (T_CMB / 2.7 K)^-4, the temperature scaling CHW19 write their
#: z_eq and R_b coefficients with.
_T_RATIO = (Tcmb / 2.7) ** -4

#: Baryon-loading coefficient: R_b(a) = _RB_COEF * omega_b_h2 * a,
#: i.e. CHW19 Eq. (3)'s ``31500 (T_CMB/2.7 K)^-4``. This is the same
#: quantity as the more familiar 3 omega_b / (4 omega_gamma) to within
#: 0.03%; their literal coefficient is used here so the sound-horizon
#: integral matches the priors' own definition exactly.
_RB_COEF = 31500.0 * _T_RATIO

#: Coefficients of a fit to CAMB's z_star, in the basis
#: [1, ln wb, ln wm, ln^2 wb, ln^2 wm, ln wb ln wm, ln^3 wb, ln^3 wm]
#: predicting ln(z_star). Calibrated against CAMB 2.0.1 (recombination
#: via RECFAST, Y_He from BBN, one massive neutrino of 0.06 eV -- the
#: Planck 2018 fiducial the priors assume) over a 12x14 grid spanning
#: omega_b in [0.0195, 0.0250] and omega_cb in [0.105, 0.175], which
#: brackets the Planck posterior many times over.
#:
#: Max residual over that grid is 0.0018% in z_star (0.019 absolute),
#: i.e. ~0.04 sigma of the l_A prior -- negligible next to the 0.22%
#: bias of the Hu & Sugiyama formula it replaces. z_star depends only
#: on the physical densities, not on H0 (verified explicitly against
#: CAMB: identical to 7 digits for H0 = 60, 67.36 and 75).
_ZSTAR_COEF = np.array([
    6.894840626,
    -0.06043207648,
    -0.007255456586,
    -0.01563497771,
    0.004901388276,
    -0.008936335909,
    -0.00238730513,
    0.0003358198277,
])

#: Calibration range of :data:`_ZSTAR_COEF`, as (min, max) in
#: omega_b_h2 and omega_m_h2. Outside it the fit is an extrapolation.
_ZSTAR_WB_RANGE = (0.0195, 0.0250)
_ZSTAR_WM_RANGE = (0.1056, 0.1757)


class RecombinationCalculator:
    """
    Recombination-epoch quantities for a given cosmology, following
    the CMB distance-prior conventions (see the module docstring).

    Parameters
    ----------
    cosmology : Cosmology
        Cosmological model. Uses ``H0``, ``Omega_m``, ``Omega_k``,
        ``Omega_b``, and the model's own ``E(z)``/``Omega_de(z)``.
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
    def z_eq(self) -> float:
        """
        Matter-radiation equality redshift, CHW19 Eq. (6):

            z_eq = 2.5e4 * Omega_m h^2 * (T_CMB / 2.7 K)^-4
        """

        return 2.5e4 * self.omega_m_h2 * _T_RATIO

    @property
    def Omega_r(self) -> float:
        """
        Present-day radiation density parameter, defined as the
        priors define it (CHW19 Eq. 6):

            Omega_r = Omega_m / (1 + z_eq)

        This is ~0.8% below the photon + 3.046-massless-neutrino sum,
        and it leaves massive neutrinos inside ``Omega_m`` at every
        redshift. Both are deliberate: see the module docstring on why
        the prediction has to share the compression's definitions
        rather than improve on them.
        """

        return self.cosmo.Omega_m / (1.0 + self.z_eq)

    # ---------------------------------------------------------
    # z_star
    # ---------------------------------------------------------

    def z_star(self) -> float:
        """
        Redshift of photon decoupling, from a fit calibrated against
        CAMB (see :data:`_ZSTAR_COEF`).

        CHW19 take z_star from the Planck 2018 chains, so matching
        their priors needs a CAMB-consistent z_star rather than the
        older Hu & Sugiyama fit (available as :meth:`z_star_hs96`),
        which runs ~0.22% high for Planck-like parameters.
        """

        wb = require_positive(self.omega_b_h2, "omega_b_h2")
        wm = require_positive(self.omega_m_h2, "omega_m_h2")

        lb, lm = np.log(wb), np.log(wm)

        basis = np.array([
            1.0, lb, lm, lb**2, lm**2, lb * lm, lb**3, lm**3,
        ])

        return float(np.exp(basis @ _ZSTAR_COEF))

    def z_star_hs96(self) -> float:
        """
        Redshift of photon decoupling from the Hu & Sugiyama (1996)
        fitting formula, Eq. (E-1).

        Kept for comparison only. It is ~0.22% high relative to CAMB
        for Planck-like parameters (1091.91 vs 1089.94), which is a
        ~4 sigma shift in l_A -- see the module docstring.
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
    # Radiation-aware expansion rate
    # ---------------------------------------------------------

    def E_cmb(self, z):
        """
        E(z) including radiation, normalized so that ``E_cmb(0) = 1``.

        The models' own ``E(z)`` deliberately omits radiation (it is
        irrelevant to the late-time CC/BAO/SN likelihoods they target),
        and closes the budget with
        ``Omega_de0 = 1 - Omega_m - Omega_k``. The priors' E(z) instead
        closes it with ``1 - Omega_m - Omega_k - Omega_r``, so adding
        ``Omega_r (1+z)^4`` on top of the model's E(z) is not enough --
        that would leave ``E(0)^2 = 1 + Omega_r``, over-closing the
        universe by ~1e-4 and biasing every distance slightly.

        Writing the model's dark-energy evolution as
        ``f(z) = Omega_de(z) / Omega_de0`` (identically 1 for LCDM),
        the correctly-renormalized version is

            E_cmb(z)^2 = E(z)^2 + Omega_r [(1+z)^4 - f(z)]

        which is exact for every model in the library -- it reuses each
        model's own dark-energy evolution rather than assuming a
        cosmological constant.
        """

        z = np.asarray(z, dtype=float)

        E2 = (
            self.cosmo.E(z) ** 2
            + self.Omega_r * (
                (1.0 + z) * cube(1.0 + z) - self._de_evolution(z)
            )
        )

        return np.sqrt(E2)

    # ---------------------------------------------------------

    def _de_evolution(self, z):
        """
        ``f(z) = Omega_de(z) / Omega_de0``, the model's dark-energy
        density relative to today (identically 1 for a cosmological
        constant).

        ``Omega_de`` is optional -- the base class raises
        ``NotImplementedError``, and a
        :func:`~cosmology.custom.define_model` model only has one if
        the user supplied it -- so fall back to ``f = 1`` when it is
        unavailable rather than making the Planck likelihood require
        it (it did not before this renormalization was added).

        The fallback is nearly exact regardless of the true evolution:
        ``f`` only ever appears multiplied by ``Omega_r ~ 9e-5``, and
        only as a correction to a ``(1+z)^4`` term that reaches ~1e12
        by z*. At z=0 it is exact for *every* model, since
        ``f(0) = 1`` by definition -- which is precisely where the
        renormalization has to be right.
        """

        z = np.asarray(z, dtype=float)

        Omega_de0 = self.cosmo.Omega_de0

        if Omega_de0 == 0.0:
            return np.zeros_like(z)

        try:
            Omega_de = self.cosmo.Omega_de(z)
        except NotImplementedError:
            return np.ones_like(z)

        return np.asarray(Omega_de, dtype=float) / Omega_de0

    # ---------------------------------------------------------
    # Sound horizon
    # ---------------------------------------------------------

    def _sound_speed(self, z):
        """
        Baryon-photon sound speed c_s(z) [km/s], with the baryon
        loading written as CHW19 Eq. (3) does:

            R_b(z) = 31500 (T_CMB/2.7 K)^-4 omega_b_h2 / (1+z)
        """

        Rb = _RB_COEF * self.omega_b_h2 / (1.0 + z)

        return SPEED_OF_LIGHT / np.sqrt(3.0 * (1.0 + Rb))

    def sound_horizon(
        self, z: float | None = None, n_grid: int = 400, decades: float = 8.0,
    ) -> float:
        """
        Comoving sound horizon [Mpc] at redshift ``z``:

            r_s(z) = integral_z^inf c_s(z') / H(z') dz'

        Defaults to ``z = z_star()``.

        Evaluated with a fixed, vectorized grid rather than
        ``scipy.integrate.quad`` -- see the module docstring's
        "Performance" section.
        """

        if z is None:
            z = self.z_star()

        # Substitute u = 1/(1+z') to map the semi-infinite domain
        # [z, inf) onto the finite interval (0, u_upper]. The
        # integrand is smooth and finite as u -> 0 (z' -> inf, where
        # c_s -> c/sqrt(3) and E_cmb -> sqrt(Omega_r)/u^2, so the u^-2
        # Jacobian and E_cmb's u^-2 cancel) -- but it only *reaches*
        # that asymptote gradually, over several decades of u. A grid
        # linear in u wastes almost all its points near u_upper and
        # badly under-resolves that approach: it needs >10x the points
        # for the same accuracy and is still only first-order
        # convergent, versus the log-spaced grid's near-machine
        # convergence by a few hundred points.
        u_upper = 1.0 / (1.0 + z)

        u = np.logspace(
            np.log10(u_upper) - decades, np.log10(u_upper), n_grid,
        )

        zp = 1.0 / u - 1.0

        integrand = (
            self._sound_speed(zp)
            / (self.cosmo.H0 * self.E_cmb(zp))
            / u**2
        )

        return float(simpson(integrand, x=u))

    # ---------------------------------------------------------
    # chi_star
    # ---------------------------------------------------------

    def chi_star(self, z_star: float | None = None, n_grid: int = 400) -> float:
        """
        Dimensionless comoving distance to ``z_star``,

            chi_star = integral_0^z_star dz' / E_cmb(z')

        used by :class:`~likelihoods.planck.PlanckLikelihood` for the
        CMB shift parameter R and acoustic scale l_A (D_M(z*) needs
        this out to z* ~ 1090, well beyond the low-z distance-
        integrator grid every other likelihood uses).

        Defaults to ``z_star = self.z_star()``.
        """

        if z_star is None:
            z_star = self.z_star()

        # Substitute x = ln(1+z'): a linear grid in z' under-resolves
        # the integrand near z'~0 relative to its curvature there while
        # wasting points on the smooth radiation-dominated tail -- it
        # needs ~8000 points for 1e-5 relative accuracy where this
        # needs a few hundred for 1e-8, because the integrand's
        # structure is much closer to log-uniform in (1+z').
        x_max = np.log1p(z_star)

        x = np.linspace(0.0, x_max, n_grid)

        z = np.expm1(x)

        integrand = (1.0 + z) / self.E_cmb(z)

        return float(simpson(integrand, x=x))

    # ---------------------------------------------------------
    # Comparison-only fitting formula
    # ---------------------------------------------------------

    def sound_horizon_eh98(self) -> float:
        """
        Comoving sound horizon [Mpc] via the closed-form
        Eisenstein & Hu (1998) Eq. (26) fitting formula:

            r_s = 44.5 ln(9.83 / omega_m_h2)
                  / sqrt(1 + 10 omega_b_h2^(3/4))    [Mpc]

        This is a fit to the sound horizon at the *drag* epoch
        (z_drag ~ 1060), not z_star (~1090), and does not account for
        radiation domination explicitly. Provided for
        comparison/testing; :meth:`sound_horizon` is what
        :class:`likelihoods.planck.PlanckLikelihood` uses.
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
            "z_star_hs96": self.z_star_hs96(),
            "r_s": self.sound_horizon(z_star),
            "r_s_eh98": self.sound_horizon_eh98(),
            "omega_m_h2": self.omega_m_h2,
            "omega_b_h2": self.omega_b_h2,
            "z_eq": self.z_eq,
            "Omega_r": self.Omega_r,
        }
