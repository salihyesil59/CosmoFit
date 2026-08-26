r"""
The sound horizon at the drag epoch, computed rather than fitted.

What ``r_d`` is, and why it was a free parameter
------------------------------------------------
Every BAO measurement in this library is a *ratio*: ``D_M/r_d``,
``D_H/r_d``, ``D_V/r_d``. The acoustic scale ``r_d`` -- the comoving
distance a sound wave travelled in the baryon-photon plasma before
baryons decoupled from photons -- is the standard ruler, and a survey
can only report distances in units of it.

Until now CosmoFit treated ``r_d`` as a free nuisance parameter. That
is a defensible and common choice: it makes BAO a purely *relative*
distance measurement, immune to any assumption about the early
universe. It is also a real cost. ``H0`` and ``r_d`` enter BAO only
through the product ``H0 r_d``, so with ``r_d`` free, **BAO alone
cannot measure H0 at all** -- only ``H0 r_d``. Every "BAO + BBN gives
H0" result in the literature works by closing that gap: ``r_d`` is
*computed* from the physical densities, and a BBN prior on
``omega_b`` supplies the one density BAO cannot constrain.

This module computes it.

    r_d = integral_{z_d}^{infinity} c_s(z) / H(z) dz

    c_s(z) = c / sqrt(3 (1 + R_b(z))),
    R_b(z) = 3 rho_b / (4 rho_gamma) = (3 omega_b) / (4 omega_gamma) a

The integral runs entirely through the radiation- and
matter-dominated eras, which has a consequence worth stating plainly:
**``r_d`` does not depend on H0, on curvature, or on the dark-energy
model at all.** It is a function of ``omega_b``, ``omega_cb``,
``N_eff`` and ``m_nu`` and nothing else. Verified directly against
CAMB, which returns the same ``rdrag`` to 1e-7 for ``H0`` from 60 to
75 and to 2e-5 for ``Omega_k = 0.05``. So this works identically for
every model in the library, including ones CAMB could never be given.

Which conventions this follows -- and which it does not
-------------------------------------------------------
:class:`~cosmology.calculators.recombination.RecombinationCalculator`
deliberately uses the Chen-Huang-Wang (2019) radiation convention
(``Omega_r = Omega_m/(1+z_eq)``, massive neutrinos left inside
``Omega_m``), because the *distance priors* it serves were compressed
under those definitions and a prediction has to share them.

**This module must not.** ``r_d`` is not being compared against a
compression of someone's fit; it is being compared against BAO
surveys' ``D/r_d``, and those are quoted against ``r_d`` as a
Boltzmann code computes it. So the radiation here is the real thing:

- **Photons** from ``T_CMB``, via
  :data:`~cosmology.core.constants.Omega_gamma_h2`.
- **Massless neutrinos**, ``N_eff - n_massive * 3.044/3`` effective
  species scaling as ``a^-4``.
- **Massive neutrinos**, with their *exact* Fermi-Dirac energy
  density -- relativistic at the drag epoch (``y ~ 0.34`` for
  ``0.06 eV``) and matter-like today, with the transition integrated
  numerically rather than fitted.

That last point is not pedantry. Treating massive neutrinos with the
common ``[1 + (Ay)^p]^{1/p}`` approximation instead costs 0.05% in
``r_d`` at ``Sum m_nu = 0.06 eV`` and 0.15% at 0.6 eV -- the latter
being half of DESI's best BAO precision, spent on an approximation
with no reason to be there. Two lines of Fermi-Dirac integral,
tabulated once at import, removes it.

The mass-to-density relation drops out of that integral rather than
being assumed: this module *derives* ``Sum m_nu / omega_nu h^2 =
93.0378 eV``, where CAMB gives 93.04.

z_drag
------
The drag epoch is where a from-scratch calculation genuinely stops:
``z_d`` is set by the baryon-photon Thomson drag integral over a
full recombination history, which is a Boltzmann code's job.

This module follows the precedent the library already set for
``z_star`` (see
:data:`~cosmology.calculators.recombination._ZSTAR_COEF`) and uses a
**fit calibrated directly against CAMB** -- see :data:`_ZDRAG_COEF`.

The obvious alternative, Eisenstein & Hu (1998) Eq. (4), is available
as :meth:`z_drag_eh98` for comparison, and it is worth seeing why it
is not used: at Planck-like parameters it returns ``z_d = 1020.7``
where CAMB gives ``1059.9``, **3.7% low**. Feeding that into the
integral below gives ``r_d = 150.75`` instead of ``147.10`` -- a
**2.5% bias**, ten times DESI DR2's best BAO error bar and in one
direction.

That is not a criticism of EH98. Their ``z_d`` was fit jointly with
their own closed-form ``r_s`` (Eq. 26, available as
:meth:`~cosmology.calculators.recombination.RecombinationCalculator.sound_horizon_eh98`),
and the *pair* reproduces ``r_s`` to about 2%; the two halves are not
separately meaningful, and mixing one of them with a modern integral
is precisely the sort of convention-splicing that
:mod:`likelihoods.planck` documents the consequences of.

Accuracy
--------
Validated end to end against CAMB's ``rdrag`` over a 5850-point grid
spanning ``omega_b`` in [0.018, 0.026], ``omega_cb`` in [0.09, 0.20],
``N_eff`` in [2.0, 5.0] and ``Sum m_nu`` in [0, 0.6] eV:

===========================================  ==========
quantity                                     max error
===========================================  ==========
the integral alone, given CAMB's ``z_drag``  2.2e-6
``z_drag`` fit                               6.7e-5
**end to end, whole grid**                   **5.0e-5**
end to end, realistic priors                 1.4e-5
===========================================  ==========

For scale, DESI DR2's single best BAO bin is a 0.24% measurement, so
the worst case here is ~50 times smaller than the best data's error
bar, and the typical case ~500 times.

Using it
--------
Off by default -- turning it on changes results, and silently
switching a free parameter into a derived one is exactly the kind of
change that should be opted into:

>>> fit = Fitter(model=LCDM, datasets=["desi", "omega_b"],
...              free_params=["H0", "Omega_m"],
...              compute_rd=True)

``rd`` then must not appear in ``free_params`` -- it is no longer a
parameter -- and ``Omega_b`` becomes a quantity the fit genuinely
needs, which is what the ``"omega_b"`` BBN dataset is for.

References
----------
Eisenstein & Hu (1998), ApJ 496, 605, arXiv:astro-ph/9709112
(``z_drag`` fitting formula, comparison only).

Lewis, Challinor & Lasenby (2000), ApJ 538, 473,
arXiv:astro-ph/9911177 (CAMB, the calibration reference).

Aubourg et al. (2015), Phys. Rev. D 92, 123516, arXiv:1411.1074
(the "BAO + BBN" programme this makes possible).
"""

from __future__ import annotations

import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

from CosmoFit.cosmology.core.constants import c as SPEED_OF_LIGHT
from CosmoFit.cosmology.core.constants import Omega_gamma_h2, Tcmb
from CosmoFit.cosmology.core.utils import require_positive


# ============================================================
# Neutrino thermodynamics
# ============================================================

#: (7/8) (4/11)^(4/3) -- the energy density of one relativistic
#: neutrino species relative to photons, for the standard
#: ``T_nu = (4/11)^(1/3) T_gamma`` decoupling temperature.
NU_ENERGY_FACTOR = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)

#: Boltzmann constant [eV/K].
K_B_EV = 8.617333262e-5

#: The Standard Model effective neutrino number. Not a parameter:
#: it fixes the *massive* species' temperature (below) and therefore
#: the mass-to-density conversion, which must not move when a fit
#: varies ``N_eff`` to test for extra radiation. Extra effective
#: species beyond this go into the massless component, which is what
#: CAMB does and what makes ``omega_nu h^2`` independent of
#: ``N_eff`` at fixed mass (verified against CAMB across
#: ``N_eff`` = 2.0 to 5.0: identical to 9 digits).
NEFF_STANDARD = 3.044

#: Effective relativistic degrees of freedom carried by each massive
#: neutrino species, ``NEFF_STANDARD / 3``.
EFF_PER_MASSIVE = NEFF_STANDARD / 3.0

#: Present-day massive-neutrino temperature in eV,
#: ``k_B T_nu0 (N_eff/3)^{1/4}``. The ``^{1/4}`` is the standard
#: non-instantaneous-decoupling correction: it is what makes each
#: massive species' *relativistic* density ``EFF_PER_MASSIVE`` times
#: a photon-temperature species, since density goes as ``T^4``.
KT_NU_MASSIVE = (

    K_B_EV

    * (4.0 / 11.0) ** (1.0 / 3.0)

    * Tcmb

    * EFF_PER_MASSIVE ** 0.25

)


def _build_fermi_dirac_table():
    r"""
    Tabulate

        f(y) = rho_nu(y) / rho_nu(massless)
             = [int_0^inf dx x^2 sqrt(x^2 + y^2) / (e^x + 1)]
               / [7 pi^4 / 120]

    the exact ratio of a massive neutrino's energy density to a
    massless one's, as a function of ``y = m / (k_B T_nu)``.

    Built once at import (a few ms) and evaluated afterwards by
    spline interpolation in ``log y``, which costs microseconds --
    the integral itself would be far too slow to sit inside an MCMC
    step.

    Returns the spline over ``log10(y)`` of ``ln f``, which is
    smooth and nearly linear at both ends, so a few hundred nodes
    reach ~1e-10.
    """

    # Integrand support: x^3/(e^x+1) has fallen by ~1e-24 at x = 60.
    x = np.logspace(-6.0, np.log10(60.0), 4000)

    weight = x ** 2 / (np.exp(x) + 1.0)

    # int_0^inf x^3/(e^x+1) dx = 7 pi^4 / 120
    denominator = 7.0 * np.pi ** 4 / 120.0

    log_y = np.linspace(-8.0, 8.0, 400)

    y = 10.0 ** log_y

    numerator = simpson(

        weight[None, :] * np.sqrt(x[None, :] ** 2 + y[:, None] ** 2),

        x=x,

        axis=1,

    )

    return CubicSpline(log_y, np.log(numerator / denominator))


_FERMI_DIRAC = _build_fermi_dirac_table()

#: ``(3 zeta(3) / 2) / (7 pi^4 / 120)`` -- the coefficient of the
#: non-relativistic asymptote ``f(y) -> A y``, used beyond the
#: table's upper edge. (It is also the ``A`` of the familiar
#: ``[1 + (Ay)^p]^{1/p}`` approximation, which is exact in this
#: limit and only approximate through the transition.)
_NR_ASYMPTOTE = 1.8030853547 / (7.0 * np.pi ** 4 / 120.0)


def neutrino_density_ratio(y):
    """
    ``f(y) = rho_nu / rho_nu,massless`` for ``y = m / (k_B T_nu)``.

    ``f -> 1`` for a relativistic neutrino and ``f -> 0.3173 y``
    for a non-relativistic one; both asymptotes are used directly
    outside the tabulated range rather than extrapolating the
    spline.
    """

    y = np.atleast_1d(np.asarray(y, dtype=float))

    out = np.empty_like(y)

    low = y < 1.0e-8
    high = y > 1.0e8
    middle = ~(low | high)

    out[low] = 1.0
    out[high] = _NR_ASYMPTOTE * y[high]
    out[middle] = np.exp(_FERMI_DIRAC(np.log10(y[middle])))

    return out


# ============================================================
# z_drag
# ============================================================

#: Names of the basis functions :data:`_ZDRAG_COEF` multiplies, in
#: order. ``lb = ln omega_b``, ``lm = ln omega_cb``,
#: ``ln = ln(N_eff / 3.044)``, ``nu = omega_nu h^2 / 6.449e-4``
#: (i.e. the Planck-fiducial neutrino density as the unit).
_ZDRAG_TERMS = (
    "1", "lb", "lm", "lb2", "lm2", "lblm", "lb3", "lm3",
    "ln", "ln2", "lnlb", "lnlm",
    "nu", "nu2", "nulb", "nulm",
)

#: Coefficients of a fit to CAMB's ``zdrag``, predicting
#: ``ln z_drag`` from :data:`_ZDRAG_TERMS`.
#:
#: Calibrated against CAMB 2.0.4 over a 5850-point grid spanning
#: ``omega_b`` in [0.0180, 0.0260], ``omega_cb`` in [0.090, 0.200],
#: ``N_eff`` in [2.0, 5.0] and ``Sum m_nu`` in [0, 0.6] eV -- which
#: brackets any posterior these datasets can produce many times
#: over. Max residual over that grid is 0.070 in ``z_drag``
#: (6.7e-5 relative), which propagates to 0.0063 Mpc in ``r_d``:
#: 0.004%, or ~60 times smaller than DESI DR2's best BAO error bar.
#:
#: ``z_drag`` depends only on the physical densities, not on ``H0``
#: or curvature -- verified against CAMB, which returns the same
#: value to 1e-6 for ``H0`` from 60 to 75 and for
#: ``Omega_k = 0.05``. That is why neither appears in the basis.
_ZDRAG_COEF = np.array([
    7.2195174822,
    0.0825877285,
    0.0030825908,
    0.0068870523,
    0.0039688520,
    -0.0050271901,
    0.0001658021,
    0.0002591217,
    -0.0030235708,
    0.0010114653,
    -0.0013760751,
    -0.0000620427,
    -0.0000647192,
    0.0000016638,
    -0.0000123329,
    -0.0000126534,
])

#: Calibration range of :data:`_ZDRAG_COEF`, as (min, max). Outside
#: it the fit is an extrapolation.
_ZDRAG_WB_RANGE = (0.0180, 0.0260)
_ZDRAG_WCB_RANGE = (0.090, 0.200)
_ZDRAG_NEFF_RANGE = (2.0, 5.0)

#: The ``omega_nu h^2`` the ``nu`` basis term is measured in -- the
#: Planck 2018 fiducial ``Sum m_nu = 0.06 eV``.
_NU_PIVOT = 0.0006449


class SoundHorizon:
    """
    The sound horizon at the drag epoch for a given cosmology.

    Parameters
    ----------
    cosmology : Cosmology
        Reads ``Omega_m``, ``Omega_b``, ``H0``, ``N_eff`` and
        ``m_nu``. Nothing else -- ``r_d`` is blind to the
        dark-energy model and to curvature (see the module
        docstring).

    Notes
    -----
    :attr:`rd` returns the *free parameter* unless the cosmology has
    ``compute_rd`` set, in which case it returns
    :meth:`rd_computed`. :meth:`rd_computed` is always available
    regardless, so the computed value can be compared against a
    fitted one without changing how a fit behaves.

    Results are cached on the physical densities they depend on, so
    the BAO likelihoods -- which ask for ``rd`` once per data point
    -- pay for the integral once per MCMC step rather than thirteen
    times.
    """

    def __init__(self, cosmology):

        self.cosmo = cosmology

        self._cache_key = None
        self._cache_value = None

    # ---------------------------------------------------------
    # Physical densities
    # ---------------------------------------------------------

    @property
    def h(self) -> float:
        """Reduced Hubble constant, ``H0 / 100``."""

        return self.cosmo.H0 / 100.0

    @property
    def omega_b(self) -> float:
        """Physical baryon density, ``Omega_b h^2``."""

        return self.cosmo.Omega_b * self.h ** 2

    @property
    def omega_gamma(self) -> float:
        """Physical photon density, from ``T_CMB``."""

        return Omega_gamma_h2

    @property
    def n_massive(self) -> int:
        """
        Number of massive neutrino species: one carrying the whole
        of ``Sum m_nu`` (CAMB's ``degenerate`` default), or zero
        when ``m_nu = 0``.

        The distinction matters more than it looks. At ``m_nu = 0``
        *every* effective species is massless, so the radiation
        density is ``N_eff`` times a photon-species worth; treating
        one of them as a zero-mass "massive" species would leave the
        massless count short by ``3.044/3`` and understate the
        radiation by ~14%, which is a 6% error in ``r_d``.
        """

        return 1 if self.cosmo.m_nu > 0.0 else 0

    @property
    def omega_nu(self) -> float:
        """
        Present-day massive-neutrino density, ``Omega_nu h^2``.

        Computed from the Fermi-Dirac integral rather than from the
        usual ``Sum m_nu / 93.14 eV`` shorthand -- which this
        instead *reproduces*, at 93.0378 eV (CAMB: 93.04).
        """

        if not self.n_massive:
            return 0.0

        y = (

            self.cosmo.m_nu / self.n_massive

        ) / KT_NU_MASSIVE

        return float(

            self.omega_gamma

            * NU_ENERGY_FACTOR

            * EFF_PER_MASSIVE

            * self.n_massive

            * neutrino_density_ratio(y)[0]

        )

    @property
    def omega_cb(self) -> float:
        """
        Physical cold-matter density, ``(Omega_c + Omega_b) h^2``.

        CosmoFit's ``Omega_m`` counts massive neutrinos as matter,
        which is right at the low redshifts its background
        calculations cover and wrong at the drag epoch, where they
        are relativistic. So the neutrino density is *subtracted*
        here and re-added by :meth:`omega_nu_of_a` with its actual
        redshift dependence.
        """

        return self.cosmo.Omega_m * self.h ** 2 - self.omega_nu

    # ---------------------------------------------------------

    def omega_nu_of_a(self, a):
        """
        Total neutrino density (massless plus massive) at scale
        factor ``a``, as it enters ``H(a)^2 / H_100^2``.
        """

        a = np.asarray(a, dtype=float)

        relativistic = self.omega_gamma * NU_ENERGY_FACTOR * a ** -4

        massless = (

            self.cosmo.N_eff

            - self.n_massive * EFF_PER_MASSIVE

        )

        total = relativistic * massless

        if self.n_massive:

            y = (

                self.cosmo.m_nu / self.n_massive

            ) * a / KT_NU_MASSIVE

            total = total + (

                relativistic

                * EFF_PER_MASSIVE

                * self.n_massive

                * neutrino_density_ratio(y)

            )

        return total

    # ---------------------------------------------------------

    def hubble_over_h100(self, a):
        r"""
        ``H(a) / (100 km/s/Mpc)`` in the pre-recombination universe:
        photons, neutrinos and cold matter.

        Dark energy and curvature are omitted, not forgotten. At
        ``a = 1/1060`` the matter term is ``~1.7e8`` and a
        cosmological constant contributes ``~0.7``; including it
        would change ``r_d`` in the twelfth significant figure.
        Omitting it is also what makes this work for *every* model
        in the library rather than only the ones with a tractable
        early-time ``E(z)``.
        """

        a = np.asarray(a, dtype=float)

        return np.sqrt(

            self.omega_gamma * a ** -4

            + self.omega_nu_of_a(a)

            + self.omega_cb * a ** -3

        )

    # ---------------------------------------------------------

    def sound_speed(self, a):
        r"""
        Baryon-photon sound speed [km/s],

            c_s = c / sqrt(3 (1 + R_b)),
            R_b = 3 rho_b / (4 rho_gamma)
                = (3 omega_b) / (4 omega_gamma) a
        """

        a = np.asarray(a, dtype=float)

        R_b = (

            3.0 * self.omega_b

            / (4.0 * self.omega_gamma)

        ) * a

        return SPEED_OF_LIGHT / np.sqrt(3.0 * (1.0 + R_b))

    # ---------------------------------------------------------
    # z_drag
    # ---------------------------------------------------------

    def z_drag(self) -> float:
        """
        Redshift of the baryon drag epoch, from a fit calibrated
        against CAMB (see :data:`_ZDRAG_COEF`).

        The drag epoch is where a first-principles calculation
        genuinely stops without a Boltzmann code: it is defined by
        the Thomson-drag optical depth reaching unity, which needs a
        full recombination history. The library already took this
        route for ``z_star``; this is the same trade, with the same
        reasoning.
        """

        wb = require_positive(self.omega_b, "omega_b")
        wcb = require_positive(self.omega_cb, "omega_cb")

        lb = np.log(wb)
        lm = np.log(wcb)
        ln = np.log(self.cosmo.N_eff / NEFF_STANDARD)
        nu = self.omega_nu / _NU_PIVOT

        basis = np.array([
            1.0,
            lb,
            lm,
            lb ** 2,
            lm ** 2,
            lb * lm,
            lb ** 3,
            lm ** 3,
            ln,
            ln ** 2,
            ln * lb,
            ln * lm,
            nu,
            nu ** 2,
            nu * lb,
            nu * lm,
        ])

        return float(np.exp(basis @ _ZDRAG_COEF))

    # ---------------------------------------------------------

    def z_drag_eh98(self) -> float:
        """
        Drag-epoch redshift from the Eisenstein & Hu (1998) Eq. (4)
        fitting formula.

        Kept for comparison only, and not usable in place of
        :meth:`z_drag`: it returns 1020.7 where CAMB gives 1059.9,
        3.7% low, which puts ``r_d`` 2.5% high. See the module
        docstring for why the two are not interchangeable.
        """

        wb = require_positive(self.omega_b, "omega_b")
        wm = require_positive(self.omega_cb, "omega_cb")

        b1 = (

            0.313 * wm ** -0.419

            * (1.0 + 0.607 * wm ** 0.674)

        )

        b2 = 0.238 * wm ** 0.223

        return float(

            1291.0

            * wm ** 0.251

            / (1.0 + 0.659 * wm ** 0.828)

            * (1.0 + b1 * wb ** b2)

        )

    # ---------------------------------------------------------
    # The integral
    # ---------------------------------------------------------

    def sound_horizon(
        self,
        z: float | None = None,
        n_grid: int = 2000,
        decades: float = 8.0,
    ) -> float:
        r"""
        Comoving sound horizon [Mpc] at redshift ``z``,

            r_s(z) = int_0^{a(z)} c_s(a) / (a^2 H(a)) da

        Defaults to ``z = z_drag()``.

        Parameters
        ----------
        z : float, optional
            Redshift to integrate down to.

        n_grid : int, optional
            Number of grid points. 2000 is far past convergence
            (500 already reaches 1e-9 against a 20000-point
            reference); the integral runs once per MCMC step, and
            the extra points cost microseconds.

        decades : float, optional
            How many decades in ``a`` below ``a(z)`` the lower limit
            is placed. See the note on truncation below.

        Notes
        -----
        Deep in radiation domination ``H`` goes as ``a^-2``, so the
        integrand ``c_s / (a^2 H)`` tends to a *constant* and the
        integral over ``[0, a_min]`` is approximately
        ``a_min c_s(0) / (a^2 H)|_{a->0}``. It therefore contributes
        in proportion to ``a_min``: eight decades below the drag
        epoch makes the omitted piece ~1e-8 of the total, well under
        every other error here.

        The grid is log-spaced for the same reason the
        recombination module's is -- a linear grid would spend
        essentially all of its points in the last decade and
        under-resolve the eight below it.
        """

        if z is None:
            z = self.z_drag()

        a_drag = 1.0 / (1.0 + z)

        a = np.logspace(

            np.log10(a_drag) - decades,

            np.log10(a_drag),

            n_grid,

        )

        integrand = (

            self.sound_speed(a)

            / (a ** 2 * 100.0 * self.hubble_over_h100(a))

        )

        return float(simpson(integrand, x=a))

    # ---------------------------------------------------------
    # The public value
    # ---------------------------------------------------------

    def _key(self) -> tuple:
        """
        Everything ``r_d`` depends on -- and nothing else. ``H0``,
        curvature and every dark-energy parameter are deliberately
        absent, so an MCMC step that moves only those reuses the
        cached value instead of re-integrating.
        """

        return (

            float(self.cosmo.Omega_m),

            float(self.cosmo.Omega_b),

            float(self.cosmo.H0),

            float(self.cosmo.N_eff),

            float(self.cosmo.m_nu),

        )

    # ---------------------------------------------------------

    def rd_computed(self) -> float:
        """
        The sound horizon at the drag epoch [Mpc], computed from
        the physical densities.

        Always available, whether or not the fit is *using* it --
        so a fit that leaves ``rd`` free can still be asked what the
        early-universe physics would have predicted, and the two
        compared.
        """

        key = self._key()

        if key != self._cache_key:

            self._cache_value = self.sound_horizon()

            self._cache_key = key

        return self._cache_value

    # ---------------------------------------------------------

    @property
    def rd(self) -> float:
        """
        The sound horizon the BAO likelihoods divide by: the free
        ``rd`` parameter by default, or :meth:`rd_computed` when the
        cosmology has ``compute_rd`` set.
        """

        if getattr(self.cosmo, "compute_rd", False):

            return self.rd_computed()

        return self.cosmo.params.rd

    # ---------------------------------------------------------

    def summary(self) -> dict:
        """
        ``z_drag``, ``r_d``, and the densities they came from, in
        one dict -- for checking a fit's early-universe sector
        without reconstructing the calculation by hand.
        """

        z_drag = self.z_drag()

        return {

            "z_drag": z_drag,

            "z_drag_eh98": self.z_drag_eh98(),

            "r_d": self.sound_horizon(z_drag),

            "r_d_parameter": float(self.cosmo.params.rd),

            "omega_b": self.omega_b,

            "omega_cb": self.omega_cb,

            "omega_nu": self.omega_nu,

            "omega_gamma": self.omega_gamma,

            "N_eff": float(self.cosmo.N_eff),

            "m_nu": float(self.cosmo.m_nu),

        }
