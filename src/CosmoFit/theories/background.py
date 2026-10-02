"""
The expansion history, with everything in it.

The old models each wrote out ``E(z)^2 = Omega_m (1+z)^3 + Omega_k
(1+z)^2 + Omega_de(z)`` -- no photons, no neutrinos -- and the parts of
the library that could not do without radiation built their own: the
CMB distance priors one way, the sound horizon another. Three
expansion histories for one cosmology, agreeing only where radiation
did not matter.

Here there is one::

    E(z)^2 = [omega_gamma a^-4 + omega_nu(a) + omega_cb a^-3] / h^2
             + Omega_k a^-2 + Omega_de f(z)

with photons from ``T_CMB``, neutrinos with their exact Fermi-Dirac
density (one massive species carrying ``m_nu``, as CAMB's default), and
the dark energy's ``f(z) = rho_de(z)/rho_de(0)`` from
:mod:`theories.dark_energy`. ``Omega_de`` closes the budget, radiation
included. Every downstream quantity -- distances, the sound horizon,
the growth of structure -- reads this one ``E(z)``.

Options
-------
dark_energy : str or dict
    ``"lambda"`` (default), ``"wcdm"``, ``"cpl"``, ... -- see
    :data:`~theories.dark_energy.DARK_ENERGY`.
parameterization : {"fractional", "physical"}
    ``"fractional"`` (default): ``H0``, ``Omega_m``, ``Omega_b``, with
    ``Omega_m`` counting massive neutrinos as matter today -- the
    library's long-standing convention. ``"physical"``: ``H0``,
    ``omega_b``, ``omega_cdm``, the densities a CMB fit samples.
radiation : bool
    Default ``True``. ``False`` drops photons and neutrinos and treats
    ``Omega_m`` as pure ``(1+z)^3`` matter: the old models' expansion
    history, exactly, for comparison with them.
zmax : float
    Initial range of the distance table (default 5). It extends itself
    when asked for more.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import PPoly

from CosmoFit.core.component import ComponentError, Theory
from CosmoFit.cosmology.core.constants import Mpc, Omega_gamma_h2, c, km, year
from CosmoFit.cosmology.calculators.sound_horizon import (
    EFF_PER_MASSIVE,
    KT_NU_MASSIVE,
    NEFF_STANDARD,
    NU_ENERGY_FACTOR,
    neutrino_density_ratio,
)
from CosmoFit.cosmology.numerics.hermite import hermite_spline

from .dark_sector import ExpansionContext
from .sectors import get_dark_sector


__all__ = ["Background", "neutrino_density", "neutrino_pressure"]


#: Seconds in a gigayear.
_GYR = 1.0e9 * year

#: ``1 / (100 km/s/Mpc)`` in gigayears: ages come out as this over ``h``.
_HUBBLE_TIME_100 = Mpc / (100.0 * km) / _GYR


def neutrino_density(a, N_eff: float, m_nu: float, omega_gamma: float = Omega_gamma_h2):
    """
    Total neutrino density ``omega_nu(a)`` (massless plus massive) in
    units of the critical density for ``H = 100 km/s/Mpc``, as it enters
    ``H(a)^2``.

    One massive species carries the whole of ``m_nu`` when it is
    positive; the remaining ``N_eff - 3.044/3`` effective species are
    massless.
    """

    a = np.asarray(a, dtype=float)

    relativistic = omega_gamma * NU_ENERGY_FACTOR / a ** 4

    n_massive = 1 if m_nu > 0.0 else 0

    total = relativistic * (N_eff - n_massive * EFF_PER_MASSIVE)

    if n_massive:

        y = m_nu * a / KT_NU_MASSIVE

        total = total + relativistic * EFF_PER_MASSIVE * neutrino_density_ratio(y).reshape(a.shape)

    return total


def neutrino_pressure(a, N_eff: float, m_nu: float, omega_gamma: float = Omega_gamma_h2):
    """
    Total neutrino pressure, in the units of :func:`neutrino_density`,
    from the continuity equation ``p = -rho - (1/3) d rho / d ln a`` --
    a third of the density while relativistic, falling to zero as the
    massive species slows down.
    """

    a = np.asarray(a, dtype=float)

    step = 1.0e-5

    slope = (
        neutrino_density(a * math.exp(step), N_eff, m_nu, omega_gamma)
        - neutrino_density(a * math.exp(-step), N_eff, m_nu, omega_gamma)
    ) / (2.0 * step)

    return -neutrino_density(a, N_eff, m_nu, omega_gamma) - slope / 3.0


def _massive_today(m_nu: float, omega_gamma: float = Omega_gamma_h2) -> float:
    """Present-day density of the massive species, ``omega_nu h^2``."""

    if m_nu <= 0.0:
        return 0.0

    return float(
        omega_gamma * NU_ENERGY_FACTOR * EFF_PER_MASSIVE
        * neutrino_density_ratio(m_nu / KT_NU_MASSIVE)[0]
    )


class Background(Theory):
    """
    The background expansion; see the module docstring.

    Provides
    --------
    ``E``, ``H`` [km/s/Mpc], ``comoving_distance``, ``DM``, ``DH``,
    ``DV``, ``DA`` [Mpc] (each ``z=...``), ``w`` and ``Omega_de_z``
    (``z=...``), and ``background_densities`` -- the physical
    densities, for the early-universe and growth theories.

    Derived parameters
    ------------------
    ``h``, ``Omega_m``, ``Omega_b``, ``omega_b``, ``omega_cdm``,
    ``omega_m``, ``Omega_de``, ``Omega_nu`` (massive, today),
    ``Omega_r`` (photons and massless neutrinos, today), ``age`` [Gyr].
    """

    def initialize(self) -> None:

        unknown = set(self.info) - {
            "dark_energy", "parameterization", "radiation", "zmax",
        }

        if unknown:
            raise ComponentError(f"{self.name}: unknown option(s) {sorted(unknown)}.")

        try:
            self.sector = get_dark_sector(self.info.get("dark_energy", "lambda"))
        except (ValueError, TypeError) as error:
            raise ComponentError(f"{self.name}: {error}") from None

        self.parameterization = self.info.get("parameterization", "fractional")

        if self.parameterization not in ("fractional", "physical"):
            raise ComponentError(
                f"{self.name}: parameterization is 'fractional' or "
                f"'physical', not {self.parameterization!r}."
            )

        self.radiation = bool(self.info.get("radiation", True))
        self.zmax = float(self.info.get("zmax", 5.0))

        if self.radiation and not self.sector.radiation_ok:
            raise ComponentError(
                f"{self.name}: {self.sector!r} has no form with radiation; "
                f"use radiation: false."
            )

        if self.parameterization == "fractional":
            matter, baryons = "Omega_m", "Omega_b"
        else:
            matter, baryons = "omega_cdm", "omega_b"

        self.matter_name = None if self.sector.derives_matter else matter
        self.baryon_name = baryons

        self.required = tuple(
            name for name in ("H0", self.matter_name, baryons, *self.sector.params)
            if name is not None
        )
        self.optional = ("Omega_k", "N_eff", "m_nu", *self.sector.defaults)

    # ---------------------------------------------------------

    def get_default_params(self) -> dict:

        return {
            "Omega_k": 0.0, "N_eff": NEFF_STANDARD, "m_nu": 0.06,
            **self.sector.defaults,
        }

    def accepts(self, name: str) -> bool:
        return name in self.required or name in self.optional

    def get_required_params(self) -> list[str]:
        return list(self.required)

    def get_can_provide(self) -> list[str]:

        return [
            "E", "H", "comoving_distance", "DM", "DH", "DV", "DA",
            "w", "Omega_de_z", "background_densities", "background_jumps",
        ]

    def get_derived_params(self) -> list[str]:

        return [
            "h", "Omega_m", "Omega_b", "omega_b", "omega_cdm", "omega_m",
            "Omega_de", "Omega_nu", "Omega_r", "age",
        ]

    def check_model(self, model) -> None:

        if not self.sector.flat_only:
            return

        spec = model.parameters.specs.get("Omega_k")

        if spec is not None and (spec.role != "fixed" or spec.value != 0.0):

            state = spec.role if spec.role != "fixed" else f"fixed at {spec.value!r}"

            raise ComponentError(
                f"{self.name}: {self.sector!r} is defined for a flat universe "
                f"only, so Omega_k must be fixed at 0 (it is {state})."
            )

    # ---------------------------------------------------------

    def _context(self, Omega_cb: float, h: float, N_eff: float, m_nu: float,
                 Omega_k: float, H0: float) -> ExpansionContext:
        """The standard fluids, for a given cold-matter density today."""

        def rho_cb(z):
            return Omega_cb * (1.0 + np.asarray(z, dtype=float)) ** 3

        if not self.radiation:

            def zero(z):
                return np.zeros_like(np.asarray(z, dtype=float))

            return ExpansionContext(
                rho_std=rho_cb, p_std=zero, rho_cb=rho_cb, rho_rel=zero,
                p_rel=zero, rho_std0=Omega_cb, rho_rel0=0.0,
                Omega_cb=Omega_cb, Omega_k=Omega_k, radiation=False, H0=H0,
            )

        h2 = h * h
        omega_gamma = Omega_gamma_h2

        def rho_rel(z):

            a = 1.0 / (1.0 + np.asarray(z, dtype=float))

            return (omega_gamma / a ** 4 + neutrino_density(a, N_eff, m_nu, omega_gamma)) / h2

        def p_rel(z):

            a = 1.0 / (1.0 + np.asarray(z, dtype=float))

            return (omega_gamma / (3.0 * a ** 4)
                    + neutrino_pressure(a, N_eff, m_nu, omega_gamma)) / h2

        def rho_std(z):
            return rho_cb(z) + rho_rel(z)

        rho_rel0 = float(rho_rel(0.0))

        return ExpansionContext(
            rho_std=rho_std, p_std=p_rel, rho_cb=rho_cb, rho_rel=rho_rel,
            p_rel=p_rel, rho_std0=Omega_cb + rho_rel0, rho_rel0=rho_rel0,
            Omega_cb=Omega_cb, Omega_k=Omega_k, radiation=True, H0=H0,
        )

    def calculate(self, state: dict, want_derived: bool = True, **p):

        H0 = float(p["H0"])
        h = H0 / 100.0
        Omega_k = float(p.get("Omega_k", 0.0))
        N_eff = float(p.get("N_eff", NEFF_STANDARD))
        m_nu = float(p.get("m_nu", 0.06))

        sector_params = {
            name: float(p[name])
            for name in (*self.sector.params, *self.sector.defaults)
        }

        omega_gamma = Omega_gamma_h2
        omega_nu_massive = _massive_today(m_nu, omega_gamma) if self.radiation else 0.0

        if self.parameterization == "fractional":
            omega_b = float(p["Omega_b"]) * h * h
        else:
            omega_b = float(p["omega_b"])

        def make_context(Omega_cb):
            return self._context(Omega_cb, h, N_eff, m_nu, Omega_k, H0)

        if self.sector.derives_matter:

            Omega_cb = self.sector.matter_density(make_context, **sector_params)

            if not np.isfinite(Omega_cb) or Omega_cb <= 0.0:
                return False

        elif self.parameterization == "fractional":

            # Omega_m counts massive neutrinos as matter today; without
            # radiation there is nothing to separate them from.
            Omega_cb = float(p["Omega_m"]) - omega_nu_massive / (h * h)

        else:

            Omega_cb = (omega_b + float(p["omega_cdm"])) / (h * h)

        ctx = make_context(Omega_cb)

        E2 = self.sector.solve(ctx, **sector_params)

        if E2 is None:
            return False

        def E(z):

            with np.errstate(invalid="ignore"):
                return np.sqrt(E2(z))

        omega_cb = Omega_cb * h * h
        Omega_m = Omega_cb + omega_nu_massive / (h * h)

        state.update(
            jumps=tuple(self.sector.jumps(**sector_params)),
            E=E,
            E2=E2,
            context=ctx,
            H0=H0,
            Omega_k=Omega_k,
            sector_params=sector_params,
            densities={
                "h": h, "omega_b": omega_b, "omega_cb": omega_cb,
                "omega_gamma": omega_gamma, "omega_nu_massive": omega_nu_massive,
                "N_eff": N_eff, "m_nu": m_nu, "radiation": self.radiation,
                "Omega_m": Omega_m,
            },
        )

        if not self._build_table(state, self.zmax):
            return False

        if want_derived:

            massive = omega_nu_massive / (h * h)

            state["derived"] = {
                "h": h,
                "Omega_m": Omega_m,
                "Omega_b": omega_b / (h * h),
                "omega_b": omega_b,
                "omega_cdm": omega_cb - omega_b,
                "omega_m": Omega_m * h * h,
                "Omega_de": 1.0 - Omega_k - ctx.rho_std0,
                "Omega_nu": massive,
                "Omega_r": ctx.rho_rel0 - massive,
                "age": self._age(E, h),
            }

        return True

    # ---------------------------------------------------------

    @staticmethod
    def _build_table(state: dict, zmax: float) -> bool:
        """
        The comoving-distance table, ``chi(z) = int_0^z dz/E``, on a grid
        uniform in ``u = ln(1+z)``, interpolated with a cubic Hermite
        spline whose slopes are the exact ``d chi/du = (1+z)/E``.
        Returns ``False`` where ``E^2 <= 0`` -- no expanding solution.

        Where ``E(z)`` jumps (``state["jumps"]``), the grid is split there
        and each piece integrated on its own, with ``E`` taken from the
        correct side at the break. ``chi`` itself only kinks; a single
        grid across the jump would smear it over one cell, an error of
        order 1e-4 in every distance beyond it.
        """

        u_max = math.log1p(zmax)

        jumps = sorted(z for z in state.get("jumps", ()) if 0.0 < z < zmax)

        edges = [0.0, *(math.log1p(z) for z in jumps), u_max]

        total = 2 * int(400 * max(1.0, u_max))

        xs, coefficients = [], []

        offset = 0.0

        for index, (u0, u1) in enumerate(zip(edges[:-1], edges[1:])):

            n = 2 * max(8, int(total * (u1 - u0) / u_max / 2)) + 1

            u = np.linspace(u0, u1, n)
            z = np.expm1(u)

            # The piece above a jump starts just past it -- from the jump
            # itself, not from expm1(log1p(z)), which can round to a hair
            # below it and land on the wrong side.
            if index > 0:
                z[0] = np.nextafter(jumps[index - 1], np.inf)

            E = state["E"](z)

            if not np.all(np.isfinite(E)) or np.any(E <= 0.0):
                return False

            slope = (1.0 + z) / E

            chi = offset + cumulative_simpson(slope, x=u, initial=0.0)

            piece = hermite_spline(u, chi, slope)

            coefficients.append(piece.c)
            xs.append(u if index == 0 else u[1:])

            offset = float(chi[-1])

        state["chi_table"] = PPoly.construct_fast(
            np.concatenate(coefficients, axis=1), np.concatenate(xs),
            extrapolate=False,
        )
        state["table_zmax"] = zmax

        return True

    @staticmethod
    def _age(E, h: float) -> float:
        """``t_0 = int_0^1 da / (a H)``, in Gyr."""

        log_a = np.linspace(math.log(1.0e-12), 0.0, 6001)
        a = np.exp(log_a)

        with np.errstate(over="ignore", invalid="ignore"):
            integrand = 1.0 / E(1.0 / a - 1.0)

        integrand = np.where(np.isfinite(integrand), integrand, 0.0)

        return float(simpson(integrand, x=log_a) * _HUBBLE_TIME_100 / h)

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def _state(self):

        if self.current_state is None:
            raise RuntimeError(f"{self.name} has not been computed yet.")

        return self.current_state

    def _chi(self, z):
        """Dimensionless comoving distance, extending the table if needed."""

        state = self._state()

        z = np.asarray(z, dtype=float)

        top = float(np.max(z)) if z.size else 0.0

        if top > state["table_zmax"]:
            self._build_table(state, 1.1 * top)

        return state["chi_table"](np.log1p(z))

    def get_E(self, z):
        return self._state()["E"](z)

    def get_H(self, z):
        return self._state()["H0"] * self.get_E(z)

    def get_comoving_distance(self, z):
        return c / self._state()["H0"] * self._chi(z)

    def get_DM(self, z):

        state = self._state()

        chi = self._chi(z)
        Omega_k = state["Omega_k"]

        if Omega_k > 0.0:
            root = math.sqrt(Omega_k)
            transverse = np.sinh(root * chi) / root
        elif Omega_k < 0.0:
            root = math.sqrt(-Omega_k)
            transverse = np.sin(root * chi) / root
        else:
            transverse = chi

        return c / state["H0"] * transverse

    def get_DH(self, z):
        return c / self.get_H(z)

    def get_DA(self, z):
        return self.get_DM(z) / (1.0 + np.asarray(z, dtype=float))

    def get_DV(self, z):

        z = np.asarray(z, dtype=float)

        return np.cbrt(z * self.get_DM(z) ** 2 * self.get_DH(z))

    def get_w(self, z):
        """
        The dark energy's equation of state -- for sectors that have one
        (a fluid on top of general relativity). A modified Friedmann
        equation has no ``w``; its *effective* dark density is
        :meth:`get_Omega_de_z`.
        """

        if not hasattr(self.sector, "w"):
            raise ComponentError(f"{self.sector!r} has no equation of state.")

        return self.sector.w(z, **self._state()["sector_params"])

    def get_Omega_de_z(self, z):
        """
        Everything in ``E^2`` beyond the standard fluids and curvature,
        in units of today's critical density: the dark energy's density,
        or for a modified Friedmann equation the effective one a GR
        analysis would attribute its expansion to.
        """

        state = self._state()
        ctx = state["context"]

        return state["E2"](z) - ctx.rho_std(z) - ctx.curvature(z)

    def get_background_jumps(self) -> tuple:
        """Redshifts at which ``E(z)`` is discontinuous."""

        return tuple(self.sector.jumps(**self._state()["sector_params"]))

    def get_background_densities(self) -> dict:
        return dict(self._state()["densities"])
