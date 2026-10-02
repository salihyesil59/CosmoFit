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

from .dark_energy import get_dark_energy


__all__ = ["Background", "neutrino_density"]


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

        de = self.info.get("dark_energy", "lambda")

        if isinstance(de, dict):
            de = dict(de).get("name", "lambda")

        try:
            self.dark_energy = get_dark_energy(de)
        except ValueError as error:
            raise ComponentError(f"{self.name}: {error}") from None

        self.parameterization = self.info.get("parameterization", "fractional")

        if self.parameterization not in ("fractional", "physical"):
            raise ComponentError(
                f"{self.name}: parameterization is 'fractional' or "
                f"'physical', not {self.parameterization!r}."
            )

        self.radiation = bool(self.info.get("radiation", True))
        self.zmax = float(self.info.get("zmax", 5.0))

        if self.parameterization == "fractional":
            self.matter_names = ("Omega_m", "Omega_b")
        else:
            self.matter_names = ("omega_cdm", "omega_b")

        self.required = ("H0", *self.matter_names, *self.dark_energy.params)
        self.optional = ("Omega_k", "N_eff", "m_nu")

    # ---------------------------------------------------------

    def get_default_params(self) -> dict:

        return {"Omega_k": 0.0, "N_eff": NEFF_STANDARD, "m_nu": 0.06}

    def accepts(self, name: str) -> bool:
        return name in self.required or name in self.optional

    def get_required_params(self) -> list[str]:
        return list(self.required)

    def get_can_provide(self) -> list[str]:

        return [
            "E", "H", "comoving_distance", "DM", "DH", "DV", "DA",
            "w", "Omega_de_z", "background_densities",
        ]

    def get_derived_params(self) -> list[str]:

        return [
            "h", "Omega_m", "Omega_b", "omega_b", "omega_cdm", "omega_m",
            "Omega_de", "Omega_nu", "Omega_r", "age",
        ]

    # ---------------------------------------------------------

    def calculate(self, state: dict, want_derived: bool = True, **p):

        H0 = float(p["H0"])
        h = H0 / 100.0
        Omega_k = float(p.get("Omega_k", 0.0))
        N_eff = float(p.get("N_eff", NEFF_STANDARD))
        m_nu = float(p.get("m_nu", 0.06))

        de_params = {name: float(p[name]) for name in self.dark_energy.params}

        omega_gamma = Omega_gamma_h2
        omega_nu_massive = _massive_today(m_nu, omega_gamma)

        if self.parameterization == "fractional":

            Omega_m = float(p["Omega_m"])
            omega_b = float(p["Omega_b"]) * h * h
            omega_cb = Omega_m * h * h - omega_nu_massive

        else:

            omega_b = float(p["omega_b"])
            omega_cb = omega_b + float(p["omega_cdm"])
            Omega_m = (omega_cb + omega_nu_massive) / (h * h)

        density = self.dark_energy.density

        if self.radiation:

            radiation_today = (
                omega_gamma + float(neutrino_density(1.0, N_eff, m_nu, omega_gamma))
                - omega_nu_massive
            )

            Omega_de = 1.0 - Omega_k - (
                omega_gamma + float(neutrino_density(1.0, N_eff, m_nu, omega_gamma))
                + omega_cb
            ) / (h * h)

            def E2(z):

                z = np.asarray(z, dtype=float)
                a = 1.0 / (1.0 + z)

                early = (
                    omega_gamma / a ** 4
                    + neutrino_density(a, N_eff, m_nu, omega_gamma)
                    + omega_cb / a ** 3
                ) / (h * h)

                return early + Omega_k / a ** 2 + Omega_de * density(z, **de_params)

        else:

            radiation_today = 0.0

            Omega_de = 1.0 - Omega_m - Omega_k

            def E2(z):

                z = np.asarray(z, dtype=float)

                return (
                    Omega_m * (1.0 + z) ** 3 + Omega_k * (1.0 + z) ** 2
                    + Omega_de * density(z, **de_params)
                )

        def E(z):

            with np.errstate(invalid="ignore"):
                return np.sqrt(E2(z))

        state.update(
            E=E,
            H0=H0,
            Omega_k=Omega_k,
            de_params=de_params,
            Omega_de=Omega_de,
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

            state["derived"] = {
                "h": h,
                "Omega_m": Omega_m,
                "Omega_b": omega_b / (h * h),
                "omega_b": omega_b,
                "omega_cdm": omega_cb - omega_b,
                "omega_m": Omega_m * h * h,
                "Omega_de": Omega_de,
                "Omega_nu": omega_nu_massive / (h * h),
                "Omega_r": radiation_today / (h * h),
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
        """

        u_max = math.log1p(zmax)

        n = 2 * int(400 * max(1.0, u_max)) + 1

        u = np.linspace(0.0, u_max, n)
        z = np.expm1(u)

        E = state["E"](z)

        if not np.all(np.isfinite(E)) or np.any(E <= 0.0):
            return False

        slope = (1.0 + z) / E

        chi = cumulative_simpson(slope, x=u, initial=0.0)

        state["chi_table"] = hermite_spline(u, chi, slope, extrapolate=False)
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
        return self.dark_energy.w(z, **self._state()["de_params"])

    def get_Omega_de_z(self, z):
        """``Omega_de rho_de(z)/rho_de(0)``, in units of today's critical density."""

        state = self._state()

        return state["Omega_de"] * self.dark_energy.density(z, **state["de_params"])

    def get_background_densities(self) -> dict:
        return dict(self._state()["densities"])
