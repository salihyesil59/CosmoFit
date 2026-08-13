"""
Physical and astronomical constants used throughout CosmoFit.

All values are given in commonly used cosmological units.
"""

from __future__ import annotations

# ============================================================
# Fundamental constants
# ============================================================

#: Speed of light in vacuum [km s^-1]
c = 299792.458

#: Newton gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11


# ============================================================
# Unit conversions
# ============================================================

#: 1 parsec [m]
pc = 3.085677581491367e16

#: 1 megaparsec [m]
Mpc = 1.0e6 * pc

#: Seconds in one Julian year
year = 365.25 * 24 * 3600

#: Kilometer in meters
km = 1000.0


# ============================================================
# CMB
# ============================================================

#: Present CMB temperature [K]
Tcmb = 2.7255

#: Present-day photon density parameter times h^2, Omega_gamma * h^2.
#:
#: Derived from `Tcmb` rather than hard-coded, so the two cannot
#: drift apart:
#:
#:     Omega_gamma h^2 = (4 sigma_SB / c^3) T^4 / rho_crit,100
#:
#: with rho_crit,100 = 3 (100 km/s/Mpc)^2 / (8 pi G). For
#: Tcmb = 2.7255 K (Fixsen 2009) this gives 2.4728e-5, the standard
#: literature value.
#:
#: The previous hard-coded 2.469e-5 corresponds to Tcmb ~ 2.716 K,
#: not to the 2.7255 K declared just above -- a 0.15% internal
#: inconsistency that propagated into the radiation density and
#: baryon loading R_b used by the sound-horizon integral, and hence
#: into the Planck distance-prior likelihood.
def _omega_gamma_h2(T_cmb: float) -> float:

    sigma_SB = 5.670374419e-8          # W m^-2 K^-4
    _c_si = 2.99792458e8               # m s^-1
    _G = G
    _Mpc = Mpc

    # Photon energy density -> mass density [kg m^-3]
    rho_gamma = 4.0 * sigma_SB * T_cmb**4 / _c_si**3

    # Critical density for H = 100 km/s/Mpc [kg m^-3]
    H100 = 100.0 * km / _Mpc           # s^-1
    rho_crit_100 = 3.0 * H100**2 / (8.0 * 3.141592653589793 * _G)

    return rho_gamma / rho_crit_100


Omega_gamma_h2 = _omega_gamma_h2(Tcmb)

#: Effective number of relativistic neutrino species (Standard
#: Model prediction, Planck 2018 fiducial value).
N_eff = 3.046


# ============================================================
# Solar quantities
# ============================================================

#: Solar mass [kg]
M_sun = 1.98847e30


# ============================================================
# Namespace object
# ============================================================
#
# `cosmology.calculators.distances` and other modules do
# `from CosmoFit.cosmology.core import constants` and then access
# attributes like `constants.c`. That requires a single
# namespace object called `constants`, not a bag of loose
# module-level names. We build it here from the module's own
# globals so the two stay in sync automatically.

from types import SimpleNamespace as _SimpleNamespace

constants = _SimpleNamespace(
    c=c,
    G=G,
    pc=pc,
    Mpc=Mpc,
    km=km,
    year=year,
    Tcmb=Tcmb,
    Omega_gamma_h2=Omega_gamma_h2,
    N_eff=N_eff,
    M_sun=M_sun,
)

# ============================================================
# Exported names
# ============================================================

__all__ = [
    "c",
    "G",
    "pc",
    "Mpc",
    "km",
    "year",
    "Tcmb",
    "Omega_gamma_h2",
    "N_eff",
    "M_sun",
    "constants",
]