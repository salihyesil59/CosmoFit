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

#: Present-day photon density parameter times h^2, Omega_gamma * h^2,
#: for Tcmb = 2.7255 K (Fixsen 2009 / standard value used
#: throughout the CMB literature, e.g. Eisenstein & Hu 1998).
Omega_gamma_h2 = 2.469e-5

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