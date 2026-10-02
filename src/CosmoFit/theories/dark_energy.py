"""
Dark energy on top of general relativity, as a component of the
background.

A model here is only what makes it different from a cosmological
constant: the density of its dark energy relative to today's,
``f(z) = rho_de(z) / rho_de(0)``, and its equation of state ``w(z)``.
Everything else -- matter, radiation, massive neutrinos, curvature,
the closure that fixes ``Omega_de`` -- is the :class:`~theories.
background.Background`'s, written once instead of once per model.

The formulas are those of the original model classes, verbatim, and
the tests check them against those classes.
"""

from __future__ import annotations

import numpy as np


__all__ = ["DarkEnergy", "DARK_ENERGY", "get_dark_energy"]


_LN10 = np.log(10.0)


class DarkEnergy:
    """
    A dark-energy density history.

    Attributes
    ----------
    name : str
    params : tuple of str
        Parameters it reads, all required.
    """

    name = "lambda"
    params: tuple = ()

    def density(self, z, **p):
        """``rho_de(z) / rho_de(0)``."""

        return np.ones_like(np.asarray(z, dtype=float))

    def w(self, z, **p):
        """The equation of state."""

        return np.full_like(np.asarray(z, dtype=float), -1.0)


class WCDM(DarkEnergy):
    """Constant ``w = w0``."""

    name = "wcdm"
    params = ("w0",)

    def density(self, z, w0):
        return (1.0 + np.asarray(z, dtype=float)) ** (3.0 * (1.0 + w0))

    def w(self, z, w0):
        return np.full_like(np.asarray(z, dtype=float), w0)


class CPL(DarkEnergy):
    """Chevallier-Polarski-Linder, ``w = w0 + wa z/(1+z)``."""

    name = "cpl"
    params = ("w0", "wa")

    def density(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(
            -3.0 * wa * z / (1.0 + z)
        )

    def w(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return w0 + wa * z / (1.0 + z)


class JBP(DarkEnergy):
    """Jassal-Bagla-Padmanabhan, ``w = w0 + wa z/(1+z)^2``."""

    name = "jbp"
    params = ("w0", "wa")

    def density(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + w0)) * np.exp(
            1.5 * wa * (z / (1.0 + z)) ** 2
        )

    def w(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return w0 + wa * z / (1.0 + z) ** 2


class BA(DarkEnergy):
    """Barboza-Alcaniz, ``w = w0 + wa z(1+z)/(1+z^2)``."""

    name = "ba"
    params = ("w0", "wa")

    def density(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return (1.0 + z) ** (3.0 * (1.0 + w0)) * (1.0 + z ** 2) ** (1.5 * wa)

    def w(self, z, w0, wa):

        z = np.asarray(z, dtype=float)

        return w0 + wa * z * (1.0 + z) / (1.0 + z ** 2)


class Logarithmic(DarkEnergy):
    """Efstathiou, ``w = w0 + wa ln(1+z)``."""

    name = "logarithmic"
    params = ("w0", "wa")

    def density(self, z, w0, wa):

        L = np.log1p(np.asarray(z, dtype=float))

        return np.exp(3.0 * (1.0 + w0) * L + 1.5 * wa * L ** 2)

    def w(self, z, w0, wa):

        return w0 + wa * np.log1p(np.asarray(z, dtype=float))


class PEDE(DarkEnergy):
    """
    Phenomenologically emergent dark energy:
    ``f = 1 - tanh(log10(1+z))``, no free parameter.
    """

    name = "pede"

    def density(self, z):

        return 1.0 - np.tanh(np.log10(1.0 + np.asarray(z, dtype=float)))

    def w(self, z):

        u = np.log10(1.0 + np.asarray(z, dtype=float))

        return -1.0 - (1.0 + np.tanh(u)) / (3.0 * _LN10)


class GEDE(DarkEnergy):
    """
    Generalized emergent dark energy:
    ``f = [1 - tanh(Delta log10((1+z)/(1+z_t)))] / [1 + tanh(Delta log10(1+z_t))]``.
    """

    name = "gede"
    params = ("Delta", "z_t")

    @staticmethod
    def _v(z, Delta, z_t):

        z = np.asarray(z, dtype=float)

        return Delta * (np.log10(1.0 + z) - np.log10(1.0 + z_t))

    def density(self, z, Delta, z_t):

        norm = 1.0 + np.tanh(Delta * np.log10(1.0 + z_t))

        return (1.0 - np.tanh(self._v(z, Delta, z_t))) / norm

    def w(self, z, Delta, z_t):

        return -1.0 - Delta * (1.0 + np.tanh(self._v(z, Delta, z_t))) / (3.0 * _LN10)


#: Every dark-energy model by name, with the old class names as aliases.
DARK_ENERGY = {
    cls.name: cls
    for cls in (DarkEnergy, WCDM, CPL, JBP, BA, Logarithmic, PEDE, GEDE)
}

_ALIASES = {
    "lcdm": "lambda", "cosmological_constant": "lambda",
    "logarithmicde": "logarithmic", "log": "logarithmic",
}


def get_dark_energy(name) -> DarkEnergy:
    """
    A dark-energy model from its name (case-insensitive; ``"LCDM"``,
    ``"CPL"`` and the other old class names work), an instance, or a
    :class:`DarkEnergy` subclass.
    """

    if isinstance(name, DarkEnergy):
        return name

    if isinstance(name, type) and issubclass(name, DarkEnergy):
        return name()

    key = str(name).lower()
    key = _ALIASES.get(key, key)

    if key not in DARK_ENERGY:
        raise ValueError(
            f"Unknown dark energy {name!r}; known: {sorted(DARK_ENERGY)}."
        )

    return DARK_ENERGY[key]()
