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

from .dark_sector import DarkSector


__all__ = ["DarkEnergy", "DARK_ENERGY", "get_dark_energy"]


_LN10 = np.log(10.0)


class DarkEnergy(DarkSector):
    """
    A dark-energy density history, added to the standard fluids:

        E(z)^2 = rho_std(z) + Omega_k (1+z)^2 + Omega_de f(z),
        Omega_de = 1 - Omega_k - rho_std(0)

    ``name`` and ``params`` (required, read by :meth:`density` and
    :meth:`w`) declare it.
    """

    name = "lambda"
    params: tuple = ()

    def density(self, z, **p):
        """``rho_de(z) / rho_de(0)``."""

        return np.ones_like(np.asarray(z, dtype=float))

    def solve(self, ctx, **p):

        Omega_de = 1.0 - ctx.Omega_k - ctx.rho_std0

        def E2(z):
            return ctx.rho_std(z) + ctx.curvature(z) + Omega_de * self.density(z, **p)

        return E2

    def dark_energy_today(self, ctx) -> float:
        """``Omega_de``, the closure."""

        return 1.0 - ctx.Omega_k - ctx.rho_std0

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


class SignSwitchingLambda(DarkEnergy):
    """
    Lambda_s CDM (Akarsu et al. 2021): a cosmological constant that is
    negative above ``z_dagger`` and positive below, ``f = sgn(z_dagger - z)``,
    with ``+1`` at ``z_dagger`` itself. ``E(z)`` jumps there; see
    :meth:`jumps`.
    """

    name = "lscdm"
    params = ("z_dagger",)

    def density(self, z, z_dagger):
        return np.where(np.asarray(z, dtype=float) <= z_dagger, 1.0, -1.0)

    def w(self, z, z_dagger):
        return np.full_like(np.asarray(z, dtype=float), -1.0)

    def jumps(self, z_dagger):
        return (float(z_dagger),)


class GCG(DarkEnergy):
    """
    The generalized Chaplygin gas, ``p = -A / rho^alpha``:

        f(z) = [A_gcg + (1 - A_gcg)(1+z)^{3(1+alpha_gcg)}]^{1/(1+alpha_gcg)}

    Its parameters are ``A_gcg`` and ``alpha_gcg`` -- the old model's
    ``A_s`` and ``alpha``, renamed so that ``A_s`` is free for the
    primordial amplitude.
    """

    name = "gcg"
    params = ("A_gcg", "alpha_gcg")

    @staticmethod
    def _g(z, A_gcg, alpha_gcg):

        z = np.asarray(z, dtype=float)

        return A_gcg + (1.0 - A_gcg) * (1.0 + z) ** (3.0 * (1.0 + alpha_gcg))

    def density(self, z, A_gcg, alpha_gcg):
        return self._g(z, A_gcg, alpha_gcg) ** (1.0 / (1.0 + alpha_gcg))

    def w(self, z, A_gcg, alpha_gcg):
        return -A_gcg / self._g(z, A_gcg, alpha_gcg)


def _expm1_ratio(x):
    """``expm1(x)/x``, equal to 1 at 0 and accurate near it."""

    x = np.asarray(x, dtype=float)

    small = np.abs(x) < 1.0e-8
    safe = np.where(small, 1.0, x)

    return np.where(small, 1.0 + 0.5 * x, np.expm1(safe) / safe)


class InteractingDarkEnergy(DarkEnergy):
    """
    Interacting dark energy, ``Q = 3 xi H rho_de`` from dark energy into
    cold matter, constant ``w0``:

        E^2 = rho_std + Omega_k (1+z)^2 + T(z) + Omega_de (1+z)^{3(1+w0+xi)}

    with ``T(z)`` the energy moved into matter, written so that it holds
    through the resonance ``w0 + xi = 0`` (see the old model's
    ``_transfer_term``, which this reproduces). The coupling feeds the
    cold matter; photons and neutrinos are untouched.
    """

    name = "ide"
    params = ("w0", "xi")

    def density(self, z, w0, xi):
        return (1.0 + np.asarray(z, dtype=float)) ** (3.0 * (1.0 + w0 + xi))

    def w(self, z, w0, xi):
        return np.full_like(np.asarray(z, dtype=float), w0)

    def transfer(self, z, Omega_de, w0, xi):
        """Energy moved into matter, ``T(z)``, in units of today's critical density."""

        z = np.asarray(z, dtype=float)

        L = np.log1p(z)

        return (
            -3.0 * xi * Omega_de * (1.0 + z) ** 3 * L
            * _expm1_ratio(3.0 * (w0 + xi) * L)
        )

    def clustering_matter(self, z, growth, w0, xi):
        """Cold matter plus what the interaction has moved into it."""

        Omega_de = self.dark_energy_today(growth.ctx)

        return growth.ctx.rho_cb(z) + self.transfer(z, Omega_de, w0, xi)

    def matter_exchange(self, z, growth, w0, xi):
        """``psi = 3 xi rho_de / rho_m``."""

        Omega_de = self.dark_energy_today(growth.ctx)

        return (
            3.0 * xi * Omega_de * self.density(z, w0, xi)
            / self.clustering_matter(z, growth, w0, xi)
        )

    def solve(self, ctx, w0, xi):

        Omega_de = self.dark_energy_today(ctx)

        def E2(z):
            return (
                ctx.rho_std(z) + ctx.curvature(z)
                + self.transfer(z, Omega_de, w0, xi)
                + Omega_de * self.density(z, w0, xi)
            )

        return E2


class HuSawicki(DarkEnergy):
    """
    f(R) gravity, Hu-Sawicki: the background is LCDM's by construction,
    and the model lives in the growth of structure, through the linear
    (unscreened) coupling

        mu(a, k) = 1 + (1/3) Y^2 / (Y^2 + Mhat^2(a)),   Y = k c / (a H_100),
        Mhat^2   = -u^{n+2} / ((n + 1) f_R0 u0^{n+1}),

    ``u = R / H0^2 / 3``, read off the trace of the standard fluids plus
    four times the cosmological constant -- the old model's
    ``Omega_m a^-3 + 4 Omega_Lambda`` with radiation (traceless) adding
    nothing. Parameters ``f_R0`` and ``n_hs``.
    """

    name = "hu_sawicki"
    params = ("f_R0", "n_hs")

    def mu(self, z, k, growth, f_R0, n_hs):

        from cosmofit.cosmology.core.constants import c

        ctx = growth.ctx

        z = np.asarray(z, dtype=float)

        Omega_L = self.dark_energy_today(ctx)

        def u(x):
            return ctx.rho_std(x) - 3.0 * ctx.p_std(x) + 4.0 * Omega_L

        Mhat2 = -(u(z) ** (n_hs + 2.0)) / (
            (n_hs + 1.0) * f_R0 * u(0.0) ** (n_hs + 1.0)
        )

        Y2 = (k * (c / 100.0) * (1.0 + z)) ** 2

        return 1.0 + (Y2 / 3.0) / (Y2 + Mhat2)


#: Every dark-energy model by name, with the old class names as aliases.
DARK_ENERGY = {
    cls.name: cls
    for cls in (
        DarkEnergy, WCDM, CPL, JBP, BA, Logarithmic, PEDE, GEDE,
        SignSwitchingLambda, GCG, InteractingDarkEnergy, HuSawicki,
    )
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
