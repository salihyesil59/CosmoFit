"""
f(R) modified gravity (Hu-Sawicki), background level.
"""

from __future__ import annotations

import numpy as np

from .lcdm import LCDM

from CosmoFit.cosmology.core import constants


class FRHuSawicki(LCDM):
    r"""
    f(R) gravity, Hu & Sawicki (2007) model.

    f(R) gravity replaces the Ricci scalar R in the Einstein-Hilbert
    action with an arbitrary function f(R). The Hu-Sawicki form,

        f(R) = -m^2 c1 (R/m^2)^n / (c2 (R/m^2)^n + 1),

    is the standard benchmark model in the literature, tuned (via
    ``n`` and the present-day scalaron value ``f_R0``) to satisfy
    Solar System tests through chameleon screening while still
    producing cosmic acceleration.

    **This class subclasses :class:`~cosmology.models.lcdm.LCDM`
    directly and does not override ``E``/``dEdz``/``Omega_de`` --
    its background expansion history *is* LCDM's, unchanged.** This
    is not a simplification or a placeholder bug: it is the actual
    physics of the standard "designer f(R)" construction, which
    builds f(R) to reproduce an assumed target background (usually
    LCDM's) essentially exactly. **``f_R0``/``n`` are therefore
    invisible to any background/expansion-history probe** (CC, BAO,
    SNe, Planck distance priors) -- fitting them against those alone
    cannot meaningfully constrain them.

    **Growth of structure is where this model actually differs from
    LCDM**, and is implemented here via ``mu(a, k)``: the standard
    chameleon-screened, scale- and time-dependent effective
    gravitational coupling for designer f(R) (Hu & Sawicki 2007;
    the general parametrized-``mu`` framework of Pogosian &
    Silvestri 2008, arXiv:0709.0296),

        mu(a,k) = 1 + (1/3) * Y^2 / (Y^2 + Mhat^2(a))

    where ``Y = k * (c/100) / a`` is the wavenumber (``k`` in
    h/Mpc) in units of ``H0/c``, and ``Mhat^2(a) = M^2(a)/(H0/c)^2``
    is the scalaron's (Hubble-units) mass-squared,

        Mhat^2(a) = -u(a)^(n+2) / [(n+1) f_R0 u0^(n+1)]

    with ``u(a) = Omega_m a^-3 + 4 Omega_de0``, ``u0 = u(a=1)``,
    obtained from ``M^2 = 1/(3 f_RR)`` and this model's own
    (LCDM) ``R(a) = 3 H0^2 u(a)`` and ``f_R(a) = f_R0 (u0/u(a))^(n+1)``
    (the standard designer-f(R) closure, chosen so that
    ``f_R(a=1) = f_R0`` exactly) -- derived here by direct analytic
    differentiation rather than transcribed from a secondary source,
    and numerically self-consistent: ``k -> 0`` gives ``mu -> 1``
    (no force on super-horizon scales), ``k -> infinity`` gives
    ``mu -> 4/3`` (the well-known maximal f(R) enhancement), and
    ``f_R0 -> 0`` gives ``mu -> 1`` at any ``k`` (GR recovered as
    the scalaron mass diverges).

    ``k`` is held at a fixed, documented fiducial pivot
    (:class:`~cosmology.calculators.growth.GrowthCalculator`
    defaults to ``k = 0.1`` h/Mpc, a representative galaxy-survey
    RSD scale) rather than a free parameter -- CosmoFit's fsigma8
    data here are single per-redshift points, not a P(k) shape, so a
    fixed representative scale is the honest ceiling of what a
    single-number-per-z comparison can use; a real k-by-k P(k)
    analysis would need the full scale dependence above, which
    ``mu(a,k)`` does provide if called directly at other ``k``.

    **What this means in practice:** ``f_R0``/``n`` are still inert
    for background-only fits (CC/BAO/SNe/Planck alone), but now
    genuinely shape the ``"fsigma8"``/``"s8"`` predictions -- fit
    those datasets (alongside or instead of the background ones) to
    actually constrain them.

    Parameters
    ----------
    Adds ``f_R0`` (present-day scalaron value, default -1e-6,
    typically negative and small -- see the reference for viability
    bounds) and ``n`` (default 1) via ``EXTRA_PARAMS``. Neither
    affects ``E(z)`` -- see above; both now shape ``mu(a,k)`` and
    therefore growth-of-structure predictions.

    References
    ----------
    Hu & Sawicki (2007), "Models of f(R) Cosmic Acceleration that
    Evade Solar-System Tests", Phys. Rev. D 76, 064004,
    arXiv:0705.1158.

    Pogosian & Silvestri (2008), "The pattern of growth in viable
    f(R) cosmologies", Phys. Rev. D 77, 023503, arXiv:0709.0296.
    """

    MODEL_NAME = "FRHuSawicki"
    MODEL_LABEL = r"$f(R)$ Hu-Sawicki"

    EXTRA_PARAMS = {
        "f_R0": {
            "default": -1e-6, "bounds": (-1e-4, -1e-8),
            "label": r"$f_{R0}$",
        },
        "n": {
            "default": 1.0, "bounds": (0.1, 4.0), "label": r"$n$",
        },
    }

    #: Fiducial pivot wavenumber [h/Mpc] used when ``mu()`` is
    #: called without an explicit ``k`` -- see the class docstring.
    _DEFAULT_K = 0.1

    # ---------------------------------------------------------

    def mu(self, a, k=None):
        """
        Chameleon-screened effective gravitational coupling
        G_eff(a,k)/G_N -- see the class docstring for the
        derivation.
        """

        if k is None:
            k = self._DEFAULT_K

        a = np.asarray(a, dtype=float)

        Om0 = self.Omega_m
        OL0 = self.Omega_de0
        n = self.n
        f_R0 = self.f_R0

        u = Om0 * a ** (-3) + 4.0 * OL0
        u0 = Om0 + 4.0 * OL0

        Mhat2 = -(u ** (n + 2.0)) / (
            (n + 1.0) * f_R0 * u0 ** (n + 1.0)
        )

        Y = k * (constants.c / 100.0) / a
        Y2 = Y ** 2

        return 1.0 + (Y2 / 3.0) / (Y2 + Mhat2)
