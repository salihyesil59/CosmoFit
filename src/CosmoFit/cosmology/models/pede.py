"""
Phenomenologically Emergent Dark Energy (PEDE).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


#: ln(10), the conversion between the log10 that PEDE's functional
#: form is written with and the natural log its derivative needs.
_LN10 = np.log(10.0)


class PEDE(Cosmology):
    r"""
    Phenomenologically Emergent Dark Energy (Li & Shafieloo 2019).

        Omega_de(z) = Omega_de0 [1 - tanh(log10(1 + z))]

    Dark energy that is *absent* at high redshift and "emerges"
    toward the present: the bracket goes to 0 as z grows and to 1
    at z = 0, so the model has no dark-energy density during matter
    domination at all.

    What makes it worth having in a library that already carries
    six dark-energy parametrizations is that it has **no free
    parameter for dark energy whatsoever**. CPL, JBP and BA each
    buy their extra flexibility with two parameters (``w0``,
    ``wa``); PEDE has the same number of free parameters as LCDM --
    ``H0`` and ``Omega_m`` -- and a completely different expansion
    history. That makes the model comparison honest in a way a
    nested-model comparison is not: AIC/BIC penalties are
    identical, so a difference in chi2 is a difference in fit,
    full stop.

    Its other claim is on the Hubble tension. The effective
    equation of state,

        w(z) = -1 - (1 / (3 ln 10)) [1 + tanh(log10(1 + z))]

    is phantom at every redshift and equals -1.145 today, which
    raises the inferred ``H0`` for a CMB-anchored fit relative to
    LCDM. Whether that survives the full data set is exactly the
    kind of question this library exists to ask, not something to
    assert here.

    Notes
    -----
    ``Omega_de0 = 1 - Omega_m - Omega_k`` as usual, so
    ``E(z=0) = 1`` holds identically, including for curved cases.

    References
    ----------
    Li & Shafieloo (2019), "A Simple Phenomenological Emergent Dark
    Energy Model can Resolve the Hubble Tension", ApJ 883, L3,
    arXiv:1906.08275.
    """

    MODEL_NAME = "PEDE"

    # ---------------------------------------------------------

    @staticmethod
    def _u(z):
        """
        ``log10(1 + z)``, the variable PEDE's tanh is written in.
        """

        return np.log10(1.0 + np.asarray(z, dtype=float))

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:

        return self.Omega_de0 * (

            1.0 - np.tanh(self._u(z))

        )

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * cube(1.0 + z)

            +

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_de(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        u = self._u(z)

        # d/dz [Omega_de0 (1 - tanh u)] with u = ln(1+z)/ln(10),
        # so du/dz = 1 / ((1+z) ln 10) and d(tanh u)/du = sech^2 u.
        d_omega_de = -(

            self.Omega_de0

            / (np.cosh(u) ** 2)

            / ((1.0 + z) * _LN10)

        )

        return (

            (

                3.0 * self.Omega_m * (1.0 + z) ** 2

                +

                2.0 * self.Omega_k * (1.0 + z)

                +

                d_omega_de

            )

            /

            (2.0 * self.E(z))

        )

    # ---------------------------------------------------------

    def w(self, z: Redshift) -> Array:
        r"""
        Effective dark-energy equation of state,

            w(z) = -1 - [1 + tanh(log10(1+z))] / (3 ln 10)

        Obtained from the continuity equation,
        ``w = (1/3) dln rho_de / dln(1+z) - 1``, using
        ``sech^2 u / (1 - tanh u) = 1 + tanh u``. Matches Li &
        Shafieloo's Eq. (3).

        Phantom (``w < -1``) at every redshift: ``-1.145`` today,
        tending to ``-1`` in the far future (``z -> -1``, where
        ``tanh -> -1``) and to ``-1 - 2/(3 ln 10) = -1.2895`` deep
        in the past. This is what lets the model be handed
        to a Boltzmann code (see
        :class:`~cosmology.boltzmann.CAMBBackend`), which needs an
        equation of state rather than only an ``E(z)``.
        """

        return -1.0 - (

            1.0 + np.tanh(self._u(z))

        ) / (3.0 * _LN10)
