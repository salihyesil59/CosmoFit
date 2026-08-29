"""
Generalized Emergent Dark Energy (GEDE).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


_LN10 = np.log(10.0)


class GEDE(Cosmology):
    r"""
    Generalized Emergent Dark Energy (Li & Shafieloo 2020).

    ::

        Omega_de(z) = Omega_de0
                      * [1 - tanh(Delta log10((1+z)/(1+z_t)))]
                      / [1 + tanh(Delta log10(1 + z_t))]

    The one-parameter family that contains both of the models it
    generalizes, which is what makes it useful:

    - ``Delta -> 0`` gives a constant ``Omega_de`` -- **LCDM**.
    - ``Delta = 1``, ``z_t = 0`` gives
      :class:`~cosmology.models.pede.PEDE`.

    So a fit for ``Delta`` is a direct, continuous test of how far
    the data pushes away from a cosmological constant, with LCDM
    sitting at a specific value of a real parameter rather than at
    the boundary of a different model. ``Delta`` controls how
    sharply dark energy emerges; ``z_t`` sets when.

    The denominator is not decoration -- it is the normalization
    that enforces ``Omega_de(z=0) = Omega_de0`` for any
    ``(Delta, z_t)``, and hence ``E(z=0) = 1``. Dropping it (an
    easy transcription slip) leaves a model that silently violates
    the Friedmann constraint by tens of percent.

    Notes
    -----
    In Li & Shafieloo's original presentation ``z_t`` is a
    *derived* quantity -- the redshift at which the dark-energy and
    matter densities are equal -- fixed by ``Delta`` and
    ``Omega_m`` rather than sampled. Here it is a free parameter,
    which is the form used in most follow-up work: it makes the
    model a genuine two-parameter extension that can be compared
    against CPL on equal footing, at the cost of the two being
    somewhat degenerate.

    References
    ----------
    Li & Shafieloo (2020), "Evidence for Emergent Dark Energy",
    ApJ 902, 58, arXiv:2001.05103.
    """

    MODEL_NAME = "GEDE"

    EXTRA_PARAMS = {

        "Delta": {
            "default": 1.0,
            "bounds": (-10.0, 10.0),
            "label": r"$\Delta$",
        },

        "z_t": {
            "default": 0.0,
            "bounds": (0.0, 5.0),
            "label": r"$z_t$",
        },

    }

    # ---------------------------------------------------------

    @property
    def _norm(self) -> float:
        """
        The ``1 + tanh(Delta log10(1 + z_t))`` denominator, which
        pins ``Omega_de(0)`` to ``Omega_de0``.
        """

        return 1.0 + np.tanh(

            self.Delta * np.log10(1.0 + self.z_t),

        )

    # ---------------------------------------------------------

    def _v(self, z):
        """
        ``Delta * log10((1 + z) / (1 + z_t))``.
        """

        z = np.asarray(z, dtype=float)

        return self.Delta * (

            np.log10(1.0 + z)

            - np.log10(1.0 + self.z_t)

        )

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:

        return (

            self.Omega_de0

            * (1.0 - np.tanh(self._v(z)))

            / self._norm

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

        v = self._v(z)

        # dv/dz = Delta / ((1+z) ln 10).
        d_omega_de = -(

            self.Omega_de0

            / self._norm

            / (np.cosh(v) ** 2)

            * self.Delta

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

            w(z) = -1 - (Delta / (3 ln 10)) [1 + tanh(v)]

        with ``v = Delta log10((1+z)/(1+z_t))``. The same
        derivation as :meth:`~cosmology.models.pede.PEDE.w`, with
        the extra factor of ``Delta`` from the chain rule -- and
        with the normalization cancelling, since it is a constant
        multiplying ``rho_de``.

        ``Delta = 0`` gives ``w = -1`` exactly, the LCDM limit.
        """

        return -1.0 - (

            self.Delta

            * (1.0 + np.tanh(self._v(z)))

            / (3.0 * _LN10)

        )
