"""
Lambda_s CDM: the sign-switching cosmological constant.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


class LsCDM(Cosmology):
    r"""
    Lambda_s CDM (Akarsu et al. 2021): LCDM, except that the
    cosmological constant *changes sign* at a redshift
    ``z_dagger``.

        E(z)^2 = Omega_m (1+z)^3 + Omega_k (1+z)^2
                 + Omega_Ls0 sgn(z_dagger - z)

    Below ``z_dagger`` this is ordinary LCDM with a positive
    (de Sitter) Lambda. Above it, Lambda is *negative* -- an
    anti-de Sitter vacuum -- which suppresses the expansion rate in
    the pre-transition universe.

    Why a discontinuous model is taken seriously
    --------------------------------------------
    It looks like a hack, and the mechanism (a rapid AdS-to-dS
    vacuum transition, as in graduated dark energy or certain
    string-swampland constructions) is genuinely speculative. What
    earns it a place is that it addresses the H0 and S8 tensions
    *and* the BAO sound-horizon problem with a single parameter, by
    the one route that is otherwise hard to arrange: a lower
    expansion rate before ``z_dagger ~ 2`` shrinks the sound
    horizon ``r_d``, and a smaller ``r_d`` raises the ``H0``
    inferred from a fixed BAO angular scale. Parametrizations that
    only alter late-time dark energy cannot do this, because they
    leave ``r_d`` untouched.

    The extra parameter is ``z_dagger``, with fits typically
    preferring ``z_dagger ~ 1.7-2.3``.

    Numerical note
    --------------
    ``E(z)`` is genuinely discontinuous at ``z_dagger`` -- that is
    the model, not an artifact. The comoving distance
    ``chi(z) = int dz/E`` remains continuous (it acquires a kink,
    not a jump), but the trapezoidal grid in
    :class:`~cosmology.numerics.integrals.DistanceIntegrator`
    smears the step across one grid cell. Measured against a
    piecewise-exact integration that splits the integral at
    ``z_dagger``, the resulting error in ``chi`` is below 1e-4
    relative -- an order of magnitude under the precision of any
    dataset in this library. It is quantified rather than assumed;
    see ``tests/test_models.py``.

    ``dEdz`` returns the derivative of the smooth part only. The
    distributional delta at the transition is not representable as
    a float and is not used by anything: ``chi`` integrates ``E``
    rather than differentiating it, and the deceleration parameter
    ``q(z)`` is undefined exactly at a discontinuity anyway.

    References
    ----------
    Akarsu, Kumar, Ozulker & Vazquez (2021), "Relaxing cosmological
    tensions with a sign switching cosmological constant",
    Phys. Rev. D 104, 123512, arXiv:2108.09239.

    Akarsu, Kumar, Ozulker, Vazquez & Yadav (2023), Phys. Rev. D
    108, 023513, arXiv:2211.05742 (updated constraints).
    """

    MODEL_NAME = "LsCDM"
    MODEL_LABEL = r"$\Lambda_{\rm s}$CDM"

    EXTRA_PARAMS = {

        "z_dagger": {
            "default": 1.8,
            "bounds": (0.5, 4.0),
            "label": r"$z_\dagger$",
        },

    }

    # ---------------------------------------------------------

    def _sign(self, z):
        """
        ``sgn(z_dagger - z)``: +1 below the transition, -1 above.

        ``np.sign`` returns 0 exactly at ``z = z_dagger``, which
        would put a spurious zero-Lambda point in the middle of the
        expansion history; ``+1`` is used there instead, matching
        the convention that the transition happens *at* z_dagger
        rather than in an infinitesimal interval around it.
        """

        z = np.asarray(z, dtype=float)

        return np.where(z <= self.z_dagger, 1.0, -1.0)

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:

        return self.Omega_de0 * self._sign(z)

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        e2 = (

            self.Omega_m * cube(1.0 + z)

            +

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_de0 * self._sign(z)

        )

        # A negative Lambda above z_dagger can in principle
        # overwhelm the matter term for a low enough Omega_m, which
        # would mean a universe that recollapses before z_dagger --
        # not a numerical problem but a parameter region the model
        # does not describe. Surface it rather than returning nan
        # from the sqrt and letting the sampler wander there.
        if np.any(e2 <= 0.0):

            raise ValueError(

                f"LsCDM: E(z)^2 <= 0 for Omega_m="

                f"{self.Omega_m:.4f}, Omega_k={self.Omega_k:.4f}, "

                f"z_dagger={self.z_dagger:.3f}. Above z_dagger the "

                f"negative (AdS) Lambda exceeds the matter density, "

                f"so there is no expanding solution there.",

            )

        return np.sqrt(e2)

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        return (

            (

                3.0 * self.Omega_m * (1.0 + z) ** 2

                +

                2.0 * self.Omega_k * (1.0 + z)

            )

            /

            (2.0 * self.E(z))

        )
