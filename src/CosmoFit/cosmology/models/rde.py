"""
Ricci dark energy (Gao, Chen & Shen 2009).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


class RDE(Cosmology):
    r"""
    Ricci dark energy: the holographic cutoff is the Ricci scalar.

    :class:`~cosmology.models.hde.HDE` takes the future event
    horizon as its infrared cutoff, which makes today's dark-energy
    density depend on the entire future expansion history -- unusual
    as physics, and the thing most often objected to. Ricci dark
    energy takes a *local* curvature scale instead,

        rho_DE = 3 gamma M_p^2 (Hdot + 2 H^2),

    which is the Ricci scalar of a spatially flat universe up to a
    constant. Same holographic idea, no reference to the future.

    Unlike HDE this has a closed form. Solving the Friedmann
    equation with that density gives

        E(z)^2 = A (1+z)^3 + (1 - A) (1+z)^(4 - 2/gamma),
        A = 2 Omega_m / (2 - gamma),

    so the dark sector behaves as a power law in ``(1+z)`` whose
    exponent the single parameter ``gamma_rde`` sets. Flatness fixes
    the normalization: ``E(0) = 1`` identically, for every
    ``gamma``.

    Reading the exponent is the quickest way to see what the model
    does. ``gamma = 1/2`` gives ``(1+z)^0``, a constant dark-energy
    density -- so the model is LCDM there, though with an
    *effective* matter density ``A = (4/3) Omega_m`` rather than
    ``Omega_m``, because part of the Ricci density scales like
    matter. Below that the exponent is negative and the dark
    energy *grows* into the future; above it the density falls with
    redshift like a decaying component. Fits have consistently
    preferred ``gamma`` slightly above 1/2 -- and, more awkwardly, a
    low matter density: this model wants ``Omega_m ~ 0.22`` where
    everything else in this library wants 0.31, which is its main
    observational problem rather than a detail.

    Caveats
    -------
    Its perturbations are outside what a standard Boltzmann code
    solves, so it is refused by
    :class:`~cosmology.boltzmann.CAMBBackend`; use the compressed
    distance priors for its CMB constraint.

    Flat universes only, for the same reason as
    :class:`~cosmology.models.hde.HDE`: the closed form above is
    derived from the flat Friedmann equation, and curvature changes
    the Ricci scalar the cutoff is built from rather than adding a
    term.

    References
    ----------
    Gao, Chen, Shen & Saridakis (2009), Phys. Rev. D 79, 043511,
    `arXiv:0712.1394 <https://arxiv.org/abs/0712.1394>`_.
    """

    MODEL_NAME = "RDE"
    MODEL_LABEL = "RDE"

    EXTRA_PARAMS = {

        "gamma_rde": {
            "default": 0.45,
            "bounds": (0.15, 1.2),
            "label": r"$\gamma$",
        },

    }

    # ---------------------------------------------------------

    def _check_flat(self):

        if abs(self.Omega_k) > 1.0e-12:

            raise ValueError(

                f"RDE is implemented for a flat universe only, but "
                f"Omega_k = {self.Omega_k:.4g}. The closed form it "
                f"uses is derived from the flat Friedmann equation, "
                f"and curvature changes the Ricci scalar the cutoff "
                f"is built from rather than adding a term.",

            )

    # ---------------------------------------------------------

    @property
    def _amplitude(self) -> float:
        """
        ``A = 2 Omega_m / (2 - gamma)``, the weight of the
        matter-like term.
        """

        gamma = float(self.params.gamma_rde)

        if gamma <= 0.0 or abs(2.0 - gamma) < 1.0e-12:

            raise ValueError(
                f"RDE needs 0 < gamma != 2, got "
                f"gamma_rde = {gamma:.4g}.",
            )

        return 2.0 * self.Omega_m / (2.0 - gamma)

    @property
    def _exponent(self) -> float:
        """
        ``4 - 2/gamma``: the power of ``(1+z)`` the dark sector
        follows. Zero at ``gamma = 1/2``, which is a cosmological
        constant.
        """

        return 4.0 - 2.0 / float(self.params.gamma_rde)

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:
        r"""
        Dark-energy density in units of today's critical density --
        ``E(z)^2 - Omega_m (1+z)^3``, as every other model here
        returns, so that ``Omega_de(0) = 1 - Omega_m``.

        Not the same as the second term of :meth:`E`. Ricci dark
        energy carries a piece that *scales like matter*: the
        coefficient of ``(1+z)^3`` in ``E^2`` is
        ``A = 2 Omega_m / (2 - gamma)``, which exceeds ``Omega_m``
        for ``0 < gamma < 2``. Reading the two terms of ``E^2`` as
        "matter" and "dark energy" would therefore misattribute
        ``(A - Omega_m)(1+z)^3`` -- about a third of the matter
        density at the fitted ``gamma`` -- and would break
        ``Omega_de(0) = 1 - Omega_m``, which the recombination
        module relies on.
        """

        self._check_flat()

        z = np.asarray(z, dtype=float)

        return self.E(z) ** 2 - self.Omega_m * cube(1.0 + z)

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:

        self._check_flat()

        z = np.asarray(z, dtype=float)

        amplitude = self._amplitude

        return np.sqrt(

            amplitude * cube(1.0 + z)

            + (1.0 - amplitude) * (1.0 + z) ** self._exponent

        )

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        amplitude = self._amplitude

        exponent = self._exponent

        return (

            3.0 * amplitude * (1.0 + z) ** 2

            + (1.0 - amplitude) * exponent * (1.0 + z) ** (exponent - 1.0)

        ) / (2.0 * self.E(z))
