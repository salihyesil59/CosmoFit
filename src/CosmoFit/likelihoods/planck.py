"""
Planck CMB distance-prior likelihood.

Implements the standard "compressed CMB" likelihood used
throughout the dark-energy literature: instead of the full
Planck power-spectrum likelihood, the CMB constraint is reduced
to a 3-vector

    p = (R, l_A, omega_b_h2)

    R      = shift parameter    = sqrt(Omega_m) * H0 * D_M(z*) / c
    l_A    = acoustic scale     = pi * D_M(z*) / r_s(z*)
    omega_b_h2 = physical baryon density = Omega_b * h^2

evaluated against a Gaussian likelihood with the Planck-derived
mean vector and covariance matrix (see :mod:`data.loader`).

z* and r_s(z*) are obtained from
:class:`cosmology.calculators.recombination.RecombinationCalculator`
(fitting formulas -- see that module's docstring for the
approximations involved and their literature sources).

References
----------
Chen, Huang & Wang (2019), JCAP 02 (2019) 028, arXiv:1808.05724.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core.constants import c as SPEED_OF_LIGHT

from CosmoFit.data.loader import load_planck

from .base import BaseLikelihood


class PlanckLikelihood(BaseLikelihood):

    def __init__(
        self,
        cosmology,
        version="planck2018",
    ):

        dataset = load_planck(

            version,

        )

        super().__init__(

            name="Planck",

            dataset=dataset,

            cosmology=cosmology,

        )

    # --------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "(R, l_A, omega_b_h2)"

    # --------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Compute the theoretical Planck distance-prior vector
        (R, l_A, omega_b_h2) for the current cosmology.

        D_M(z*) is obtained from :meth:`RecombinationCalculator.chi_star`,
        which integrates chi(z) = int dz/E(z) directly out to
        z* ~ 1090, rather than through the cosmology's cached low-z
        distance-integrator grid (which only spans z=0..5 by
        default, sized for the low-redshift CC/BAO/SN likelihoods).
        This keeps the low-z grid's resolution -- and the
        performance of every other likelihood -- unaffected by the
        CMB's much larger redshift range.
        """

        recomb = self.cosmology.recombination

        z_star = recomb.z_star()
        r_s = recomb.sound_horizon(z_star)
        chi_star = recomb.chi_star(z_star)

        dm_star = self.cosmology.distance.DM_from_chi(chi_star)

        R = (
            np.sqrt(self.cosmology.Omega_m)
            * self.cosmology.H0
            * dm_star
            / SPEED_OF_LIGHT
        )

        lA = np.pi * dm_star / r_s

        omega_b_h2 = recomb.omega_b_h2

        return np.array(

            [R, lA, omega_b_h2],

            dtype=float,

        )

    # --------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Compute residuals.
        """

        return (

            self.data.values

            - self.model()

        )

    # --------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Compute chi-square.
        """

        return self.covariance.chi2(

            self.residuals(),

        )

