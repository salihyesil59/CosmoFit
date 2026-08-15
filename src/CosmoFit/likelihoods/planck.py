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

z*, r_s(z*) and the radiation-aware E(z) behind D_M(z*) all come from
:class:`cosmology.calculators.recombination.RecombinationCalculator`,
which follows Chen, Huang & Wang (2019)'s *own* definitions
(their Eqs. 1-6) rather than a more detailed independent
calculation.

That is deliberate and it matters. These priors are not a measurement
of the sky -- they are a compression of Planck's own fit, computed
under a specific set of conventions, and the theory prediction has to
share those conventions or the comparison is inconsistent. An earlier
version of this module did not, and returned chi2 ~ 100 for 3 data
points at *Planck's own best-fit LCDM* (l_A off by -8.9 sigma), which
biased every joint fit that included this dataset. It now returns
chi2 ~ 0.4 there. See the recombination module's docstring for the
full diagnosis.

Validation
----------
The prediction has been cross-checked two ways: against CAMB 2.0.1
(z_star agrees to 0.002%), and against an independent ``scipy.quad``
implementation of the CHW19 recipe across flat/open/closed LCDM and
CPL (R and l_A agree to <0.01 sigma).

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

