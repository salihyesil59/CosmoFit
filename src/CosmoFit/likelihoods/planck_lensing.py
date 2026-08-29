"""
Planck 2018 CMB lensing likelihood.

What this measures that nothing else here does
----------------------------------------------
The CMB we observe has been gravitationally lensed by everything it
passed through. That deflection is imprinted as a specific
non-Gaussian correlation between multipoles, and it can be inverted
to reconstruct the lensing potential ``phi`` -- a map of the
integrated matter distribution back to z ~ 1100, peaking around
z ~ 2.

Its power spectrum is therefore a *growth* measurement made by the
CMB itself. Every other CMB dataset in this library
(:mod:`likelihoods.planck`, :mod:`likelihoods.planck_lite`)
constrains the universe at recombination and reaches the present only
through a distance; this one constrains ``sigma_8 Omega_m^{0.25}``
directly, from the same photons.

That makes it the piece needed to pose the S8 question from the CMB
side rather than from a weak-lensing survey's, and it is why every
modern Planck analysis quotes results with and without it.

The likelihood
--------------
Nine bandpowers of ``[L(L+1)]^2 C_L^{phiphi} / 2 pi`` over
``8 <= L <= 400`` (Planck's conservative baseline), Gaussian against
their 9x9 covariance:

    chi2 = (d - t)^T C^-1 (d - t)

    t_b = sum_L W_bL C_L^{phiphi}
          + sum_X sum_L M^X_bL C_L^X  -  f_b

The first term is the ordinary binning. The second is what makes this
likelihood different, and dropping it is the obvious mistake:

**The reconstruction is normalized against an assumed cosmology.**
The lensing estimator weights multipole pairs by the CMB spectra it
expects to see, so the recovered ``C_L^{phiphi}`` depends on the
fiducial TT/EE/TE used to build it. Move away from that fiducial and
the normalization is wrong. ``M^X_bL`` propagates that dependence to
first order, and ``f_b`` is the same quantity at the fiducial
cosmology, subtracted so the correction is zero there by
construction.

At the fiducial point the correction is worth ``chi2`` 0.02 -- which
is exactly the point: it is *designed* to vanish there, so testing
only at the fiducial cosmology cannot tell you whether you
implemented it. It grows as the fit moves away, which is where it
matters and where nothing would flag its absence.

Validation
----------
At Planck's own best-fit LCDM this returns ``chi2 = 8.8`` for 9
bandpowers, against the ~9 Planck 2018 reports. See
``tests/test_planck_lensing.py``.

References
----------
Planck Collaboration (2020), *Planck 2018 results. VIII. Gravitational
lensing*, A&A 641, A8, arXiv:1807.06210.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.boltzmann import CAMBBackend
from CosmoFit.data.loader import load_planck_lensing

from .base import BaseLikelihood


class PlanckLensingLikelihood(BaseLikelihood):
    """
    Planck 2018 lensing-potential bandpowers against a
    Boltzmann-computed ``C_L^{phiphi}``.

    Parameters
    ----------
    cosmology
        Cosmology model instance. Must be one CAMB can represent
        (LCDM, or any model with a ``w(z)``); anything else raises
        :class:`~cosmology.boltzmann.BoltzmannError` at
        construction rather than mid-chain.

    version : str, optional
        Dataset version.

    lens_potential_accuracy : int, optional
        CAMB's lensing accuracy setting. Default 4 here, not the 1
        used for the temperature and polarization spectra: this
        likelihood *is* the lensing spectrum, and 1 is calibrated
        for lensing's effect on TT/TE/EE rather than for the
        potential itself. The difference is well under the
        bandpower errors, but it costs little and there is no
        reason to accept avoidable numerical error in the one
        quantity being fitted.

    Notes
    -----
    Combining this with ``"planck"`` or ``"planck_lite"`` is
    normal and is what Planck's own analyses do. The lensing
    reconstruction is built from the same maps as the power spectra,
    so the two are not strictly independent, but the correlation is
    small enough that Planck distributes and combines them as
    separate likelihoods -- which is why no conflict is registered
    for the pair.
    """

    def __init__(
        self,
        cosmology,
        version: str = "planck2018",
        lens_potential_accuracy: int = 4,
    ):

        dataset = load_planck_lensing(version)

        # Constructed before `super().__init__` so an
        # unrepresentable model fails here, at setup, rather than
        # thousands of MCMC steps in.
        # Shared with any other CAMB-based likelihood on the same
        # cosmology -- see `CAMBBackend.shared`.
        self.backend = CAMBBackend.shared(

            cosmology,

            lmax=dataset.lmax,

            lens_potential_accuracy=lens_potential_accuracy,

        )

        low, high = dataset.ell_range

        super().__init__(

            name=f"Planck lensing (L {low}-{high})",

            dataset=dataset,

            cosmology=cosmology,

        )

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "C_L^phiphi"

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted lensing bandpowers for the current cosmology.
        """

        data = self.data

        spectra = self.backend.lensing_spectra(data.lmax)

        # The plain binning of the lensing spectrum.
        binned = data.windows @ spectra["PP"]

        if not data.has_linear_correction:
            return binned

        # The normalization correction: one window per contributing
        # spectrum, summed, minus its value at the fiducial
        # cosmology.
        correction = np.zeros_like(binned)

        for index, name in enumerate(data.CORRECTION_SPECTRA):

            correction = correction + (

                data.delta_windows[:, index, :] @ spectra[name]

            )

        return binned + correction - data.fiducial_correction

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Data minus model bandpower residuals.
        """

        return (

            self.data.value

            - self.model()

        )

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        return self.covariance.chi2(

            self.residuals(),

        )
