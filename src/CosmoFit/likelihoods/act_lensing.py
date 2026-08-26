"""
ACT DR6 CMB lensing likelihood.

An independent, tighter lensing measurement
-------------------------------------------
:mod:`likelihoods.planck_lensing` is one reconstruction of the
lensing potential; this is another, from a different telescope, a
different sky area and a different pipeline. ACT DR6 measures the
lensing power spectrum amplitude to 2.3% -- better than Planck's --
over ``40 <= L <= 763`` with 10 bandpowers.

Two independent measurements of the same thing is worth more than
either alone, and not only for the extra precision: CMB lensing is
the cleanest handle anyone has on ``sigma_8 Omega_m^{0.25}``, and it
is the quantity the S8 tension is about.

The likelihood
--------------
A binned Gaussian, and simpler than Planck's because ACT ships a
binning *matrix* rather than per-bin window files:

    chi2 = (d - B C^{kappakappa})^T C^{-1} (d - B C^{kappakappa})

Three details decide whether it is right.

**The convergence, not the potential.** ACT's products are built on
``C_L^{kappakappa}``, a raw ``C_L``; Planck's are built on
``[L(L+1)]^2 C_L^{phiphi} / 2 pi``. The two describe identical
physics and differ by ``2 pi / 4``, which is exactly the kind of
mistake that survives: a smooth rescaling of the prediction is
something a fit absorbs into the amplitude, leaving a plausible
posterior in the wrong place. The dataset carries which spectrum its
windows act on rather than leaving it to be remembered.

**The Hartlap correction.** The covariance is estimated from 796
simulations, so its inverse is biased. The standard correction,
``(n_sim - n_bin - 2) / (n_sim - 1)``, multiplies the *inverse* --
applied here by dividing the covariance itself, which is equivalent
and lets the library's own covariance machinery handle the rest.

**The bin range.** The released vector runs wider than the range ACT
adopts. The baseline variant uses bins ``[2:-6]``, and the same rows
and columns have to come out of the covariance and the binning
matrix. Dropping them from one and not another is silent.

What this implements, and what it does not
------------------------------------------
ACT distributes two modes. ``lens_only`` uses a
**CMB-marginalized covariance** that already accounts for the
reconstruction's dependence on the primary CMB spectra it was
normalized against; the other mode uses an unmarginalized covariance
plus an explicit normalization correction, whose response matrix is
a ``(n_bin, 4001, 4001)`` object that cannot reasonably be shipped
inside a library.

This implements the first. ACT recommend it when *not* combining
with primary CMB data. Combining anyway -- with ``"planck_lite"``,
say -- is **conservative rather than wrong**: the CMB uncertainty is
marginalized once in this covariance and constrained again by the
primary data, so the lensing error bar ends up slightly too wide.
That is the safe direction, and it is stated rather than hidden. For
the exact treatment, ACT's own ``act_dr6_lenslike`` package is the
reference.

References
----------
Madhavacheril et al. (ACT Collaboration, 2024), *The Atacama
Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological
Parameters*, ApJ 962, 113, arXiv:2304.05203.

Qu et al. (ACT Collaboration, 2024), *The Atacama Cosmology
Telescope: A Measurement of the DR6 CMB Lensing Power Spectrum and
its Implications for Structure Growth*, ApJ 962, 112,
arXiv:2304.05202.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.boltzmann import CAMBBackend
from CosmoFit.data.loader import load_act_lensing

from .base import BaseLikelihood


#: ``C_L^{kappakappa} = [L(L+1)]^2 C_L^{phiphi} / 4``, and CAMB
#: hands back ``[L(L+1)]^2 C_L^{phiphi} / 2 pi`` -- so the two differ
#: by this factor. Written out rather than inlined because it is the
#: single conversion this whole module turns on.
POTENTIAL_TO_CONVERGENCE = np.pi / 2.0


class ACTDR6LensingLikelihood(BaseLikelihood):
    """
    ACT DR6 lensing bandpowers against a Boltzmann-computed
    ``C_L^{kappakappa}``.

    Parameters
    ----------
    cosmology
        Cosmology model instance. Must be one CAMB can represent.

    version : str, optional
        ``"act_baseline"`` (default, ``40 <= L <= 763``) or
        ``"act_extended"`` (out to ``L < 1250``). ACT's own baseline
        is the first; the extended range includes multipoles whose
        systematics they flag as less well controlled.

    lens_potential_accuracy : int, optional
        CAMB's lensing accuracy. ACT recommend 4 or higher, which is
        the default here.

    Notes
    -----
    Combining with ``"planck_lensing"`` is *not* double-counting --
    different telescopes, different skies -- but the two are not
    perfectly independent either, since ACT's map overlaps Planck's
    on the sky. ACT provide a proper joint variant
    (``actplanck_*``) for exactly this; it is not implemented here,
    so combining the two separate likelihoods slightly overstates
    the joint constraint. :class:`~stats.fitter.Fitter` warns.
    """

    def __init__(
        self,
        cosmology,
        version: str = "act_baseline",
        lens_potential_accuracy: int = 4,
    ):

        dataset = load_act_lensing(version)

        self.backend = CAMBBackend.shared(

            cosmology,

            lmax=dataset.lmax,

            lens_potential_accuracy=lens_potential_accuracy,

        )

        low, high = dataset.ell_range

        super().__init__(

            name=f"ACT DR6 lensing (L {low}-{high})",

            dataset=dataset,

            cosmology=cosmology,

        )

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "C_L^kappakappa"

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted lensing bandpowers.
        """

        data = self.data

        spectra = self.backend.lensing_spectra(data.lmax)

        convergence = spectra["PP"] * POTENTIAL_TO_CONVERGENCE

        return data.windows @ convergence

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
