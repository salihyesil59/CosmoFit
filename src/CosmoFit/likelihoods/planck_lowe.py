"""
Planck 2018 low-multipole EE likelihood (SimAll).

The one place a Gaussian will not do
------------------------------------
Everything else in this library compares a prediction against a mean
and a covariance. That works because most measurements average over
enough modes for the central limit theorem to apply.

Below ``l = 30`` it does not. There are only ``2l + 1`` modes on the
sky at multipole ``l`` -- five of them at ``l = 2`` -- so the C_l
distribution is a strongly skewed chi-square-like thing, not a
Gaussian. And this is precisely the regime that carries essentially
all of the CMB's information about the reionization optical depth
``tau``, because reionization's signature is a bump in EE at exactly
these multipoles.

So Planck ships a *table*: for each multipole ``l = 2..29`` and each
value of ``D_l^EE`` on a grid of ``1e-4 muK^2``, the log-probability.
The likelihood is a lookup and a sum:

    log L = sum_l  table[int(D_l^EE / step), l - 2]

No mean, no covariance, no assumption of symmetry.

Relation to the ``"tau"`` dataset
---------------------------------
CosmoFit's ``"tau"`` dataset is the familiar Gaussian shorthand,
``tau = 0.0544 +- 0.0073``, which is what most dark-energy papers
use and what this library used before this module existed. It is a
compression *of this*, and the two must not both be in one fit --
that would be Planck's low-l polarization counted twice, once in
full and once in summary. :class:`~stats.fitter.Fitter` refuses the
combination.

Which to use is a real choice. The Gaussian prior costs nothing and
is accurate enough for most purposes. This is the actual
measurement, it is what pins ``tau`` when the fit is allowed to
explore, and it is the only version that represents the skewness --
which matters most exactly where a fit wants to push ``tau`` low,
since the true likelihood falls off far less steeply there than a
Gaussian does.

Validation
----------
Scanning ``tau`` at Planck's best-fit LCDM recovers a maximum at
``tau = 0.054`` with a width matching the published
``0.0544 +- 0.0073`` -- which is the strongest available check,
since it reconstructs the published constraint from the table
rather than assuming it. See ``tests/test_planck_lowe.py``.

References
----------
Planck Collaboration (2020), *Planck 2018 results. V. CMB power
spectra and likelihoods*, A&A 641, A5, arXiv:1907.12875.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.boltzmann import CAMBBackend
from CosmoFit.data.loader import load_planck_lowe

from .base import BaseLikelihood


class PlanckLowEELikelihood(BaseLikelihood):
    """
    Planck 2018 low-multipole EE, via its released probability
    table.

    Parameters
    ----------
    cosmology
        Cosmology model instance. Must be one CAMB can represent.

    version : str, optional
        Dataset version.

    Notes
    -----
    Unlike every other likelihood here, this one has no residuals
    and no covariance -- there is no mean to subtract from. It
    reports ``chi2 = -2 log L`` so that it composes with the rest
    (``JointLikelihood`` sums chi2, ``AIC``/``BIC`` consume it), but
    that number is *not* a sum of squared pulls and should not be
    read as one. In particular it does not go to zero at a perfect
    fit: the table's log-probabilities are normalized densities, so
    the best achievable value is some finite negative log-likelihood
    set by the data, not zero.

    :meth:`residuals` therefore returns the difference between the
    predicted ``D_l^EE`` and the table's own maximum-probability
    value at each multipole. That is a genuine diagnostic -- it says
    where the model sits relative to the most likely value -- but it
    is not what the chi2 is built from.
    """

    def __init__(
        self,
        cosmology,
        version: str = "planck2018",
    ):

        dataset = load_planck_lowe(version)

        # `lmax + 500` of headroom inside the backend, and the
        # bandpower likelihood may want far more -- `shared` widens
        # to whatever the largest requester needs.
        self.backend = CAMBBackend.shared(

            cosmology,

            lmax=max(dataset.lmax, 30),

        )

        super().__init__(

            name=f"Planck lowE (l {dataset.lmin}-{dataset.lmax})",

            dataset=dataset,

            cosmology=cosmology,

        )

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return "D_l^EE (l < 30)"

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        r"""
        Predicted ``D_l^EE = l(l+1) C_l^EE / 2 pi`` [muK^2] for
        ``l = lmin..lmax``.

        Divided by ``A_planck^2``, the same absolute calibration the
        bandpower likelihood applies -- Planck's own low-l EE
        likelihood carries it too.
        """

        data = self.data

        spectra = self.backend.lensing_spectra(

            max(data.lmax, 30),

        )

        ee = spectra["EE"][data.lmin: data.lmax + 1]

        return ee / self.cosmology.A_planck ** 2

    # ---------------------------------------------------------

    def log_likelihood(
        self,
    ) -> float:
        """
        Log-likelihood, read straight out of the table.

        A predicted ``D_l^EE`` outside the tabulated range is not a
        numerical problem to clamp away -- it is a cosmology the
        data excludes -- so it returns ``-inf`` and lets the sampler
        move on, which is what the reference implementation does.
        """

        data = self.data

        index = (self.model() / data.step).astype(int)

        if np.any(index < 0) or np.any(index >= data.n_step):

            return -np.inf

        return float(

            data.table[index, np.arange(data.size)].sum(),

        )

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Predicted ``D_l^EE`` minus the table's most probable value
        at each multipole.

        A diagnostic, not the quantity the likelihood is built from
        -- see the class docstring.
        """

        data = self.data

        peak = data.table.argmax(axis=0) * data.step

        return self.model() - peak

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        ``-2 log L``, so this composes with the Gaussian
        likelihoods it is summed with.

        Not a sum of squared pulls, and not zero at the best fit --
        see the class docstring.
        """

        log_like = self.log_likelihood()

        if not np.isfinite(log_like):

            return np.inf

        return -2.0 * log_like
