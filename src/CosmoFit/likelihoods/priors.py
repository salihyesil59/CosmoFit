"""
External single-number Gaussian constraints.

These are the measurements that a dark-energy fit borrows from
elsewhere in cosmology rather than derives itself: a local
distance-ladder ``H0``, a BBN ``omega_b h^2``, a reionization
``tau``. Each is one number with one error bar, and each enters
the fit as a dataset with one data point.

Why a dataset and not a prior
-----------------------------
It would be shorter to express "SH0ES measured H0 = 73.04 +- 1.04"
as a Gaussian prior on the ``H0`` parameter, and mathematically the
posterior is identical either way. It is still the wrong place to
put it. A prior is a statement about belief before seeing data; the
SH0ES measurement *is* data, from a telescope, with a paper and a
systematic error budget. Treating it as a dataset keeps it visible
where it should be visible: in the per-dataset chi2 breakdown
(:meth:`~likelihoods.joint.JointLikelihood.summary`), in the
degrees-of-freedom count that AIC/BIC use, in the dataset list a
figure legend prints, and in the chain metadata that decides whether
a saved chain may be resumed. A fit that quietly assumed the local
distance ladder should not look, from the outside, like a fit that
did not.

It also makes the H0-tension question askable in the form it is
actually argued: run the same model against ``["desi", "planck"]``
and against ``["desi", "planck", "h0"]``, and compare the chi2 each
dataset contributes. If the local H0 is a prior, it never shows up
in that table.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import load_gaussian_prior

from .base import BaseLikelihood


#: How each constrained quantity is predicted from a cosmology.
#:
#: Keys are the ``quantity`` field of the corresponding
#: :class:`~data.dataset.GaussianPriorDataset`; values take the
#: cosmology and return a scalar.
QUANTITY_MAP = {

    "H0": lambda m: float(m.H0),

    # Omega_b h^2, with h = H0/100 -- the same combination
    # `RecombinationCalculator` uses, and the one BBN actually
    # constrains (BBN is sensitive to the baryon-to-photon ratio,
    # which depends on the physical density, not on Omega_b and H0
    # separately).
    "omega_b_h2": lambda m: float(m.Omega_b * (m.H0 / 100.0) ** 2),

    "tau_reio": lambda m: float(m.params.tau_reio),

}


#: LaTeX label for each quantity, for figure axes and tables.
QUANTITY_LABELS = {
    "H0": r"$H_0$",
    "omega_b_h2": r"$\omega_b h^2$",
    "tau_reio": r"$\tau$",
}


#: Display name for each prior dataset.
PRIOR_NAMES = {
    "h0": "H0",
    "omega_b": "BBN",
    "tau": "tau",
}


class GaussianPriorLikelihood(BaseLikelihood):
    """
    A single Gaussian constraint on one derived or free quantity.

        chi2 = (value - prediction)^2 / sigma^2

    Parameters
    ----------
    cosmology
        Cosmology model instance.

    dataset : str
        Which constraint: ``"h0"``, ``"omega_b"`` or ``"tau"``.

    version : str, optional
        Which measurement of it (see
        :func:`~data.loader.available_versions`). Defaults to that
        dataset's first registered version.

    Warning
    -------
    Do not combine two versions of the same prior in one fit (two
    ``H0`` measurements, say). They are independent measurements of
    the same number, not a joint constraint, and stacking them
    multiplies two likelihoods that are each already the full
    statement about that quantity -- the same rule that applies to
    the two ``"s8"`` versions.

    Combining an ``"h0"`` prior with ``"pantheon"`` is a subtler
    version of the same trap: Pantheon+SH0ES's Cepheid calibrators
    (loaded only with ``include_cepheid=True``) *are* the SH0ES
    measurement. With the default ``include_cepheid=False`` there is
    no double counting, since the calibrators are excluded and the
    SN absolute magnitude is marginalized -- but turn Cepheids on
    and the ``"h0"`` dataset is the same information twice.
    """

    def __init__(
        self,
        cosmology,
        dataset: str = "h0",
        version: str | None = None,
    ):

        data = load_gaussian_prior(

            dataset,

            version,

        )

        if data.quantity not in QUANTITY_MAP:

            raise ValueError(

                f"No model prediction registered for quantity "

                f"'{data.quantity}'. "

                f"Available: {list(QUANTITY_MAP)}",

            )

        self.dataset_name = dataset

        self.quantity = data.quantity

        super().__init__(

            name=PRIOR_NAMES.get(dataset, dataset),

            dataset=data,

            cosmology=cosmology,

        )

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return self.quantity

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted value of the constrained quantity.
        """

        return np.array(

            [

                QUANTITY_MAP[self.quantity](

                    self.cosmology,

                ),

            ],

            dtype=float,

        )

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Data minus model residual (a length-1 vector).
        """

        return (

            np.array([self.data.value], dtype=float)

            - self.model()

        )

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        residual = float(self.residuals()[0])

        return (residual / self.data.sigma) ** 2


# ============================================================
# Named subclasses, one per constrained quantity
# ============================================================

class H0Likelihood(GaussianPriorLikelihood):
    """
    A local distance-ladder H0 measurement.

    Versions: ``"sh0es2022"`` (default; Riess et al. 2022,
    73.04 +- 1.04), ``"sh0es2024"`` (Breuval et al. 2024, the
    JWST/HST Cepheid re-calibration, 73.17 +- 0.86), and
    ``"tdcosmo2025"`` (Birrer et al. 2025, strong-lensing time
    delays, 71.6 +- 3.6 -- independent of the Cepheid ladder
    entirely, and the one to reach for when the question is
    whether the tension survives dropping SH0ES).

    Do not combine two of them: they measure the same number.
    """

    def __init__(self, cosmology, version: str = "sh0es2022"):

        super().__init__(cosmology, dataset="h0", version=version)


class OmegaBLikelihood(GaussianPriorLikelihood):
    """
    A BBN constraint on the physical baryon density
    ``omega_b h^2``.

    This is the piece that makes a BAO-only fit able to say
    anything about H0 at all. BAO measures ``D/r_d``, and ``r_d``
    depends on ``omega_b`` and ``omega_m`` -- so with ``r_d``
    computed rather than left free (see
    :class:`~cosmology.calculators.sound_horizon.SoundHorizon`), a
    BBN prior on ``omega_b`` closes the loop and turns BAO into an
    absolute distance measurement. That is exactly how the DESI
    "BAO + BBN" constraints are produced, and it is a genuinely
    CMB-independent route to H0.

    Versions: ``"bbn2024"`` (default; Schoeneberg 2024, the
    conservative PDG-abundance value DESI adopts, 0.02218 +-
    0.00055) and ``"cooke2018"`` (0.02166 +- 0.00019 -- three
    times tighter because it propagates only the deuterium
    measurement error, not the nuclear-reaction-rate spread; a
    deliberate choice, not a better default).
    """

    def __init__(self, cosmology, version: str = "bbn2024"):

        super().__init__(cosmology, dataset="omega_b", version=version)


class TauLikelihood(GaussianPriorLikelihood):
    """
    The Planck 2018 large-scale-polarization constraint on the
    reionization optical depth, ``tau = 0.0544 +- 0.0073``.

    Only meaningful alongside
    :class:`~likelihoods.planck_lite.PlanckLiteLikelihood`, which
    covers l >= 30 only. There, ``tau`` enters solely through the
    ``exp(-2 tau)`` damping of the spectra and is therefore
    degenerate with the primordial amplitude ``A_s``; this prior is
    what breaks that degeneracy, and it is the standard companion
    to a high-l-only CMB fit rather than an optional extra. Without
    it, ``sigma8`` and anything derived from it are unconstrained.
    """

    def __init__(self, cosmology, version: str = "planck2018"):

        super().__init__(cosmology, dataset="tau", version=version)
