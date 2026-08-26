"""
Dataset containers.

This module defines lightweight dataclasses used throughout the
project to store observational datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CosmoFit.data.covariance import CovarianceBase


# ============================================================
# Cosmic Chronometers
# ============================================================

@dataclass(slots=True)
class CCDataset:
    """
    Cosmic Chronometer measurements.
    """

    z: np.ndarray

    H: np.ndarray

    sigma: np.ndarray

    covariance: "CovarianceBase | None" = None

    reference: str = ""


    def __post_init__(
        self,
    ):

        n = len(

            self.z,

        )

        if not (

            len(self.H)

            == n

            == len(self.sigma)

        ):

            raise ValueError(

                "CC arrays have inconsistent lengths.",

            )

        if (

            self.covariance is not None

            and self.covariance.shape != (

                n,

                n,

            )

        ):

            raise ValueError(

                "CC covariance has wrong shape.",

            )

# -----------------------------------------------------------
    
    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z,

        )

# -----------------------------------------------------------

    @property
    def has_covariance(
        self,
    ) -> bool:
        """
        Whether the dataset has a covariance matrix.
        """

        return self.covariance is not None


# ============================================================
# BAO (DESI, SDSS)
# ============================================================

@dataclass(slots=True)
class DESIDataset:
    """
    Generic BAO distance-ratio measurements: a vector of
    (redshift, value, observable-type) triplets plus their
    covariance, where ``observable`` labels which theory quantity
    each value corresponds to (e.g. ``"DM_over_rs"``,
    ``"DH_over_rs"``, ``"DV_over_rs"`` -- see
    :data:`likelihoods.desi.MODEL_MAP`).

    Named for DESI (the first survey CosmoFit supported), but the
    container itself is survey-agnostic and is reused as-is for
    :func:`~data.loader.load_sdss_bao` (BOSS DR12 + eBOSS DR16
    LRG/QSO), which uses the same three-column format and the same
    observable-type strings.
    """

    z: np.ndarray

    value: np.ndarray

    observable: np.ndarray

    covariance: "CovarianceBase"

    reference: str = ""

    #: Per-point factor the *theory* sound horizon must be multiplied
    #: by before it is compared against ``value``. ``None`` (every
    #: DESI/SDSS entry) means 1 everywhere.
    #:
    #: This is not a fudge factor -- it is a units conversion. A BAO
    #: paper reports ``D_V/r_d`` in the units of whatever ``r_d``
    #: *its own* fiducial cosmology assigned, and older analyses
    #: computed that with the Eisenstein & Hu (1998) fitting formula
    #: rather than by integrating the sound speed as a Boltzmann code
    #: does. The two differ by ~2.7% (6dFGS: 153.9 vs 149.8 Mpc), so
    #: comparing a modern ``r_d`` against a measurement calibrated on
    #: the EH98 one is the same class of definitional mismatch that
    #: :mod:`likelihoods.planck` documents at length -- and at 2.7%
    #: on a 4.5%-precision measurement, it is not a rounding error.
    #: The rescale is taken from that survey's entry in
    #: CobayaSampler/cobaya, which applies exactly this factor.
    rs_rescale: "np.ndarray | None" = None

# -----------------------------------------------------------

    def __post_init__(
        self,
    ):

        n = len(

            self.z,

        )

        if not (

            len(self.observable)

            == n

            == len(self.value)

        ):

            raise ValueError(

                "BAO dataset arrays have inconsistent lengths.",

            )

        if self.rs_rescale is not None:

            self.rs_rescale = np.asarray(

                self.rs_rescale,

                dtype=float,

            )

            if len(self.rs_rescale) != n:

                raise ValueError(

                    "BAO dataset rs_rescale has wrong length.",

                )

        if self.covariance.shape != (

            n,

            n,

        ):

            raise ValueError(

                "BAO dataset covariance has wrong shape.",

            )

# -----------------------------------------------------------

    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z,

        )


# ============================================================
# Pantheon+
# ============================================================

@dataclass(slots=True)
class PantheonDataset:
    """
    Pantheon+SH0ES Supernova sample.

    Three redshifts are kept per supernova, matching the official
    Pantheon+SH0ES data release columns:

    z_hd
        Hubble-diagram redshift (CMB-frame, corrected for the
        local peculiar-velocity/bulk-flow model). This is the
        redshift the *cosmological* comoving/transverse distance
        should be evaluated at.
    z_cmb
        CMB-frame redshift, without the peculiar-velocity
        correction applied to get ``z_hd``. Kept for reference;
        not used by :class:`likelihoods.pantheon.PantheonLikelihood`.
    z_hel
        Heliocentric redshift. This is the redshift that sets the
        observed time/flux dilation of the light curve, so it is
        what the ``(1 + z)`` factor in the luminosity distance
        D_L = (1 + z_hel) * D_M(z_hd) should use -- *not* z_hd or
        z_cmb (Brout et al. 2022, Pantheon+SH0ES; see also the
        official analysis code's ``dl_at_zhel_zhd`` convention).
    """

    z_hd: np.ndarray

    z_cmb: np.ndarray

    z_hel: np.ndarray

    m_b_corr: np.ndarray

    covariance: "CovarianceBase"

    cepheid: Optional[np.ndarray] = None

    reference: str = ""

# -----------------------------------------------------------

    def __post_init__(
    self,
    ):

        n = len(

            self.z_cmb,

        )

        if not (

            len(self.z_hd)

            == n

            == len(self.z_hel)

            == len(self.m_b_corr)

        ):

            raise ValueError(

                "Pantheon arrays have inconsistent lengths.",

            )

        if self.covariance.shape != (

            n,

            n,

        ):

            raise ValueError(

                "Pantheon covariance has wrong shape.",

            )

        if (

            self.cepheid is not None

            and len(self.cepheid) != n

        ):

            raise ValueError(

                "Cepheid mask has wrong length.",

            )

    # -----------------------------------------------------------

    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.z_cmb,

        )

    # -----------------------------------------------------------

    @property
    def has_cepheid(
        self,
    ) -> bool:
        """
        Whether the dataset contains Cepheid calibration flags.
        """

        return self.cepheid is not None


# ============================================================
# DES-SN5YR
# ============================================================

@dataclass(slots=True)
class DESSN5YRDataset:
    """
    DES-SN5YR (Dark Energy Survey 5-year) Supernova sample.

    Unlike Pantheon+ (which distributes the SALT-corrected
    *apparent magnitude* ``m_b_corr``, requiring the stretch/color
    standardization to have already been applied by the caller),
    DES-SN5YR distributes the already-standardized *distance
    modulus* ``mu`` directly (computed assuming a fiducial H0=70 --
    see :class:`~likelihoods.des_sn5yr.DESSN5YRLikelihood`, which
    marginalizes the resulting additive offset exactly as
    :class:`~likelihoods.pantheon.PantheonLikelihood` marginalizes
    ``M_B``).

    z_hd
        Hubble-diagram redshift (CMB-frame, corrected for the
        local peculiar-velocity/bulk-flow model) -- as with
        Pantheon+, the redshift the cosmological distance should
        be evaluated at.
    z_hel
        Heliocentric redshift -- the ``(1 + z)`` light-curve
        dilation factor should use this, not ``z_hd``.
    mu
        Distance modulus (bias- and contamination-corrected).
    mu_err
        Diagonal statistical uncertainty on ``mu``, as tabulated in
        the data release for reference. Not used by
        :class:`~likelihoods.des_sn5yr.DESSN5YRLikelihood` (which
        uses the full covariance matrix instead); provided for
        quick-look diagnostics only, since it does not include the
        systematic contribution folded into ``covariance``.
    """

    z_hd: np.ndarray

    z_hel: np.ndarray

    mu: np.ndarray

    mu_err: np.ndarray

    covariance: "CovarianceBase"

    reference: str = ""

    # -----------------------------------------------------------

    def __post_init__(self):

        n = len(self.z_hd)

        if not (
            len(self.z_hel) == n
            == len(self.mu)
            == len(self.mu_err)
        ):

            raise ValueError(
                "DES-SN5YR arrays have inconsistent lengths.",
            )

        if self.covariance.shape != (n, n):

            raise ValueError(
                "DES-SN5YR covariance has wrong shape.",
            )

    # -----------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of measurements.
        """

        return len(self.z_hd)


# ============================================================
# Planck Distance Prior
# ============================================================

@dataclass(slots=True)
class PlanckDataset:
    """
    Planck distance-prior measurements.
    """

    values: np.ndarray

    covariance: "CovarianceBase"

    labels: tuple[str, ...] = (
        "R",
        "lA",
        "omega_b_h2",
    )

    reference: str = ""


    @property
    def size(
        self,
    ) -> int:
        """
        Number of measurements.
        """

        return len(

            self.values,

        )


# ============================================================
# Growth of structure (fsigma8)
# ============================================================

@dataclass(slots=True)
class GrowthDataset:
    """
    Growth-rate (fsigma8) measurements, e.g. the "Gold-2018" RSD
    compilation (:func:`~data.loader.load_fsigma8`).

    z
        Effective redshift of each measurement.
    fsigma8
        f(z) * sigma8(z) measurement.
    sigma
        Diagonal statistical uncertainty (used to build the
        covariance's diagonal before any correlated blocks --
        e.g. WiggleZ, SDSS -- are written in on top of it; see
        the loader).
    HdAz
        Fiducial-cosmology H(z)*D_A(z) product [km/s] each survey
        assumed when converting its raw RSD measurement into
        fsigma8 -- used by
        :class:`~likelihoods.fsigma8.FSigma8Likelihood` for the
        Alcock-Paczynski correction every precision RSD analysis
        applies (comparing a different test cosmology's H(z)*D_A(z)
        against this fiducial value).
    """

    z: np.ndarray

    fsigma8: np.ndarray

    sigma: np.ndarray

    HdAz: np.ndarray

    covariance: "CovarianceBase | None" = None

    reference: str = ""

    # -----------------------------------------------------------

    def __post_init__(self):

        n = len(self.z)

        if not (
            len(self.fsigma8) == n
            == len(self.sigma)
            == len(self.HdAz)
        ):

            raise ValueError(
                "Growth dataset arrays have inconsistent lengths.",
            )

        if (
            self.covariance is not None
            and self.covariance.shape != (n, n)
        ):

            raise ValueError(
                "Growth dataset covariance has wrong shape.",
            )

    # -----------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of measurements.
        """

        return len(self.z)


# ============================================================
# S8 weak-lensing prior
# ============================================================

@dataclass(slots=True)
class S8Dataset:
    """
    A single Gaussian S8 = sigma8 * sqrt(Omega_m / 0.3) constraint
    from a weak-lensing survey (e.g. KiDS-1000, DES Y3), used by
    :class:`~likelihoods.s8.S8Likelihood`.
    """

    value: float

    sigma: float

    covariance: "CovarianceBase | None" = None

    reference: str = ""

    @property
    def size(self) -> int:
        return 1


# ============================================================
# Union3 binned supernovae
# ============================================================

@dataclass(slots=True)
class Union3Dataset:
    """
    The Union3 supernova compilation in its released *binned* form:
    22 distance moduli on a redshift grid, with a 22x22 magnitude
    covariance.

    Union3 is distributed this way on purpose. The full sample is
    2087 supernovae fit with the UNITY1.5 Bayesian hierarchical
    model, whose light-curve, host-mass and selection nuisance
    parameters are marginalized *inside* that fit; what comes out
    is a binned distance-modulus vector and its covariance, not a
    per-supernova catalogue like
    :class:`PantheonDataset` (apparent magnitudes) or
    :class:`DESSN5YRDataset` (distance moduli). So this container
    holds bins, and 22 of them carry most of the constraining power
    of 2087 objects.

    Like both other supernova samples, the overall distance-modulus
    zero point is degenerate with H0 and is analytically
    marginalized by :class:`~likelihoods.union3.Union3Likelihood`
    rather than fit.

    References
    ----------
    Rubin et al. (2023), "Union Through UNITY: Cosmology with
    2,000 SNe Using a Unified Bayesian Framework",
    arXiv:2311.12098 (ApJ, accepted).
    """

    z_cmb: np.ndarray

    z_hel: np.ndarray

    mu: np.ndarray

    covariance: "CovarianceBase"

    reference: str = ""

    def __post_init__(self):

        n = len(self.z_cmb)

        if not (len(self.z_hel) == n == len(self.mu)):

            raise ValueError(

                "Union3 dataset arrays have inconsistent lengths.",

            )

        if self.covariance.shape != (n, n):

            raise ValueError(

                "Union3 dataset covariance has wrong shape.",

            )

    @property
    def size(self) -> int:
        """
        Number of redshift bins.
        """

        return len(self.z_cmb)

    @property
    def z_hd(self) -> np.ndarray:
        """
        Alias for :attr:`z_cmb`, under the name Pantheon+ and
        DES-SN5YR use for the same thing.

        Those releases call it ``zHD`` -- the "Hubble diagram"
        redshift, CMB-frame and peculiar-velocity-corrected --
        while Union3's file column is ``zcmb``. The field keeps its
        released name so provenance stays readable; this alias
        exists so the shared supernova plotting code
        (:meth:`~plots.FitPlotter._sn_hubble_diagram`) can take all
        three datasets without a per-dataset branch.
        """

        return self.z_cmb


# ============================================================
# Single-number external priors
# ============================================================

@dataclass(slots=True)
class GaussianPriorDataset:
    """
    A single Gaussian constraint on one named quantity -- a local
    distance-ladder ``H0``, a BBN ``omega_b h^2``, a reionization
    ``tau`` -- used by
    :class:`~likelihoods.priors.GaussianPriorLikelihood`.

    These are *external* measurements entering the fit as one data
    point each, not compressions of a larger dataset the way
    :class:`PlanckDataset` is. Keeping them as a dataset rather than
    as a prior on the parameter is deliberate: a prior is a
    statement about belief before seeing data, while
    "SH0ES measured 73.04 +- 1.04" is data, and it should show up in
    the chi2 accounting, the degrees-of-freedom count and the
    dataset list like every other measurement does.

    ``quantity`` names what is constrained; the likelihood maps it
    to a model prediction (see
    :data:`likelihoods.priors.QUANTITY_MAP`).
    """

    quantity: str

    value: float

    sigma: float

    covariance: "CovarianceBase | None" = None

    reference: str = ""

    @property
    def size(self) -> int:
        return 1


# ============================================================
# Binned CMB power spectra
# ============================================================

@dataclass(slots=True)
class CMBSpectrumDataset:
    """
    Binned CMB angular power-spectrum bandpowers and their
    covariance -- the actual measured spectra, as opposed to the
    three-number compression in :class:`PlanckDataset`.

    Holds the Planck 2018 ``plik_lite`` TT/TE/EE bandpowers used by
    :class:`~likelihoods.planck_lite.PlanckLiteLikelihood`, together
    with the binning operator (``blmin``/``blmax``/``weights``) that
    turns a theory C_l array into the same bandpowers, since the
    data vector is meaningless without the window functions that
    defined it.

    Attributes
    ----------
    ell : np.ndarray
        Effective multipole of each bandpower.
    value : np.ndarray
        Bandpower values, C_l in muK^2, ordered TT then TE then EE.
    sigma : np.ndarray
        Per-bandpower uncertainty (the covariance's diagonal; kept
        for plotting error bars, not used in the chi2).
    covariance : CovarianceBase
        Full bandpower covariance across all three spectra.
    n_bin : tuple[int, int, int]
        Number of bandpowers in TT, TE, EE respectively, so the
        concatenated vector can be split back apart.
    blmin, blmax : np.ndarray
        First/last multipole index contributing to each bandpower,
        relative to ``lmin``.
    weights : np.ndarray
        Binning weights, indexed by the flattened window.
    lmin, lmax : int
        Multipole range the binning operator spans.
    """

    ell: np.ndarray

    value: np.ndarray

    sigma: np.ndarray

    covariance: "CovarianceBase"

    n_bin: tuple[int, int, int]

    blmin: np.ndarray

    blmax: np.ndarray

    weights: np.ndarray

    lmin: int = 30

    lmax: int = 2508

    reference: str = ""

    def __post_init__(self):

        n = len(self.value)

        if sum(self.n_bin) != n:

            raise ValueError(

                f"CMB bandpower counts {self.n_bin} sum to "

                f"{sum(self.n_bin)}, but the data vector has "

                f"{n} entries.",

            )

        if self.covariance.shape != (n, n):

            raise ValueError(

                "CMB bandpower covariance has wrong shape.",

            )

    @property
    def size(self) -> int:
        """
        Total number of bandpowers.
        """

        return len(self.value)

    @property
    def slices(self) -> dict[str, slice]:
        """
        Where each spectrum sits in the concatenated data vector.
        """

        n_tt, n_te, n_ee = self.n_bin

        return {

            "TT": slice(0, n_tt),

            "TE": slice(n_tt, n_tt + n_te),

            "EE": slice(n_tt + n_te, n_tt + n_te + n_ee),

        }


# ============================================================
# Generic container (optional)
# ============================================================

@dataclass(slots=True)
class GenericDataset:
    """
    Generic dataset container.
    """

    name: str

    data: dict