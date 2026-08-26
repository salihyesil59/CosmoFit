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

    #: TT-specific binning, used only when the low-multipole
    #: temperature bins are included. ``None`` means TT shares the
    #: windows above with TE and EE, which is the case for
    #: ``plik_lite`` on its own.
    #:
    #: They have to be separable because the two low-l bins come
    #: from a *different* likelihood (Commander, l = 2-29) with its
    #: own windows, and prepending them shifts every high-l TT
    #: window index -- while TE and EE, which have no low-l
    #: counterpart here, must keep the original indexing. One
    #: shared window set cannot describe both.
    blmin_tt: "np.ndarray | None" = None

    blmax_tt: "np.ndarray | None" = None

    weights_tt: "np.ndarray | None" = None

    #: First multipole the TT windows are indexed from. 2 with the
    #: low-l bins, 30 without.
    lmin_tt: int = 30

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
    def tt_windows(self) -> tuple:
        """
        ``(blmin, blmax, weights, lmin)`` for the TT spectrum --
        the TT-specific set where one exists, otherwise the shared
        one.
        """

        if self.blmin_tt is None:

            return self.blmin, self.blmax, self.weights, self.lmin

        return self.blmin_tt, self.blmax_tt, self.weights_tt, self.lmin_tt

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
# CMB lensing
# ============================================================

@dataclass(slots=True)
class CMBLensingDataset:
    """
    Planck's CMB lensing bandpowers and everything needed to predict
    them -- see
    :class:`~likelihoods.planck_lensing.PlanckLensingLikelihood`.

    The lensing likelihood is a binned Gaussian like the bandpower
    one, with one addition that is easy to overlook and wrong to
    drop. The lensing reconstruction is *normalized* using an
    assumed set of CMB spectra; change the cosmology and that
    normalization changes too. The ``delta_windows`` propagate that
    dependence to first order, and ``fiducial_correction`` is the
    same quantity evaluated at the fiducial cosmology, subtracted so
    the correction vanishes there.

    Attributes
    ----------
    ell : np.ndarray
        Effective multipole of each bandpower.
    value : np.ndarray
        Bandpowers of ``[L(L+1)]^2 C_L^{phiphi} / 2 pi``.
    sigma : np.ndarray
        Per-bandpower error (the covariance's diagonal; for plots).
    covariance : CovarianceBase
        The bandpower covariance.
    windows : np.ndarray
        ``(n_bin, lmax + 1)``: ``W_bL``, mapping the theory lensing
        spectrum to bandpower ``b``.
    delta_windows : np.ndarray
        ``(n_bin, 4, lmax + 1)``: the linear-correction windows, in
        the column order ``TT, EE, TE, PP`` the released files use.
    fiducial_correction : np.ndarray
        The linear correction at the fiducial cosmology.
    lmax : int
        Highest multipole the windows span.
    ell_range : tuple[int, int]
        The reconstruction's multipole range, for labelling.
    """

    ell: np.ndarray

    value: np.ndarray

    sigma: np.ndarray

    covariance: "CovarianceBase"

    windows: np.ndarray

    lmax: int = 2500

    ell_range: tuple = (8, 400)

    reference: str = ""

    #: Which lensing spectrum ``windows`` multiplies. Planck's
    #: released products are built on the *potential*,
    #: ``[L(L+1)]^2 C_L^{phiphi} / 2 pi``; ACT's are built on the
    #: *convergence*, ``C_L^{kappakappa}`` as a raw C_L. The two
    #: differ by a factor of ``2 pi / 4`` and describe the same
    #: physics, so the distinction is easy to lose and impossible
    #: to notice afterwards -- a wrong choice is a smooth rescaling
    #: of the prediction, which a fit absorbs into the amplitude.
    spectrum: str = "PP"

    #: The linear-correction machinery, present only where the
    #: release provides it (Planck). ``None`` means the covariance
    #: already accounts for the reconstruction's dependence on the
    #: fiducial CMB -- which is what ACT's CMB-marginalized
    #: covariance does instead.
    delta_windows: "np.ndarray | None" = None

    fiducial_correction: "np.ndarray | None" = None

    #: Column order of ``delta_windows``' second axis, matching the
    #: released ``linear_correction_bin_window_in_order``.
    CORRECTION_SPECTRA = ("TT", "EE", "TE", "PP")

    def __post_init__(self):

        n = len(self.value)

        if not (len(self.ell) == n == len(self.sigma)):

            raise ValueError(

                "CMB lensing dataset arrays have inconsistent "

                "lengths.",

            )

        if self.spectrum not in ("PP", "KK"):

            raise ValueError(

                f"Unknown lensing spectrum '{self.spectrum}'; "

                f"expected 'PP' (potential) or 'KK' (convergence).",

            )

        has_correction = (

            self.delta_windows is not None

            or self.fiducial_correction is not None

        )

        if has_correction:

            if self.delta_windows is None or self.fiducial_correction is None:

                raise ValueError(

                    "The linear correction needs both its windows "

                    "and its fiducial values; one was given without "

                    "the other.",

                )

            if len(self.fiducial_correction) != n:

                raise ValueError(

                    "The fiducial correction has the wrong length.",

                )

        if self.covariance.shape != (n, n):

            raise ValueError(

                "CMB lensing covariance has wrong shape.",

            )

        if self.windows.shape != (n, self.lmax + 1):

            raise ValueError(

                f"Expected windows of shape "

                f"({n}, {self.lmax + 1}), got {self.windows.shape}.",

            )

        if (

            self.delta_windows is not None

            and self.delta_windows.shape != (n, 4, self.lmax + 1)

        ):

            raise ValueError(

                f"Expected delta windows of shape "

                f"({n}, 4, {self.lmax + 1}), got "

                f"{self.delta_windows.shape}.",

            )

    @property
    def has_linear_correction(self) -> bool:
        """
        Whether this release provides the normalization correction.
        """

        return self.delta_windows is not None

    @property
    def size(self) -> int:
        """
        Number of bandpowers.
        """

        return len(self.value)


# ============================================================
# Low-multipole CMB polarization
# ============================================================

@dataclass(slots=True)
class LowEllEEDataset:
    """
    Planck's low-multipole EE likelihood as a *tabulated*
    probability, used by
    :class:`~likelihoods.planck_lowe.PlanckLowEELikelihood`.

    Not a data vector with a covariance -- a lookup table. ``table``
    is ``(n_step, n_ell)`` of log-probabilities: column ``j`` is
    multipole ``lmin + j``, row ``i`` is
    ``D_l^EE = i * step`` in muK^2.

    The shape of the thing is the point. Below ``l = 30`` there are
    only ``2l + 1`` modes on the sky, so the C_l distribution is
    strongly non-Gaussian -- and that regime carries essentially all
    of the CMB's information about the reionization optical depth.
    A mean and an error bar cannot represent it, which is why Planck
    ships a table and why CosmoFit's Gaussian ``"tau"`` prior is a
    documented approximation to this rather than an equivalent of
    it.
    """

    table: np.ndarray

    lmin: int = 2

    lmax: int = 29

    step: float = 1.0e-4

    reference: str = ""

    def __post_init__(self):

        n_ell = self.lmax - self.lmin + 1

        if self.table.ndim != 2 or self.table.shape[1] != n_ell:

            raise ValueError(

                f"Expected a probability table with {n_ell} columns "

                f"(l = {self.lmin}..{self.lmax}), got "

                f"{self.table.shape}.",

            )

        if np.any(self.table > 0.0):

            raise ValueError(

                "The table holds log-probabilities and must be "

                "non-positive throughout; positive entries mean it "

                "was read as something else.",

            )

    @property
    def size(self) -> int:
        """
        Number of multipoles constrained -- the data points this
        contributes to a fit's degrees-of-freedom count.
        """

        return self.lmax - self.lmin + 1

    @property
    def n_step(self) -> int:
        return self.table.shape[0]

    @property
    def max_value(self) -> float:
        """
        Largest ``D_l^EE`` the table covers [muK^2].
        """

        return (self.n_step - 1) * self.step


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

# ============================================================
# Tabulated BAO likelihoods
# ============================================================

@dataclass(slots=True)
class TabulatedBAODataset:
    """
    A BAO measurement distributed as a likelihood *surface* rather
    than a mean and a covariance, used by
    :class:`~likelihoods.eboss_dr16.TabulatedBAOLikelihood`.

    eBOSS DR16 releases two of its tracers this way, and in both
    cases the reason is that a Gaussian would misrepresent them:

    * **ELG** (``observable = ("DV_over_rs",)``) has only a 1.4-sigma
      BAO detection, so the likelihood is asymmetric and does not
      decay before the low edge of the released table.
    * **Lyman-alpha** (``observable = ("DM_over_rs", "DH_over_rs")``)
      is a genuinely two-dimensional surface whose degeneracy is not
      an ellipse.

    ``axes`` holds one grid vector per observable, and ``log_prob``
    is the log-likelihood on the outer product of them -- shape
    ``(len(axes[0]),)`` in 1D and ``(len(axes[0]), len(axes[1]))`` in
    2D. It is stored as a log because the released probabilities
    span thirty orders of magnitude, and because interpolating the
    log is what reproduces the published error bars (interpolating
    the probability itself does not, and can go negative between
    nodes).

    The grid edges are a hard bound: outside them the likelihood is
    zero, not extrapolated. For the Lyman-alpha grids that is
    harmless -- the surface has long decayed. For ELG it is a real
    prior, and :attr:`edge_delta_chi2` records how real.
    """

    z_eff: float

    observable: tuple[str, ...]

    axes: tuple[np.ndarray, ...]

    log_prob: np.ndarray

    reference: str = ""

    def __post_init__(self):

        if len(self.axes) != len(self.observable):

            raise ValueError(

                f"{len(self.observable)} observables but "
                f"{len(self.axes)} grid axes -- they index the same "
                f"dimensions and must match.",

            )

        expected = tuple(len(a) for a in self.axes)

        if self.log_prob.shape != expected:

            raise ValueError(

                f"Grid shape {self.log_prob.shape} does not match "
                f"the axes {expected}.",

            )

        for name, axis in zip(self.observable, self.axes):

            if np.any(np.diff(axis) <= 0.0):

                raise ValueError(

                    f"The {name} axis must be strictly increasing.",

                )

        if not np.all(np.isfinite(self.log_prob)):

            raise ValueError(

                "The grid holds log-probabilities and must be finite "
                "throughout -- an exact zero in the released "
                "probability would become -inf here and make the "
                "spline meaningless over its whole support.",

            )

    # ---------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of observables constrained -- 1 for ELG, 2 for
        Lyman-alpha. *Not* the number of grid points, which is a
        property of how finely the surface was sampled and not of
        how much was measured.
        """

        return len(self.observable)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """
        ``(low, high)`` per observable: the support of the table,
        outside which the likelihood is zero.
        """

        return tuple((float(a[0]), float(a[-1])) for a in self.axes)

    @property
    def peak(self) -> tuple[float, ...]:
        """
        Grid point of maximum likelihood, one value per observable.
        """

        index = np.unravel_index(

            np.argmax(self.log_prob),

            self.log_prob.shape,

        )

        return tuple(

            float(axis[i]) for axis, i in zip(self.axes, index)

        )

    @property
    def edge_delta_chi2(self) -> tuple[tuple[float, float], ...]:
        """
        ``chi2 - chi2_min`` at each end of each axis.

        A large number means the surface has decayed and truncating
        it costs nothing. A small one means the table's edge is
        acting as a prior -- eBOSS's ELG table reaches its low edge
        at ``delta chi2 = 3.3``, still well inside 2 sigma.
        """

        peak = np.max(self.log_prob)

        out = []

        for axis_index in range(len(self.axes)):

            moved = np.moveaxis(self.log_prob, axis_index, 0)

            flat = moved.reshape(moved.shape[0], -1)

            # Profile, not marginalize: the best the surface can do
            # at each end, which is what "is the edge reachable?"
            # asks.
            best = flat.max(axis=1)

            out.append(
                (
                    float(-2.0 * (best[0] - peak)),
                    float(-2.0 * (best[-1] - peak)),
                ),
            )

        return tuple(out)
