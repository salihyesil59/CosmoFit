"""
The CMB: distance priors, plik_lite spectra, low-l EE, and the Planck
and ACT lensing reconstructions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dataset import PlanckDataset
from ..dataset import CMBSpectrumDataset
from ..dataset import CMBLensingDataset
from ..dataset import LowEllEEDataset
from ..covariance import make_covariance

from ._common import _check_file_exists, _get_dataset_path, _load_covariance, _load_txt, _validate_version


PLANCK_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "planck2018",

        "data": "distance_prior.txt",

        "covariance": "distance_prior_cov.txt",

        "reference": "Chen, Huang & Wang (2019), JCAP 02 (2019) 028, arXiv:1808.05724",

    },

}


#: Planck 2018 ``plik_lite`` binned TT/TE/EE bandpowers -- the
#: foreground-marginalized high-l likelihood, i.e. the measured
#: CMB spectra themselves rather than the three-number compression
#: in :data:`PLANCK_FILES`. Used by
#: :class:`~likelihoods.planck_lite.PlanckLiteLikelihood`, which
#: needs a Boltzmann code to predict C_l and so is the one dataset
#: here with an optional dependency (CAMB).
PLIK_LITE_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "plik_lite",

        "data": "cl_cmb_plik_v22.dat",

        # A Fortran unformatted record holding the 613x613
        # bandpower covariance -- see `_load_plik_covariance`.
        "covariance": "c_matrix_plik_v22.dat",

        "blmin": "blmin.dat",

        "blmax": "blmax.dat",

        "weights": "bweight.dat",

        #: TT, TE, EE bandpower counts. TT spans l = 30-2508,
        #: TE and EE l = 30-1996.
        "n_bin": (215, 199, 199),

        "lmin": 30,

        "lmax": 2508,

        #: The two Commander low-multipole temperature bins
        #: (l = 2-29), which `load_plik_lite(use_low_ell=True)`
        #: prepends to the TT block. A separate likelihood from
        #: plik_lite, with its own windows and an uncorrelated
        #: (diagonal) covariance.
        "low_ell": {

            "folder": "low_ell",

            "data": "CTT_bin_low_ell_2018.dat",

            "blmin": "blmin_low_ell.dat",

            "blmax": "blmax_low_ell.dat",

            "weights": "bweight_low_ell.dat",

            "lmin": 2,

            "reference": (
                "Planck Collaboration (2020), A&A 641, A5, "
                "arXiv:1907.12875 (Commander, l = 2-29)"
            ),

        },

        "reference": (
            "Planck Collaboration (2020), A&A 641, A5, arXiv:1907.12875 "
            "(likelihood); data as redistributed by "
            "heatherprince/planck-lite-py from the Planck Legacy Archive"
        ),

    },

}


#: Planck 2018 CMB lensing -- the reconstructed lensing-potential
#: bandpowers. A different measurement from the temperature and
#: polarization spectra, and the CMB's own handle on how much
#: structure grew between recombination and today.
PLANCK_LENSING_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "lensing2018",

        "data": "bandpowers.dat",

        "covariance": "cov.dat",

        "fiducial_correction": "lensing_fiducial_correction.dat",

        "windows": "window/window{bin}.dat",

        "delta_windows": "lens_delta_window/window{bin}.dat",

        "n_bin": 9,

        "lmax": 2500,

        "ell_range": (8, 400),

        "reference": (
            "Planck Collaboration (2020), A&A 641, A8, arXiv:1807.06210 "
            "(lensing); conservative 8 <= L <= 400 baseline, data as "
            "redistributed by CobayaSampler/planck_supp_data_and_covmats"
        ),

    },

}


#: ACT DR6 CMB lensing. A second, independent reconstruction of the
#: lensing potential -- different telescope, different sky,
#: different pipeline -- and a tighter one than Planck's.
#:
#: ``bins`` is the slice of the released bandpower vector each
#: variant adopts, applied identically to the data, the covariance
#: and the binning matrix.
ACT_LENSING_FILES = {

    "act_baseline": {

        "parent": "cmb",

        "folder": "act_dr6_lensing",

        "data": "clkk_bandpowers_act.txt",

        "binning_matrix": "binning_matrix_act.txt",

        # The CMB-marginalized covariance: it already accounts for
        # the reconstruction's dependence on the primary CMB
        # spectra, which is what lets this be used without the
        # (unshippable) explicit normalization correction. See
        # `likelihoods.act_lensing`.
        "covariance": "covmat_act_cmbmarg.txt",

        "bins": (2, -6),

        "n_sims": 796,

        "lmax": 2999,

        "ell_range": (40, 763),

        "reference": (
            "Madhavacheril et al. (ACT Collaboration, 2024), ApJ 962, 113, "
            "arXiv:2304.05203; Qu et al. (ACT Collaboration, 2024), "
            "ApJ 962, 112, arXiv:2304.05202"
        ),

    },

    "act_extended": {

        "parent": "cmb",

        "folder": "act_dr6_lensing",

        "data": "clkk_bandpowers_act.txt",

        "binning_matrix": "binning_matrix_act.txt",

        "covariance": "covmat_act_cmbmarg.txt",

        "bins": (2, -3),

        "n_sims": 796,

        "lmax": 2999,

        "ell_range": (40, 1250),

        "reference": (
            "Madhavacheril et al. (ACT Collaboration, 2024), ApJ 962, 113, "
            "arXiv:2304.05203; Qu et al. (ACT Collaboration, 2024), "
            "ApJ 962, 112, arXiv:2304.05202 (extended multipole range)"
        ),

    },

}


LOWE_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "lowE2018",

        "data": "prob_table.txt",

        "lmin": 2,

        "lmax": 29,

        "step": 1.0e-4,

        "reference": (
            "Planck Collaboration (2020), A&A 641, A5, arXiv:1907.12875 "
            "(SimAll low-l EE); Python-translated table as distributed "
            "by CobayaSampler/planck_native_data"
        ),

    },

}


# ============================================================
# Planck CMB lensing
# ============================================================

def load_planck_lensing(
    version: str = "planck2018",
) -> CMBLensingDataset:
    """
    Load the Planck 2018 CMB lensing bandpowers, their covariance,
    and the two sets of window functions the likelihood needs.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    CMBLensingDataset
    """

    entry = _validate_version("planck_lensing", version)
    dataset_path = _get_dataset_path("planck_lensing", version)

    n_bin = int(entry["n_bin"])
    lmax = int(entry["lmax"])

    # bin, L_min, L_max, L_av, PP, Error, Ahat
    table = _load_txt(dataset_path / entry["data"])

    ell = np.asarray(table[:, 3], dtype=float)
    value = np.asarray(table[:, 4], dtype=float)
    sigma = np.asarray(table[:, 5], dtype=float)

    if len(value) != n_bin:

        raise ValueError(

            f"'{entry['data']}': expected {n_bin} bandpowers, "

            f"found {len(value)}.",

        )

    covariance = _load_covariance(dataset_path / entry["covariance"])

    fiducial = _load_txt(

        dataset_path / entry["fiducial_correction"],

    )[:, 1]

    # Both window sets are stored one file per bin, listing only the
    # multipoles that bin actually touches -- so they are scattered
    # into dense (lmax + 1) rows here rather than read as blocks.
    windows = np.zeros((n_bin, lmax + 1), dtype=float)
    delta_windows = np.zeros((n_bin, 4, lmax + 1), dtype=float)

    for b in range(n_bin):

        path = dataset_path / entry["windows"].format(bin=b + 1)

        _check_file_exists(path)

        rows = np.loadtxt(path, ndmin=2)

        windows[b, rows[:, 0].astype(int)] = rows[:, 1]

        path = dataset_path / entry["delta_windows"].format(bin=b + 1)

        _check_file_exists(path)

        rows = np.loadtxt(path, ndmin=2)

        index = rows[:, 0].astype(int)

        for column in range(4):

            delta_windows[b, column, index] = rows[:, 1 + column]

    return CMBLensingDataset(

        ell=ell,

        value=value,

        sigma=sigma,

        covariance=make_covariance(cov=covariance),

        windows=windows,

        delta_windows=delta_windows,

        fiducial_correction=np.asarray(fiducial, dtype=float),

        lmax=lmax,

        ell_range=tuple(entry["ell_range"]),

        reference=entry["reference"],

    )


# ============================================================
# ACT DR6 CMB lensing
# ============================================================

def load_act_lensing(
    version: str = "act_baseline",
) -> CMBLensingDataset:
    """
    Load the ACT DR6 lensing bandpowers, binning matrix and
    CMB-marginalized covariance.

    Parameters
    ----------
    version : str, optional
        ``"act_baseline"`` or ``"act_extended"``.

    Returns
    -------
    CMBLensingDataset
        With ``spectrum="KK"``: the windows act on the lensing
        *convergence* ``C_L^{kappakappa}``, not on Planck's
        potential convention.
    """

    entry = _validate_version("act_lensing", version)
    dataset_path = _get_dataset_path("act_lensing", version)

    start, end = entry["bins"]
    lmax = int(entry["lmax"])

    bandpowers = np.atleast_1d(

        _load_txt(dataset_path / entry["data"]),

    )

    binning = _load_txt(dataset_path / entry["binning_matrix"])

    full_covariance = _load_covariance(

        dataset_path / entry["covariance"],

    )

    n_total = len(bandpowers)

    if binning.shape[0] != n_total:

        raise ValueError(

            f"'{entry['binning_matrix']}': has {binning.shape[0]} "

            f"rows against {n_total} bandpowers.",

        )

    # The same bins come out of all three, which is the whole point
    # of doing it in one place: dropping them from the data and not
    # from the covariance produces a chi2 that is merely wrong.
    keep = np.arange(n_total)[start:end]

    value = bandpowers[keep]

    covariance = full_covariance[np.ix_(keep, keep)]

    # `standardize` in ACT's own loader pads/trims the binning
    # matrix to l = 0..lmax; the released matrix is at least that
    # wide, so this is the trim half.
    windows = np.zeros((len(keep), lmax + 1), dtype=float)

    width = min(binning.shape[1], lmax + 1)

    windows[:, :width] = binning[np.ix_(keep, np.arange(width))]

    # Effective multipole of each bin, from the binning matrix
    # itself rather than assumed.
    ell = windows @ np.arange(lmax + 1)

    # Hartlap: the covariance is estimated from a finite number of
    # simulations, so its *inverse* is biased high by
    # (n_sim - 1) / (n_sim - n_bin - 2). ACT's own code multiplies
    # the inverse by the reciprocal; dividing the covariance here is
    # algebraically the same and keeps the library's covariance
    # machinery in charge of the inversion.
    n_bin = len(keep)
    n_sims = int(entry["n_sims"])

    hartlap = (n_sims - n_bin - 2.0) / (n_sims - 1.0)

    if hartlap <= 0.0:

        raise ValueError(

            f"Hartlap factor is non-positive for {n_bin} bins from "

            f"{n_sims} simulations; the covariance cannot be "

            f"inverted meaningfully.",

        )

    covariance = covariance / hartlap

    return CMBLensingDataset(

        ell=ell,

        value=value,

        sigma=np.sqrt(np.diag(covariance)),

        covariance=make_covariance(cov=covariance),

        windows=windows,

        lmax=lmax,

        ell_range=tuple(entry["ell_range"]),

        spectrum="KK",

        reference=entry["reference"],

    )


# ============================================================
# Planck low-multipole EE
# ============================================================

def load_planck_lowe(
    version: str = "planck2018",
) -> LowEllEEDataset:
    """
    Load Planck's low-multipole EE probability table.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    LowEllEEDataset
    """

    entry = _validate_version("planck_lowe", version)
    dataset_path = _get_dataset_path("planck_lowe", version)

    table = _load_txt(dataset_path / entry["data"])

    return LowEllEEDataset(

        table=np.asarray(table, dtype=float),

        lmin=int(entry["lmin"]),

        lmax=int(entry["lmax"]),

        step=float(entry["step"]),

        reference=entry["reference"],

    )


# ============================================================
# Planck plik_lite binned TT/TE/EE bandpowers
# ============================================================

def _load_plik_covariance(
    path: Path,
    n: int,
) -> np.ndarray:
    """
    Read the ``plik_lite`` bandpower covariance.

    The released file is a single Fortran unformatted record: a
    4-byte length marker, ``n*n`` little-endian float64s, and a
    closing marker. Only the lower triangle is filled in, so the
    upper triangle is mirrored onto it here.

    ``scipy.io.FortranFile`` would read this in one line, but that
    would make SciPy's I/O module a hard import for a dataset most
    fits never touch; the record layout is fixed and three lines of
    NumPy, so it is read directly.
    """

    _check_file_exists(path)

    expected = n * n

    raw = path.read_bytes()

    # 4-byte opening marker + n*n float64s + 4-byte closing marker.
    if len(raw) != 8 + 8 * expected:

        raise ValueError(

            f"'{path.name}': expected a Fortran record holding "

            f"{n}x{n} float64s ({8 + 8 * expected} bytes with its "

            f"markers), but the file is {len(raw)} bytes.",

        )

    # The payload starts 4 bytes in, i.e. half a float64 -- so the
    # offset has to be applied to the byte buffer, not to a float
    # view of it.
    cov = np.frombuffer(

        raw,

        dtype=np.float64,

        count=expected,

        offset=4,

    ).reshape(n, n).copy()

    # Released with only one triangle populated.
    cov = np.tril(cov) + np.tril(cov, -1).T

    return cov


# ------------------------------------------------------------

def _prepend_low_ell_bins(
    folder: Path,
    spec: dict,
    *,
    ell,
    value,
    sigma,
    covariance,
    n_bin,
    blmin,
    blmax,
    weights,
):
    """
    Prepend the two Commander low-multipole temperature bandpowers
    to a ``plik_lite`` data vector.

    ``plik_lite`` starts at l = 30. Planck's own low-l temperature
    likelihood (Commander, l = 2-29) is a separate product, and
    ``planck-lite-py`` distributes a two-bin Gaussian compression of
    it that can be bolted onto the front.

    Three things have to move together, and getting any one of them
    wrong produces a chi2 that is merely wrong:

    - **The data vector.** The two bins go at the *front*, so the
      TT block becomes 217 long while TE and EE are untouched.
    - **The covariance.** The low-l bins are uncorrelated with the
      high-l block and with each other, so the result is
      block-diagonal with ``diag(sigma^2)`` in the top-left corner.
    - **The windows.** The low-l bins have their own, indexed from
      l = 2, and prepending them shifts every high-l TT window
      index by the length of the low-l weight array. TE and EE keep
      the original indexing, which is why the dataset carries a
      separate TT window set (see
      :class:`~data.dataset.CMBSpectrumDataset`).

    Returns the updated ``(ell, value, sigma, covariance, n_bin,
    extra)``, where ``extra`` holds the TT-specific window fields.
    """

    low_ell_table = np.loadtxt(folder / spec["data"])

    ell_low, value_low, sigma_low = (

        np.atleast_2d(low_ell_table).T

    )

    n_low = len(value_low)

    blmin_low = np.loadtxt(folder / spec["blmin"]).astype(int)
    blmax_low = np.loadtxt(folder / spec["blmax"]).astype(int)
    weights_low = np.loadtxt(folder / spec["weights"])

    n_tt, n_te, n_ee = n_bin

    # Data vector: low-l TT in front of high-l TT, then TE and EE.
    def _insert(low, high):

        return np.concatenate([

            low,

            high[:n_tt],

            high[n_tt:],

        ])

    ell = _insert(ell_low, ell)
    value = _insert(value_low, value)
    sigma = _insert(sigma_low, sigma)

    n_total = len(value)

    combined = np.zeros((n_total, n_total), dtype=float)

    combined[:n_low, :n_low] = np.diag(sigma_low ** 2)
    combined[n_low:, n_low:] = covariance

    extra = {

        # The high-l windows are indexed from `lmin`; after
        # prepending `len(weights_low)` low-l weights they start
        # that much further into the flat weight array.
        "blmin_tt": np.concatenate([blmin_low, blmin + len(weights_low)]),

        "blmax_tt": np.concatenate([blmax_low, blmax + len(weights_low)]),

        "weights_tt": np.concatenate([weights_low, weights]),

        "lmin_tt": int(spec["lmin"]),

    }

    return (

        ell,

        value,

        sigma,

        combined,

        (n_tt + n_low, n_te, n_ee),

        extra,

    )


# ------------------------------------------------------------

def load_plik_lite(
    version: str = "planck2018",
    use_low_ell: bool = False,
) -> CMBSpectrumDataset:
    """
    Load the Planck 2018 ``plik_lite`` binned TT/TE/EE bandpowers,
    their joint covariance, and the binning operator that produced
    them.

    This is the *measured CMB power spectrum*, not the
    three-number distance-prior compression that
    :func:`load_planck` returns -- see
    :class:`~likelihoods.planck_lite.PlanckLiteLikelihood` for
    what that buys and what it costs.

    ``plik_lite`` is the foreground-marginalized variant of the
    Planck high-l likelihood: the ~20 nuisance parameters
    describing dust, point sources and the SZ effect have already
    been marginalized over by the Planck team, leaving a single
    calibration parameter (``A_planck``). That is what makes it
    usable outside a full Planck pipeline.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    CMBSpectrumDataset
    """

    entry = _validate_version("planck_lite", version)
    dataset_path = _get_dataset_path("planck_lite", version)

    ell, value, sigma = np.loadtxt(

        dataset_path / entry["data"],

        unpack=True,

    )

    n = len(value)

    if sum(entry["n_bin"]) != n:

        raise ValueError(

            f"'{entry['data']}': registry declares "

            f"{entry['n_bin']} bandpowers ({sum(entry['n_bin'])} "

            f"total) but the file holds {n}.",

        )

    covariance = _load_plik_covariance(

        dataset_path / entry["covariance"],

        n=n,

    )

    blmin = np.loadtxt(dataset_path / entry["blmin"]).astype(int)
    blmax = np.loadtxt(dataset_path / entry["blmax"]).astype(int)
    weights = np.loadtxt(dataset_path / entry["weights"])

    n_bin = tuple(entry["n_bin"])

    extra = {}

    if use_low_ell:

        (
            ell, value, sigma, covariance, n_bin, extra,
        ) = _prepend_low_ell_bins(

            dataset_path / entry["low_ell"]["folder"],

            entry["low_ell"],

            ell=ell,

            value=value,

            sigma=sigma,

            covariance=covariance,

            n_bin=n_bin,

            blmin=blmin,

            blmax=blmax,

            weights=weights,

        )

    return CMBSpectrumDataset(

        ell=ell,

        value=value,

        sigma=sigma,

        covariance=make_covariance(cov=covariance),

        n_bin=n_bin,

        blmin=blmin,

        blmax=blmax,

        weights=weights,

        lmin=int(entry["lmin"]),

        lmax=int(entry["lmax"]),

        reference=entry["reference"],

        **extra,

    )


# ============================================================
# Planck CMB distance priors
# ============================================================

def load_planck(
    version: str = "planck2018",
) -> PlanckDataset:
    """
    Load a Planck CMB distance-prior dataset.

    The data vector is (R, l_A, omega_b_h2) -- the CMB shift
    parameter, acoustic scale, and physical baryon density -- as
    described in :mod:`likelihoods.planck`.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    PlanckDataset
    """

    entry = _validate_version(

        "planck",

        version,

    )

    dataset_path = _get_dataset_path(

        "planck",

        version,

    )

    table = _load_txt(

        dataset_path / entry["data"],

        dtype=None,

    )

    covariance = _load_covariance(

        dataset_path / entry["covariance"],

    )

    labels = tuple(str(label) for label in table["f0"])

    values = np.asarray(table["f1"], dtype=float)

    if covariance.shape != (len(values), len(values)):

        raise ValueError(

            f"Expected covariance shape ({len(values)}, {len(values)}), "
            f"but found {covariance.shape}.",

        )

    return PlanckDataset(

        values=values,

        covariance=make_covariance(

            cov=covariance,

        ),

        labels=labels,

        reference=entry["reference"],

    )
