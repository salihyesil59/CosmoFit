"""
Dataset loading utilities.

This module provides a single interface for loading all
observational datasets used throughout the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.linalg import block_diag

from .dataset import CCDataset
from .dataset import DESIDataset
from .dataset import PantheonDataset
from .dataset import DESSN5YRDataset
from .dataset import PlanckDataset
from .dataset import GrowthDataset
from .dataset import S8Dataset

from .covariance import make_covariance


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).parent


# ============================================================
# Dataset registry
# ============================================================

CC_FILES = {

    "favale2023": {

        "folder": "favale2023",

        "data": "CC_32_Favale2023_data.txt",

        "correlation": "CC_32_Favale2023_Moresco2020_correlation.txt",

        "reference": "Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406, arXiv:2301.09591",

    },

}


DESI_FILES = {

    "desi2024": {

        "parent": "bao",

        "folder": "desi2024",

        "data": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",

        "covariance": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",

        "reference": "DESI Collaboration (2024), JCAP 02 (2025) 021, arXiv:2404.03002",

    },

}


SDSS_BAO_FILES = {

    "dr12+dr16": {

        "parent": "bao",

        "folder": "sdss",

        # Three independent (non-overlapping-redshift) BAO-only
        # measurements, each with its own (z, value, observable)
        # data file and covariance -- combined into one dataset
        # with a block-diagonal covariance (no cross-survey
        # correlations) by `load_sdss_bao()`. BOSS DR12's usual
        # third bin (z_eff=0.61) is deliberately omitted: its
        # redshift range overlaps the eBOSS DR16 LRG sample
        # (0.6 < z < 1.0), so including both would double-count
        # galaxies -- the same reasoning as the Pantheon+/
        # DES-SN5YR overlap noted in likelihoods/des_sn5yr.py.
        "components": (

            {
                "data": "sdss_DR12_LRG_BAO_DMDH.dat",
                "covariance": "sdss_DR12_LRG_BAO_DMDH_covtot.txt",
            },

            {
                "data": "sdss_DR16_LRG_BAO_DMDH.dat",
                "covariance": "sdss_DR16_LRG_BAO_DMDH_covtot.txt",
            },

            {
                "data": "sdss_DR16_QSO_BAO_DMDH.txt",
                "covariance": "sdss_DR16_QSO_BAO_DMDH_covtot.txt",
            },

        ),

        "reference": (
            "Alam et al. (2017), arXiv:1607.03155 (BOSS DR12, z=0.38/0.51); "
            "eBOSS Collaboration / Alam et al. (2021), arXiv:2007.08991 "
            "(eBOSS DR16 LRG z=0.698, QSO z=1.48); "
            "data as distributed by CobayaSampler/bao_data "
            "(originally with CosmoMC)"
        ),

    },

}


PANTHEON_FILES = {

    "pantheon+sh0es": {

        "parent": "sn",

        "folder": "pantheon-plus-sh0es",

        "data": "Pantheon+SH0ES.dat",

        "covariance": "Pantheon+SH0ES_STAT+SYS.cov",

        "reference": "Brout et al. (2022), ApJ 938, 110, arXiv:2202.04077",

    },

}


DES_SN5YR_FILES = {

    "des-sn5yr": {

        "parent": "sn",

        "folder": "des-sn5yr",

        "data": "DES-SN5YR_HD.csv",

        "covariance": "DES-SN5YR_STAT+SYS.npz",

        "reference": (
            "Sanchez et al. (2024), arXiv:2406.05046; "
            "DES Collaboration (2024), arXiv:2401.02929"
        ),

    },

}


GROWTH_FILES = {

    "gold2018": {

        "parent": "growth",

        "folder": "gold2018",

        "data": "fsigma8_gold2018.txt",

        # Row ranges (0-indexed, half-open) in `data` that are
        # internally correlated -- overwritten as dense blocks on
        # top of the diagonal(sigma^2) covariance by
        # `load_fsigma8()`. Matches the reference MontePython
        # likelihood (snesseris/RSD-growth) exactly: WiggleZ's
        # three z<1 points (Blake et al. 2012) and eBOSS DR14
        # quasars' four tomographic bins (Zhao et al. 2018).
        "blocks": (
            {"rows": (12, 15), "covariance": "Cij_WiggleZ.txt"},
            {"rows": (18, 22), "covariance": "Cij_SDSS.txt"},
        ),

        "reference": "Sagredo, Nesseris & Sapone (2018), Phys. Rev. D 98, 083543, arXiv:1806.10822",

    },

}


S8_FILES = {

    "kids1000": {

        "parent": "s8",

        "folder": "kids1000",

        "data": "s8_kids1000.txt",

        "reference": "Asgari et al. (2021), A&A 645, A104, arXiv:2007.15633",

    },

    "des_y3": {

        "parent": "s8",

        "folder": "des_y3",

        "data": "s8_des_y3.txt",

        "reference": "DES Collaboration / Abbott et al. (2022), Phys. Rev. D 105, 023520, arXiv:2105.13549",

    },

}


PLANCK_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "planck2018",

        "data": "distance_prior.txt",

        "covariance": "distance_prior_cov.txt",

        "reference": "Chen, Huang & Wang (2019), JCAP 02 (2019) 028, arXiv:1808.05724",

    },

}


# ============================================================
# Registry lookup
# ============================================================

_REGISTRIES = {

    "cc": CC_FILES,

    "desi": DESI_FILES,

    "sdss_bao": SDSS_BAO_FILES,

    "pantheon": PANTHEON_FILES,

    "des_sn5yr": DES_SN5YR_FILES,

    "fsigma8": GROWTH_FILES,

    "s8": S8_FILES,

    "planck": PLANCK_FILES,

}


# ============================================================
# Helper functions
# ============================================================

def _validate_version(
    dataset: str,
    version: str,
) -> dict:
    """
    Validate a dataset version and return its registry entry.
    """

    if dataset not in _REGISTRIES:

        raise ValueError(

            f"Unknown dataset '{dataset}'."

        )

    registry = _REGISTRIES[dataset]

    if version not in registry:

        available = ", ".join(

            registry.keys()

        )

        raise ValueError(

            f"Unknown {dataset} version "

            f"'{version}'. "

            f"Available versions: {available}"

        )

    return registry[version]


# ------------------------------------------------------------

def _get_dataset_path(
    dataset: str,
    version: str,
) -> Path:
    """
    Return the directory corresponding to a dataset version.
    """

    entry = _validate_version(

        dataset,

        version,

    )

    parent = entry.get(

        "parent",

        dataset,

    )

    return (

        DATA_DIR

        / parent

        / entry["folder"]

    )


# ------------------------------------------------------------

def _check_file_exists(
    path: Path,
) -> None:
    """
    Check that a file exists.
    """

    if not path.exists():

        raise FileNotFoundError(

            f"Dataset file not found:\n{path}"

        )


# ------------------------------------------------------------

def _load_txt(
    path: Path,
    *,
    names: bool = False,
    dtype=float,
    **kwargs,
):
    """
    Load an ASCII text table.

    Parameters
    ----------
    path
        Path to the file.

    names
        If True, read the first line as column names and return a
        structured NumPy array.

    dtype
        Data type passed to NumPy.
    """

    _check_file_exists(

        path,

    )

    if names:

        return np.genfromtxt(

            path,

            names=True,

            dtype=dtype,

            encoding="utf-8",

            **kwargs,

        )

    return np.genfromtxt(

        path,

        dtype=dtype,

        encoding="utf-8",

        comments="#",

        **kwargs,

    )

# ------------------------------------------------------------

def _load_covariance(
    path: Path,
):
    """
    Load a covariance matrix.
    """

    _check_file_exists(

        path,

    )

    return np.loadtxt(

        path,

    )

# ------------------------------------------------------------

def _load_pantheon_covariance(
    path: Path,
) -> np.ndarray:
    """
    Load a Pantheon covariance matrix.

    The first entry gives the matrix dimension.
    The remaining values are stored sequentially.
    """

    _check_file_exists(

        path,

    )

    data = np.loadtxt(

        path,

    )

    n = int(

        data[0],

    )

    expected = n * n

    if len(

        data,

    ) != expected + 1:

        raise ValueError(

            f"Expected {expected} covariance values "

            f"but found {len(data) - 1}.",

        )

    covariance = data[1:].reshape(

        n,

        n,

    )

    return covariance

# ------------------------------------------------------------

def _build_pantheon_mask(
    is_calibrator: np.ndarray,
    include_cepheid: bool,
) -> np.ndarray:
    """
    Build the Pantheon sample mask.
    """

    if include_cepheid:

        return np.ones(

            is_calibrator.size,

            dtype=bool,

        )

    return (

        is_calibrator

        == 0

    )

# ------------------------------------------------------------

def _load_snana_fitres(path: Path) -> dict[str, np.ndarray]:
    """
    Load a SNANA "FITRES"-style table, the format DES-SN5YR
    distributes its Hubble-diagram table in: ``#``-prefixed
    comment lines, a ``VARNAMES:`` line giving the column names,
    and one ``SN:``-prefixed data row per supernova (the ``SN:``
    token itself is not a data column).

    Returns
    -------
    dict[str, ndarray]
        Column name -> array of string values (each column is
        cast to the appropriate dtype by the caller, since not
        every column here is numeric, e.g. ``CID``).
    """

    _check_file_exists(path)

    columns: list[str] | None = None
    rows: list[list[str]] = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("VARNAMES:"):
                columns = line.split()[1:]
                continue

            if line.startswith("SN:"):

                if columns is None:
                    raise ValueError(
                        f"'{path}': found a data row before "
                        "'VARNAMES:'."
                    )

                rows.append(line.split()[1:])

    if columns is None or not rows:
        raise ValueError(
            f"'{path}': no VARNAMES/SN: rows found -- "
            "not a SNANA FITRES-format file?"
        )

    for i, row in enumerate(rows):
        if len(row) != len(columns):
            raise ValueError(
                f"'{path}': row {i} has {len(row)} values, "
                f"expected {len(columns)} (matching VARNAMES)."
            )

    return {
        name: np.array(values)
        for name, values in zip(columns, zip(*rows))
    }

# ------------------------------------------------------------

def _load_des_precision_covariance(path: Path) -> np.ndarray:
    """
    Load a DES-SN5YR precision (inverse covariance) matrix.

    Stored as an ``.npz`` archive with ``nsn`` (matrix dimension)
    and ``cov`` (the upper triangular part, including the
    diagonal, flattened in ``numpy.triu_indices`` order) --
    reconstructed here into the full symmetric matrix. Despite the
    array's name, this is the *precision* matrix, not the
    covariance itself (see :class:`~likelihoods.covariance.PrecisionCovariance`,
    and the DES-SN5YR data release's own README, which flags this
    explicitly).
    """

    _check_file_exists(path)

    archive = np.load(path)

    n = int(archive["nsn"][0])

    expected = n * (n + 1) // 2

    flat = archive["cov"]

    if len(flat) != expected:

        raise ValueError(
            f"'{path}': expected {expected} upper-triangular "
            f"precision-matrix values for n={n}, "
            f"but found {len(flat)}.",
        )

    precision = np.zeros((n, n), dtype=float)

    precision[np.triu_indices(n)] = flat

    lower = np.tril_indices(n, -1)

    precision[lower] = precision.T[lower]

    return precision

# ------------------------------------------------------------

def available_versions(
    dataset: str,
) -> list[str]:
    """
    Return all available versions of a dataset.
    """

    if dataset not in _REGISTRIES:

        raise ValueError(

            f"Unknown dataset '{dataset}'."

        )

    return list(

        _REGISTRIES[dataset].keys()

    )

# ------------------------------------------------------------

def available_datasets():

    return {

        key: list(

            value.keys(),

        )

        for key, value

        in _REGISTRIES.items()

    }


# ============================================================
# Cosmic Chronometers
# ============================================================

def load_cc(
    version: str = "favale2023",
) -> CCDataset:
    """
    Load a Cosmic Chronometer dataset.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    CCDataset
    """

    entry = _validate_version(

        "cc",

        version,

    )

    dataset_path = _get_dataset_path(

        "cc",

        version,

    )

    data = _load_txt(

        dataset_path / entry["data"],

    )

    z = data[:, 0]

    H = data[:, 1]

    sigma = data[:, 2]

    covariance = None

    if "correlation" in entry:

        corr_path = (

            dataset_path

            / entry["correlation"]

        )

        if corr_path.exists():

            correlation = _load_txt(

                corr_path,

            )

            covariance = make_covariance(

                cov=correlation * np.outer(

                    sigma,

                    sigma,

                ),

            )

    return CCDataset(

        z=z,

        H=H,

        sigma=sigma,

        covariance=covariance,

        reference=entry["reference"],

    )


# ============================================================
# BAO (DESI)
# ============================================================

def load_desi(
    version: str = "desi2024",
) -> DESIDataset:
    """
    Load a DESI BAO dataset.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    DESIDataset
    """

    entry = _validate_version(

        "desi",

        version,

    )

    dataset_path = _get_dataset_path(

        "desi",

        version,

    )

    data = _load_txt(

    dataset_path / entry["data"],

    dtype=None,

    )

    covariance = _load_covariance(

        dataset_path / entry["covariance"],

    )

    return DESIDataset(

        z = np.asarray(data["f0"], dtype=float),

        value = np.asarray(data["f1"], dtype=float),

        observable = np.asarray(data["f2"], dtype=str),

        covariance=make_covariance(

            cov=covariance,

        ),

        reference=entry["reference"],

    )

# ============================================================
# BAO (SDSS: BOSS DR12 + eBOSS DR16)
# ============================================================

def load_sdss_bao(
    version: str = "dr12+dr16",
) -> DESIDataset:
    """
    Load the combined SDSS BAO dataset (BOSS DR12 + eBOSS DR16
    LRG + eBOSS DR16 QSO), in the same (z, value, observable-type)
    format DESI uses (see :data:`likelihoods.desi.MODEL_MAP`).

    Each component is measured from an independent (non-
    overlapping-redshift) galaxy/quasar sample, so the combined
    covariance is block-diagonal: no cross-survey correlation
    terms, but each component's own internal correlation (e.g.
    between its DM/rs and DH/rs) is preserved.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    DESIDataset
    """

    entry = _validate_version(

        "sdss_bao",

        version,

    )

    dataset_path = _get_dataset_path(

        "sdss_bao",

        version,

    )

    z_parts = []
    value_parts = []
    observable_parts = []
    cov_blocks = []

    for component in entry["components"]:

        data = _load_txt(

            dataset_path / component["data"],

            dtype=None,

        )

        cov = _load_covariance(

            dataset_path / component["covariance"],

        )

        n = len(data["f0"])

        if cov.shape != (n, n):

            raise ValueError(

                f"'{component['data']}': expected a ({n}, {n}) "

                f"covariance from '{component['covariance']}', "

                f"but found {cov.shape}.",

            )

        z_parts.append(np.asarray(data["f0"], dtype=float))
        value_parts.append(np.asarray(data["f1"], dtype=float))
        observable_parts.append(np.asarray(data["f2"], dtype=str))
        cov_blocks.append(cov)

    covariance = block_diag(*cov_blocks)

    return DESIDataset(

        z=np.concatenate(z_parts),

        value=np.concatenate(value_parts),

        observable=np.concatenate(observable_parts),

        covariance=make_covariance(

            cov=covariance,

        ),

        reference=entry["reference"],

    )

# ============================================================
# Pantheon+ / Pantheon+SH0ES
# ============================================================

def load_pantheon(
    version: str = "pantheon+sh0es",
    include_cepheid: bool = False,
) -> PantheonDataset:
    """
    Load a Pantheon+ / Pantheon+SH0ES supernova dataset.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    include_cepheid : bool, optional
        If True, include Cepheid calibrator supernovae.

    Returns
    -------
    PantheonDataset
    """

    entry = _validate_version(

        "pantheon",

        version,

    )

    dataset_path = _get_dataset_path(

        "pantheon",

        version,

    )

    table = _load_txt(

        dataset_path / entry["data"],

        names=True,

        dtype=None,

    )

    covariance = _load_pantheon_covariance(

        dataset_path / entry["covariance"],

    )

    z_hd = table["zHD"].astype(float)

    z_cmb = table["zCMB"].astype(float)

    z_hel = table["zHEL"].astype(float)

    m_b_corr = table["m_b_corr"].astype(float)

    is_calibrator = table["IS_CALIBRATOR"].astype(int)

    mask = _build_pantheon_mask(

        is_calibrator,

        include_cepheid,

    )

    z_hd = z_hd[mask]

    z_cmb = z_cmb[mask]

    z_hel = z_hel[mask]

    m_b_corr = m_b_corr[mask]

    is_calibrator = is_calibrator[mask]

    covariance = covariance[

        np.ix_(

            mask,

            mask,

        )

    ]

    expected = mask.sum()

    if covariance.shape != (

        expected,

        expected,

    ):

        raise ValueError(

            f"Expected covariance shape ({expected}, {expected}), "

            f"but found {covariance.shape}.",

        )

    covariance = make_covariance(

        cov=covariance,

    )

    return PantheonDataset(

        z_hd=z_hd,

        z_cmb=z_cmb,

        z_hel=z_hel,

        m_b_corr=m_b_corr,

        covariance=covariance,

        cepheid=is_calibrator,

        reference=entry["reference"],

    )


# ============================================================
# DES-SN5YR
# ============================================================

def load_des_sn5yr(
    version: str = "des-sn5yr",
) -> DESSN5YRDataset:
    """
    Load the DES-SN5YR (Dark Energy Survey 5-year) supernova
    dataset.

    Unlike Pantheon+, DES-SN5YR distributes the already-computed
    distance modulus (``MU``) rather than a SALT-corrected
    apparent magnitude, and its covariance file stores the
    *precision* (inverse covariance) matrix directly rather than
    the covariance itself -- see :class:`~data.dataset.DESSN5YRDataset`
    and :class:`~data.covariance.PrecisionCovariance`.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    DESSN5YRDataset
    """

    entry = _validate_version(

        "des_sn5yr",

        version,

    )

    dataset_path = _get_dataset_path(

        "des_sn5yr",

        version,

    )

    table = _load_snana_fitres(

        dataset_path / entry["data"],

    )

    precision = _load_des_precision_covariance(

        dataset_path / entry["covariance"],

    )

    z_hd = table["zHD"].astype(float)

    z_hel = table["zHEL"].astype(float)

    mu = table["MU"].astype(float)

    mu_err = table["MUERR"].astype(float)

    n = len(z_hd)

    if precision.shape != (n, n):

        raise ValueError(

            f"Expected precision matrix shape ({n}, {n}), "

            f"but found {precision.shape}.",

        )

    covariance = make_covariance(

        precision=precision,

    )

    return DESSN5YRDataset(

        z_hd=z_hd,

        z_hel=z_hel,

        mu=mu,

        mu_err=mu_err,

        covariance=covariance,

        reference=entry["reference"],

    )


# ============================================================
# Growth rate (fsigma8)
# ============================================================

def load_fsigma8(
    version: str = "gold2018",
) -> GrowthDataset:
    """
    Load an fsigma8(z) growth-rate ("RSD") dataset.

    The default ``"gold2018"`` version is the Sagredo, Nesseris &
    Sapone (2018) "Gold-2018" compilation (22 points spanning
    6dFGS/SDSS/WiggleZ/BOSS/VIPERS/FastSound/eBOSS DR14Q), as
    bundled with the public MontePython likelihood
    snesseris/RSD-growth -- see ``data/growth/gold2018/`` and
    REFERENCES.md for provenance. Three of WiggleZ's points
    (Blake et al. 2012) and all four of eBOSS DR14Q's tomographic
    bins (Zhao et al. 2018) are internally correlated -- the
    combined covariance is block-diagonal-on-top-of-diagonal:
    ``diag(sigma^2)`` everywhere, with those two blocks overwritten
    by their own dense sub-covariance, exactly as the reference
    likelihood builds it.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    GrowthDataset
    """

    entry = _validate_version("fsigma8", version)
    dataset_path = _get_dataset_path("fsigma8", version)

    data = _load_txt(dataset_path / entry["data"])

    z = data[:, 0]
    fsigma8 = data[:, 1]
    sigma = data[:, 2]
    HdAz = data[:, 3]

    n = len(z)
    cov = np.diag(sigma ** 2)

    for block in entry.get("blocks", ()):

        lo, hi = block["rows"]

        block_cov = _load_covariance(dataset_path / block["covariance"])
        # The bundled blocks are symmetric up to float-formatting
        # noise in the source file (e.g. "0.0032857439999999997" vs
        # "0.003285744") -- symmetrize rather than trust either
        # triangle exactly.
        block_cov = 0.5 * (block_cov + block_cov.T)

        m = hi - lo

        if block_cov.shape != (m, m):

            raise ValueError(
                f"'{block['covariance']}': expected a ({m}, {m}) "
                f"covariance block, but found {block_cov.shape}.",
            )

        cov[lo:hi, lo:hi] = block_cov

    return GrowthDataset(

        z=z,

        fsigma8=fsigma8,

        sigma=sigma,

        HdAz=HdAz,

        covariance=make_covariance(cov=cov),

        reference=entry["reference"],

    )


# ============================================================
# S8 weak-lensing prior
# ============================================================

def load_s8(
    version: str = "kids1000",
) -> S8Dataset:
    """
    Load a single Gaussian S8 = sigma8 * sqrt(Omega_m / 0.3)
    weak-lensing constraint.

    Parameters
    ----------
    version : str, optional
        Dataset version -- ``"kids1000"`` (default) or
        ``"des_y3"``. Don't combine the two in the same fit: they
        are independent surveys, not a single joint constraint, and
        :class:`~likelihoods.s8.S8Likelihood` treats whichever
        version is loaded as the only S8 measurement in the fit.

    Returns
    -------
    S8Dataset
    """

    entry = _validate_version("s8", version)
    dataset_path = _get_dataset_path("s8", version)

    data = _load_txt(dataset_path / entry["data"])

    value = float(data[0])
    sigma = float(data[1])

    return S8Dataset(

        value=value,

        sigma=sigma,

        covariance=make_covariance(sigma=np.array([sigma])),

        reference=entry["reference"],

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