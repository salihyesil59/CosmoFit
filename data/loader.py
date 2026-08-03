"""
Dataset loading utilities.

This module provides a single interface for loading all
observational datasets used throughout the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import CCDataset
from .dataset import DESIDataset
from .dataset import PantheonDataset
from .dataset import PlanckDataset

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

        "reference": "Favale et al. (2023)",

    },

}


DESI_FILES = {

    "desi2024": {

        "parent": "bao",

        "folder": "desi2024",

        "data": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",

        "covariance": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",

        "reference": "DESI Collaboration (2024)",

    },

}


PANTHEON_FILES = {

    "pantheon+sh0es": {

        "parent": "sn",

        "folder": "pantheon-plus-sh0es",

        "data": "Pantheon+SH0ES.dat",

        "covariance": "Pantheon+SH0ES_STAT+SYS.cov",

        "reference": "Brout et al. (2022)",

    },

}


PLANCK_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "planck2018",

        "data": "distance_prior.txt",

        "covariance": "distance_prior_cov.txt",

        "reference": "Chen, Kumar & Ratra (2019), arXiv:1808.05724",

    },

}


# ============================================================
# Registry lookup
# ============================================================

_REGISTRIES = {

    "cc": CC_FILES,

    "desi": DESI_FILES,

    "pantheon": PANTHEON_FILES,

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