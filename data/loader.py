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

        "folder": "desi2024",

        "data": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",

        "covariance": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",

        "reference": "DESI Collaboration (2024)",

    },

}


PANTHEON_FILES = {

    "pantheonplus": {

        "folder": "pantheonplus",

        "data": "Pantheon+SH0ES.dat.txt",

        "covariance": "Pantheon+SH0ES_STAT+SYS.cov.txt",

        "reference": "Scolnic et al. (2022)",

    },

}


PLANCK_FILES = {

    "planck2018": {

        "folder": "planck2018",

        "data": "distance_prior.txt",

        "covariance": "distance_prior_cov.txt",

        "reference": "Planck Collaboration VI (2020)",

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

    return (

        DATA_DIR

        / dataset

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
    **kwargs,
):
    """
    Wrapper around numpy.loadtxt with file checking.
    """

    _check_file_exists(

        path,

    )

    return np.loadtxt(

        path,

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

            covariance = (

                correlation

                * np.outer(

                    sigma,

                    sigma,

                )

            )

    return CCDataset(

        z=z,

        H=H,

        sigma=sigma,

        covariance=covariance,

        reference=entry["reference"],

    )