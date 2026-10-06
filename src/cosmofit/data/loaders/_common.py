"""
Paths, version lookup and the file readers every loader shares.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

# The data directory: this package's parent.
DATA_DIR = Path(__file__).parent.parent


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

    # Here rather than at the top: the registries are every family's
    # file tables, and every family imports this module.
    from ._index import _REGISTRIES

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
