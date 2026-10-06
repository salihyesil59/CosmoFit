"""
Type Ia supernovae: Pantheon+, DES-SN5YR and Union3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dataset import PantheonDataset
from ..dataset import DESSN5YRDataset
from ..dataset import Union3Dataset
from ..covariance import make_covariance

from ._common import _check_file_exists, _get_dataset_path, _load_txt, _validate_version


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
            "Popovic et al. (2026), MNRAS, arXiv:2511.07517 "
            "(DES-Dovekie recalibration, 1820 SNe; supersedes "
            "Vincenzi et al. 2024 / DES Collaboration 2024, "
            "arXiv:2401.02929)"
        ),

    },

}


UNION3_FILES = {

    "union3": {

        "parent": "sn",

        "folder": "union3",

        "data": "union3_lcparam_full.txt",

        "covariance": "union3_mag_covmat.txt",

        "reference": "Rubin et al. (2023), arXiv:2311.12098 (Union3 / UNITY1.5)",

    },

}


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

#: Lower redshift limit of the Pantheon+ Hubble-flow sample.
#:
#: Below z = 0.01 a supernova's redshift is dominated by its
#: peculiar velocity rather than by the expansion, and Brout et al.
#: (2022) cut those light curves from every cosmological fit: 1590
#: of the 1701 survive. This loader used to keep all 1624
#: non-calibrators instead, including 44 at 0.001 < z_HD < 0.01, and
#: to drop the 10 calibrator light curves that sit above the cut --
#: a sample nobody had published a fit to.
PANTHEON_Z_MIN = 0.01


def _build_pantheon_mask(
    z_hd: np.ndarray,
    is_calibrator: np.ndarray,
    include_cepheid: bool,
    z_min: float = PANTHEON_Z_MIN,
) -> np.ndarray:
    """
    Build the Pantheon+ sample mask.

    Without Cepheids this is the Hubble-flow cut alone,
    ``z_HD > z_min``. A calibrator above the cut is an ordinary
    Hubble-flow supernova in that fit and stays in; the ones below
    it go with every other low-z light curve.

    With Cepheids, every calibrator is added back regardless of
    redshift, since their distances come from the Cepheids rather
    than from the redshift. This is the 1657-row Pantheon+SH0ES
    sample.
    """

    mask = z_hd > z_min

    if include_cepheid:

        mask = mask | (is_calibrator == 1)

    return mask


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


# ============================================================
# Pantheon+ / Pantheon+SH0ES
# ============================================================

def load_pantheon(
    version: str = "pantheon+sh0es",
    include_cepheid: bool = False,
    z_min: float = PANTHEON_Z_MIN,
) -> PantheonDataset:
    """
    Load a Pantheon+ / Pantheon+SH0ES supernova dataset.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    include_cepheid : bool, optional
        If True, add the Cepheid-calibrator supernovae below
        ``z_min`` back in, together with their Cepheid distance
        moduli (``CEPH_DIST``) -- the Pantheon+SH0ES sample, 1657
        light curves. If False (default), the Hubble-flow sample
        alone, 1590 light curves.

    z_min : float, optional
        Hubble-flow cut on ``z_HD``. Default 0.01, as in every
        published Pantheon+ fit.

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

    ceph_dist = table["CEPH_DIST"].astype(float)

    mask = _build_pantheon_mask(

        z_hd,

        is_calibrator,

        include_cepheid,

        z_min,

    )

    z_hd = z_hd[mask]

    z_cmb = z_cmb[mask]

    z_hel = z_hel[mask]

    m_b_corr = m_b_corr[mask]

    is_calibrator = is_calibrator[mask]

    ceph_dist = ceph_dist[mask]

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

        ceph_dist=ceph_dist,

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
# Union3
# ============================================================

def _load_cosmomc_covmat(
    path: Path,
    expected: int,
) -> np.ndarray:
    """
    Read a CosmoMC-style supernova magnitude covariance: one
    integer giving the matrix dimension, followed by that many
    squared entries in row-major order.

    Parameters
    ----------
    path
        Path to the file.

    expected
        Number of data rows the covariance must match.
    """

    _check_file_exists(path)

    flat = np.loadtxt(path).ravel()

    n = int(flat[0])

    if n != expected:

        raise ValueError(

            f"'{path.name}': covariance declares dimension {n}, "

            f"but the data file has {expected} rows.",

        )

    if flat.size != 1 + n * n:

        raise ValueError(

            f"'{path.name}': expected {1 + n * n} numbers for a "

            f"{n}x{n} covariance, but found {flat.size}.",

        )

    return flat[1:].reshape(n, n)


# ------------------------------------------------------------

def load_union3(
    version: str = "union3",
) -> Union3Dataset:
    """
    Load the Union3 binned supernova compilation (Rubin et al.
    2023): 22 distance-modulus bins and their 22x22 magnitude
    covariance.

    Distributed in the CosmoMC ``lcparam``/``mag_covmat`` format,
    where the ``mb`` column holds the binned *distance modulus*
    (~36-46 mag) rather than an apparent magnitude, since UNITY1.5
    has already marginalized the light-curve standardization
    internally. Only ``zcmb``, ``zhel`` and ``mb`` carry
    information; the remaining SALT2 columns are zero-filled
    placeholders kept so the file matches the format.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    Union3Dataset
    """

    entry = _validate_version("union3", version)
    dataset_path = _get_dataset_path("union3", version)

    data_path = dataset_path / entry["data"]

    _check_file_exists(data_path)

    # Read by column position, not by header name. The released
    # file's header lists 19 columns but its rows carry 18 (the
    # trailing `biascor` field is absent), which `names=True`
    # rejects outright. Only three columns are used anyway, and
    # they are the first ones, ahead of the mismatch.
    z_cmb, z_hel, mu = np.loadtxt(

        data_path,

        usecols=(1, 2, 4),

        unpack=True,

        ndmin=2,

    )

    z_cmb = np.atleast_1d(z_cmb)
    z_hel = np.atleast_1d(z_hel)
    mu = np.atleast_1d(mu)

    covariance = _load_cosmomc_covmat(

        dataset_path / entry["covariance"],

        expected=len(z_cmb),

    )

    return Union3Dataset(

        z_cmb=z_cmb,

        z_hel=z_hel,

        mu=mu,

        covariance=make_covariance(cov=covariance),

        reference=entry["reference"],

    )
