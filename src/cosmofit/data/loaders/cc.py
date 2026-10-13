"""
Cosmic chronometers: H(z), and the covariance built from the primary
sources.
"""

from __future__ import annotations

import numpy as np

from ..dataset import CCDataset
from ..covariance import make_covariance

from ._common import _get_dataset_path, _load_txt, _validate_version


# ============================================================
# Dataset registry
# ============================================================

CC_FILES = {

    "favale2023": {

        "folder": "favale2023",

        "data": "CC_32_Favale2023_data.txt",

        # Moresco et al. (2020) systematic error budget, in percent of
        # H(z); the off-diagonal covariance of the Moresco measurements
        # is built from it -- see `_cc_covariance`.
        "systematics": "data_MM20.dat",

        "reference": "Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406, arXiv:2301.09591",

    },

    # The same 32, with Jiao et al. (2023) at z = 0.8 and Tomasetti et
    # al. (2023) at z = 1.26 -- full-spectrum fits published since.
    # Their errors carry their own systematics and they enter on the
    # diagonal alone, as every non-Moresco point of the 32 does.
    "favale2023_extended": {

        "folder": "favale2023",

        "data": "CC_34_Favale2023_Jiao2023_Tomasetti2023_data.txt",

        "systematics": "data_MM20.dat",

        "reference": (
            "Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406, arXiv:2301.09591; "
            "Jiao et al. (2023), ApJS 265, 48, arXiv:2205.05701; "
            "Tomasetti et al. (2023), A&A 679, A96, arXiv:2305.16387"
        ),

    },

}


# ============================================================
# Cosmic Chronometers
# ============================================================

def _cc_covariance(
    z: np.ndarray,
    H: np.ndarray,
    sigma: np.ndarray,
    systematics: np.ndarray,
    correlated: np.ndarray,
) -> np.ndarray:
    r"""
    Covariance of the cosmic chronometer H(z) measurements, as Favale,
    Gomez-Valent & Migliaccio (2023) built theirs.

    The tabulated errors ``sigma`` are each measurement's *total* error
    and stay on the diagonal. The 15 measurements of Moresco et al.
    (2012, 2015, 2016) -- ``correlated`` -- share the method's
    systematics, the initial mass function and the stellar population
    synthesis model of the Moresco et al. (2020) budget, which are the
    same assumption at every redshift and so fully correlated between
    them::

        C_ii = sigma_i^2
        C_ij = sum_k s_k(z_i) s_k(z_j)       (i != j, both Moresco's)
        s_k(z) = H(z) f_k(z)

    with ``f_k`` the fractional error of component ``k`` (``IMF`` and
    ``mod_ooo``), interpolated in the table and held at its end values
    beyond it. Every other measurement is uncorrelated with the rest.

    This is the construction of the authors' own script (A. Favale,
    private communication): ``C_diag + C_IMF + C_SPS`` over the Moresco
    points, where ``C_diag`` holds their statistical errors, and the
    tabulated error squared for the others. The diagonal of that sum is
    the tabulated ``sigma^2`` -- 6.2 at z = 0.1791 is the statistical
    4.5 and the systematic 4.2 in quadrature -- so adding ``s^2`` to
    ``sigma^2`` would count the systematic twice.

    References
    ----------
    Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406;
    Moresco et al. (2020), ApJ 898, 82 (2020ApJ...898...82M);
    https://gitlab.com/mmoresco/CCcovariance
    """

    z_table = systematics[:, 0]

    shared = np.zeros((len(z), len(z)))

    for column in (1, 4):                       # IMF, mod_ooo

        fractional = np.interp(z, z_table, systematics[:, column]) / 100.0

        error = np.where(correlated, H * fractional, 0.0)

        shared += np.outer(error, error)

    covariance = shared - np.diag(np.diag(shared))

    covariance += np.diag(sigma ** 2)

    return covariance


# ------------------------------------------------------------

def load_cc(
    version: str = "favale2023",
) -> CCDataset:
    """
    Load a Cosmic Chronometer dataset.

    The covariance keeps the tabulated errors on its diagonal and adds
    the correlation the method's systematics induce between the
    Moresco et al. measurements -- see :func:`_cc_covariance`.

    Parameters
    ----------
    version : str, optional
        ``"favale2023"`` (the default; 32 points) or
        ``"favale2023_extended"`` (34: with Jiao et al. 2023 at
        z = 0.8 and Tomasetti et al. 2023 at z = 1.26).

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

    correlated = data[:, 3].astype(bool)

    systematics = _load_txt(

        dataset_path / entry["systematics"],

    )

    covariance = make_covariance(

        cov=_cc_covariance(

            z,

            H,

            sigma,

            systematics,

            correlated,

        ),

    )

    return CCDataset(

        z=z,

        H=H,

        sigma=sigma,

        covariance=covariance,

        reference=entry["reference"],

    )
