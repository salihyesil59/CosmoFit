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
        # H(z); the off-diagonal covariance is built from it -- see
        # `_cc_covariance`. The distributed correlation matrix is kept
        # next to it as the check on that construction (tests only).
        "systematics": "data_MM20.dat",

        "correlation": "CC_32_Favale2023_Moresco2020_correlation.txt",

        "reference": "Favale, Gomez-Valent & Migliaccio (2023), MNRAS 523, 3406, arXiv:2301.09591",

    },

    # The same 32, with Jiao et al. (2023) at z = 0.8 and Tomasetti et
    # al. (2023) at z = 1.26 -- full-spectrum fits published since. They
    # are correlated with the rest by the same IMF and SPS budget, as
    # every one of the 32 is: the stellar-population models behind
    # their ages are the same ones.
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
) -> np.ndarray:
    r"""
    Covariance of the cosmic chronometer H(z) measurements.

    The tabulated errors ``sigma`` are taken as each measurement's
    *total* error and stay on the diagonal unchanged. What they cannot
    carry is the correlation: the method's systematics -- the initial
    mass function and the stellar population synthesis model, from the
    budget of Moresco et al. (2020) -- are the same assumption at every
    redshift, so they are fully correlated between measurements::

        C_ii = sigma_i^2
        C_ij = sum_k s_k(z_i) s_k(z_j)       (i != j)
        s_k(z) = H(z) f_k(z)

    with ``f_k`` the fractional error of component ``k`` (``IMF`` and
    ``mod_ooo``), interpolated in the table and held at its end values
    beyond it.

    Why the diagonal is not ``sigma^2 + s^2``
    -----------------------------------------
    That is the other reading, and it is what Moresco's own notebook
    does with *his* table, whose errors are statistical only. For the
    15 Moresco points in the Favale et al. (2023) compilation the
    tabulated errors are already close to the statistical and SPS
    errors in quadrature (6.2 against sqrt(4.3^2 + 4.2^2) = 6.1 at
    z = 0.1791, median ratio 1.02), so adding ``s^2`` again would count
    the systematic twice. Favale et al. do not release their covariance
    -- the correlation file bundled with the data is not an official
    product -- so the diagonal cannot be settled from the release
    itself; this keeps it at the published errors.

    The off-diagonal is not in question either way. The bundled
    correlation's off-diagonal factorizes as ``a_i a_j``, and ``s``
    from this budget reproduces it under the construction its author
    used. Turning that correlation back into a covariance with
    ``outer(sigma, sigma)``, as this library did, shrank every
    off-diagonal term below ``s_i s_j`` -- by the factor
    ``sigma_i sigma_j / sqrt((sigma_i^2 + s_i^2)(sigma_j^2 + s_j^2))``
    -- which is wrong under both readings.

    References
    ----------
    Moresco et al. (2020), ApJ 898, 82 (2020ApJ...898...82M);
    https://gitlab.com/mmoresco/CCcovariance
    """

    z_table = systematics[:, 0]

    correlated = np.zeros((len(z), len(z)))

    for column in (1, 4):                       # IMF, mod_ooo

        fractional = np.interp(z, z_table, systematics[:, column]) / 100.0

        error = H * fractional

        correlated += np.outer(error, error)

    covariance = correlated - np.diag(np.diag(correlated))

    covariance += np.diag(sigma ** 2)

    return covariance


# ------------------------------------------------------------

def load_cc(
    version: str = "favale2023",
) -> CCDataset:
    """
    Load a Cosmic Chronometer dataset.

    The covariance keeps the tabulated errors on its diagonal and adds
    the correlation the method's systematics induce between
    measurements -- see :func:`_cc_covariance`.

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

    systematics = _load_txt(

        dataset_path / entry["systematics"],

    )

    covariance = make_covariance(

        cov=_cc_covariance(

            z,

            H,

            sigma,

            systematics,

        ),

    )

    return CCDataset(

        z=z,

        H=H,

        sigma=sigma,

        covariance=covariance,

        reference=entry["reference"],

    )
