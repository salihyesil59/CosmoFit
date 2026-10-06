"""
Every dataset's versions and references, across the families.
"""

from __future__ import annotations

from ._common import _validate_version
from .bao import BAO_LOWZ_FILES, DESI_FILES, EBOSS_ELG_FILES, EBOSS_ELG_FS_FILES, EBOSS_LYA_FILES, SDSS_BAO_FILES, SDSS_FSBAO_FILES
from .cc import CC_FILES
from .cmb import ACT_LENSING_FILES, LOWE_FILES, PLANCK_FILES, PLANCK_LENSING_FILES, PLIK_LITE_FILES
from .growth import GROWTH_FILES, S8_FILES
from .priors import _PRIOR_REGISTRIES
from .sn import DES_SN5YR_FILES, PANTHEON_FILES, UNION3_FILES


# ============================================================
# Registry lookup
# ============================================================

_REGISTRIES = {

    "cc": CC_FILES,

    "desi": DESI_FILES,

    "sdss_bao": SDSS_BAO_FILES,

    "sdss_fsbao": SDSS_FSBAO_FILES,

    "pantheon": PANTHEON_FILES,

    "des_sn5yr": DES_SN5YR_FILES,

    "fsigma8": GROWTH_FILES,

    "s8": S8_FILES,

    "planck": PLANCK_FILES,

    "bao_lowz": BAO_LOWZ_FILES,

    "union3": UNION3_FILES,

    "planck_lite": PLIK_LITE_FILES,

    "planck_lensing": PLANCK_LENSING_FILES,

    "planck_lowe": LOWE_FILES,

    "act_lensing": ACT_LENSING_FILES,

    "eboss_elg": EBOSS_ELG_FILES,

    "eboss_lya": EBOSS_LYA_FILES,

    "eboss_elg_fs": EBOSS_ELG_FS_FILES,

    **_PRIOR_REGISTRIES,

}


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

def dataset_reference(
    dataset: str,
    version: str | None = None,
) -> str:
    """
    The citation string for a dataset version, without loading any
    of its files.

    Every registry entry already carries the paper its numbers come
    from; this exposes it, so a caller that wants to *show* the
    provenance -- a GUI panel, a figure caption, a log line -- does
    not have to read a 1600x1600 covariance matrix off disk first,
    and does not have to reach into the private registry to avoid
    that.

    Parameters
    ----------
    dataset : str
        Dataset name, as in :func:`available_datasets`.

    version : str, optional
        Which version. Defaults to that dataset's first registered
        one -- the same default the corresponding ``load_*``
        function uses.

    Returns
    -------
    str

    Examples
    --------
    >>> dataset_reference("desi", "desi2025")
    'DESI Collaboration (2025), arXiv:2503.14738 (DESI DR2 Results II)'
    """

    if dataset not in _REGISTRIES:

        raise ValueError(

            f"Unknown dataset '{dataset}'. "

            f"Available: {list(_REGISTRIES)}",

        )

    if version is None:

        version = next(iter(_REGISTRIES[dataset]))

    return _validate_version(dataset, version).get("reference", "")


# ------------------------------------------------------------

def available_datasets() -> dict[str, list[str]]:

    return {

        key: list(

            value.keys(),

        )

        for key, value

        in _REGISTRIES.items()

    }
