"""
Growth of structure: f sigma8(z) and the weak-lensing S8 priors.
"""

from __future__ import annotations

import numpy as np

from ..dataset import GrowthDataset
from ..dataset import S8Dataset
from ..covariance import make_covariance

from ._common import _get_dataset_path, _load_covariance, _load_txt, _validate_version


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

    "kids_legacy": {

        "parent": "s8",

        "folder": "kids_legacy",

        "data": "s8_kids_legacy.txt",

        "reference": "Wright et al. (2025), A&A 703, A158, arXiv:2503.19441",

    },

    "hsc_y3": {

        "parent": "s8",

        "folder": "hsc_y3",

        "data": "s8_hsc_y3.txt",

        "reference": "Li et al. (2023), Phys. Rev. D, arXiv:2304.00702",

    },

    "desy3_kids1000": {

        "parent": "s8",

        "folder": "desy3_kids1000",

        "data": "s8_desy3_kids1000.txt",

        "reference": "DES & KiDS Collaborations / Abbott et al. (2023), OJAp, arXiv:2305.17173",

    },

}


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
        Dataset version -- ``"kids1000"`` (default), ``"des_y3"``
        (3x2pt), ``"kids_legacy"`` (the complete KiDS, superseding
        KiDS-1000), ``"hsc_y3"`` or ``"desy3_kids1000"`` (the two
        surveys' shear analysed jointly). Load one:
        :class:`~likelihoods.s8.S8Likelihood` treats whichever
        version is loaded as the only S8 measurement in the fit, and
        several of them share data -- KiDS-Legacy contains KiDS-1000,
        the joint analysis contains both KiDS-1000 and DES Y3.

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
