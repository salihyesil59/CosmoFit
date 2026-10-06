"""
BAO: DESI, SDSS/BOSS/eBOSS consensus and full shape, 6dFGS + MGS, and
the tabulated eBOSS likelihoods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from ..dataset import DESIDataset
from ..dataset import TabulatedBAODataset
from ..covariance import make_covariance

from ._common import _check_file_exists, _get_dataset_path, _load_covariance, _load_txt, _validate_version


DESI_FILES = {

    "desi2024": {

        "parent": "bao",

        "folder": "desi2024",

        "data": "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",

        "covariance": "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",

        "reference": "DESI Collaboration (2024), JCAP 02 (2025) 021, arXiv:2404.03002",

    },

    # DESI Data Release 2: three years of observations, >14 million
    # galaxies and quasars, twice the DR1 sample. Same 13-element
    # (z, value, observable) format and same provenance
    # (CobayaSampler/bao_data) as DR1 above, so it drops straight
    # into the same loader -- only the numbers change.
    #
    # Not a superset to be stacked with DR1: DR2 *includes* every
    # DR1 galaxy, so combining the two versions double-counts the
    # entire DR1 sample. Pick one.
    "desi2025": {

        "parent": "bao",

        "folder": "desi_dr2",

        "data": "desi_dr2_gaussian_bao_ALL_GCcomb_mean.txt",

        "covariance": "desi_dr2_gaussian_bao_ALL_GCcomb_cov.txt",

        "reference": "DESI Collaboration (2025), arXiv:2503.14738 (DESI DR2 Results II)",

    },

}


#: Pre-DESI, pre-BOSS low-redshift BAO: the two single-point
#: measurements that anchor the BAO distance ladder below z = 0.2,
#: where every other BAO dataset in this library has no coverage
#: at all (DESI's lowest bin is z = 0.295, SDSS's is z = 0.38).
#: Independent of both, so unlike DESI-vs-SDSS these *can* be
#: combined with either.
BAO_LOWZ_FILES = {

    "6dfgs+mgs": {

        "parent": "bao",

        "folder": "lowz",

        "components": (

            {
                "data": "sixdfgs_2011_bao.txt",
                "sigma": "sixdfgs_2011_bao_sigma.txt",
                # Beutler et al. quote r_s/D_V in units of the
                # Eisenstein & Hu (1998) fitting-formula sound
                # horizon (153.9 Mpc for their fiducial), where a
                # Boltzmann code gives 149.8 Mpc for the same
                # cosmology. See `DESIDataset.rs_rescale`.
                "rs_rescale": 153.9 / 149.8,
            },

            {
                "data": "sdss_dr7_mgs_bao.txt",
                "sigma": "sdss_dr7_mgs_bao_sigma.txt",
            },

        ),

        "reference": (
            "Beutler et al. (2011), MNRAS 416, 3017, arXiv:1106.3366 "
            "(6dFGS, z=0.106); "
            "Ross et al. (2015), MNRAS 449, 835, arXiv:1409.3242 "
            "(SDSS DR7 MGS, z=0.15)"
        ),

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


#: Planck 2018 low-multipole EE (SimAll), as a tabulated
#: probability rather than a mean and a covariance.
# ------------------------------------------------------------
# eBOSS DR16 tabulated BAO likelihoods
# ------------------------------------------------------------
#
# The two DR16 tracers that are *not* Gaussian. Everything else in
# `SDSS_BAO_FILES` is a mean and a covariance; these are likelihood
# surfaces, released as a grid because a mean and a covariance would
# misrepresent them (see `TabulatedBAODataset`).
#
# `observable` names what the coordinate columns hold, in order,
# followed in the file by the probability. Both grids are written
# with the last coordinate varying fastest.
EBOSS_ELG_FILES = {

    "dr16": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {"data": "sdss_DR16_ELG_BAO_DVtable.txt"},

        ),

        "z_eff": 0.845,

        "observable": ("DV_over_rs",),

        "reference": (
            "eBOSS DR16 ELG BAO -- de Mattia et al. (2020), "
            "MNRAS 501, 5616, arXiv:2007.09008. "
            "D_V/r_d = 18.33 (+0.57/-0.62) at z_eff = 0.845, from a "
            "1.4-sigma BAO detection."
        ),

    },

}


# The auto-correlation and the quasar cross-correlation, multiplied.
#
# eBOSS release them separately and quote a *combined* constraint
# obtained by fitting them together; there is no combined grid.
# Multiplying the two treats them as independent, which is what
# Cobaya does and which `tests/test_eboss_tables.py` justifies rather
# than assumes: the product reproduces the published
# D_M/r_d = 37.5 +- 1.1 and D_H/r_d = 8.99 +- 0.19 to better than 1%.
# Had the neglected correlation mattered, the recovered errors would
# have come out too tight.
#
# The halves are kept as their own versions so that claim stays
# checkable, and because the auto-correlation alone is occasionally
# what a comparison wants.
EBOSS_LYA_FILES = {

    "dr16": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {"data": "sdss_DR16_LYAUTO_BAO_DMDHgrid.txt"},

            {"data": "sdss_DR16_LYxQSO_BAO_DMDHgrid.txt"},

        ),

        "z_eff": 2.334,

        "observable": ("DM_over_rs", "DH_over_rs"),

        "reference": (
            "eBOSS DR16 Lyman-alpha BAO -- du Mas des Bourboux et "
            "al. (2020), ApJ 901, 153, arXiv:2007.08995. "
            "D_M/r_d = 37.5 +- 1.1, D_H/r_d = 8.99 +- 0.19 at "
            "z_eff = 2.334, from the forest auto-correlation "
            "combined with its cross-correlation with quasars."
        ),

    },

    "dr16_auto": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {"data": "sdss_DR16_LYAUTO_BAO_DMDHgrid.txt"},

        ),

        "z_eff": 2.334,

        "observable": ("DM_over_rs", "DH_over_rs"),

        "reference": (
            "eBOSS DR16 Lyman-alpha forest auto-correlation only -- "
            "du Mas des Bourboux et al. (2020), arXiv:2007.08995."
        ),

    },

    "dr16_cross": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {"data": "sdss_DR16_LYxQSO_BAO_DMDHgrid.txt"},

        ),

        "z_eff": 2.334,

        "observable": ("DM_over_rs", "DH_over_rs"),

        "reference": (
            "eBOSS DR16 Lyman-alpha x quasar cross-correlation only "
            "-- du Mas des Bourboux et al. (2020), arXiv:2007.08995."
        ),

    },

}


# The ELG sample again, this time as the full-shape analysis: a
# 100x100x100 grid in (D_M/r_d, D_H/r_d, f*sigma8). Unlike everything
# else in `data/`, this is shipped in a converted form -- the release
# is 60 MB of ASCII with 10.3% of its probabilities underflowed to
# exact zero. `tools/convert_eboss_elg_fs_grid.py` does the
# conversion, is committed, and documents both lossy steps and the
# check that the marginals survive them unchanged.
EBOSS_ELG_FS_FILES = {

    "dr16": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {"data": "sdss_DR16_ELG_FSBAO_DMDHfs8grid.npz"},

        ),

        "z_eff": 0.845,

        "observable": ("DM_over_rs", "DH_over_rs", "fsigma8"),

        "reference": (
            "eBOSS DR16 ELG full-shape (RSD + BAO) -- de Mattia et "
            "al. (2020), MNRAS 501, 5616, arXiv:2007.09008. "
            "D_M/r_d = 19.5 +- 1.0, D_H/r_d = 19.6 (-2.1/+2.2), "
            "f*sigma8 = 0.315 +- 0.095 at z_eff = 0.85, from the "
            "consensus of the Fourier- and configuration-space "
            "analyses."
        ),

    },

}


# BOSS DR12 + eBOSS DR16, analysed for the full anisotropic shape
# rather than the BAO peak alone: (D_M/r_d, D_H/r_d, f*sigma8) per
# bin, with the covariance *between* geometry and growth. The BAO-only
# `SDSS_BAO_FILES` above and the `GROWTH_FILES` compilation together
# cover the same galaxies while treating those as independent, which
# they are not -- this is the product that does not.
#
# The same z = 0.61 omission as `SDSS_BAO_FILES`: the released
# BAOplus LRG file already excludes it, for the same overlap with
# the eBOSS DR16 LRG sample.
SDSS_FSBAO_FILES = {

    "dr16": {

        "parent": "bao",

        "folder": "sdss",

        "components": (

            {
                "data": "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat",
                "covariance":
                    "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt",
            },

            {
                "data": "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat",
                "covariance":
                    "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt",
            },

        ),

        "reference": (
            "SDSS BAO + full-shape consensus -- eBOSS Collaboration "
            "/ Alam et al. (2021), Phys. Rev. D 103, 083533, "
            "arXiv:2007.08991. BOSS DR12 (z = 0.38, 0.51), eBOSS "
            "DR16 LRG (z = 0.698) and eBOSS DR16 QSO (z = 1.48), "
            "each giving D_M/r_d, D_H/r_d and f*sigma8 with their "
            "joint covariance."
        ),

    },

}


# ============================================================
# BAO (DESI)
# ============================================================

def load_desi(
    version: str = "desi2025",
) -> DESIDataset:
    """
    Load a DESI BAO dataset.

    Parameters
    ----------
    version : str, optional
        Dataset version: ``"desi2025"`` (DR2, the default) or
        ``"desi2024"`` (DR1).

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

def _load_blockdiag_bao(
    family: str,
    version: str,
    rename: dict[str, str] | None = None,
) -> DESIDataset:
    """
    Load a set of ``(z, value, observable)`` components into one
    dataset with a block-diagonal covariance.

    Each component is measured from an independent, non-overlapping
    sample, so there are no cross-component terms -- but each
    component's own internal correlation is preserved, which for a
    full-shape component means the correlation between its
    geometry and its growth rate.

    ``rename`` maps observable names as they appear in the released
    files onto the keys of :data:`likelihoods.desi.MODEL_MAP`. Only
    the full-shape files need it: they write ``f_sigma8`` where the
    library says ``fsigma8``.
    """

    entry = _validate_version(family, version)

    dataset_path = _get_dataset_path(family, version)

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

        observables = np.asarray(data["f2"], dtype=str)

        if rename:

            observables = np.array(

                [rename.get(name, name) for name in observables],

                dtype=str,

            )

        z_parts.append(np.asarray(data["f0"], dtype=float))
        value_parts.append(np.asarray(data["f1"], dtype=float))
        observable_parts.append(observables)
        cov_blocks.append(cov)

    return DESIDataset(

        z=np.concatenate(z_parts),

        value=np.concatenate(value_parts),

        observable=np.concatenate(observable_parts),

        covariance=make_covariance(

            cov=block_diag(*cov_blocks),

        ),

        reference=entry["reference"],

    )


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

    return _load_blockdiag_bao("sdss_bao", version)


def load_sdss_fsbao(
    version: str = "dr16",
) -> DESIDataset:
    """
    Load the SDSS **BAO + full-shape** consensus: ``D_M/r_d``,
    ``D_H/r_d`` and ``f sigma_8`` per redshift bin, with the
    covariance between them.

    This is the same galaxies as :func:`load_sdss_bao`, analysed
    for their full anisotropic clustering rather than the BAO peak
    alone -- so it adds the growth rate, and, more importantly,
    the correlation between growth and geometry. Combining
    ``"sdss_bao"`` with the separate ``"fsigma8"`` compilation
    instead treats those as independent, which they are not: the
    released covariance has correlations of 0.19 to 0.64 between
    ``D_M/r_d`` and ``f sigma_8`` within a bin, strongest for the
    quasars.

    The released files name the growth observable ``f_sigma8``;
    it is renamed on load to the library's ``fsigma8``.

    Note that ``f sigma_8`` here is compared with the model's own
    ``fsigma8(z)`` *without* an Alcock-Paczynski rescaling, unlike
    :class:`~likelihoods.fsigma8.FSigma8Likelihood`. A full-shape
    fit varies the geometry alongside the growth rate, so the
    fiducial it was measured against is already a fitted quantity
    rather than something to correct back to -- applying the
    correction as well would count it twice.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    DESIDataset
    """

    return _load_blockdiag_bao(

        "sdss_fsbao",

        version,

        rename={"f_sigma8": "fsigma8"},

    )


# ============================================================
# BAO (low-z: 6dFGS + SDSS DR7 MGS)
# ============================================================

def load_bao_lowz(
    version: str = "6dfgs+mgs",
) -> DESIDataset:
    """
    Load the low-redshift BAO anchors (6dFGS z=0.106, SDSS DR7 MGS
    z=0.15), in the same (z, value, observable-type) format DESI
    and SDSS use.

    Both are single measurements from independent surveys with no
    published cross-correlation, so the covariance is diagonal.

    One of them (6dFGS) reports ``r_s/D_V`` rather than the
    ``D_V/r_s`` every other BAO dataset here uses. That is kept as
    its own observable type rather than inverted: 0.336 +- 0.015 is
    Gaussian in ``r_s/D_V``, and 1/x of a Gaussian is neither
    Gaussian nor centred on 1/mean. It also carries an
    ``rs_rescale`` (see :class:`~data.dataset.DESIDataset`), since
    it was calibrated against an Eisenstein & Hu (1998) sound
    horizon rather than an integrated one.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    DESIDataset
    """

    entry = _validate_version("bao_lowz", version)
    dataset_path = _get_dataset_path("bao_lowz", version)

    z_parts = []
    value_parts = []
    observable_parts = []
    sigma_parts = []
    rescale_parts = []

    for component in entry["components"]:

        data = _load_txt(

            dataset_path / component["data"],

            dtype=None,

        )

        sigma = np.atleast_1d(

            _load_txt(dataset_path / component["sigma"]),

        ).astype(float)

        # A one-row file comes back from genfromtxt as a 0-d
        # structured scalar rather than a length-1 array, so every
        # field needs atleast_1d before it can be concatenated.
        z = np.atleast_1d(np.asarray(data["f0"], dtype=float))
        value = np.atleast_1d(np.asarray(data["f1"], dtype=float))
        observable = np.atleast_1d(np.asarray(data["f2"], dtype=str))

        if not (len(z) == len(value) == len(observable) == len(sigma)):

            raise ValueError(

                f"'{component['data']}': data and sigma files "

                f"disagree on the number of measurements.",

            )

        z_parts.append(z)
        value_parts.append(value)
        observable_parts.append(observable)
        sigma_parts.append(sigma)

        rescale_parts.append(

            np.full(

                len(z),

                float(component.get("rs_rescale", 1.0)),

            ),

        )

    sigma = np.concatenate(sigma_parts)

    return DESIDataset(

        z=np.concatenate(z_parts),

        value=np.concatenate(value_parts),

        observable=np.concatenate(observable_parts),

        covariance=make_covariance(sigma=sigma),

        rs_rescale=np.concatenate(rescale_parts),

        reference=entry["reference"],

    )


# ============================================================
# Tabulated BAO likelihood surfaces
# ============================================================

def _load_grid_npz(path: Path, observable: tuple[str, ...]):
    """
    Read a pre-converted likelihood grid.

    Stored as ``log_prob`` (the log-likelihood on the grid, float32,
    floored 200 below its peak) plus one axis array per observable,
    named after it. See ``tools/convert_eboss_elg_fs_grid.py`` for
    why this one is shipped converted rather than as released.
    """

    _check_file_exists(path)

    archive = np.load(path)

    axis_names = {
        "DM_over_rs": "dm_over_rs",
        "DH_over_rs": "dh_over_rs",
        "DV_over_rs": "dv_over_rs",
        "fsigma8": "fsigma8",
    }

    missing = [
        name for name in observable
        if axis_names.get(name, name) not in archive
    ]

    if missing:

        raise ValueError(
            f"'{path}' has no axis for {missing}; it holds "
            f"{sorted(archive.files)}.",
        )

    grids = tuple(
        np.asarray(archive[axis_names.get(name, name)], dtype=float)
        for name in observable
    )

    log_prob = np.asarray(archive["log_prob"], dtype=float)

    expected = tuple(len(g) for g in grids)

    if log_prob.shape != expected:

        raise ValueError(
            f"'{path}': log_prob has shape {log_prob.shape}, but its "
            f"axes describe {expected}.",
        )

    return grids, log_prob


def load_eboss_table(
    family: str,
    version: str = "dr16",
) -> TabulatedBAODataset:
    """
    Load an eBOSS DR16 BAO likelihood released as a grid.

    Parameters
    ----------
    family : str
        ``"eboss_elg"`` or ``"eboss_lya"``.
    version : str, optional
        ``"dr16"`` (the default). For ``"eboss_lya"`` this is the
        auto-correlation times the cross-correlation; the halves are
        available as ``"dr16_auto"`` and ``"dr16_cross"``.

    Returns
    -------
    TabulatedBAODataset

    Notes
    -----
    Two things are done here rather than in the likelihood, because
    both are properties of the *files* and getting either wrong is
    silent.

    The released column is a probability, and it is converted to a
    log once, at load. Interpolating the probability directly spans
    thirty orders of magnitude, does not reproduce the published
    error bars, and can return negative values between nodes that
    then become NaN under a later log.

    Multi-component versions are combined by **adding** the logs,
    which multiplies the likelihoods. That is only legitimate if the
    components are independent; see ``EBOSS_LYA_FILES["dr16"]`` for
    why it holds here and ``tests/test_eboss_tables.py`` for the
    check that it does.
    """

    entry = _validate_version(family, version)
    dataset_path = _get_dataset_path(family, version)

    observable = tuple(entry["observable"])
    n_axes = len(observable)

    axes: tuple[np.ndarray, ...] | None = None
    total: np.ndarray | None = None

    for component in entry["components"]:

        path = dataset_path / component["data"]

        if path.suffix == ".npz":

            grids, log_prob = _load_grid_npz(path, observable)

            if axes is None:
                axes, total = grids, log_prob
            else:
                for existing, incoming in zip(axes, grids):
                    if not np.array_equal(existing, incoming):
                        raise ValueError(
                            f"{component['data']} is on a different "
                            f"grid from the components before it.",
                        )
                total = total + log_prob

            continue

        raw = _load_txt(path)

        raw = np.atleast_2d(np.asarray(raw, dtype=float))

        if raw.shape[1] != n_axes + 1:

            raise ValueError(

                f"{component['data']} has {raw.shape[1]} columns; "
                f"expected {n_axes + 1} for observables "
                f"{observable} plus a probability.",

            )

        grids = tuple(np.unique(raw[:, i]) for i in range(n_axes))

        shape = tuple(len(g) for g in grids)

        if np.prod(shape) != raw.shape[0]:

            raise ValueError(

                f"{component['data']} holds {raw.shape[0]} rows, "
                f"which is not the {shape} rectangular grid its "
                f"coordinate columns describe.",

            )

        probability = raw[:, n_axes]

        if np.any(probability <= 0.0):

            raise ValueError(

                f"{component['data']} contains a non-positive "
                f"probability, which has no logarithm. The released "
                f"tables are strictly positive throughout.",

            )

        # `np.unique` sorts, so the reshape has to be checked
        # against the file's own ordering rather than assumed.
        order = np.lexsort(tuple(raw[:, i] for i in range(n_axes - 1, -1, -1)))

        log_prob = np.log(probability[order]).reshape(shape)

        if axes is None:

            axes, total = grids, log_prob

        else:

            for existing, incoming in zip(axes, grids):

                if not np.array_equal(existing, incoming):

                    raise ValueError(

                        f"{component['data']} is on a different grid "
                        f"from the components before it -- they can "
                        f"only be combined point by point.",

                    )

            total = total + log_prob

    return TabulatedBAODataset(

        z_eff=float(entry["z_eff"]),

        observable=observable,

        axes=axes,

        log_prob=total,

        reference=entry["reference"],

    )
