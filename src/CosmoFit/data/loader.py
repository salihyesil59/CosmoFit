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
from .dataset import Union3Dataset
from .dataset import GaussianPriorDataset
from .dataset import CMBSpectrumDataset
from .dataset import TabulatedBAODataset
from .dataset import CMBLensingDataset
from .dataset import LowEllEEDataset

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


UNION3_FILES = {

    "union3": {

        "parent": "sn",

        "folder": "union3",

        "data": "union3_lcparam_full.txt",

        "covariance": "union3_mag_covmat.txt",

        "reference": "Rubin et al. (2023), arXiv:2311.12098 (Union3 / UNITY1.5)",

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


#: Planck 2018 ``plik_lite`` binned TT/TE/EE bandpowers -- the
#: foreground-marginalized high-l likelihood, i.e. the measured
#: CMB spectra themselves rather than the three-number compression
#: in :data:`PLANCK_FILES`. Used by
#: :class:`~likelihoods.planck_lite.PlanckLiteLikelihood`, which
#: needs a Boltzmann code to predict C_l and so is the one dataset
#: here with an optional dependency (CAMB).
PLIK_LITE_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "plik_lite",

        "data": "cl_cmb_plik_v22.dat",

        # A Fortran unformatted record holding the 613x613
        # bandpower covariance -- see `_load_plik_covariance`.
        "covariance": "c_matrix_plik_v22.dat",

        "blmin": "blmin.dat",

        "blmax": "blmax.dat",

        "weights": "bweight.dat",

        #: TT, TE, EE bandpower counts. TT spans l = 30-2508,
        #: TE and EE l = 30-1996.
        "n_bin": (215, 199, 199),

        "lmin": 30,

        "lmax": 2508,

        #: The two Commander low-multipole temperature bins
        #: (l = 2-29), which `load_plik_lite(use_low_ell=True)`
        #: prepends to the TT block. A separate likelihood from
        #: plik_lite, with its own windows and an uncorrelated
        #: (diagonal) covariance.
        "low_ell": {

            "folder": "low_ell",

            "data": "CTT_bin_low_ell_2018.dat",

            "blmin": "blmin_low_ell.dat",

            "blmax": "blmax_low_ell.dat",

            "weights": "bweight_low_ell.dat",

            "lmin": 2,

            "reference": (
                "Planck Collaboration (2020), A&A 641, A5, "
                "arXiv:1907.12875 (Commander, l = 2-29)"
            ),

        },

        "reference": (
            "Planck Collaboration (2020), A&A 641, A5, arXiv:1907.12875 "
            "(likelihood); data as redistributed by "
            "heatherprince/planck-lite-py from the Planck Legacy Archive"
        ),

    },

}


#: Planck 2018 CMB lensing -- the reconstructed lensing-potential
#: bandpowers. A different measurement from the temperature and
#: polarization spectra, and the CMB's own handle on how much
#: structure grew between recombination and today.
PLANCK_LENSING_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "lensing2018",

        "data": "bandpowers.dat",

        "covariance": "cov.dat",

        "fiducial_correction": "lensing_fiducial_correction.dat",

        "windows": "window/window{bin}.dat",

        "delta_windows": "lens_delta_window/window{bin}.dat",

        "n_bin": 9,

        "lmax": 2500,

        "ell_range": (8, 400),

        "reference": (
            "Planck Collaboration (2020), A&A 641, A8, arXiv:1807.06210 "
            "(lensing); conservative 8 <= L <= 400 baseline, data as "
            "redistributed by CobayaSampler/planck_supp_data_and_covmats"
        ),

    },

}


#: ACT DR6 CMB lensing. A second, independent reconstruction of the
#: lensing potential -- different telescope, different sky,
#: different pipeline -- and a tighter one than Planck's.
#:
#: ``bins`` is the slice of the released bandpower vector each
#: variant adopts, applied identically to the data, the covariance
#: and the binning matrix.
ACT_LENSING_FILES = {

    "act_baseline": {

        "parent": "cmb",

        "folder": "act_dr6_lensing",

        "data": "clkk_bandpowers_act.txt",

        "binning_matrix": "binning_matrix_act.txt",

        # The CMB-marginalized covariance: it already accounts for
        # the reconstruction's dependence on the primary CMB
        # spectra, which is what lets this be used without the
        # (unshippable) explicit normalization correction. See
        # `likelihoods.act_lensing`.
        "covariance": "covmat_act_cmbmarg.txt",

        "bins": (2, -6),

        "n_sims": 796,

        "lmax": 2999,

        "ell_range": (40, 763),

        "reference": (
            "Madhavacheril et al. (ACT Collaboration, 2024), ApJ 962, 113, "
            "arXiv:2304.05203; Qu et al. (ACT Collaboration, 2024), "
            "ApJ 962, 112, arXiv:2304.05202"
        ),

    },

    "act_extended": {

        "parent": "cmb",

        "folder": "act_dr6_lensing",

        "data": "clkk_bandpowers_act.txt",

        "binning_matrix": "binning_matrix_act.txt",

        "covariance": "covmat_act_cmbmarg.txt",

        "bins": (2, -3),

        "n_sims": 796,

        "lmax": 2999,

        "ell_range": (40, 1250),

        "reference": (
            "Madhavacheril et al. (ACT Collaboration, 2024), ApJ 962, 113, "
            "arXiv:2304.05203; Qu et al. (ACT Collaboration, 2024), "
            "ApJ 962, 112, arXiv:2304.05202 (extended multipole range)"
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


LOWE_FILES = {

    "planck2018": {

        "parent": "cmb",

        "folder": "lowE2018",

        "data": "prob_table.txt",

        "lmin": 2,

        "lmax": 29,

        "step": 1.0e-4,

        "reference": (
            "Planck Collaboration (2020), A&A 641, A5, arXiv:1907.12875 "
            "(SimAll low-l EE); Python-translated table as distributed "
            "by CobayaSampler/planck_native_data"
        ),

    },

}


#: External single-number Gaussian constraints (see
#: :class:`~data.dataset.GaussianPriorDataset`). Each entry names
#: the *quantity* it constrains, which
#: :class:`~likelihoods.priors.GaussianPriorLikelihood` maps to a
#: model prediction.
PRIOR_FILES = {

    "h0": {

        "quantity": "H0",

        "parent": "priors",

        "folder": "h0",

        "versions": {

            "sh0es2022": {
                "data": "h0_sh0es2022.txt",
                "reference": "Riess et al. (2022), ApJ 934, L7, arXiv:2112.04510",
            },

            "sh0es2024": {
                "data": "h0_sh0es2024.txt",
                "reference": "Breuval et al. (2024), ApJ 973, 30, arXiv:2404.08038",
            },

            "tdcosmo2025": {
                "data": "h0_tdcosmo2025.txt",
                "reference": "TDCOSMO Collaboration / Birrer et al. (2025), A&A 704, A63, arXiv:2506.03023",
            },

        },

    },

    "omega_b": {

        "quantity": "omega_b_h2",

        "parent": "priors",

        "folder": "omega_b",

        "versions": {

            "bbn2024": {
                "data": "omega_b_bbn2024.txt",
                "reference": "Schoeneberg (2024), JCAP 06 (2024) 006, arXiv:2401.15054",
            },

            "cooke2018": {
                "data": "omega_b_cooke2018.txt",
                "reference": "Cooke, Pettini & Steidel (2018), ApJ 855, 102, arXiv:1710.11129",
            },

        },

    },

    "tau": {

        "quantity": "tau_reio",

        "parent": "priors",

        "folder": "tau",

        "versions": {

            "planck2018": {
                "data": "tau_planck2018_lowe.txt",
                "reference": "Planck Collaboration (2020), A&A 641, A6, arXiv:1807.06209",
            },

        },

    },

}


#: Flattened ``PRIOR_FILES``, so each prior dataset ("h0",
#: "omega_b", "tau") looks like every other registry to
#: :func:`_validate_version` and :func:`available_versions`.
_PRIOR_REGISTRIES = {

    name: {

        version: {
            **spec,
            "parent": entry["parent"],
            "folder": entry["folder"],
            "quantity": entry["quantity"],
        }

        for version, spec in entry["versions"].items()

    }

    for name, entry in PRIOR_FILES.items()

}


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
# Planck CMB lensing
# ============================================================

def load_planck_lensing(
    version: str = "planck2018",
) -> CMBLensingDataset:
    """
    Load the Planck 2018 CMB lensing bandpowers, their covariance,
    and the two sets of window functions the likelihood needs.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    CMBLensingDataset
    """

    entry = _validate_version("planck_lensing", version)
    dataset_path = _get_dataset_path("planck_lensing", version)

    n_bin = int(entry["n_bin"])
    lmax = int(entry["lmax"])

    # bin, L_min, L_max, L_av, PP, Error, Ahat
    table = _load_txt(dataset_path / entry["data"])

    ell = np.asarray(table[:, 3], dtype=float)
    value = np.asarray(table[:, 4], dtype=float)
    sigma = np.asarray(table[:, 5], dtype=float)

    if len(value) != n_bin:

        raise ValueError(

            f"'{entry['data']}': expected {n_bin} bandpowers, "

            f"found {len(value)}.",

        )

    covariance = _load_covariance(dataset_path / entry["covariance"])

    fiducial = _load_txt(

        dataset_path / entry["fiducial_correction"],

    )[:, 1]

    # Both window sets are stored one file per bin, listing only the
    # multipoles that bin actually touches -- so they are scattered
    # into dense (lmax + 1) rows here rather than read as blocks.
    windows = np.zeros((n_bin, lmax + 1), dtype=float)
    delta_windows = np.zeros((n_bin, 4, lmax + 1), dtype=float)

    for b in range(n_bin):

        path = dataset_path / entry["windows"].format(bin=b + 1)

        _check_file_exists(path)

        rows = np.loadtxt(path, ndmin=2)

        windows[b, rows[:, 0].astype(int)] = rows[:, 1]

        path = dataset_path / entry["delta_windows"].format(bin=b + 1)

        _check_file_exists(path)

        rows = np.loadtxt(path, ndmin=2)

        index = rows[:, 0].astype(int)

        for column in range(4):

            delta_windows[b, column, index] = rows[:, 1 + column]

    return CMBLensingDataset(

        ell=ell,

        value=value,

        sigma=sigma,

        covariance=make_covariance(cov=covariance),

        windows=windows,

        delta_windows=delta_windows,

        fiducial_correction=np.asarray(fiducial, dtype=float),

        lmax=lmax,

        ell_range=tuple(entry["ell_range"]),

        reference=entry["reference"],

    )


# ============================================================
# ACT DR6 CMB lensing
# ============================================================

def load_act_lensing(
    version: str = "act_baseline",
) -> CMBLensingDataset:
    """
    Load the ACT DR6 lensing bandpowers, binning matrix and
    CMB-marginalized covariance.

    Parameters
    ----------
    version : str, optional
        ``"act_baseline"`` or ``"act_extended"``.

    Returns
    -------
    CMBLensingDataset
        With ``spectrum="KK"``: the windows act on the lensing
        *convergence* ``C_L^{kappakappa}``, not on Planck's
        potential convention.
    """

    entry = _validate_version("act_lensing", version)
    dataset_path = _get_dataset_path("act_lensing", version)

    start, end = entry["bins"]
    lmax = int(entry["lmax"])

    bandpowers = np.atleast_1d(

        _load_txt(dataset_path / entry["data"]),

    )

    binning = _load_txt(dataset_path / entry["binning_matrix"])

    full_covariance = _load_covariance(

        dataset_path / entry["covariance"],

    )

    n_total = len(bandpowers)

    if binning.shape[0] != n_total:

        raise ValueError(

            f"'{entry['binning_matrix']}': has {binning.shape[0]} "

            f"rows against {n_total} bandpowers.",

        )

    # The same bins come out of all three, which is the whole point
    # of doing it in one place: dropping them from the data and not
    # from the covariance produces a chi2 that is merely wrong.
    keep = np.arange(n_total)[start:end]

    value = bandpowers[keep]

    covariance = full_covariance[np.ix_(keep, keep)]

    # `standardize` in ACT's own loader pads/trims the binning
    # matrix to l = 0..lmax; the released matrix is at least that
    # wide, so this is the trim half.
    windows = np.zeros((len(keep), lmax + 1), dtype=float)

    width = min(binning.shape[1], lmax + 1)

    windows[:, :width] = binning[np.ix_(keep, np.arange(width))]

    # Effective multipole of each bin, from the binning matrix
    # itself rather than assumed.
    ell = windows @ np.arange(lmax + 1)

    # Hartlap: the covariance is estimated from a finite number of
    # simulations, so its *inverse* is biased high by
    # (n_sim - 1) / (n_sim - n_bin - 2). ACT's own code multiplies
    # the inverse by the reciprocal; dividing the covariance here is
    # algebraically the same and keeps the library's covariance
    # machinery in charge of the inversion.
    n_bin = len(keep)
    n_sims = int(entry["n_sims"])

    hartlap = (n_sims - n_bin - 2.0) / (n_sims - 1.0)

    if hartlap <= 0.0:

        raise ValueError(

            f"Hartlap factor is non-positive for {n_bin} bins from "

            f"{n_sims} simulations; the covariance cannot be "

            f"inverted meaningfully.",

        )

    covariance = covariance / hartlap

    return CMBLensingDataset(

        ell=ell,

        value=value,

        sigma=np.sqrt(np.diag(covariance)),

        covariance=make_covariance(cov=covariance),

        windows=windows,

        lmax=lmax,

        ell_range=tuple(entry["ell_range"]),

        spectrum="KK",

        reference=entry["reference"],

    )


# ============================================================
# Planck low-multipole EE
# ============================================================

def load_planck_lowe(
    version: str = "planck2018",
) -> LowEllEEDataset:
    """
    Load Planck's low-multipole EE probability table.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    LowEllEEDataset
    """

    entry = _validate_version("planck_lowe", version)
    dataset_path = _get_dataset_path("planck_lowe", version)

    table = _load_txt(dataset_path / entry["data"])

    return LowEllEEDataset(

        table=np.asarray(table, dtype=float),

        lmin=int(entry["lmin"]),

        lmax=int(entry["lmax"]),

        step=float(entry["step"]),

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


# ============================================================
# External single-number Gaussian constraints
# ============================================================

def load_gaussian_prior(
    dataset: str,
    version: str | None = None,
) -> GaussianPriorDataset:
    """
    Load a single external Gaussian constraint -- a local
    distance-ladder ``H0``, a BBN ``omega_b h^2``, or a
    reionization ``tau``.

    These enter a fit as one-point datasets rather than as priors
    on the parameter; see
    :class:`~data.dataset.GaussianPriorDataset` for why.

    Parameters
    ----------
    dataset : str
        Which constraint: ``"h0"``, ``"omega_b"`` or ``"tau"``.

    version : str, optional
        Which measurement of it. Defaults to the first entry
        registered for that dataset (``"sh0es2022"``,
        ``"bbn2024"``, ``"planck2018"``).

    Returns
    -------
    GaussianPriorDataset
    """

    if dataset not in _PRIOR_REGISTRIES:

        raise ValueError(

            f"Unknown prior dataset '{dataset}'. "

            f"Available: {list(_PRIOR_REGISTRIES)}",

        )

    if version is None:

        version = next(iter(_PRIOR_REGISTRIES[dataset]))

    entry = _validate_version(dataset, version)
    dataset_path = _get_dataset_path(dataset, version)

    data = np.atleast_1d(

        _load_txt(dataset_path / entry["data"]),

    ).astype(float)

    if data.size != 2:

        raise ValueError(

            f"'{entry['data']}': expected one '<value> <sigma>' "

            f"row, but found {data.size} numbers.",

        )

    value = float(data[0])
    sigma = float(data[1])

    if sigma <= 0.0:

        raise ValueError(

            f"'{entry['data']}': sigma must be positive.",

        )

    return GaussianPriorDataset(

        quantity=entry["quantity"],

        value=value,

        sigma=sigma,

        covariance=make_covariance(sigma=np.array([sigma])),

        reference=entry["reference"],

    )


# ============================================================
# Planck plik_lite binned TT/TE/EE bandpowers
# ============================================================

def _load_plik_covariance(
    path: Path,
    n: int,
) -> np.ndarray:
    """
    Read the ``plik_lite`` bandpower covariance.

    The released file is a single Fortran unformatted record: a
    4-byte length marker, ``n*n`` little-endian float64s, and a
    closing marker. Only the lower triangle is filled in, so the
    upper triangle is mirrored onto it here.

    ``scipy.io.FortranFile`` would read this in one line, but that
    would make SciPy's I/O module a hard import for a dataset most
    fits never touch; the record layout is fixed and three lines of
    NumPy, so it is read directly.
    """

    _check_file_exists(path)

    expected = n * n

    raw = path.read_bytes()

    # 4-byte opening marker + n*n float64s + 4-byte closing marker.
    if len(raw) != 8 + 8 * expected:

        raise ValueError(

            f"'{path.name}': expected a Fortran record holding "

            f"{n}x{n} float64s ({8 + 8 * expected} bytes with its "

            f"markers), but the file is {len(raw)} bytes.",

        )

    # The payload starts 4 bytes in, i.e. half a float64 -- so the
    # offset has to be applied to the byte buffer, not to a float
    # view of it.
    cov = np.frombuffer(

        raw,

        dtype=np.float64,

        count=expected,

        offset=4,

    ).reshape(n, n).copy()

    # Released with only one triangle populated.
    cov = np.tril(cov) + np.tril(cov, -1).T

    return cov


# ------------------------------------------------------------

def _prepend_low_ell_bins(
    folder: Path,
    spec: dict,
    *,
    ell,
    value,
    sigma,
    covariance,
    n_bin,
    blmin,
    blmax,
    weights,
):
    """
    Prepend the two Commander low-multipole temperature bandpowers
    to a ``plik_lite`` data vector.

    ``plik_lite`` starts at l = 30. Planck's own low-l temperature
    likelihood (Commander, l = 2-29) is a separate product, and
    ``planck-lite-py`` distributes a two-bin Gaussian compression of
    it that can be bolted onto the front.

    Three things have to move together, and getting any one of them
    wrong produces a chi2 that is merely wrong:

    - **The data vector.** The two bins go at the *front*, so the
      TT block becomes 217 long while TE and EE are untouched.
    - **The covariance.** The low-l bins are uncorrelated with the
      high-l block and with each other, so the result is
      block-diagonal with ``diag(sigma^2)`` in the top-left corner.
    - **The windows.** The low-l bins have their own, indexed from
      l = 2, and prepending them shifts every high-l TT window
      index by the length of the low-l weight array. TE and EE keep
      the original indexing, which is why the dataset carries a
      separate TT window set (see
      :class:`~data.dataset.CMBSpectrumDataset`).

    Returns the updated ``(ell, value, sigma, covariance, n_bin,
    extra)``, where ``extra`` holds the TT-specific window fields.
    """

    low_ell_table = np.loadtxt(folder / spec["data"])

    ell_low, value_low, sigma_low = (

        np.atleast_2d(low_ell_table).T

    )

    n_low = len(value_low)

    blmin_low = np.loadtxt(folder / spec["blmin"]).astype(int)
    blmax_low = np.loadtxt(folder / spec["blmax"]).astype(int)
    weights_low = np.loadtxt(folder / spec["weights"])

    n_tt, n_te, n_ee = n_bin

    # Data vector: low-l TT in front of high-l TT, then TE and EE.
    def _insert(low, high):

        return np.concatenate([

            low,

            high[:n_tt],

            high[n_tt:],

        ])

    ell = _insert(ell_low, ell)
    value = _insert(value_low, value)
    sigma = _insert(sigma_low, sigma)

    n_total = len(value)

    combined = np.zeros((n_total, n_total), dtype=float)

    combined[:n_low, :n_low] = np.diag(sigma_low ** 2)
    combined[n_low:, n_low:] = covariance

    extra = {

        # The high-l windows are indexed from `lmin`; after
        # prepending `len(weights_low)` low-l weights they start
        # that much further into the flat weight array.
        "blmin_tt": np.concatenate([blmin_low, blmin + len(weights_low)]),

        "blmax_tt": np.concatenate([blmax_low, blmax + len(weights_low)]),

        "weights_tt": np.concatenate([weights_low, weights]),

        "lmin_tt": int(spec["lmin"]),

    }

    return (

        ell,

        value,

        sigma,

        combined,

        (n_tt + n_low, n_te, n_ee),

        extra,

    )


# ------------------------------------------------------------

def load_plik_lite(
    version: str = "planck2018",
    use_low_ell: bool = False,
) -> CMBSpectrumDataset:
    """
    Load the Planck 2018 ``plik_lite`` binned TT/TE/EE bandpowers,
    their joint covariance, and the binning operator that produced
    them.

    This is the *measured CMB power spectrum*, not the
    three-number distance-prior compression that
    :func:`load_planck` returns -- see
    :class:`~likelihoods.planck_lite.PlanckLiteLikelihood` for
    what that buys and what it costs.

    ``plik_lite`` is the foreground-marginalized variant of the
    Planck high-l likelihood: the ~20 nuisance parameters
    describing dust, point sources and the SZ effect have already
    been marginalized over by the Planck team, leaving a single
    calibration parameter (``A_planck``). That is what makes it
    usable outside a full Planck pipeline.

    Parameters
    ----------
    version : str, optional
        Dataset version.

    Returns
    -------
    CMBSpectrumDataset
    """

    entry = _validate_version("planck_lite", version)
    dataset_path = _get_dataset_path("planck_lite", version)

    ell, value, sigma = np.loadtxt(

        dataset_path / entry["data"],

        unpack=True,

    )

    n = len(value)

    if sum(entry["n_bin"]) != n:

        raise ValueError(

            f"'{entry['data']}': registry declares "

            f"{entry['n_bin']} bandpowers ({sum(entry['n_bin'])} "

            f"total) but the file holds {n}.",

        )

    covariance = _load_plik_covariance(

        dataset_path / entry["covariance"],

        n=n,

    )

    blmin = np.loadtxt(dataset_path / entry["blmin"]).astype(int)
    blmax = np.loadtxt(dataset_path / entry["blmax"]).astype(int)
    weights = np.loadtxt(dataset_path / entry["weights"])

    n_bin = tuple(entry["n_bin"])

    extra = {}

    if use_low_ell:

        (
            ell, value, sigma, covariance, n_bin, extra,
        ) = _prepend_low_ell_bins(

            dataset_path / entry["low_ell"]["folder"],

            entry["low_ell"],

            ell=ell,

            value=value,

            sigma=sigma,

            covariance=covariance,

            n_bin=n_bin,

            blmin=blmin,

            blmax=blmax,

            weights=weights,

        )

    return CMBSpectrumDataset(

        ell=ell,

        value=value,

        sigma=sigma,

        covariance=make_covariance(cov=covariance),

        n_bin=n_bin,

        blmin=blmin,

        blmax=blmax,

        weights=weights,

        lmin=int(entry["lmin"]),

        lmax=int(entry["lmax"]),

        reference=entry["reference"],

        **extra,

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
