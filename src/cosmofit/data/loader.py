"""
Dataset loading: one function per dataset, ``load_<name>(version=...)``,
returning the dataset object its likelihood reads.

The loaders live in :mod:`data.loaders`, one module per family of
probes -- chronometers, BAO, supernovae, growth, the CMB, external
priors. This module re-exports every name they define, private ones
included, so ``cosmofit.data.loader.load_pantheon`` and every import
written against the single 3400-line module this used to be keep
working.
"""

from __future__ import annotations

from .loaders._common import (
    DATA_DIR,
    _validate_version,
    _get_dataset_path,
    _check_file_exists,
    _load_txt,
    _load_covariance,
)

from .loaders.cc import (
    CC_FILES,
    _cc_covariance,
    load_cc,
)

from .loaders.bao import (
    DESI_FILES,
    BAO_LOWZ_FILES,
    SDSS_BAO_FILES,
    EBOSS_ELG_FILES,
    EBOSS_LYA_FILES,
    EBOSS_ELG_FS_FILES,
    SDSS_FSBAO_FILES,
    load_desi,
    _load_blockdiag_bao,
    load_sdss_bao,
    load_sdss_fsbao,
    load_bao_lowz,
    _load_grid_npz,
    load_eboss_table,
)

from .loaders.sn import (
    PANTHEON_FILES,
    DES_SN5YR_FILES,
    UNION3_FILES,
    _load_pantheon_covariance,
    PANTHEON_Z_MIN,
    _build_pantheon_mask,
    _load_snana_fitres,
    _load_des_precision_covariance,
    load_pantheon,
    load_des_sn5yr,
    _load_cosmomc_covmat,
    load_union3,
)

from .loaders.growth import (
    GROWTH_FILES,
    S8_FILES,
    load_fsigma8,
    load_s8,
)

from .loaders.cmb import (
    PLANCK_FILES,
    PLIK_LITE_FILES,
    PLANCK_LENSING_FILES,
    ACT_LENSING_FILES,
    LOWE_FILES,
    load_planck_lensing,
    load_act_lensing,
    load_planck_lowe,
    _load_plik_covariance,
    _prepend_low_ell_bins,
    load_plik_lite,
    load_planck,
)

from .loaders.priors import (
    PRIOR_FILES,
    _PRIOR_REGISTRIES,
    load_gaussian_prior,
)

from .loaders._index import (
    _REGISTRIES,
    available_versions,
    dataset_reference,
    available_datasets,
)

__all__ = [
    "DATA_DIR",
    "CC_FILES",
    "load_cc",
    "DESI_FILES",
    "BAO_LOWZ_FILES",
    "SDSS_BAO_FILES",
    "EBOSS_ELG_FILES",
    "EBOSS_LYA_FILES",
    "EBOSS_ELG_FS_FILES",
    "SDSS_FSBAO_FILES",
    "load_desi",
    "load_sdss_bao",
    "load_sdss_fsbao",
    "load_bao_lowz",
    "load_eboss_table",
    "PANTHEON_FILES",
    "DES_SN5YR_FILES",
    "UNION3_FILES",
    "PANTHEON_Z_MIN",
    "load_pantheon",
    "load_des_sn5yr",
    "load_union3",
    "GROWTH_FILES",
    "S8_FILES",
    "load_fsigma8",
    "load_s8",
    "PLANCK_FILES",
    "PLIK_LITE_FILES",
    "PLANCK_LENSING_FILES",
    "ACT_LENSING_FILES",
    "LOWE_FILES",
    "load_planck_lensing",
    "load_act_lensing",
    "load_planck_lowe",
    "load_plik_lite",
    "load_planck",
    "PRIOR_FILES",
    "load_gaussian_prior",
    "available_versions",
    "dataset_reference",
    "available_datasets",
]
