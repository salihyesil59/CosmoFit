"""
What each dataset is: one place for its names, its family, its sources,
and the samples it is built from.

Everything else that needs to know about a dataset reads it here: the
legend labels figures print, the titles and groups the GUI shows, the
native likelihood's name, and -- the part that matters for the science
-- which datasets must not be combined.

Which datasets must not be combined
-----------------------------------
That used to be a hand-written list of pairs, and the GUI kept its own
copy, which had drifted to 7 of the 17 by the time anyone compared
them. Here it follows from what each dataset *is*: every dataset lists
the samples it measures -- the same galaxies, the same patch of sky and
redshift range, the same supernovae, the same CMB measurement in
another form -- and two datasets that share a sample are not
independent. A new dataset declares its samples, and every pair it
should not be combined with follows; no list to remember.

A sample counts only when combining the two would multiply a likelihood
by part of itself. Neighbouring but independent data do not share one:
the low-redshift BAO of 6dFGS and SDSS MGS lies below anything DESI or
BOSS measured, and adds to either.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


__all__ = [
    "DatasetInfo",
    "SAMPLES",
    "FAMILIES",
    "DATASETS",
    "conflicts",
    "CONFLICTING_DATASETS",
]


#: Every sample a dataset can be built from, with what it is -- phrased
#: to complete "Both use ...".
SAMPLES = {

    "boss_eboss_galaxies": (
        "the same BOSS DR12 and eBOSS DR16 galaxies and quasars, so their "
        "geometry and growth are correlated (0.19 to 0.64 within a "
        "redshift bin in SDSS's released covariance) -- the BAO+FS "
        "consensus measures both jointly"
    ),

    "boss_eboss_volume": (
        "the sky and redshift range BOSS and eBOSS mapped, whose "
        "structure DESI observes again"
    ),

    "eboss_lya_forest": (
        "the eBOSS Lyman-alpha forest, whose quasars DESI largely "
        "re-observes at z ~ 2.33; DESI and eBOSS publish a joint "
        "Lyman-alpha likelihood for exactly this reason"
    ),

    "eboss_elg_volume": (
        "the eBOSS emission-line-galaxy footprint, which DESI's ELG "
        "sample covers again"
    ),

    "eboss_elg_galaxies": (
        "the same eBOSS DR16 emission-line galaxies, once for the BAO "
        "scale and once for the full anisotropic shape"
    ),

    "eboss_growth_z085": (
        "growth measured in the same eBOSS volume near z ~ 0.85, from "
        "the compilation's DR14 quasars and from the DR16 ELGs"
    ),

    "sdss_mgs_galaxies": (
        "the SDSS DR7 Main Galaxy Sample: the compilation's z = 0.15 "
        "growth rate (Howlett et al. 2015) and the MGS BAO come from "
        "the same galaxies"
    ),

    "des_lowz_supernovae": (
        "the low-redshift supernovae DES-SN5YR anchors to (about 11% of "
        "it), which Pantheon+ also compiles"
    ),

    "literature_supernovae": (
        "substantially the same literature supernovae"
    ),

    "des_supernovae": (
        "the DES supernovae, which Union3's high-redshift half overlaps"
    ),

    "cmb_lensing_sky": (
        "the same CMB lensing: ACT's lensing map overlaps Planck's on the "
        "sky, so the two reconstructions are correlated, and ACT publish "
        "a joint variant for this"
    ),

    "planck_lowl_ee": (
        "Planck's low-l EE likelihood, which the 'tau' prior compresses"
    ),

    "planck_highl_spectra": (
        "Planck's TT/TE/EE spectra, which the distance priors compress"
    ),
}


#: Probe families, in the order they are shown.
FAMILIES = {
    "expansion": "Expansion rate",
    "bao": "BAO (standard ruler)",
    "sn": "Supernovae (standard candle)",
    "cmb": "CMB",
    "growth": "Growth of structure",
    "external": "External measurements",
}


@dataclass(frozen=True)
class DatasetInfo:
    """
    Attributes
    ----------
    likelihood : str
        Name of the native likelihood (:mod:`likelihoods.native`).
    label : str
        Short, for figure legends; several are joined to describe a fit.
    title : str
        Descriptive, for the GUI.
    family : str
        A key of :data:`FAMILIES`.
    samples : frozenset
        Keys of :data:`SAMPLES`.
    references : tuple
        arXiv identifiers of the measurement (see ``REFERENCES.md``).
    needs_camb : bool
        Whether its predictions run a Boltzmann code.
    """

    likelihood: str
    label: str
    title: str
    family: str
    samples: frozenset = frozenset()
    references: tuple = ()
    needs_camb: bool = False


def _info(likelihood, label, title, family, samples=(), references=(), needs_camb=False):
    return DatasetInfo(
        likelihood, label, title, family, frozenset(samples), tuple(references),
        needs_camb,
    )


#: Every dataset, by its registry key, in the order the GUI lists them.
DATASETS = {

    "cc": _info(
        "cc.chronometers", "CC", "Cosmic Chronometers (CC)", "expansion",
        references=("2301.09591",),
    ),

    "desi": _info(
        "bao.desi", "DESI", "DESI BAO", "bao",
        samples=("boss_eboss_volume", "eboss_lya_forest", "eboss_elg_volume"),
        references=("2503.14738", "2404.03002"),
    ),

    "sdss_bao": _info(
        "bao.sdss", "SDSS", "SDSS BAO (BOSS DR12 + eBOSS DR16)", "bao",
        samples=("boss_eboss_galaxies", "boss_eboss_volume"),
        references=("1607.03155", "2007.08991"),
    ),

    "sdss_fsbao": _info(
        "bao.sdss_fullshape", "SDSS BAO+FS",
        "SDSS BAO + full shape (BOSS DR12 + eBOSS DR16)", "bao",
        samples=("boss_eboss_galaxies", "boss_eboss_volume"),
        references=("2007.08991",),
    ),

    "bao_lowz": _info(
        "bao.lowz", "6dFGS + MGS", "Low-z BAO (6dFGS + SDSS MGS)", "bao",
        samples=("sdss_mgs_galaxies",),
        references=("1106.3366", "1409.3242"),
    ),

    "eboss_elg": _info(
        "bao.eboss_elg", "eBOSS ELG", "eBOSS DR16 ELG BAO (tabulated)", "bao",
        samples=("eboss_elg_volume", "eboss_elg_galaxies"),
        references=("2007.09008",),
    ),

    "eboss_elg_fs": _info(
        "bao.eboss_elg_fullshape", "eBOSS ELG (full shape)",
        "eBOSS DR16 ELG full shape (tabulated)", "bao",
        samples=("eboss_elg_volume", "eboss_elg_galaxies", "eboss_growth_z085"),
        references=("2007.09008",),
    ),

    "eboss_lya": _info(
        "bao.eboss_lya", r"eBOSS Ly$\alpha$", "eBOSS DR16 Lyman-α BAO (tabulated)",
        "bao",
        samples=("eboss_lya_forest",),
        references=("2007.08995",),
    ),

    "pantheon": _info(
        "sn.pantheonplus", "Pantheon+", "Pantheon+ (SNe Ia)", "sn",
        samples=("des_lowz_supernovae", "literature_supernovae"),
        references=("2202.04077",),
    ),

    "des_sn5yr": _info(
        "sn.des_sn5yr", "DES-SN5YR", "DES-SN5YR (SNe Ia)", "sn",
        samples=("des_lowz_supernovae", "des_supernovae"),
        references=("2511.07517", "2401.02929"),
    ),

    "union3": _info(
        "sn.union3", "Union3", "Union3 (SNe Ia, binned)", "sn",
        samples=("literature_supernovae", "des_supernovae"),
        references=("2311.12098",),
    ),

    "planck": _info(
        "cmb.distance_priors", "Planck", "Planck 2018 CMB (distance priors)", "cmb",
        samples=("planck_highl_spectra",),
        references=("1808.05724",),
    ),

    "planck_lite": _info(
        "cmb.planck_lite", "Planck TTTEEE", "Planck 2018 CMB (full TT/TE/EE spectra)",
        "cmb",
        samples=("planck_highl_spectra",),
        references=("1907.12875", "1807.06209"),
        needs_camb=True,
    ),

    "planck_lowe": _info(
        "cmb.planck_lowe", "Planck lowE", "Planck 2018 low-ℓ EE (τ, tabulated)", "cmb",
        samples=("planck_lowl_ee",),
        references=("1907.12875",),
        needs_camb=True,
    ),

    "planck_lensing": _info(
        "cmb.planck_lensing", "Planck lensing", "Planck 2018 CMB lensing", "cmb",
        samples=("cmb_lensing_sky",),
        references=("1807.06210",),
        needs_camb=True,
    ),

    "act_lensing": _info(
        "cmb.act_lensing", "ACT DR6 lensing", "ACT DR6 CMB lensing", "cmb",
        samples=("cmb_lensing_sky",),
        references=("2304.05203", "2304.05202"),
        needs_camb=True,
    ),

    "fsigma8": _info(
        "rsd.fsigma8", r"$f\sigma_8$", "Growth rate fσ₈(z) (RSD)", "growth",
        samples=("boss_eboss_galaxies", "eboss_growth_z085", "sdss_mgs_galaxies"),
        references=("1806.10822",),
    ),

    "s8": _info(
        "lss.s8", r"$S_8$", "S₈ weak-lensing prior", "growth",
        references=("2007.15633", "2105.13549", "2503.19441", "2304.00702", "2305.17173"),
    ),

    "h0": _info(
        "external.h0", r"$H_0$", "Local H₀ (distance ladder)", "external",
        references=("2112.04510", "2404.08038", "2506.03023", "2408.06153"),
    ),

    "omega_b": _info(
        "external.bbn", "BBN", "BBN prior on ω_b", "external",
        references=("2401.15054", "1710.11129"),
    ),

    "tau": _info(
        "external.tau", r"$\tau$", "Reionization τ prior", "external",
        samples=("planck_lowl_ee",),
        references=("1807.06209",),
    ),
}


def conflicts(names) -> dict:
    """
    The pairs among ``names`` (dataset keys) that share a sample, each
    with why: ``{(first, second): reason}``, in the order given.
    """

    found = {}

    for first, second in combinations(dict.fromkeys(names), 2):

        shared = sorted(DATASETS[first].samples & DATASETS[second].samples)

        if shared:
            found[(first, second)] = (
                f"Both use {'; and '.join(SAMPLES[s] for s in shared)}."
            )

    return found


#: Every pair of datasets that must not be combined, with why.
CONFLICTING_DATASETS = conflicts(DATASETS)
