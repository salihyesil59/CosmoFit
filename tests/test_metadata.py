"""
Dataset metadata: one source, and the conflicts that follow from it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from CosmoFit.core import get_model
from CosmoFit.data import metadata
from CosmoFit.data.metadata import DATASETS, FAMILIES, SAMPLES, conflicts
from CosmoFit.likelihoods.native import NATIVE_LIKELIHOODS
from CosmoFit.stats.fitter import DATASET_REGISTRY


#: The pairs the hand-written list held before they were derived. A
#: change to a dataset's samples that changes this is a change to the
#: science, and should be made here on purpose.
EXPECTED = {
    frozenset(pair) for pair in [
        ("desi", "sdss_bao"), ("desi", "sdss_fsbao"), ("sdss_bao", "sdss_fsbao"),
        ("sdss_fsbao", "fsigma8"), ("sdss_bao", "fsigma8"), ("desi", "eboss_lya"),
        ("desi", "eboss_elg"), ("desi", "eboss_elg_fs"), ("eboss_elg", "eboss_elg_fs"),
        ("fsigma8", "eboss_elg_fs"), ("pantheon", "des_sn5yr"), ("pantheon", "union3"),
        ("des_sn5yr", "union3"), ("planck_lensing", "act_lensing"),
        ("fsigma8", "bao_lowz"), ("planck_lowe", "tau"), ("planck", "planck_lite"),
    ]
}


def test_conflicts_follow_from_the_samples():

    assert {frozenset(p) for p in metadata.CONFLICTING_DATASETS} == EXPECTED

    # Independent neighbours are not conflicts.
    assert not conflicts(["desi", "bao_lowz"])
    assert not conflicts(["planck", "planck_lowe"])


def test_a_new_dataset_inherits_its_conflicts(monkeypatch):

    monkeypatch.setitem(DATASETS, "boss_dr12_rsd", metadata.DatasetInfo(
        likelihood="rsd.boss", label="BOSS RSD", title="BOSS DR12 RSD",
        family="growth", samples=frozenset({"boss_eboss_galaxies"}),
    ))

    found = conflicts(DATASETS)

    assert {
        frozenset(pair) for pair in found if "boss_dr12_rsd" in pair
    } == {
        frozenset({"boss_dr12_rsd", other})
        for other in ("sdss_bao", "sdss_fsbao", "fsigma8")
    }

    assert "BOSS DR12" in found[("sdss_bao", "boss_dr12_rsd")]


def test_every_dataset_is_described_once():

    assert set(DATASETS) == set(DATASET_REGISTRY)

    for key, info in DATASETS.items():

        assert NATIVE_LIKELIHOODS[info.likelihood].dataset == key
        assert info.family in FAMILIES
        assert info.samples <= set(SAMPLES), key
        assert info.label and info.title

    # Every sample is shared, or it would be no reason for anything.
    for sample in SAMPLES:
        assert sum(sample in info.samples for info in DATASETS.values()) >= 2, sample


def test_references_are_the_verified_ones():
    """Every arXiv identifier is one REFERENCES.md lists."""

    listed = set(re.findall(
        r"arXiv:(\d{4}\.\d{4,5})",
        (Path(__file__).parents[1] / "REFERENCES.md").read_text(encoding="utf-8"),
    ))

    for key, info in DATASETS.items():

        assert info.references, key
        assert set(info.references) <= listed, key


def test_native_likelihoods_warn_about_shared_samples():

    base = {
        "theory": {"background": None, "early_universe": None},
        "params": {"H0": 67.5, "Omega_m": 0.31, "Omega_b": 0.049},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_model({**base, "likelihood": {"bao.desi": None, "bao.sdss": None}})

    messages = [str(w.message) for w in caught]

    assert any("'bao.desi' and 'bao.sdss' should not be combined" in m for m in messages)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_model({**base, "likelihood": {"bao.desi": None, "bao.lowz": None}})

    assert not [w for w in caught if "should not be combined" in str(w.message)]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_has_members(family):

    assert any(info.family == family for info in DATASETS.values())
