"""
The new core runs the library's models and datasets exactly as before.

Two references:

* ``tests/data/golden_chi2.json`` -- every built-in model's chi2 on
  every non-CAMB dataset, written by ``tools/make_golden_chi2.py``
  straight from the original likelihood classes. This file is what
  later phases' ports must reproduce.
* ``Fitter``, at sampled points with the options that change what is
  computed (``compute_rd``, dataset versions, a free nuisance
  parameter).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

import CosmoFit
from CosmoFit import Fitter
from CosmoFit.core import ComponentError, get_model, run


GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "golden_chi2.json").read_text(
        encoding="utf-8",
    )
)


def _quiet(function, *args, **kwargs):

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        return function(*args, **kwargs)


# ============================================================
# The golden reference
# ============================================================

@pytest.mark.parametrize("model_name", sorted(GOLDEN["models"]))
def test_core_reproduces_the_golden_chi2(model_name):

    entry = GOLDEN["models"][model_name]

    finite = sorted(k for k, v in entry["chi2"].items() if not isinstance(v, dict))
    failing = sorted(k for k, v in entry["chi2"].items() if isinstance(v, dict))

    def build(datasets):

        return _quiet(get_model, {
            "theory": {"legacy_cosmology": {"model": model_name}},
            "likelihood": {name: None for name in datasets},
            "params": dict(entry["point"]),
        })

    result = build(finite).logposterior({})

    assert result.rejected is None

    for dataset in finite:

        expected = entry["chi2"][dataset]

        assert result.chi2[dataset] == pytest.approx(expected, rel=1e-10), dataset

    # Where the original code had no finite chi2 (a point outside a
    # tabulated likelihood's grid, say), the core rejects the point.
    for dataset in failing:

        assert build([dataset]).logposterior({}).rejected is not None, dataset


# ============================================================
# Against Fitter
# ============================================================

CASES = [
    # model, datasets, sampled point, fixed, theory options, likelihood options
    ("CPL", ["cc", "desi", "pantheon", "planck"],
     {"H0": 68.0, "Omega_m": 0.31, "w0": -0.9, "wa": -0.3}, {"rd": 147.1},
     {}, {}),
    ("LCDM", ["desi", "omega_b", "des_sn5yr"],
     {"H0": 67.5, "Omega_m": 0.30, "Omega_b": 0.05}, {},
     {"compute_rd": True}, {"desi": {"version": "desi2024"}}),
    ("LCDM", ["pantheon", "cc"],
     {"H0": 73.0, "Omega_m": 0.33, "MB": -19.25}, {},
     {}, {"pantheon": {"include_cepheid": True}}),
    ("IDE", ["cc", "sdss_fsbao", "union3"],
     {"H0": 69.0, "Omega_m": 0.30, "xi": 0.05, "sigma8": 0.8}, {"rd": 148.0},
     {}, {}),
    ("LsCDM", ["desi", "fsigma8", "bao_lowz"],
     {"H0": 70.0, "Omega_m": 0.30, "z_dagger": 2.0}, {"rd": 145.0},
     {}, {}),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] + ":" + "+".join(c[1]) for c in CASES])
def test_core_matches_fitter(case):

    model_name, datasets, point, fixed, theory_options, lk_options = case

    model_cls = getattr(CosmoFit, model_name)

    params = {
        name: {"prior": {"min": value - 1.0, "max": value + 1.0}}
        for name, value in point.items()
    }
    params.update(fixed)

    model = _quiet(get_model, {
        "theory": {"legacy_cosmology": {"model": model_name, **theory_options}},
        "likelihood": {d: lk_options.get(d) for d in datasets},
        "params": params,
    })

    fitter = _quiet(
        Fitter,
        model=model_cls,
        datasets=datasets,
        free_params=list(point),
        initial={**point, **fixed},
        dataset_kwargs=lk_options,
        compute_rd=theory_options.get("compute_rd", False),
    )

    for shift in (0.0, 0.01):

        moved = {k: v + shift * (1 if i % 2 else -1)
                 for i, (k, v) in enumerate(point.items())}

        new = model.logposterior(moved).chi2["total"]
        old = fitter.chi2([moved[k] for k in point])

        assert new == pytest.approx(old, rel=1e-12)


# ============================================================
# Options and refusals
# ============================================================

def test_computed_rd_is_not_an_input():

    with pytest.raises(ComponentError, match="No theory or likelihood takes"):

        _quiet(get_model, {
            "theory": {"legacy_cosmology": {"model": "LCDM", "compute_rd": True}},
            "likelihood": {"desi": None},
            "params": {"H0": 68.0, "Omega_m": 0.3, "rd": 147.0},
        })


def test_derived_sound_horizon():

    model = _quiet(get_model, {
        "theory": {"legacy_cosmology": {"model": "LCDM", "compute_rd": True}},
        "likelihood": {"desi": None},
        "params": {
            "H0": 68.0, "Omega_m": 0.3, "Omega_b": 0.049,
            "rd_computed": {"derived": True},
            "rd_h": {"derived": "lambda rd_computed, H0: rd_computed * H0 / 100"},
        },
    })

    derived = model.logposterior({}).derived

    expected = model.theories["legacy_cosmology"].cosmology.sound_horizon.rd_computed()

    assert derived["rd_computed"] == pytest.approx(expected)
    assert derived["rd_h"] == pytest.approx(expected * 0.68)


def test_combining_overlapping_datasets_warns():

    with pytest.warns(UserWarning, match="should not be combined"):

        get_model({
            "theory": {"legacy_cosmology": {"model": "LCDM"}},
            "likelihood": {"desi": None, "sdss_bao": None},
            "params": {"H0": 68.0, "Omega_m": 0.3},
        })


def test_unknown_theory_option_is_refused():

    with pytest.raises(ComponentError, match="unknown option"):

        get_model({
            "theory": {"legacy_cosmology": {"model": "LCDM", "compute_rdd": True}},
            "likelihood": {"cc": None},
            "params": {"H0": 68.0, "Omega_m": 0.3},
        })


def test_an_unphysical_point_is_rejected_and_counted():
    """
    LsCDM raises for E(z)^2 <= 0; the old path turned that into -inf
    silently. The new core says why, and counts it.
    """

    model = _quiet(get_model, {
        "theory": {"legacy_cosmology": {"model": "LsCDM"}},
        "likelihood": {"cc": None},
        "params": {
            "H0": 68.0, "z_dagger": 1.0,
            "Omega_m": {"prior": {"min": 0.01, "max": 0.6}},
        },
    })

    result = model.logposterior({"Omega_m": 0.02})

    assert result.logpost == -math.inf
    assert "ValueError" in result.rejected
    assert sum(model.rejections.values()) == 1


def test_minimize_agrees_with_best_fit():

    info = {
        "theory": {"legacy_cosmology": {"model": "LCDM"}},
        "likelihood": {"cc": None, "desi": None},
        "params": {
            "H0": {"prior": {"min": 50, "max": 90}},
            "Omega_m": {"prior": {"min": 0.1, "max": 0.6}},
            "rd": {"prior": {"min": 120, "max": 170}},
        },
        "sampler": {"minimize": {"ignore_prior": True, "starts": 2}},
    }

    _, sampler = _quiet(run, info, seed=0)

    new = sampler.products()

    fitter = _quiet(
        Fitter, model=CosmoFit.LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "rd"],
        initial={"H0": 70.0, "Omega_m": 0.3, "rd": 147.0},
    )

    old = _quiet(fitter.best_fit)

    assert new["chi2"]["total"] == pytest.approx(old.fun, abs=1e-3)
    assert new["chi2"]["total"] <= old.fun + 1e-6
