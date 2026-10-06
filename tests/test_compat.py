"""
``Fitter.to_info``: the 1.x interface written as a 2.0 input.

``exact=True`` must give the fitter's chi2 at every point -- it runs the
fitter's own classes. The native input runs the new theories, which put
radiation in the late-time expansion; with that switched off it must
give the fitter's chi2 too, so that what remains is radiation and
nothing else.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import yaml

import cosmofit
from cosmofit import Fitter
from cosmofit.compat import RENAMED
from cosmofit.core import get_model, run


def quiet(function, *args, **kwargs):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


CASES = {
    "lcdm": dict(
        model=cosmofit.LCDM, datasets=["cc", "desi", "pantheon"],
        free_params=["H0", "Omega_m", "rd"],
        initial={"H0": 68.0, "Omega_m": 0.3, "rd": 147.0},
    ),
    "cpl_rd_computed": dict(
        model=cosmofit.CPL, datasets=["desi", "des_sn5yr", "planck", "omega_b"],
        free_params=["H0", "Omega_m", "Omega_b", "w0", "wa"],
        initial={"H0": 68.0, "Omega_m": 0.3, "w0": -0.9, "wa": -0.3},
        compute_rd=True,
    ),
    "hde_growth": dict(
        model=cosmofit.HDE, datasets=["cc", "fsigma8", "s8", "sdss_fsbao"],
        free_params=["H0", "Omega_m", "sigma8", "c_hde"],
        initial={"H0": 68.0, "Omega_m": 0.3, "sigma8": 0.8, "c_hde": 0.7},
    ),
    "gcg_renamed": dict(
        model=cosmofit.GCG, datasets=["union3", "bao_lowz", "h0"],
        free_params=["H0", "Omega_m", "A_s", "alpha"],
        initial={"H0": 70.0, "Omega_m": 0.3, "A_s": 0.75, "alpha": 0.1, "rd": 147.0},
        dataset_kwargs={"h0": {"version": "sh0es2022"}},
    ),
}


def points(fitter, n=4):

    rng = np.random.default_rng(0)
    centre = np.array([fitter.initial[p] for p in fitter.free_params])

    return [centre * (1.0 + 0.01 * rng.standard_normal(len(centre))) for _ in range(n)]


def new_chi2(model, fitter, theta, rename=None):

    rename = rename or {}
    point = {rename.get(p, p): t for p, t in zip(fitter.free_params, theta)}

    return model.logposterior(point).chi2["total"]


# ============================================================

@pytest.mark.parametrize("case", sorted(CASES))
def test_exact_gives_the_fitters_chi2(case):

    fitter = quiet(Fitter, **CASES[case])
    model = quiet(get_model, fitter.to_info(exact=True))

    for theta in points(fitter):
        assert new_chi2(model, fitter, theta) == pytest.approx(fitter.chi2(theta), rel=1e-12)


@pytest.mark.parametrize("case", ["lcdm", "hde_growth", "gcg_renamed"])
def test_native_differs_by_radiation_alone(case):

    fitter = quiet(Fitter, **CASES[case])
    info = fitter.to_info()
    rename = RENAMED.get(fitter.model_cls.__name__, {})

    with_radiation = quiet(get_model, info)

    info["theory"]["background"]["radiation"] = False
    without = quiet(get_model, info)

    for theta in points(fitter):

        old = fitter.chi2(theta)

        assert new_chi2(without, fitter, theta, rename) == pytest.approx(old, abs=1e-5)
        assert abs(new_chi2(with_radiation, fitter, theta, rename) - old) < 2.0


def test_native_writes_the_new_names():

    fitter = quiet(Fitter, **CASES["gcg_renamed"])
    info = fitter.to_info()

    assert info["theory"]["background"] == {"dark_energy": "gcg"}
    assert set(info["likelihood"]) == {"sn.union3", "bao.lowz", "external.h0"}
    assert info["likelihood"]["bao.lowz"] == {"rd": "free"}
    assert info["likelihood"]["external.h0"] == {"version": "sh0es2022"}

    params = info["params"]

    assert "A_gcg" in params and "alpha_gcg" in params and "A_s" not in params
    assert params["A_gcg"]["prior"] == {"min": fitter.prior.lower[2], "max": fitter.prior.upper[2]}
    assert params["rd"] == 147.0

    # What nothing takes is left out: GCG reads no w0, and no CMB here.
    assert "w0" not in params and "A_planck" not in params


def test_computing_rd_adds_the_early_universe():

    info = quiet(Fitter, **CASES["cpl_rd_computed"]).to_info()

    assert set(info["theory"]) == {"background", "early_universe"}
    assert info["likelihood"]["bao.desi"] == {}
    assert "rd" not in info["params"]


def test_the_input_is_yaml_and_runs(tmp_path):

    fitter = quiet(Fitter, **CASES["lcdm"])

    info = fitter.to_info()
    info["sampler"] = {"minimize": None}

    path = tmp_path / "lcdm.yaml"
    path.write_text(yaml.safe_dump(info), encoding="utf-8")

    _, sampler = quiet(run, str(path))

    point = sampler.products()["point"]

    best = quiet(fitter.best_fit)

    # Radiation moves the best fit a little, not the answer.
    assert point["Omega_m"] == pytest.approx(best.x[1], abs=0.01)


def test_a_model_outside_the_library_is_wrapped():

    def E(p, z):
        return np.sqrt(p["Omega_m"] * (1 + z) ** 3 + (1 - p["Omega_m"]) * (1 + z) ** 0.3)

    custom = cosmofit.define_model("Thawing", E=E)

    fitter = quiet(
        Fitter, model=custom, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"], initial={"H0": 68.0, "Omega_m": 0.3},
    )

    info = fitter.to_info()

    assert info["theory"]["background"]["dark_energy"]["name"] == "legacy"
    assert info["theory"]["background"]["radiation"] is False

    model = quiet(get_model, info)

    for theta in points(fitter):
        assert new_chi2(model, fitter, theta) == pytest.approx(fitter.chi2(theta), abs=1e-5)

    rd_fitter = quiet(
        Fitter, model=custom, datasets=["desi", "omega_b"],
        free_params=["H0", "Omega_m"], initial={"H0": 68.0, "Omega_m": 0.3},
        compute_rd=True,
    )

    with pytest.raises(ValueError, match="exact=True"):
        rd_fitter.to_info()

    model = quiet(get_model, rd_fitter.to_info(exact=True))

    for theta in points(rd_fitter):
        assert new_chi2(model, rd_fitter, theta) == pytest.approx(rd_fitter.chi2(theta), rel=1e-12)


def test_a_free_parameter_nothing_takes_is_refused():

    fitter = quiet(
        Fitter, model=cosmofit.ADE, datasets=["cc"],
        free_params=["H0", "Omega_m", "n_ade"],
        initial={"H0": 68.0, "Omega_m": 0.3, "n_ade": 3.0},
    )

    with pytest.raises(ValueError, match="'Omega_m'"):
        fitter.to_info()
