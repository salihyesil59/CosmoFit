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


# ============================================================
# A Fitter's calculations on the core
# ============================================================

from cosmofit.compat import (  # noqa: E402
    CoreChains,
    evidence_on_core,
    fisher_on_core,
    profile_on_core,
    sample_on_core,
)


def lcdm():

    fitter = quiet(
        Fitter, model=cosmofit.LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "rd"],
        initial={"H0": 68.0, "Omega_m": 0.3, "rd": 147.0},
    )

    quiet(fitter.best_fit)

    return fitter


@pytest.mark.parametrize("sampler, options", [
    ("mcmc", {"Rminus1_stop": 0.02}),
    ("emcee", {"walkers": 16, "max_samples": 3000}),
])
def test_core_chains_are_the_fitters_own(sampler, options):

    fitter = lcdm()

    quiet(sample_on_core, fitter, sampler, options, seed=1)

    assert isinstance(fitter.sampler, CoreChains)

    chain = fitter.sampler.get_chain()

    assert chain.shape[1:] == (fitter.sampler.nwalkers, 3)
    assert fitter.flat_samples().shape[1] == 3
    assert 0 < fitter.burnin < fitter.sampler.iteration

    summary = quiet(fitter.summary)

    # CC + DESI: Omega_m to about 0.01.
    assert summary["Omega_m"]["median"] == pytest.approx(0.30, abs=0.03)
    assert 0.003 < summary["Omega_m"]["plus"] < 0.03

    convergence = fitter.convergence()

    assert "stopping_rule" in convergence
    assert convergence["converged"] == fitter.sampler.stopped_by["converged"]

    # The figures read it like any chain.
    assert quiet(fitter.plots.corner) is not None


def test_a_saved_run_is_read_back_not_sampled_again(tmp_path):

    output = str(tmp_path / "lcdm" / "run")

    first = lcdm()
    quiet(sample_on_core, first, "mcmc", {"Rminus1_stop": 0.05}, output=output, seed=2)

    assert first.sampler.stopped_by["converged"]
    assert not first.sampler.reused

    rows = np.loadtxt(tmp_path / "lcdm" / "run.1.txt")

    again = lcdm()
    quiet(sample_on_core, again, "mcmc", {"Rminus1_stop": 0.05}, output=output, seed=3)

    assert again.sampler.reused
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "lcdm" / "run.1.txt"), rows)
    # Read back from the files, which keep eleven significant digits.
    np.testing.assert_allclose(again.flat_samples(), first.flat_samples(), rtol=1e-9)


def test_the_core_profile_is_the_fitters():

    fitter = lcdm()
    values = np.linspace(0.27, 0.33, 4)

    new = quiet(profile_on_core, fitter, "Omega_m", values)
    old = quiet(fitter.profile, "Omega_m", values)

    np.testing.assert_allclose(new["delta_chi2"], old["delta_chi2"], atol=1e-3)
    assert set(new["params"][0]) == {"H0", "rd"}


def test_the_core_fisher_agrees_with_the_fitters():

    fitter = lcdm()

    new = quiet(fisher_on_core, fitter)
    old = quiet(fitter.fisher)

    assert new["free_params"] == old["free_params"]
    np.testing.assert_allclose(new["errors"], old["errors"], rtol=0.05)
    np.testing.assert_allclose(new["theta"], old["theta"])


def test_the_core_evidence_agrees_with_the_fitters():

    pytest.importorskip("dynesty")

    from cosmofit.stats.nested import run_nested

    fitter = lcdm()

    new = quiet(evidence_on_core, fitter, nlive=150, seed=4)
    old = quiet(run_nested, fitter.logpost, fitter.prior, fitter.free_params,
                n_live=150, progress=False, seed=4)

    error = np.hypot(new.log_evidence_error, old.log_evidence_error)

    assert abs(new.log_evidence - old.log_evidence) < 4 * error
    assert new.prior_volume == pytest.approx(old.prior_volume)
    assert new.samples.shape[1] == 3


# ============================================================
# Fitter's own samplers, deprecated
# ============================================================

@pytest.mark.parametrize("call, replacement", [
    (lambda f: f.run_mcmc(nwalkers=8, nsteps=20, burnin=0, progress=False), "sample_on_core"),
    (lambda f: f.profile("Omega_m", [0.3]), "profile_on_core"),
    (lambda f: f.fisher(theta=[68.0, 0.3, 147.0], check_steps=False), "fisher_on_core"),
])
def test_fitters_own_samplers_point_to_the_core(call, replacement):

    fitter = quiet(
        Fitter, model=cosmofit.LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "rd"],
        initial={"H0": 68.0, "Omega_m": 0.3, "rd": 147.0},
    )

    with pytest.warns(DeprecationWarning, match=replacement):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            call(fitter)


def test_run_nested_points_to_the_core():

    pytest.importorskip("dynesty")

    fitter = quiet(
        Fitter, model=cosmofit.LCDM, datasets=["cc"],
        free_params=["H0", "Omega_m"], initial={"H0": 68.0, "Omega_m": 0.3},
    )

    with pytest.warns(DeprecationWarning, match="evidence_on_core"):
        fitter.run_nested(n_live=30, dlogz=5.0, progress=False)


def test_a_profile_reaches_the_minimum_a_gradient_method_stops_short_of():
    """
    Profiling CPL's w0 on CC + DESI + Pantheon+, L-BFGS-B alone stopped
    8 units of chi2 above the minimum at w0 = -0.7 (1440.9 against
    1432.7), and the profile came out with a kink. The Nelder-Mead
    polish after it finds the minimum, and the profile through the best
    fit is the best fit's chi2.
    """

    fitter = quiet(
        Fitter, model=cosmofit.CPL, datasets=["cc", "desi", "pantheon"],
        free_params=["H0", "Omega_m", "w0", "wa", "rd"],
        initial={"H0": 70.0, "Omega_m": 0.3, "w0": -1.0, "wa": 0.0, "rd": 147.0},
    )

    quiet(fitter.best_fit, restarts=3, seed=0)

    w0 = fitter.best_fit_params["w0"]

    profile = quiet(profile_on_core, fitter, "w0", [w0, -0.7])

    assert profile["chi2"][0] == pytest.approx(fitter.best_fit_chi2, abs=0.05)
    assert profile["chi2"][1] == pytest.approx(1432.7, abs=0.2)
