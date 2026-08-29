"""
`FitResult` and the two results it carries.

Everything here is what a user gets back from `Fitter.best_fit()`,
`Fitter.run_mcmc()` and `Fitter.result` -- the object a script prints,
stores, and reads back in a later session. 58% of the module had ever
run, and the half that had not was the half that leaves the process:
`save_json`, `load_json`, and the encoder standing between numpy and
a `json` module that rejects everything numpy produces.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from CosmoFit import LCDM, BestFitResult, FitResult, Fitter, MCMCResult


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def best_fit():

    return BestFitResult(
        params={"H0": 70.0, "Omega_m": 0.3},
        chi2=25.5,
        ndim=2,
        n_data=40,
        success=True,
        message="Converged",
    )


@pytest.fixture
def mcmc():

    return MCMCResult(
        summary={
            "H0": {"median": 70.1, "plus": 1.3, "minus": 1.2},
            "Omega_m": {"median": 0.31, "plus": 0.02, "minus": 0.02},
        },
        convergence={"tau": [12.0, 14.0], "converged": True},
        nwalkers=32,
        nsteps=3000,
        burnin=500,
        ndim=2,
        acceptance_fraction=0.42,
    )


# ============================================================
# The information criteria
# ============================================================


def test_aic_and_bic_are_the_textbook_formulae(best_fit):
    """
    Two definitions are in circulation -- one on chi-squared and one
    on the log-likelihood, differing by a factor of two. This library
    uses the chi-squared form throughout, and every model comparison
    it reports depends on which one this is.
    """

    assert best_fit.aic() == pytest.approx(25.5 + 2 * 2)

    assert best_fit.bic() == pytest.approx(25.5 + 2 * np.log(40))


def test_bic_penalises_more_than_aic_for_any_real_dataset(best_fit):
    """
    `ln(n) > 2` for n > 7, which is every dataset here. If the two
    ever came out the other way round the formulae would have been
    swapped.
    """

    assert best_fit.bic() > best_fit.aic()


# ============================================================
# to_dict
# ============================================================


def test_best_fit_to_dict_carries_the_derived_numbers(best_fit):
    """
    `aic` and `bic` are methods on the object and *keys* in the
    dictionary -- so a consumer of the JSON does not have to know the
    formulae, and cannot get them wrong.
    """

    d = best_fit.to_dict()

    assert d["aic"] == best_fit.aic()
    assert d["bic"] == best_fit.bic()
    assert d["params"] == {"H0": 70.0, "Omega_m": 0.3}


def test_to_dict_copies_rather_than_aliases(best_fit):

    d = best_fit.to_dict()

    d["params"]["H0"] = 0.0

    assert best_fit.params["H0"] == 70.0


def test_fit_result_to_dict_with_nothing_run():

    d = FitResult(model="LCDM").to_dict()

    assert d["best_fit"] is None
    assert d["mcmc"] is None
    assert d["datasets"] == []


# ============================================================
# The JSON round trip
# ============================================================


def test_round_trip_preserves_everything(best_fit, mcmc, tmp_path):

    original = FitResult(
        model="LCDM",
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        best_fit=best_fit,
        mcmc=mcmc,
    )

    path = tmp_path / "fit.json"

    original.save_json(path)

    restored = FitResult.load_json(path)

    assert restored.to_dict() == original.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"best_fit": "best_fit"},
        {"mcmc": "mcmc"},
    ],
    ids=["neither", "best-fit only", "mcmc only"],
)
def test_round_trip_with_each_half_missing(kwargs, best_fit, mcmc, tmp_path):
    """
    `Fitter.result` is available before either `best_fit()` or
    `run_mcmc()` has been called, so both halves are genuinely
    optional and both `None` branches are reachable from a script.
    """

    filled = {
        key: {"best_fit": best_fit, "mcmc": mcmc}[value]
        for key, value in kwargs.items()
    }

    original = FitResult(model="LCDM", **filled)

    path = tmp_path / "fit.json"

    original.save_json(path)

    assert FitResult.load_json(path).to_dict() == original.to_dict()


def test_numpy_scalars_survive_the_save(tmp_path):
    """
    A chain read back from HDF5 reports `np.int64` counters, and
    `json` rejects those outright -- `np.float64` only survives
    because it subclasses `float`. Without the encoder, saving a
    result that had been through a saved chain would fail.
    """

    result = FitResult(
        model="LCDM",
        best_fit=BestFitResult(
            params={"H0": np.float32(70.0)},
            chi2=np.float64(25.5),
            ndim=np.int64(1),
            n_data=np.int32(40),
            success=True,
            message="ok",
        ),
    )

    path = tmp_path / "fit.json"

    result.save_json(path)

    data = json.loads(path.read_text())

    assert data["best_fit"]["ndim"] == 1

    assert isinstance(data["best_fit"]["ndim"], int)

    assert data["best_fit"]["params"]["H0"] == pytest.approx(70.0)


def test_an_unserializable_object_raises_rather_than_being_stringified(tmp_path):
    """
    The encoder coerces numpy and re-raises for everything else. The
    alternative -- `default=str` -- would write `"<object at 0x...>"`
    into the file and call the save a success.
    """

    result = FitResult(
        model="LCDM",
        best_fit=BestFitResult(
            params={"H0": object()},
            chi2=1.0,
            ndim=1,
            n_data=2,
            success=True,
            message="ok",
        ),
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        result.save_json(tmp_path / "fit.json")


# ============================================================
# repr
# ============================================================


def test_best_fit_repr_shows_the_numbers(best_fit):

    text = repr(best_fit)

    assert "H0=70" in text
    assert "chi2=25.5" in text
    assert "AIC" in text and "BIC" in text


def test_mcmc_repr_shows_every_parameter(mcmc):

    text = repr(mcmc)

    for name in mcmc.summary:
        assert name in text


def test_fit_result_repr_nests_both_halves(best_fit, mcmc):

    text = repr(FitResult(model="LCDM", best_fit=best_fit, mcmc=mcmc))

    assert "LCDM" in text
    assert "chi2" in text
    assert "H0" in text


# ============================================================
# Against a real fit
# ============================================================


def test_a_real_fit_produces_a_saveable_result(tmp_path):
    """
    The fixtures above build the dataclasses by hand. This is the
    path a user actually takes, and the one where a numpy type from
    the sampler can reach the encoder.
    """

    fitter = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 70.0, "Omega_m": 0.3},
    )

    fitter.best_fit()

    fitter.run_mcmc(nwalkers=8, nsteps=80, burnin=10, progress=False)

    result = fitter.result

    assert result.model == "LCDM"
    assert result.datasets == ["cc"]
    assert result.best_fit is not None
    assert result.mcmc is not None

    path = tmp_path / "fit.json"

    result.save_json(path)

    restored = FitResult.load_json(path)

    assert restored.best_fit.chi2 == pytest.approx(result.best_fit.chi2)

    assert restored.mcmc.nwalkers == 8
