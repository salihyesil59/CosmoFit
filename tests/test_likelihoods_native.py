"""
The datasets as native likelihoods.

Every one must give the old likelihood's chi2 when the theories give the
old models' predictions -- radiation off, each model wrapped by the
``legacy`` sector -- at the golden reference point
(``tests/data/golden_chi2.json``). Then what is new: the nuisance
parameters belong to the likelihoods, the sound horizon comes from the
early universe unless freed, and the distance priors keep the
conventions they were compressed with.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import CosmoFit
from CosmoFit.core import ComponentError, get_model
from CosmoFit.core.legacy import LegacyLikelihood
from CosmoFit.core.registry import resolve
from CosmoFit.likelihoods import PlanckLikelihood
from CosmoFit.likelihoods.native import NATIVE_LIKELIHOODS


GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "golden_chi2.json").read_text()
)["models"]

OLD_TO_NEW = {cls.dataset: name for name, cls in NATIVE_LIKELIHOODS.items()}

GROWTH = {"fsigma8", "s8", "sdss_fsbao", "eboss_elg_fs"}


def evaluate(info):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = get_model(info)
        return m, m.logposterior({})


def chi2(point):
    return -2.0 * point.loglike if point.rejected is None else math.inf


# ============================================================
# The golden reference
# ============================================================

def _golden_chi2(model_name, dataset):

    point = dict(GOLDEN[model_name]["point"])

    # ADE derives Omega_m; the background then takes none.
    if model_name == "ADE":
        del point["Omega_m"]

    new = OLD_TO_NEW[dataset]

    theory = {"background": {
        "dark_energy": {"name": "legacy", "model": model_name}, "radiation": False,
    }}

    if dataset in GROWTH:
        theory["growth"] = None

    options = {"rd": "free"} if NATIVE_LIKELIHOODS[new].uses_rd else {}

    _, result = evaluate({
        "theory": theory, "likelihood": {new: options}, "params": point,
    })

    return chi2(result)


@pytest.mark.parametrize("dataset", sorted(OLD_TO_NEW))
def test_every_model_gives_the_old_chi2(dataset):
    """
    LsCDM is left out of the distance datasets: its old distance table
    smeared the jump at z_dagger, 2e-5 in D_M beyond it, and the native
    one does not (``test_lscdm_distances_are_exact_where_the_old_were_not``).
    """

    for model_name, entry in sorted(GOLDEN.items()):

        if model_name == "LsCDM" and dataset in ("desi", "eboss_lya", "union3"):
            continue

        reference = entry["chi2"][dataset]

        if isinstance(reference, dict):
            reference = math.inf

        value = _golden_chi2(model_name, dataset)

        if math.isinf(reference):
            assert math.isinf(value), model_name
        else:
            assert value == pytest.approx(reference, rel=5e-7), model_name


def test_lscdm_distances_are_exact_where_the_old_were_not():

    from scipy.integrate import quad

    point = GOLDEN["LsCDM"]["point"]
    old = CosmoFit.LsCDM(CosmoFit.LsCDM.PARAMS_CLASS(**point))

    m, _ = evaluate({
        "theory": {"background": {
            "dark_energy": {"name": "legacy", "model": "LsCDM"}, "radiation": False,
        }},
        "likelihood": {"bao.desi": {"rd": "free"}},
        "params": point,
    })

    z, (z_dagger,) = 2.33, old.background_jumps()

    exact = sum(
        quad(lambda x: 1.0 / old.E(x), a, b, epsabs=0, epsrel=1e-12)[0]
        for a, b in ((0.0, z_dagger), (z_dagger, z))
    ) * 299792.458 / point["H0"]

    assert m.provider.get_DM(z) == pytest.approx(exact, rel=1e-10)
    assert abs(old.distance.DM(z) / exact - 1.0) > 1e-5


# ============================================================
# Parameters the likelihoods own
# ============================================================

BASE = {"H0": 67.5, "Omega_m": 0.31, "Omega_b": 0.049}


def test_rd_comes_from_the_early_universe_unless_freed():

    m, point = evaluate({
        "theory": {"background": None, "early_universe": None},
        "likelihood": {"bao.desi": None},
        "params": BASE,
    })

    rdrag = m.theories["early_universe"].current_state["rdrag"]

    _, freed = evaluate({
        "theory": {"background": None},
        "likelihood": {"bao.desi": {"rd": "free"}},
        "params": {**BASE, "rd": rdrag},
    })

    # Not to the last digit: early_universe extends the distance table to z_*.
    assert chi2(point) == pytest.approx(chi2(freed), rel=1e-9)

    with pytest.raises(ComponentError, match="rdrag"):
        get_model({
            "theory": {"background": None},
            "likelihood": {"bao.desi": None},
            "params": BASE,
        })

    with pytest.raises(ComponentError, match="needs the parameter"):
        get_model({
            "theory": {"background": None},
            "likelihood": {"bao.desi": {"rd": "free"}},
            "params": BASE,
        })

    with pytest.raises(ComponentError, match="rd must be"):
        get_model({
            "theory": {"background": None},
            "likelihood": {"bao.desi": {"rd": "fitted"}},
            "params": BASE,
        })


def test_two_bao_likelihoods_share_one_free_rd():

    m = get_model({
        "theory": {"background": None},
        "likelihood": {"bao.desi": {"rd": "free"}, "bao.lowz": {"rd": "free"}},
        "params": {**BASE, "rd": {"prior": {"min": 120, "max": 170}}},
    })

    assert m.sampled_params == ["rd"]
    assert np.isfinite(m.logposterior({"rd": 147.0}).logpost)


@pytest.mark.parametrize("name, options", [
    ("sn.pantheonplus", {"marginalize_MB": False}),
    ("sn.pantheonplus", {"include_cepheid": True}),
    ("sn.des_sn5yr", {"marginalize_offset": False}),
    ("sn.union3", {"marginalize_offset": False}),
])
def test_supernovae_take_MB_when_it_is_not_marginalized(name, options):

    with pytest.raises(ComponentError, match="MB"):
        get_model({
            "theory": {"background": None},
            "likelihood": {name: options},
            "params": BASE,
        })

    _, point = evaluate({
        "theory": {"background": None},
        "likelihood": {name: options},
        "params": {**BASE, "MB": -19.25},
    })

    assert math.isfinite(chi2(point))


def test_marginalized_supernovae_take_no_MB():

    with pytest.raises(ComponentError, match="No theory or likelihood takes"):
        get_model({
            "theory": {"background": None},
            "likelihood": {"sn.pantheonplus": None},
            "params": {**BASE, "MB": -19.25},
        })


def test_external_priors_read_the_background():

    _, point = evaluate({
        "theory": {"background": None},
        "likelihood": {"external.h0": None, "external.bbn": None,
                       "external.tau": None},
        "params": {**BASE, "tau_reio": 0.06},
    })

    sh0es = CosmoFit.likelihoods.H0Likelihood(None).data
    bbn = CosmoFit.likelihoods.OmegaBLikelihood(None).data
    tau = CosmoFit.likelihoods.TauLikelihood(None).data

    expected = (
        ((67.5 - sh0es.value) / sh0es.sigma) ** 2
        + ((0.049 * 0.675 ** 2 - bbn.value) / bbn.sigma) ** 2
        + ((0.06 - tau.value) / tau.sigma) ** 2
    )

    assert chi2(point) == pytest.approx(float(np.squeeze(expected)), rel=1e-10)


def test_S8_is_the_growth_theorys():

    m, point = evaluate({
        "theory": {"background": None, "growth": None},
        "likelihood": {"lss.s8": None},
        "params": {**BASE, "sigma8": 0.8},
    })

    Omega_m = m.provider.get_background_densities()["Omega_m"]

    assert m.provider.get_S8() == pytest.approx(0.8 * math.sqrt(Omega_m / 0.3))
    assert math.isfinite(chi2(point))


# ============================================================
# The CMB distance priors
# ============================================================

#: Planck 2018 best-fit LCDM, the cosmology the priors summarize.
PLANCK = {"H0": 67.36, "Omega_m": 0.3153, "Omega_b": 0.02237 / 0.6736 ** 2, "m_nu": 0.06}


def test_distance_priors_keep_their_own_conventions():
    """
    At Planck's best fit, in CHW19's conventions, the priors are met
    (chi2 ~ 0.4 for three numbers) -- as the old likelihood met them.
    The exact sound horizon of the native background would put l_A at
    pi/theta_*, 2.7 sigma from the prior.
    """

    m, point = evaluate({
        "theory": {"background": None, "early_universe": None},
        "likelihood": {"cmb.distance_priors": None},
        "params": PLANCK,
    })

    cls = CosmoFit.LCDM.PARAMS_CLASS
    old = PlanckLikelihood(CosmoFit.LCDM(cls(**{**cls.defaults(), **PLANCK})))

    assert chi2(point) == pytest.approx(old.chi2(), rel=1e-6)
    assert chi2(point) < 1.0

    data = m.likelihoods["cmb.distance_priors"].legacy.data.values
    thetastar = m.theories["early_universe"].current_state["thetastar"]

    assert (math.pi / (thetastar / 100.0) - data[1]) / 0.0895 > 2.5


# ============================================================
# Names
# ============================================================

def test_native_and_old_names():

    for name, cls in NATIVE_LIKELIHOODS.items():
        assert resolve("likelihood", name) is cls

    # The old names still reach the old likelihoods, for the old models.
    assert resolve("likelihood", "desi") is LegacyLikelihood


def test_unknown_options_name_the_likelihood():

    with pytest.raises(ComponentError, match="bao.desi"):
        get_model({
            "theory": {"background": None, "early_universe": None},
            "likelihood": {"bao.desi": {"versoin": "desi2025"}},
            "params": BASE,
        })
