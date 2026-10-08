"""
DESI DR1 full-shape (ShapeFit) likelihood.

The data were extracted from the paper's LaTeX; a few numbers are
pinned here against the printed appendix. The prediction is held to
the paper's own table 11: at DESI's fiducial cosmology, the
``sigma_s8`` and ``f sigma_s8`` it computes must be the ones DESI
printed -- which is what fixes the radius ``sigma_s8`` is read at.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cosmofit.core import ComponentError, get_model
from cosmofit.likelihoods.desi_fullshape import load_desi_shapefit


# ============================================================
# The data
# ============================================================

@pytest.mark.parametrize("version", ["shapefit", "shapefit_bao"])
def test_six_tracers_with_positive_definite_covariances(version):

    data = load_desi_shapefit(version)

    assert data["tracers"] == ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
    np.testing.assert_array_equal(data["z"], [0.295, 0.510, 0.706, 0.919, 1.317, 1.491])

    for covariance in data["covariance"]:
        np.testing.assert_array_equal(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_values_are_the_papers():
    """A corner of each version, read off DESI 2024 V, appendix A."""

    alone = load_desi_shapefit("shapefit")

    np.testing.assert_allclose(alone["data"][0], [7.788174, 3.053800, 0.377174])
    assert alone["covariance"][0][0, 0] == pytest.approx(1314.664401e-4)
    assert alone["covariance"][0][1, 2] == pytest.approx(-158.101737e-4)

    combined = load_desi_shapefit("shapefit_bao")

    np.testing.assert_allclose(combined["data"][-1], [25.768648, 0.425629, 0.436849])
    assert combined["covariance"][-1][2, 2] == pytest.approx(21.028703e-4)
    assert combined["covariance"][-1][0, 2] == pytest.approx(40.332929e-4)


def test_unknown_version():

    with pytest.raises(ValueError, match="shapefit_bao"):
        load_desi_shapefit("fullmodelling")


# ============================================================
# The prediction
# ============================================================

H = 0.6736

#: DESI's fiducial cosmology (DESI 2024 V, table 6, Planck LCDM).
FIDUCIAL = {
    "H0": 100 * H, "Omega_b": 0.02237 / H**2,
    "Omega_m": (0.02237 + 0.1200 + 0.06 / 93.14) / H**2,
    "m_nu": 0.06, "N_eff": 3.046, "ln1e10As": math.log(20.830), "n_s": 0.9649,
}

#: DESI 2024 V, table 11: D_V/r_d, D_H/D_M, sigma_s8, f sigma_s8.
TABLE_11 = np.array([
    [8.0663, 3.1180, 0.6936, 0.4723],
    [12.8269, 1.6858, 0.6210, 0.4733],
    [16.4597, 1.1399, 0.5638, 0.4608],
    [19.7356, 0.8162, 0.5108, 0.4398],
    [24.4318, 0.5029, 0.4320, 0.3944],
    [26.0292, 0.4228, 0.4042, 0.3750],
])


def model(likelihood=None, params=None, **options):

    pytest.importorskip("camb")

    return get_model({
        "theory": {
            "background": {"dark_energy": "lambda"}, "early_universe": None,
            "camb": {"lmax": 30}, "growth": {"amplitude": "boltzmann"},
        },
        "likelihood": likelihood or {"bao.desi_fullshape": options or None},
        "params": params or FIDUCIAL,
    })


def evaluate(m, point=None):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = m.logposterior(point or {})

    assert result.rejected is None, result.rejected

    return result


def test_fiducial_cosmology_gives_back_table_11():

    m = model()
    evaluate(m)

    predicted = m.likelihoods["bao.desi_fullshape"].predictions()

    # The distances to the table's rounding and CAMB-vs-native background
    # differences; the growth to the four digits printed.
    np.testing.assert_allclose(predicted[:, :2], TABLE_11[:, :2], rtol=1.5e-3)
    np.testing.assert_allclose(predicted[:, 2], TABLE_11[:, 3], atol=1.5e-4)


def test_free_rd_moves_the_smoothing_radius():
    """
    With ``rd`` free, ``sigma_s8`` is read at ``8 r_d / 99.0792`` Mpc:
    a larger sound horizon, a larger sphere, a smaller ``sigma``.
    """

    m = model(params={**FIDUCIAL, "rd": 147.0}, rd="free")
    evaluate(m)

    likelihood = m.likelihoods["bao.desi_fullshape"]

    small = likelihood.predictions(rd=140.0)[:, 2]
    large = likelihood.predictions(rd=155.0)[:, 2]

    assert np.all(large < small)

    # The distances scale as 1/r_d; their ratio not at all.
    np.testing.assert_allclose(
        likelihood.predictions(rd=140.0)[:, 0] * 140.0,
        likelihood.predictions(rd=155.0)[:, 0] * 155.0,
    )


def test_combining_with_desi_bao_warns():

    with pytest.warns(UserWarning, match="same galaxies|DESI DR1's galaxies"):
        model(likelihood={"bao.desi_fullshape": None, "bao.desi": None})


@pytest.mark.parametrize("options, match", [
    ({"version": "fullmodelling"}, "Unknown DESI full-shape version"),
    ({"rd": "fixed"}, "rd must be"),
    ({"kmax": 0.2}, "unknown option"),
])
def test_bad_options(options, match):

    with pytest.raises(ComponentError, match=match):
        model(**options)
