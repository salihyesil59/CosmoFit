"""The fastcmb theory: the camb theory with fastcmb's emulator in front."""

import glob
import math
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("camb")
pytest.importorskip("fastcmb")

from cosmofit.core import get_model  # noqa: E402


PARAMS = {
    "H0": 67.36, "Omega_m": 0.3153, "Omega_b": 0.02237 / 0.6736 ** 2, "m_nu": 0.06,
    "ln1e10As": {"prior": {"min": 2.9, "max": 3.2}},
    "tau_reio": {"prior": {"min": 0.03, "max": 0.09}},
}


def _model(theory, options=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_model({
            "theory": {"background": None, theory: options},
            "likelihood": {"cmb.planck_lowe": None},
            "params": PARAMS,
        })


def _points(n, seed):
    rng = np.random.default_rng(seed)
    return [{"ln1e10As": rng.uniform(3.0, 3.09), "tau_reio": rng.uniform(0.045, 0.065)} for _ in range(n)]


def test_is_found_by_name_and_takes_camb_options():

    m = _model("fastcmb", {"tolerance": 0.01, "lens_potential_accuracy": 2})
    theory = m.theories["fastcmb"]

    assert theory.lmax == 30
    assert theory.settings["tolerance"] == 0.01
    assert theory.lens_potential_accuracy == 2


def test_learns_then_answers_like_camb():

    fast = _model("fastcmb", {"min_train": 12, "check_every": 0})
    slow = _model("camb")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for point in _points(60, seed=0):
            fast.logposterior(point)

        stats = fast.theories["fastcmb"].stats()
        assert stats["emulated"] > 20

        for point in _points(10, seed=1):
            a, b = fast.logposterior(point), slow.logposterior(point)
            assert abs(a.loglike - b.loglike) < 0.05

    Cl_fast = fast.theories["fastcmb"].get_Cl()
    Cl_slow = slow.theories["camb"].get_Cl()

    assert set(Cl_fast) == set(Cl_slow)
    assert all(Cl_fast[k].shape == Cl_slow[k].shape for k in Cl_slow)
    assert math.isfinite(fast.theories["fastcmb"].get_sigma8_0())


def test_training_points_are_shared_through_the_file(tmp_path):

    path = str(tmp_path / "emu" / "lowe")

    first = _model("fastcmb", {"min_train": 12, "training_file": path, "check_every": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for point in _points(20, seed=2):
            first.logposterior(point)

    # Another process's file, as a parallel chain or a later run sees it.
    (own,) = glob.glob(path + ".*.npz")
    os.replace(own, path + ".999999.npz")

    second = _model("fastcmb", {"min_train": 12, "training_file": path})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        second.logposterior(_points(1, seed=3)[0])

    online = second.theories["fastcmb"].emulator.online
    assert online.size >= 12
    assert online.ready



def test_ede_z_c_is_learned_in_its_logarithm():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = get_model({
            "theory": {"background": {"dark_energy": "ede"}, "fastcmb": None},
            "likelihood": {"cmb.planck_lowe": None},
            "params": {**PARAMS, "f_ede": 0.05, "z_c": 3500.0},
        })
        m.logposterior(_points(1, seed=4)[0])

    names = m.theories["fastcmb"].emulator.online.names

    assert "log10_z_c" in names and "z_c" not in names


@pytest.mark.skipif(
    not hasattr(__import__("camb").CAMBparams, "set_mgparams"),
    reason="needs MGCAMB in place of CAMB",
)
def test_f_R0_is_learned_in_the_logarithm_of_its_size():

    def model(f_R0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return get_model({
                "theory": {"background": {"dark_energy": "hu_sawicki"}, "fastcmb": None},
                "likelihood": {"cmb.planck_lowe": None},
                "params": {**PARAMS, "f_R0": f_R0, "n_hs": 1.0},
            })

    m = model(-1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.logposterior(_points(1, seed=5)[0])

    online = m.theories["fastcmb"].emulator.online
    assert "log10_f_R0" in online.names and "f_R0" not in online.names
    assert online.X[0][online.names.index("log10_f_R0")] == pytest.approx(-5.0)

    # General relativity, f_R0 = 0, is learned at the floor.
    gr = model(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert gr.logposterior(_points(1, seed=5)[0]).rejected is None

    online = gr.theories["fastcmb"].emulator.online
    assert online.X[0][online.names.index("log10_f_R0")] == pytest.approx(-12.0)
