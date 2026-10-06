"""
The ``cosmofit`` command.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import yaml

from cosmofit.cli import main


GAUSSIAN = """
likelihood:
  gauss:
    class: test_mcmc:Gaussian
params:
  a: {prior: {min: -20, max: 20}, ref: {dist: norm, loc: 0, scale: 0.5}, proposal: 0.5}
  b: {prior: {min: -20, max: 20}, ref: {dist: norm, loc: 0, scale: 0.5}, proposal: 0.5}
  c: {prior: {min: -20, max: 20}, ref: {dist: norm, loc: 0, scale: 0.5}, proposal: 0.5}
"""


def write(tmp_path, sampler: dict, **extra):

    info = yaml.safe_load(GAUSSIAN)
    info["sampler"] = sampler
    info.update(extra)

    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(info), encoding="utf-8")

    return path


# ============================================================
# run
# ============================================================

def test_run_minimizes_and_says_where(tmp_path, capsys):

    path = write(tmp_path, {"minimize": None})

    assert main(["run", str(path), "-o", str(tmp_path / "out" / "g")]) == 0

    out = capsys.readouterr().out

    assert "minimize: done." in out
    assert "a = 1" in out and "b = -2" in out
    assert "chi2[total] = 0.0000" in out
    assert "Output:" in out

    assert (tmp_path / "out" / "g.minimum.txt").exists()


def test_run_samples_and_resumes(tmp_path, capsys):

    path = write(tmp_path, {"mcmc": {"max_samples": 240}})
    prefix = str(tmp_path / "chains" / "g")

    assert main(["run", str(path), "-o", prefix, "--seed", "1"]) == 0

    out = capsys.readouterr().out

    assert "mcmc: done." in out and "R - 1:" in out and "acceptance:" in out

    before = len(np.loadtxt(tmp_path / "chains" / "g.1.txt"))

    # The same output again is refused, then continued.
    assert main(["run", str(path), "-o", prefix]) == 1
    assert "already exists" in capsys.readouterr().err

    assert main(["run", str(path), "-o", prefix, "--resume"]) == 0
    assert len(np.loadtxt(tmp_path / "chains" / "g.1.txt")) > before


def test_run_post_processes(tmp_path, capsys):

    prefix = str(tmp_path / "chains" / "g")

    main(["run", str(write(tmp_path, {"mcmc": {"max_samples": 240}})), "-o", prefix])
    capsys.readouterr()

    post = tmp_path / "post.yaml"
    post.write_text(yaml.safe_dump({
        "output": prefix,
        "post": {"suffix": "d", "add": {"params": {"d": {"derived": "lambda a, b: a - b"}}}},
    }), encoding="utf-8")

    assert main(["run", str(post)]) == 0

    out = capsys.readouterr().out

    assert "post: done." in out and "effective samples after" in out
    assert (tmp_path / "chains" / "g.post.d.1.txt").exists()


def test_an_error_is_one_line_unless_debugging(tmp_path, capsys):

    path = write(tmp_path, {"minimize": {"nonsense": 1}})

    assert main(["run", str(path)]) == 1

    err = capsys.readouterr().err

    assert err.startswith("cosmofit run: ValueError:") and "nonsense" in err
    assert "Traceback" not in err

    with pytest.raises(ValueError, match="nonsense"):
        main(["--debug", "run", str(path)])


def test_force_and_resume_exclude_each_other(tmp_path):

    with pytest.raises(SystemExit):
        main(["run", "x.yaml", "--force", "--resume"])


# ============================================================
# list and doc
# ============================================================

def test_list_names_every_kind(capsys):

    assert main(["list"]) == 0

    out = capsys.readouterr().out

    for name in ("background", "bao.desi", "mcmc", "nested"):
        assert f"  {name} " in out

    assert "(old name) prefer bao.desi" in out


def test_list_one_kind(capsys):

    main(["list", "sampler"])

    out = capsys.readouterr().out

    assert "Adaptive Metropolis-Hastings" in out
    assert "bao.desi" not in out


def test_doc_of_a_sampler_shows_its_defaults(capsys):

    assert main(["doc", "mcmc"]) == 0

    out = capsys.readouterr().out

    assert "mcmc  (sampler, cosmofit.samplers.mcmc:MCMC)" in out
    assert "Rminus1_stop: 0.01" in out


def test_doc_of_a_theory_shows_what_it_provides(capsys):

    main(["doc", "background"])

    out = capsys.readouterr().out

    assert "Parameters it needs: H0, Omega_m, Omega_b" in out
    assert "Provides:" in out and "DM" in out
    assert "m_nu: 0.06" in out


def test_doc_of_nothing_says_so():

    with pytest.raises(SystemExit, match="nothing is called"):
        main(["doc", "no.such.thing"])


def test_python_dash_m_is_the_command():

    result = subprocess.run(
        [sys.executable, "-m", "cosmofit", "list", "sampler"],
        capture_output=True, text=True, check=True,
    )

    assert "mcmc" in result.stdout
