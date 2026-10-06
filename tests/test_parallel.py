"""
Chains in several processes.

Each chain draws from its own random stream, spawned from the run's
seed, and the processes meet only to compare chains -- so a run split
over processes must give exactly the chains, files and covariance the
same run gives in one. And a process that fails must stop the run with
its error, not leave the others waiting.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cosmofit.core import run
from cosmofit.core.mpi import Serial, WorkerError, world

from test_mcmc import Gaussian, gaussian_input


#: The test process's id; a component sees a different one only when it
#: runs in a process the sampler started. Set once: a started process
#: inherits it, then imports this module again to unpickle the classes.
os.environ.setdefault("COSMOFIT_TEST_PARENT", str(os.getpid()))


def _in_child() -> bool:
    return str(os.getpid()) != os.environ["COSMOFIT_TEST_PARENT"]


class FailsInChild(Gaussian):
    """Evaluates normally, except in a started process."""

    def logp(self, a, b, c):

        if _in_child():
            raise KeyError("only the child fails")

        return super().logp(a, b, c)


class CannotStartInChild(Gaussian):

    def initialize(self):

        if _in_child():
            raise LookupError("no data in the child")


def short(output=None, **options):
    """
    600 steps, learning the proposal from the first check on, so the
    covariance the processes share changes along the way.
    """

    return gaussian_input(
        output, max_samples=600, Rminus1_stop=0.001,
        learn_proposal_Rminus1_max=100.0, **options,
    )


# ============================================================

def test_without_mpirun_the_world_is_one_process():

    assert isinstance(world(), Serial)


def test_two_processes_give_the_one_process_chains(tmp_path):

    _, serial = run(short(tmp_path / "one" / "run"), seed=21)
    _, parallel = run(short(tmp_path / "two" / "run", processes=2), seed=21)

    one, two = serial.products(), parallel.products()

    assert len(two["chains"]) == 4

    for a, b in zip(one["chains"], two["chains"]):
        np.testing.assert_array_equal(a, b)

    np.testing.assert_array_equal(one["covmat"], two["covmat"])
    assert one["Rminus1"] == two["Rminus1"]
    assert [p["acceptance"] for p in one["progress"]] == [
        p["acceptance"] for p in two["progress"]
    ]

    # The files too: each chain written by the process that ran it.
    for index in range(1, 5):
        assert (
            (tmp_path / "one" / f"run.{index}.txt").read_text()
            == (tmp_path / "two" / f"run.{index}.txt").read_text()
        )

    assert (tmp_path / "two" / "run.covmat").exists()
    assert (tmp_path / "two" / "run.progress").exists()


def test_a_parallel_run_resumes(tmp_path):

    prefix = tmp_path / "run"

    run(gaussian_input(prefix, max_samples=120, processes=2), seed=22)

    before = [len(np.loadtxt(tmp_path / f"run.{i}.txt")) for i in range(1, 5)]

    info = gaussian_input(prefix, max_samples=600, processes=2)
    info["resume"] = True

    run(info, seed=23)

    after = [len(np.loadtxt(tmp_path / f"run.{i}.txt")) for i in range(1, 5)]

    assert all(a > b for a, b in zip(after, before))


def test_an_error_in_another_process_stops_the_run():

    info = short(processes=2)
    info["likelihood"] = {"gauss": {"class": FailsInChild}}

    with pytest.raises(WorkerError, match="only the child fails"):
        run(info, seed=24)


def test_a_process_that_cannot_build_the_model_stops_the_run():

    info = short(processes=2)
    info["likelihood"] = {"gauss": {"class": CannotStartInChild}}

    with pytest.raises(WorkerError, match="no data in the child"):
        run(info, seed=25)


def test_an_error_in_the_first_process_is_its_own():

    info = short(processes=2, max_tries=50)

    for name in "abc":
        info["params"][name]["proposal"] = 1e4

    with pytest.raises(RuntimeError, match="too wide"):
        run(info, seed=26)


def test_processes_must_be_positive():

    with pytest.raises(ValueError, match="processes must be at least 1"):
        run(short(processes=0))


class _World:
    """What mpirun with three processes looks like to the first."""

    rank = 0
    size = 3


def test_under_mpirun_only_parallel_samplers_run(monkeypatch):

    import sys

    # The module, not the function `cosmofit.core.run` it exports.
    monkeypatch.setattr(sys.modules["cosmofit.core.run"], "world", _World)

    info = short()
    info["sampler"] = {"minimize": None}

    with pytest.raises(RuntimeError, match="under mpirun"):
        run(info)


def test_under_mpirun_every_process_needs_a_chain(monkeypatch):

    import cosmofit.samplers.mcmc as mcmc

    monkeypatch.setattr(mcmc, "world", _World)

    with pytest.raises(ValueError, match="chains: 3"):
        run(short(chains=2))
