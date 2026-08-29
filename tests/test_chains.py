"""
Saved chains.

`Fitter.run_mcmc(save=...)` is one of the library's headline features
-- the README gives it a section -- and 64% of `stats/chains.py` had
ever run. The untested part is the part that matters most: the
signature machinery that decides whether a chain on disk may be
resumed.

That decision is the quietest scientific error the library can make.
Continue a chain under a changed model, dataset combination, free
parameter list, prior bound or fixed parameter value, and the result
is one array of samples drawn from two different posteriors, with
nothing in the file or the output to say so. Every test in the
"refusing to resume" section below is a way that could happen.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import CPL, LCDM, ChainFile, Fitter, chain_info, open_chain
from CosmoFit.stats.chains import (
    build_metadata,
    compare_signatures,
    list_chains,
    signature_id,
)


NWALKERS = 8
NSTEPS = 60


def _fitter(model=LCDM, datasets=("cc",), free_params=("H0", "Omega_m"), **kwargs):

    return Fitter(
        model=model,
        datasets=list(datasets),
        free_params=list(free_params),
        initial={"H0": 70.0, "Omega_m": 0.3, "w0": -1.0, "wa": 0.0, "rd": 147.1},
        **kwargs,
    )


@pytest.fixture
def saved(tmp_path):
    """A finished chain on disk, and the fitter that wrote it."""

    path = tmp_path / "chain.h5"

    fitter = _fitter()

    fitter.run_mcmc(
        nwalkers=NWALKERS, nsteps=NSTEPS, burnin=10,
        save=str(path), progress=False,
    )

    return fitter, path


# ============================================================
# Reading a chain back without a Fitter
# ============================================================


def test_a_chain_can_be_read_without_touching_a_dataset(saved):
    """
    The point of `open_chain`: the cheap questions -- what did the
    posterior look like, how long did it run -- answered from the
    file alone, with no likelihood constructed and no data loaded.
    """

    fitter, path = saved

    stored = open_chain(str(path))

    assert stored.nwalkers == NWALKERS
    assert stored.ndim == fitter.ndim
    assert stored.iteration == NSTEPS
    assert stored.free_params == fitter.free_params


def test_the_stored_summary_matches_the_live_one(saved):
    """
    Two independent implementations of the same percentiles -- one
    on the fitter, one on the file. They have to agree, or a result
    quoted from a reopened chain would differ from the one quoted
    when it was run.
    """

    fitter, path = saved

    live = fitter.summary()

    stored = open_chain(str(path)).summary()

    assert set(stored) == set(live)

    for name in live:
        for key in ("median", "plus", "minus"):
            assert stored[name][key] == pytest.approx(live[name][key])


def test_samples_dict_is_the_flat_array_keyed_by_name(saved):

    _, path = saved

    stored = open_chain(str(path))

    flat = stored.flat_samples()

    as_dict = stored.samples_dict()

    assert list(as_dict) == stored.free_params

    for i, name in enumerate(stored.free_params):
        np.testing.assert_array_equal(as_dict[name], flat[:, i])


def test_burnin_comes_from_the_file_and_can_be_overridden(saved):
    """
    A chain records the burn-in it was run with, so a reader does
    not have to remember it. Passing one explicitly still wins.
    """

    _, path = saved

    stored = open_chain(str(path))

    assert stored.burnin == 10

    assert len(stored.flat_samples()) == (NSTEPS - 10) * NWALKERS

    assert len(stored.flat_samples(burnin=0)) == NSTEPS * NWALKERS


def test_chain_info_reads_without_reading_the_chain(saved):

    _, path = saved

    info = chain_info(str(path))

    assert info["exists"] is True
    assert info["model"] == "LCDM"
    assert info["datasets"] == ["cc"]
    assert info["nsteps"] == NSTEPS
    assert info["nwalkers"] == NWALKERS
    assert info["burnin"] == 10
    assert info["cosmofit_version"]
    assert info["created"] and info["updated"]


def test_repr_says_what_is_in_the_file(saved):

    _, path = saved

    assert "LCDM" in repr(ChainFile(str(path)))

    assert "LCDM" in repr(open_chain(str(path)))


# ============================================================
# A file that is not there, or is not a chain
# ============================================================


def test_a_path_that_was_never_written(tmp_path):

    chain = ChainFile(tmp_path / "nothing.h5")

    assert chain.exists is False
    assert chain.iteration == 0
    assert chain.shape is None
    assert chain.metadata == {}
    assert "empty" in repr(chain)

    assert list_chains(tmp_path / "nothing.h5") == []

    with pytest.raises(FileNotFoundError, match="run_mcmc"):
        chain.open()


def test_a_file_that_is_not_hdf5(tmp_path):
    """
    Treated as "nothing to resume from" rather than raised on, so a
    stray file in a chains directory does not stop a run before it
    starts. A later *write* still raises, with h5py's own message.
    """

    path = tmp_path / "not-a-chain.h5"

    path.write_text("this is not HDF5")

    assert ChainFile(path).exists is False

    assert ChainFile(path).metadata == {}

    assert list_chains(path) == []


def test_a_file_can_hold_several_chains(tmp_path):
    """
    `name=` is what makes one file per model comparison possible,
    and the metadata is keyed by chain name so two chains in one
    file cannot overwrite each other's.
    """

    path = tmp_path / "models.h5"

    for model, name in ((LCDM, "LCDM"), (CPL, "CPL")):

        free = ["H0", "Omega_m"] + ([] if model is LCDM else ["w0", "wa"])

        _fitter(model=model, free_params=free).run_mcmc(
            nwalkers=NWALKERS, nsteps=NSTEPS, burnin=5,
            save=ChainFile(path, name=name), progress=False,
        )

    assert list_chains(path) == ["CPL", "LCDM"]

    assert chain_info(path, name="LCDM")["model"] == "LCDM"
    assert chain_info(path, name="CPL")["model"] == "CPL"

    assert ChainFile(path, name="CPL").shape[1] == 4
    assert ChainFile(path, name="LCDM").shape[1] == 2


# ============================================================
# Resuming
# ============================================================


def test_resuming_continues_the_same_chain(saved):

    _, path = saved

    # Snapshot first: a fitter's `sampler` is backed by the file
    # itself, so it grows when anything else appends to it.
    before = open_chain(str(path)).get_chain()

    assert before.shape[0] == NSTEPS

    _fitter().run_mcmc(
        nwalkers=NWALKERS, nsteps=NSTEPS + 40, burnin=10,
        save=str(path), progress=False,
    )

    assert ChainFile(str(path)).iteration == NSTEPS + 40

    # The first NSTEPS steps are the ones already on disk, not
    # resampled -- a resume that quietly restarted would still end
    # up with the right total.
    np.testing.assert_allclose(
        open_chain(str(path)).get_chain()[:NSTEPS], before,
    )


def test_asking_for_no_more_steps_than_are_stored_does_not_resample(saved):

    _, path = saved

    before = open_chain(str(path)).get_chain()

    _fitter().run_mcmc(
        nwalkers=NWALKERS, nsteps=NSTEPS, burnin=10,
        save=str(path), progress=False,
    )

    np.testing.assert_allclose(open_chain(str(path)).get_chain(), before)


def test_resume_false_throws_the_chain_away(saved):
    """
    Destructive, and the only path that reaches `ChainFile.reset`.

    Truncation is what is asserted, not that the numbers changed:
    `run_mcmc` seeds its walker initialization from a fixed `seed`,
    so a discarded-and-rerun chain reproduces the steps it replaced.
    Sixty steps becoming twenty is the observable.
    """

    _, path = saved

    assert ChainFile(str(path)).iteration == NSTEPS

    _fitter().run_mcmc(
        nwalkers=NWALKERS, nsteps=20, burnin=5,
        save=str(path), resume=False, progress=False,
    )

    assert ChainFile(str(path)).iteration == 20


# ============================================================
# Refusing to resume: the signature
# ============================================================


def test_an_identical_fit_has_no_differences():

    signature = _fitter()._chain_signature()

    assert compare_signatures(signature, dict(signature)) == []


@pytest.mark.parametrize(
    "kwargs, expect",
    [
        ({"model": CPL, "free_params": ["H0", "Omega_m", "w0", "wa"]}, "model"),
        ({"datasets": ["cc", "desi"]}, "datasets"),
        ({"free_params": ["H0"]}, "free_params"),
        ({"compute_rd": True}, "compute_rd"),
    ],
)
def test_a_changed_posterior_is_reported(kwargs, expect):
    """
    Every one of these changes the distribution being sampled while
    leaving a chain file that looks perfectly resumable.
    """

    stored = _fitter()._chain_signature()

    current = _fitter(**kwargs)._chain_signature()

    differences = compare_signatures(stored, current)

    assert differences

    assert any(expect in line for line in differences)


def test_a_changed_prior_bound_names_the_parameter():
    """
    `bounds` holds one entry per parameter, and printing both dicts
    whole to report one changed number buries it.
    """

    stored = _fitter()._chain_signature()

    current = _fitter()._chain_signature()

    current["bounds"]["H0"] = [0.0, 200.0]

    differences = compare_signatures(stored, current)

    assert len(differences) == 1

    assert "bounds[H0]" in differences[0]

    assert "H0" in differences[0] and "200" in differences[0]


def test_a_parameter_appearing_or_disappearing_is_reported():

    stored = _fitter()._chain_signature()

    current = _fitter()._chain_signature()

    current["fixed"]["something_new"] = 1.0

    del stored["fixed"]

    stored["fixed"] = {"gone": 2.0}

    differences = compare_signatures(stored, current)

    assert any("now absent" in line for line in differences)

    assert any("not in the stored chain" in line for line in differences)


def test_a_key_missing_from_one_side_is_not_a_mismatch():
    """
    An older or foreign file simply may not carry a key. That is
    reported by the caller as "no metadata", not as a mismatch
    against a default that was never stored -- otherwise every chain
    written before a signature key was added would look incompatible.
    """

    current = _fitter()._chain_signature()

    stored = {k: v for k, v in current.items() if k != "derive_sigma8"}

    assert compare_signatures(stored, current) == []


def test_resuming_a_different_posterior_is_refused(saved):

    _, path = saved

    different = _fitter(datasets=["cc", "desi"])

    with pytest.raises(ValueError) as excinfo:
        different.run_mcmc(
            nwalkers=NWALKERS, nsteps=NSTEPS + 10, burnin=10,
            save=str(path), progress=False,
        )

    assert "datasets" in str(excinfo.value)


# ============================================================
# Metadata and identity
# ============================================================


def test_build_metadata_keeps_the_original_creation_stamp():
    """
    A chain grown over five sessions still reports when it started,
    and when it was last touched.
    """

    signature = {"model": "LCDM"}

    first = build_metadata(signature, burnin=10)

    second = build_metadata(signature, previous=first, burnin=20)

    assert second["created"] == first["created"]

    assert second["burnin"] == 20

    assert second["cosmofit_version"] and second["emcee_version"]


def test_signature_id_is_stable_and_discriminating():
    """
    Stable across processes -- it hashes the JSON encoding, not
    Python's `hash`, which is salted per process. A chain filename
    derived from it therefore names the same file tomorrow.
    """

    a = _fitter()._chain_signature()

    b = _fitter()._chain_signature()

    assert signature_id(a) == signature_id(b)

    assert len(signature_id(a)) == 8

    assert signature_id(a, extra={"seed": 1}) != signature_id(a, extra={"seed": 2})

    c = _fitter(datasets=["cc", "desi"])._chain_signature()

    assert signature_id(a) != signature_id(c)


def test_a_chain_id_names_the_fit():
    """
    Model name plus the signature hash: readable enough to
    recognize in a directory listing, and different for a
    different posterior, which is the property that stops two fits
    colliding on one file.
    """

    identifier = _fitter().chain_id()

    assert identifier.startswith("LCDM_")

    assert identifier == _fitter().chain_id()

    assert identifier != _fitter(datasets=["cc", "desi"]).chain_id()
