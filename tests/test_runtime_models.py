"""
Models built at runtime, and the two things they cannot do.

Three routes produce a ``Cosmology`` subclass that exists only in
the session that made it -- ``define_model``,
``model_from_expression`` and ``CosmoFit.theory.Action.build``.
Everything about fitting works on them, with two exceptions, both
for the same underlying reason: such a class cannot be pickled *by
reference*, because there is no importable name to pickle it to.

  multiprocessing   `run_mcmc(n_processes>1)` sends the model to
                    worker processes

  saved chains      `Fitter.from_chain` re-imports the model the
                    chain was sampled with

Both are guarded, and what is tested here is mostly the *message*:
each has exactly one workaround, and a user who cannot find it is
stuck. The messages named only two of the three routes for a
while, so someone who had written an action was told about
functions they never called.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from CosmoFit import Fitter, LCDM, define_model, model_from_expression


INITIAL = {"H0": 70.0, "Omega_m": 0.3, "rd": 147.0}


def flat_lcdm_expression():

    return model_from_expression(
        "RuntimeExpression",
        E="sqrt(Omega_m*(1+z)**3 + 1 - Omega_m)",
    )


def flat_lcdm_function():

    return define_model(
        "RuntimeFunction",
        E=lambda p, z: np.sqrt(
            p["Omega_m"] * (1 + z) ** 3 + 1 - p["Omega_m"]
        ),
    )


def flat_lcdm_action():

    pytest.importorskip("sympy")

    from CosmoFit.theory import Action

    return Action("R - 2*Lam", closure="Lam").build("RuntimeAction")


ROUTES = [
    pytest.param(flat_lcdm_function, id="define_model"),
    pytest.param(flat_lcdm_expression, id="model_from_expression"),
    pytest.param(flat_lcdm_action, id="Action.build"),
]


# ============================================================
# The shared property
# ============================================================

@pytest.mark.parametrize("route", ROUTES)
def test_a_runtime_model_cannot_be_pickled_by_reference(route):
    """
    The single fact both guards below follow from. A built-in
    model pickles to its import path; one made at runtime has no
    import path to pickle to.
    """

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(route())

    pickle.dumps(LCDM)          # the control


# ============================================================
# Multiprocessing
# ============================================================

@pytest.mark.parametrize("route", ROUTES)
def test_multiprocessing_is_refused_with_the_route_named(route):
    """
    Refused rather than attempted: the pickling failure would
    otherwise surface from inside a worker process, where its
    traceback says nothing about the model.

    The message has to name the route the user actually took --
    all three, since it cannot know which.
    """

    fit = Fitter(
        model=route(),
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial=INITIAL,
    )

    with pytest.raises(ValueError) as raised:
        fit.run_mcmc(
            nwalkers=8, nsteps=4, burnin=1, seed=0,
            progress=False, n_processes=2,
        )

    message = str(raised.value)

    assert "n_processes=1" in message

    for name in ("define_model", "model_from_expression", "Action.build"):
        assert name in message


def test_multiprocessing_still_works_for_a_built_in_model():
    """
    The control: the guard is about runtime classes, not about
    multiprocessing being broken.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial=INITIAL,
    )

    fit.run_mcmc(
        nwalkers=8, nsteps=4, burnin=1, seed=0,
        progress=False, n_processes=2,
    )

    assert fit.flat_samples().shape[1] == 2


# ============================================================
# Saved chains
# ============================================================

@pytest.mark.parametrize("route", ROUTES)
def test_a_saved_chain_reloads_when_the_model_is_supplied(route, tmp_path):
    """
    The chain itself saves and reloads. What cannot be recovered
    from the file is the model class, so `from_chain` says so and
    names its one workaround -- and that workaround has to work,
    which is the half worth testing.
    """

    path = tmp_path / "chain.h5"

    fit = Fitter(
        model=route(),
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial=INITIAL,
    )
    fit.run_mcmc(
        nwalkers=8, nsteps=20, burnin=5, seed=0,
        progress=False, save=str(path),
    )

    expected = fit.summary()

    with pytest.raises(ValueError) as raised:
        Fitter.from_chain(str(path))

    message = str(raised.value)

    assert "from_chain" in message

    for name in ("define_model", "model_from_expression", "Action.build"):
        assert name in message

    # Rebuilt the same way, it reloads and gives back the same
    # posterior -- the chain was never the problem.
    reloaded = Fitter.from_chain(str(path), model=route())

    for name in ("H0", "Omega_m"):
        assert reloaded.summary()[name]["median"] == pytest.approx(
            expected[name]["median"], rel=1e-12,
        )
