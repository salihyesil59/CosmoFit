"""
The new core, on toy components.

No cosmology here: a theory that computes a line, likelihoods that
compare it with numbers. What is tested is the machinery -- priors,
parameter roles, requirement resolution, caching, assembly errors,
the two samplers -- where an answer can be written down by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from cosmofit.core import (
    ComponentError,
    Gaussian,
    Likelihood,
    LogUniform,
    Theory,
    Uniform,
    get_model,
    load_info,
    make_prior,
    parse_param,
    resolve,
    run,
)


# ============================================================
# Toy components
# ============================================================

class Line(Theory):
    """``y = a x + b``, provided as "line"; derived "slope2" = a^2."""

    params = {"a": None, "b": None}

    def get_can_provide(self):
        return ["line"]

    def get_derived_params(self):
        return ["slope2"]

    def calculate(self, state, want_derived=True, **params):

        if params["a"] < -100:
            return False

        state["line"] = (params["a"], params["b"])
        state["derived"] = {"slope2": params["a"] ** 2}

    def get_line(self, x):
        a, b = self.current_state["line"]
        return a * np.asarray(x) + b


class Points(Likelihood):
    """Gaussian data on the line, unit errors; nuisance "shift"."""

    params = {"shift": {"prior": {"min": -1, "max": 1}}}

    X = np.array([0.0, 1.0, 2.0, 3.0])
    Y = np.array([1.0, 3.0, 5.0, 7.0])        # a = 2, b = 1

    def get_requirements(self):
        return {"line": None}

    def logp(self, shift):
        model = self.provider.get_line(x=self.X) + shift
        return -0.5 * float(np.sum((self.Y - model) ** 2))


class Square(Theory):
    """Requires "line" from another theory: a theory chain."""

    def get_requirements(self):
        return {"line": None}

    def get_can_provide(self):
        return ["square"]

    def calculate(self, state, want_derived=True, **params):
        state["square"] = float(self.provider.get_line(x=2.0)) ** 2


class UsesSquare(Likelihood):

    def get_requirements(self):
        return {"square": None}

    def logp(self):
        return -0.5 * (self.provider.get_square() - 25.0) ** 2


def toy_info(**params):

    base = {
        "theory": {"line": {"class": Line}},
        "likelihood": {"points": {"class": Points}},
        "params": {
            "a": {"prior": {"min": -5, "max": 5}},
            "b": {"prior": {"dist": "norm", "loc": 0.0, "scale": 3.0}},
            "shift": 0.0,
        },
    }

    base["params"].update(params)

    return base


# ============================================================
# Priors
# ============================================================

@pytest.mark.parametrize("prior", [
    Uniform(-1.0, 3.0),
    Gaussian(1.0, 0.5),
    Gaussian(0.0, 1.0, min=-0.5, max=2.0),
    LogUniform(1e-3, 10.0),
])
def test_priors_are_normalized(prior):

    low, high = prior.support
    low = max(low, -40.0)
    high = min(high, 40.0)

    edges = np.unique(np.concatenate([np.geomspace(low, high, 50)
                                      if low > 0 else [low, high]]))

    total = sum(
        integrate.quad(lambda x: math.exp(prior.logpdf(x)), a, b, limit=200)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )

    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("prior", [
    Uniform(-1.0, 3.0),
    Gaussian(0.0, 1.0, min=-0.5, max=2.0),
    LogUniform(1e-3, 10.0),
])
def test_ppf_inverts_the_cdf(prior):

    low, _ = prior.support

    for u in (0.1, 0.5, 0.9):

        x = prior.ppf(u)

        cdf = integrate.quad(lambda t: math.exp(prior.logpdf(t)), low, x)[0]

        assert cdf == pytest.approx(u, abs=1e-7)


def test_prior_specs():

    assert isinstance(make_prior({"min": 0, "max": 1}), Uniform)
    assert isinstance(make_prior((0, 1)), Uniform)
    assert isinstance(make_prior({"dist": "norm", "loc": 0, "scale": 1}), Gaussian)
    assert isinstance(make_prior({"dist": "loguniform", "min": 1, "max": 2}), LogUniform)

    with pytest.raises(ValueError, match="needs 'scale'"):
        make_prior({"dist": "norm", "loc": 0})

    with pytest.raises(ValueError, match="Unexpected key"):
        make_prior({"min": 0, "max": 1, "loc": 3})


# ============================================================
# Parameters
# ============================================================

def test_parameter_roles():

    assert parse_param("a", 1.5).role == "fixed"
    assert parse_param("a", {"prior": {"min": 0, "max": 1}}).role == "sampled"
    assert parse_param("a", "lambda b: 2 * b").role == "dependent"
    assert parse_param("a", {"derived": True}).role == "derived"
    assert parse_param("a", {"derived": "lambda b: b"}).role == "derived"


def test_a_parameter_cannot_be_two_things():

    with pytest.raises(ValueError, match="one of them"):
        parse_param("a", {"prior": {"min": 0, "max": 1}, "value": 2.0})

    with pytest.raises(ValueError, match="unknown key"):
        parse_param("a", {"priors": {"min": 0, "max": 1}})


def test_dependent_parameter_and_drop():
    """
    Sample ``c`` (dropped -- no component takes it) and hand the
    theory ``b = c / 2``.
    """

    info = toy_info(
        b="lambda c: c / 2",
        c={"prior": {"min": -10, "max": 10}, "drop": True},
    )

    model = get_model(info)

    assert model.sampled_params == ["a", "c"]

    result = model.logposterior({"a": 2.0, "c": 2.0})

    assert result.loglikes["points"] == pytest.approx(0.0, abs=1e-12)


def test_a_parameter_nobody_takes_is_refused():

    with pytest.raises(ComponentError, match="No theory or likelihood takes"):
        get_model(toy_info(typo={"prior": {"min": 0, "max": 1}}))


# ============================================================
# Assembly
# ============================================================

def test_missing_and_ambiguous_providers_are_refused():

    info = toy_info()
    info["theory"] = {}

    with pytest.raises(ComponentError, match="no theory provides it"):
        get_model(info)

    info = toy_info()
    info["theory"]["line2"] = {"class": Line}

    with pytest.raises(ComponentError, match="more than one theory"):
        get_model(info)


def test_theories_are_ordered_by_what_they_require():

    info = toy_info()
    info["theory"] = {"square": {"class": Square}, "line": {"class": Line}}
    info["likelihood"]["sq"] = {"class": UsesSquare}

    model = get_model(info)

    assert model.theory_order == ["line", "square"]

    result = model.logposterior({"a": 2.0, "b": 1.0})

    assert result.loglikes["sq"] == pytest.approx(0.0)


def test_a_theory_recomputes_when_the_theory_it_reads_changes():
    """
    ``Square`` takes no parameters of its own -- it reads ``Line`` -- so
    a cache keyed on its own inputs alone would serve the first state
    forever. It must follow ``Line``, and still reuse a state when
    ``Line`` comes back to a point it has seen.
    """

    info = toy_info()
    info["theory"] = {"square": {"class": Square}, "line": {"class": Line}}
    info["likelihood"]["sq"] = {"class": UsesSquare}

    model = get_model(info)

    square = model.theories["square"]

    model.logposterior({"a": 2.0, "b": 1.0})
    first = model.provider.get_square()

    model.logposterior({"a": 3.0, "b": 1.0})
    second = model.provider.get_square()

    assert first == pytest.approx(25.0)
    assert second == pytest.approx(49.0)
    assert square.n_calculations == 2

    model.logposterior({"a": 2.0, "b": 1.0})

    assert model.provider.get_square() == pytest.approx(25.0)
    assert square.n_calculations == 2


def test_input_from_yaml():

    info = load_info(
        "likelihood:\n"
        "  points: {class: tests.test_core.Points}\n"
        "params:\n"
        "  a: {prior: {min: -5, max: 5}}\n"
    )

    assert info["likelihood"]["points"]["class"] == "tests.test_core.Points"

    with pytest.raises(ValueError, match="Unknown top-level"):
        get_model({"likelihoods": {}})


def test_registry_resolves_import_paths():

    from cosmofit.core.legacy import LegacyCosmology

    assert resolve("theory", "x", {"class": Line}) is Line
    assert resolve(
        "theory", "x", {"class": "cosmofit.core.legacy:LegacyCosmology"},
    ) is LegacyCosmology
    assert resolve(
        "theory", "cosmofit.core.legacy.LegacyCosmology",
    ) is LegacyCosmology
    assert resolve("sampler", "minimize").__name__ == "Minimize"

    with pytest.raises(ValueError, match="Unknown likelihood"):
        resolve("likelihood", "no_such_dataset")


# ============================================================
# Evaluation
# ============================================================

def test_logposterior_parts():

    model = get_model(toy_info())

    result = model.logposterior({"a": 2.0, "b": 0.0})

    expected_prior = -math.log(10.0) + Gaussian(0.0, 3.0).logpdf(0.0)

    assert result.logprior == pytest.approx(expected_prior)

    # Residual 1 at each of four points.
    assert result.loglikes["points"] == pytest.approx(-2.0)
    assert result.chi2["total"] == pytest.approx(4.0)
    assert result.logpost == pytest.approx(expected_prior - 2.0)


def test_outside_the_prior_is_not_evaluated():

    model = get_model(toy_info())

    result = model.logposterior({"a": 50.0, "b": 0.0})

    assert result.logpost == -math.inf
    assert result.rejected == "prior"
    assert model.theories["line"].n_calculations == 0


def test_a_theory_that_rejects_a_point_is_counted():

    info = toy_info(a={"prior": {"min": -500, "max": 5}})

    model = get_model(info)

    result = model.logposterior({"a": -200.0, "b": 0.0})

    assert result.logpost == -math.inf
    assert model.rejections == {"line: no solution": 1}


def test_results_are_cached_by_parameter_value():
    """
    Same theory inputs, no recomputation -- and a changed nuisance
    parameter, which the theory does not take, does not invalidate it.
    """

    info = toy_info(shift={"prior": {"min": -1, "max": 1}})

    model = get_model(info)

    model.logposterior({"a": 2.0, "b": 1.0, "shift": 0.0})
    model.logposterior({"a": 2.0, "b": 1.0, "shift": 0.3})
    model.logposterior({"a": 2.0, "b": 1.0, "shift": 0.0})

    assert model.theories["line"].n_calculations == 1

    model.logposterior({"a": 2.5, "b": 1.0, "shift": 0.0})

    assert model.theories["line"].n_calculations == 2


def test_derived_parameters():

    info = toy_info(
        slope2={"derived": True},
        twice={"derived": "lambda slope2, b: 2 * slope2 + b"},
    )

    result = get_model(info).logposterior({"a": 3.0, "b": 1.0})

    assert result.derived == {"slope2": 9.0, "twice": 19.0}


def test_a_derived_parameter_nothing_computes_is_refused():

    with pytest.raises(ComponentError, match="no component outputs it"):
        get_model(toy_info(nonexistent={"derived": True}))


# ============================================================
# Samplers
# ============================================================

def test_minimize_finds_the_maximum_likelihood_line():

    info = toy_info()
    info["sampler"] = {"minimize": {"starts": 2, "ignore_prior": True}}

    _, sampler = run(info, seed=0)

    products = sampler.products()

    assert products["point"]["a"] == pytest.approx(2.0, abs=1e-4)
    assert products["point"]["b"] == pytest.approx(1.0, abs=1e-4)
    assert products["chi2"]["total"] == pytest.approx(0.0, abs=1e-6)
    assert products["maximized"] == "likelihood"


def test_minimize_maximizes_the_posterior_by_default():
    """
    A tight Gaussian prior on ``b`` at 0 pulls the MAP away from the
    maximum-likelihood b = 1 -- by exactly the amount a linear
    Gaussian problem gives.
    """

    info = toy_info(b={"prior": {"dist": "norm", "loc": 0.0, "scale": 0.1}})
    info["sampler"] = {"minimize": {"method": "Nelder-Mead"}}

    _, sampler = run(info, seed=0)

    point = sampler.products()["point"]

    # Normal equations with the prior as one more data point.
    X = np.column_stack([Points.X, np.ones(4)])
    A = X.T @ X + np.diag([0.0, 1.0 / 0.1 ** 2])
    expected = np.linalg.solve(A, X.T @ Points.Y)

    assert point["a"] == pytest.approx(expected[0], abs=1e-3)
    assert point["b"] == pytest.approx(expected[1], abs=1e-3)


def test_evaluate():

    info = toy_info()
    info["sampler"] = {"evaluate": {"override": {"a": 2.0, "b": 1.0}}}

    _, sampler = run(info)

    products = sampler.products()

    assert products["chi2"]["points"] == pytest.approx(0.0)
    assert products["finite"]

    info["sampler"] = {"evaluate": {"override": {"zz": 1.0}}}

    with pytest.raises(ValueError, match="not sampled"):
        run(info)


def test_a_cached_state_without_derived_parameters_is_not_reused_for_them():
    """
    A minimizer evaluates without derived parameters; asking for them
    at a point it visited used to be served from the cache, empty --
    the derived columns came out NaN.
    """

    info = toy_info(slope2={"derived": True})

    model = get_model(info)

    point = {"a": 3.0, "b": 1.0}

    assert model.logposterior(point, want_derived=False).derived == {}
    assert model.logposterior(point).derived == {"slope2": 9.0}

    # And the other way round, the cache does serve.
    calculations = model.theories["line"].n_calculations

    model.logposterior(point, want_derived=False)

    assert model.theories["line"].n_calculations == calculations


class AskedLine(Line):
    """Records whether it was asked for its derived parameters."""

    def calculate(self, state, want_derived=True, **params):
        self.asked = want_derived
        return super().calculate(state, want_derived, **params)


def test_a_sampler_asks_for_derived_parameters_only_if_declared():

    info = toy_info(a={"prior": {"min": -5, "max": 5}, "ref": 2.0, "proposal": 0.1},
                    b={"prior": {"min": -5, "max": 5}, "ref": 1.0, "proposal": 0.1})
    info["theory"] = {"line": {"class": AskedLine}}
    info["sampler"] = {"mcmc": {"max_samples": 10, "learn_every": 5}}

    _, sampler = run(info, seed=0)

    assert sampler.model.theories["line"].asked is False

    info["params"]["slope2"] = {"derived": True}

    _, sampler = run(info, seed=0)

    assert sampler.model.theories["line"].asked is True
    assert "slope2" in sampler.columns
