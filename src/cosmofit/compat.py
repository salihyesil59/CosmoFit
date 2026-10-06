"""
From the 1.x interface to the 2.0 core.

:meth:`stats.fitter.Fitter.to_info` writes a fitter as an input the
new core runs (:mod:`cosmofit.core`): the same model, datasets, free
parameters and bounds, fixed values and dataset options. With a
``sampler`` block added and written to YAML, ``cosmofit run`` takes
over from it. It is written one of two ways.

``to_info()``: the analysis in 2.0
----------------------------------
The native theories and likelihoods. The model becomes its dark sector
on the ``background`` theory (``CPL`` is ``dark_energy: cpl``), with
radiation and the massive neutrinos' own history in ``E(z)``; the
datasets become their native likelihoods (``desi`` is ``bao.desi``, see
:mod:`data.metadata`). That is the same analysis on the new physics,
not the fitter's numbers: with radiation in the late-time expansion,
``chi2`` moves by a few hundredths to about one -- 0.4 to 1.1 for LCDM
on CC + DESI + Pantheon+ near the best fit, most of it the supernovae. That is all radiation:
with ``radiation: false`` the native input gives the fitter's ``chi2``
to 1e-7.
Four parameters are named differently there -- the generalized
Chaplygin gas's ``A_s`` and ``alpha`` are ``A_gcg`` and ``alpha_gcg``,
the power-law ``f(T)``'s ``n`` is ``n_ft``, Hu-Sawicki's ``n`` is
``n_hs`` -- and are renamed.

A model defined at runtime or outside the library has no native sector;
it is wrapped by the ``legacy`` one, radiation off, which reproduces its
expansion exactly but cannot give the sound horizon its radiation era
-- ``compute_rd`` and the CMB distance priors are refused there, and
``exact=True`` is the way.

``to_info(exact=True)``: the fitter's numbers
---------------------------------------------
The ``legacy_cosmology`` theory and the old dataset names
(:mod:`core.legacy`): the fitter's own model and likelihood classes,
run by the new core. ``chi2`` is the fitter's at every point, for every
model and dataset -- the reference to compare the native input against.

The parameters, either way
--------------------------
Each free one gets its bounds as a uniform prior, its initial value as
the centre of a narrow ``ref``, and a hundredth of its range as a
``proposal``. Fixed values are written for every parameter some
component takes; the ones nothing takes (the fitter carries every
model's full parameter set) are left out.
"""

from __future__ import annotations

import math

import numpy as np


__all__ = [
    "fitter_to_info", "RENAMED", "CoreChains",
    "sample_on_core", "profile_on_core", "fisher_on_core", "evidence_on_core",
]


#: Old parameter names the native sectors spell differently, by model.
RENAMED = {
    "GCG": {"A_s": "A_gcg", "alpha": "alpha_gcg"},
    "FTPowerLaw": {"n": "n_ft"},
    "FRHuSawicki": {"n": "n_hs"},
}

#: Theories a likelihood may need beyond the background, in the order
#: they are tried.
_EXTRA_THEORIES = ("early_universe", "growth", "camb")


def _builtin_name(model_cls) -> str | None:

    import cosmofit

    name = model_cls.__name__

    return name if getattr(cosmofit, name, None) is model_cls else None


def _model_reference(model_cls):
    """The model as an input names it: built-in name, import path, or class."""

    import importlib

    name = _builtin_name(model_cls)

    if name is not None:
        return name

    try:
        module = importlib.import_module(model_cls.__module__)
        found = getattr(module, model_cls.__qualname__, None)
    except ImportError:
        found = None

    if found is model_cls:
        return f"{model_cls.__module__}:{model_cls.__qualname__}"

    # Built at runtime (define_model, Action.build): only the class
    # itself names it, and the input works in this session only.
    return model_cls


def _build(kind, name, options):

    from cosmofit.core.registry import resolve

    cls = resolve(kind, name, options)

    return cls({k: v for k, v in options.items() if k != "class"}, name=name)


def _free(fitter, rename) -> dict:

    lower = dict(zip(fitter.free_params, fitter.prior.lower))
    upper = dict(zip(fitter.free_params, fitter.prior.upper))

    params = {}

    for name in fitter.free_params:

        low, high = float(lower[name]), float(upper[name])
        width = 0.01 * (high - low)

        params[rename.get(name, name)] = {
            "prior": {"min": low, "max": high},
            "ref": {"dist": "norm", "loc": float(fitter.initial[name]), "scale": width},
            "proposal": width,
        }

    return params


# ============================================================

def _exact(fitter) -> dict:

    theory = {"model": _model_reference(fitter.model_cls)}

    if fitter.compute_rd:
        theory["compute_rd"] = True

    if fitter.derive_sigma8:
        theory["derive_sigma8"] = True

    params = _free(fitter, {})

    # The wrapped model takes every parameter of its class -- except
    # what it then computes itself.
    computed = {"rd"} if fitter.compute_rd else set()

    if fitter.derive_sigma8:
        computed.add("sigma8")

    for name, value in fitter._initial_all.items():
        if name not in params and name not in computed:
            params[name] = float(value)

    return {
        "theory": {"legacy_cosmology": theory},
        "likelihood": {
            name: dict(fitter.dataset_kwargs.get(name, {})) for name in fitter.dataset_names
        },
        "params": params,
    }


def _native(fitter) -> dict:

    from cosmofit.data.metadata import DATASETS
    from cosmofit.likelihoods.native import NATIVE_LIKELIHOODS
    from cosmofit.theories.sectors import get_dark_sector

    name = _builtin_name(fitter.model_cls)

    if name is not None:
        background = {"dark_energy": get_dark_sector(name).name}
        rename = RENAMED.get(name, {})
    else:
        background = {
            "dark_energy": {"name": "legacy", "model": _model_reference(fitter.model_cls)},
            "radiation": False,
        }
        rename = {}

    theories = {"background": background}

    likelihoods = {}

    for dataset in fitter.dataset_names:

        likelihood = DATASETS[dataset].likelihood
        options = dict(fitter.dataset_kwargs.get(dataset, {}))

        if NATIVE_LIKELIHOODS[likelihood].uses_rd and not fitter.compute_rd:
            options["rd"] = "free"

        likelihoods[likelihood] = options

    candidates = {
        "early_universe": {},
        "camb": {},
        "growth": {"amplitude": "boltzmann"} if fitter.derive_sigma8 else {},
    }

    components = {
        ("likelihood", n): _build("likelihood", n, o) for n, o in likelihoods.items()
    }

    def add(theory):
        theories[theory] = candidates.get(theory, theories.get(theory))
        components[("theory", theory)] = _build("theory", theory, theories[theory])

    theories_to_add = ["background"] + (["camb", "growth"] if fitter.derive_sigma8 else [])

    for theory in theories_to_add:
        add(theory)

    def missing():

        provided, required = set(), set()

        for (kind, _), component in components.items():
            if kind == "theory":
                provided.update(component.get_can_provide())
            required.update(component.get_requirements())

        return required - provided

    # Add theories for what is required and not yet provided, until
    # nothing is missing -- a theory added may require more.
    for theory in _EXTRA_THEORIES:

        needed = missing()

        if needed and theory not in theories:

            component = _build("theory", theory, candidates[theory])

            if set(component.get_can_provide()) & needed:
                theories[theory] = candidates[theory]
                components[("theory", theory)] = component

    if "early_universe" in theories and background.get("radiation") is False:
        raise ValueError(
            f"{fitter.model_cls.__name__} is outside the library, so the "
            f"native background wraps it without radiation -- and the sound "
            f"horizon these datasets need is an integral through the "
            f"radiation era. to_info(exact=True) writes the fitter's own "
            f"classes instead."
        )

    if missing():
        raise ValueError(
            f"No theory of the new core provides {sorted(missing())} for "
            f"this fitter's datasets with its model. A model outside the "
            f"library has no radiation era for the sound horizon; "
            f"to_info(exact=True) writes the fitter's own classes instead."
        )

    def taken(parameter):
        return any(c.accepts(parameter) for c in components.values())

    params = _free(fitter, rename)

    for old in fitter.free_params:

        if not taken(rename.get(old, old)):
            raise ValueError(
                f"No component of the new core takes '{old}', which this "
                f"fitter samples. A parameter the model derives, or one no "
                f"dataset here reads, has no likelihood to constrain it; "
                f"drop it from free_params."
            )

    for old, value in fitter._initial_all.items():

        new = rename.get(old, old)

        if new not in params and taken(new):
            params[new] = float(value)

    return {"theory": theories, "likelihood": likelihoods, "params": params}


def fitter_to_info(fitter, exact: bool = False) -> dict:
    """
    The input for the new core equivalent to ``fitter``: on the native
    theories (the default), or through the fitter's own classes
    (``exact=True``). See the module docstring.

    Raises
    ------
    ValueError
        For a free parameter no component takes -- one the model
        derives, which the fitter samples to no effect -- or, natively,
        for a model outside the library with ``compute_rd``.
    """

    return _exact(fitter) if exact else _native(fitter)


# ============================================================
# A Fitter's calculations, run by the 2.0 core
# ============================================================
#
# The GUI builds a `Fitter` -- for its checks, its warnings and its
# figures -- and leaves the calculations to the core. Each function
# here runs one on `fitter.to_info(exact=True)`, so the numbers are the
# fitter's own model and likelihoods, and hands the result back in the
# shape the fitter's own method returned.


class CoreChains:
    """
    The core's chains, wearing the part of ``emcee.EnsembleSampler``
    that :class:`~stats.fitter.Fitter` and its figures read:
    ``get_chain``, ``get_log_prob``, ``get_autocorr_time``,
    ``nwalkers``, ``iteration``, ``acceptance_fraction``.

    Each Metropolis-Hastings chain is one "walker": its weighted rows
    are expanded back into one row per step, and every chain takes the
    same number of steps, so together they are an array of
    ``(steps, chains, parameters)`` exactly as an ensemble's walkers
    are. emcee's walkers come back as they were.

    Parameters
    ----------
    chain : ndarray
        ``(steps, walkers, parameters)``.
    log_prob : ndarray
        ``(steps, walkers)``.
    acceptance : array_like
        Per walker.
    stopped_by : dict
        The sampler's own stopping rule: ``rule``, ``value``,
        ``target`` and whether it was met (``converged``).
        :meth:`~stats.fitter.Fitter.convergence` reports that rather
        than its own rule, which is emcee's.
    output : str or None
        The prefix the core wrote the run under.
    """

    def __init__(self, chain, log_prob, acceptance, stopped_by, output=None):

        self.chain = np.asarray(chain, dtype=float)
        self.log_prob = np.asarray(log_prob, dtype=float)
        self.acceptance_fraction = np.asarray(acceptance, dtype=float)
        self.stopped_by = dict(stopped_by)
        self.output = output

        #: Whether a saved run was read back without a new step.
        self.reused = False

    @classmethod
    def from_sampler(cls, sampler, output=None) -> "CoreChains":
        """From a finished ``mcmc`` or ``emcee`` sampler of the core."""

        from cosmofit.samplers.emcee import Emcee

        products = sampler.products(skip=0.0)
        d = len(products["names"])

        if isinstance(sampler, Emcee):

            rows = products["chains"][0].reshape(-1, sampler.walkers, 2 + len(sampler.columns))

            return cls(
                rows[:, :, 2: 2 + d], -rows[:, :, 1],
                np.full(sampler.walkers, sampler.acceptance),
                {
                    "rule": f"{sampler.options['tau_factor']:g} autocorrelation times",
                    "value": sampler.tau,
                    "target": None,
                    "converged": sampler.converged,
                },
                output,
            )

        expanded, accepted = [], []

        for rows in products["chains"]:

            weights = rows[:, 0].astype(int)

            expanded.append(np.repeat(rows, weights, axis=0))
            accepted.append(max(len(rows) - 1, 0) / max(weights.sum(), 1))

        steps = min(len(rows) for rows in expanded)
        stacked = np.stack([rows[:steps] for rows in expanded], axis=1)

        return cls(
            stacked[:, :, 2: 2 + d], -stacked[:, :, 1], accepted,
            {
                "rule": f"R - 1 < {sampler.options['Rminus1_stop']:g}",
                "value": sampler.Rminus1,
                "target": sampler.options["Rminus1_stop"],
                "converged": sampler.converged,
            },
            output,
        )

    @property
    def nwalkers(self) -> int:
        return int(self.chain.shape[1])

    @property
    def ndim(self) -> int:
        return int(self.chain.shape[2])

    @property
    def iteration(self) -> int:
        return int(self.chain.shape[0])

    def get_chain(self, discard=0, thin=1, flat=False):

        chain = self.chain[int(discard)::int(thin)]

        return chain.reshape(-1, self.ndim) if flat else chain

    def get_log_prob(self, discard=0, thin=1, flat=False):

        log_prob = self.log_prob[int(discard)::int(thin)]

        return log_prob.reshape(-1) if flat else log_prob

    def get_autocorr_time(self, discard=0, thin=1, quiet=False, **kwargs):

        import emcee

        return thin * emcee.autocorr.integrated_time(
            self.get_chain(discard=discard, thin=thin), quiet=quiet, **kwargs,
        )


def _core_info(fitter, sampler: str, options: dict | None, centre=None) -> dict:

    info = fitter_to_info(fitter, exact=True)

    # Start the core where the fitter has already been -- its best fit
    # -- rather than at its initial values.
    for name, value in (centre or {}).items():
        declaration = info["params"].get(name)
        if isinstance(declaration, dict) and isinstance(declaration.get("ref"), dict):
            declaration["ref"]["loc"] = float(value)

    info["sampler"] = {sampler: dict(options or {})}

    return info


def _best_fit(fitter) -> dict | None:
    return dict(fitter.best_fit_params) if fitter.best_fit_result is not None else None


def _needs_camb(fitter) -> bool:

    from cosmofit.data.metadata import DATASETS

    return any(DATASETS[name].needs_camb for name in fitter.dataset_names)


def sample_on_core(
    fitter, sampler: str = "mcmc", options: dict | None = None, *,
    output: str | None = None, seed: int | None = None, callback=None,
    burn_in: float = 0.3,
):
    """
    Sample ``fitter``'s posterior with the core's ``mcmc`` or ``emcee``
    sampler, and make the result the fitter's own: ``fitter.sampler``
    becomes a :class:`CoreChains`, so ``summary()``, ``convergence()``,
    ``best_fit()`` and every figure read it as they read emcee.

    With ``output``, the run is written there in getdist's format, and
    a run already there is continued -- or, if it had met its stopping
    rule, read back without a single new step.

    Parameters
    ----------
    burn_in : float
        For ``mcmc``, the fraction of each chain's steps discarded as
        burn-in. ``emcee`` discards two autocorrelation times.

    Returns
    -------
    The core's sampler, for its own products.
    """

    from cosmofit.core import run
    from cosmofit.core.output import Output

    info = _core_info(fitter, sampler, options, _best_fit(fitter))

    if output is not None:

        info["output"] = str(output)

        if Output(info).existing():
            info["resume"] = True

    _, core = run(info, seed=seed, callback=callback)

    chains = CoreChains.from_sampler(core, output=output)

    # Read back without a new step: the saved run had met its rule,
    # and the only check made was the one before moving.
    chains.reused = bool(info.get("resume")) and core.converged and len(core.progress) == 1

    fitter.sampler = chains

    if sampler == "emcee":
        tau = core.tau if math.isfinite(core.tau) else 0.0
        fitter.burnin = min(chains.iteration - 1, int(math.ceil(2 * tau)))
    else:
        fitter.burnin = int(burn_in * chains.iteration)

    return core


def profile_on_core(fitter, name: str, values) -> dict:
    """
    :meth:`~stats.fitter.Fitter.profile`, by the core's ``profile``
    sampler: ``name``, ``values``, ``chi2``, ``delta_chi2`` and
    ``params`` (a dict per value, as the fitter returned them). Starts
    from the fitter's best fit when it has one, and minimizes without
    gradients when a Boltzmann code's noise would defeat them.
    """

    from cosmofit.core import run

    options = {
        "param": name,
        "values": [float(v) for v in np.atleast_1d(values)],
        "method": "Nelder-Mead" if _needs_camb(fitter) else "L-BFGS-B",
    }

    _, core = run(_core_info(fitter, "profile", options, _best_fit(fitter)))

    products = core.products()

    return {
        "name": name,
        "values": products["values"],
        "chi2": products["chi2"],
        "delta_chi2": products["delta_chi2"],
        "params": [
            {n: float(column[i]) for n, column in products["params"].items() if n != name}
            for i in range(len(products["values"]))
        ],
    }


def fisher_on_core(fitter, point: dict | None = None) -> dict:
    """
    :meth:`~stats.fitter.Fitter.fisher`, by the core's ``fisher``
    sampler: at ``point``, else the fitter's best fit, else at a best
    fit found first. Returns the fitter's keys -- ``matrix``,
    ``covariance``, ``errors`` (an array), ``theta``, ``steps``,
    ``free_params``, ``positive_definite`` -- and the core's
    ``unstable_steps``.
    """

    from cosmofit.core import run

    point = point if point is not None else _best_fit(fitter)

    options = {"point": dict(point)} if point is not None else {}

    _, core = run(_core_info(fitter, "fisher", options))

    products = core.products()
    names = products["names"]

    return {
        "matrix": products["matrix"],
        "covariance": products["covariance"],
        "errors": np.array([products["errors"][n] for n in names]),
        "theta": np.array([products["point"][n] for n in names]),
        "steps": np.array([products["steps"][n] for n in names]),
        "free_params": list(names),
        "positive_definite": products["positive_definite"],
        "unstable_steps": products["unstable_steps"],
    }


def evidence_on_core(fitter, nlive: int = 400, seed: int | None = 42):
    """
    :func:`stats.nested.run_nested`, by the core's ``nested`` sampler,
    returned as the same :class:`~stats.nested.NestedResult`, so
    :func:`stats.evidence.bayes_factor` compares it as before.
    """

    from cosmofit.core import run
    from cosmofit.stats.nested import NestedResult

    _, core = run(_core_info(fitter, "nested", {"nlive": int(nlive)}), seed=seed)

    products = core.products()

    weights = products["weights"] / products["weights"].sum()

    # Equal weights, as the fitter's nested result carried them.
    counts = np.random.default_rng(seed).multinomial(len(weights), weights)
    samples = np.repeat(products["samples"], counts, axis=0)

    width = np.asarray(fitter.prior.upper) - np.asarray(fitter.prior.lower)

    return NestedResult(
        log_evidence=products["log_evidence"],
        log_evidence_error=products["log_evidence_error"],
        samples=samples,
        free_params=list(products["names"]),
        prior_volume=float(np.prod(width)),
        n_live=products["nlive"],
        n_evaluations=products["n_evaluations"],
        information=products["information"],
        dimensionality=products["dimensionality"],
    )
