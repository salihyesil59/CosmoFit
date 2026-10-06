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


__all__ = ["fitter_to_info", "RENAMED"]


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
