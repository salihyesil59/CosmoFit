"""
Turning a name in an input into a class.

A theory, likelihood or sampler is named in an input by one of:

* a built-in name -- ``minimize``, ``legacy_cosmology``, or any of the
  library's dataset names (``desi``, ``pantheon``, ...), which resolve
  to the wrapped classes in :mod:`core.legacy`;
* an import path, ``package.module:Class`` or ``package.module.Class``,
  for a component that lives in someone else's package;
* an entry point in the group ``cosmofit.theories``,
  ``cosmofit.likelihoods`` or ``cosmofit.samplers``, so an installed
  package can register names of its own;
* the ``class`` key of the component's options, which overrides the
  name and lets one class be listed twice under different names.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points


__all__ = ["resolve", "builtin_names", "KINDS"]


KINDS = ("theory", "likelihood", "sampler")

#: Built-in components, ``{kind: {name: import path}}``.
_BUILTIN = {
    "theory": {
        "legacy_cosmology": "CosmoFit.core.legacy:LegacyCosmology",
        "background": "CosmoFit.theories.background:Background",
        "early_universe": "CosmoFit.theories.early:EarlyUniverse",
        "growth": "CosmoFit.theories.growth:Growth",
        "camb": "CosmoFit.theories.boltzmann:CAMB",
    },
    "likelihood": {
        "bao.desi": "CosmoFit.likelihoods.native:DESI",
        "bao.sdss": "CosmoFit.likelihoods.native:SDSSBAO",
        "bao.sdss_fullshape": "CosmoFit.likelihoods.native:SDSSFullShape",
        "bao.eboss_elg": "CosmoFit.likelihoods.native:EBOSSELG",
        "bao.eboss_elg_fullshape": "CosmoFit.likelihoods.native:EBOSSELGFullShape",
        "bao.eboss_lya": "CosmoFit.likelihoods.native:EBOSSLya",
        "bao.lowz": "CosmoFit.likelihoods.native:BAOLowZ",
        "sn.pantheonplus": "CosmoFit.likelihoods.native:PantheonPlus",
        "sn.des_sn5yr": "CosmoFit.likelihoods.native:DESSN5YR",
        "sn.union3": "CosmoFit.likelihoods.native:Union3",
        "cc.chronometers": "CosmoFit.likelihoods.native:CosmicChronometers",
        "rsd.fsigma8": "CosmoFit.likelihoods.native:FSigma8",
        "lss.s8": "CosmoFit.likelihoods.native:S8",
        "cmb.distance_priors": "CosmoFit.likelihoods.native:DistancePriors",
        "cmb.planck_lite": "CosmoFit.likelihoods.native:PlanckLite",
        "cmb.planck_lowe": "CosmoFit.likelihoods.native:PlanckLowE",
        "cmb.planck_lensing": "CosmoFit.likelihoods.native:PlanckLensing",
        "cmb.act_lensing": "CosmoFit.likelihoods.native:ACTLensing",
        "external.h0": "CosmoFit.likelihoods.native:H0",
        "external.bbn": "CosmoFit.likelihoods.native:BBN",
        "external.tau": "CosmoFit.likelihoods.native:Tau",
    },
    "sampler": {
        "evaluate": "CosmoFit.samplers.evaluate:Evaluate",
        "minimize": "CosmoFit.samplers.minimize:Minimize",
        "mcmc": "CosmoFit.samplers.mcmc:MCMC",
        "emcee": "CosmoFit.samplers.emcee:Emcee",
        "nested": "CosmoFit.samplers.nested:Nested",
        "fisher": "CosmoFit.samplers.fisher:Fisher",
        "profile": "CosmoFit.samplers.profile:Profile",
    },
}


def _import(path: str):

    if ":" in path:
        module, _, attribute = path.partition(":")
    else:
        module, _, attribute = path.rpartition(".")

    if not module or not attribute:
        raise ValueError(f"Not an import path: {path!r}.")

    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Could not import {path!r}: {error}") from None


def _entry_point(kind: str, name: str):

    group = f"cosmofit.{kind}s" if kind != "theory" else "cosmofit.theories"

    for point in entry_points(group=group):
        if point.name == name:
            return point.load()

    return None


def _legacy_datasets() -> dict:

    from CosmoFit.stats.fitter import DATASET_REGISTRY

    return DATASET_REGISTRY


def builtin_names(kind: str) -> list[str]:
    """Every name ``resolve`` knows without an import path."""

    names = set(_BUILTIN[kind])

    if kind == "likelihood":
        names.update(_legacy_datasets())

    return sorted(names)


def resolve(kind: str, name: str, info: dict | None = None):
    """
    The class a component name stands for.

    Parameters
    ----------
    kind : {"theory", "likelihood", "sampler"}
    name : str
        The name it is listed under in the input.
    info : dict, optional
        Its options; a ``class`` key there takes precedence over ``name``.

    Returns
    -------
    type
        For a legacy dataset name, :class:`core.legacy.LegacyLikelihood`.
    """

    if kind not in KINDS:
        raise ValueError(f"Unknown component kind {kind!r}; one of {KINDS}.")

    target = (info or {}).get("class") or name

    if isinstance(target, type):
        return target

    if target in _BUILTIN[kind]:
        return _import(_BUILTIN[kind][target])

    if kind == "likelihood" and target in _legacy_datasets():

        from .legacy import LegacyLikelihood

        return LegacyLikelihood

    found = _entry_point(kind, target)

    if found is not None:
        return found

    if ":" in target or "." in target:
        return _import(target)

    raise ValueError(
        f"Unknown {kind} {target!r}. Built in: {builtin_names(kind)}. "
        f"Anything else is named by import path, 'package.module:Class'."
    )
