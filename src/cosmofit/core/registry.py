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
        "legacy_cosmology": "cosmofit.core.legacy:LegacyCosmology",
        "background": "cosmofit.theories.background:Background",
        "early_universe": "cosmofit.theories.early:EarlyUniverse",
        "growth": "cosmofit.theories.growth:Growth",
        "camb": "cosmofit.theories.boltzmann:CAMB",
    },
    "likelihood": {
        "bao.desi": "cosmofit.likelihoods.native:DESI",
        "bao.sdss": "cosmofit.likelihoods.native:SDSSBAO",
        "bao.sdss_fullshape": "cosmofit.likelihoods.native:SDSSFullShape",
        "bao.eboss_elg": "cosmofit.likelihoods.native:EBOSSELG",
        "bao.eboss_elg_fullshape": "cosmofit.likelihoods.native:EBOSSELGFullShape",
        "bao.desi_fullshape": "cosmofit.likelihoods.desi_fullshape:DESIFullShape",
        "bao.eboss_lya": "cosmofit.likelihoods.native:EBOSSLya",
        "bao.lowz": "cosmofit.likelihoods.native:BAOLowZ",
        "sn.pantheonplus": "cosmofit.likelihoods.native:PantheonPlus",
        "sn.des_sn5yr": "cosmofit.likelihoods.native:DESSN5YR",
        "sn.union3": "cosmofit.likelihoods.native:Union3",
        "cc.chronometers": "cosmofit.likelihoods.native:CosmicChronometers",
        "rsd.fsigma8": "cosmofit.likelihoods.native:FSigma8",
        "lss.s8": "cosmofit.likelihoods.native:S8",
        "cmb.distance_priors": "cosmofit.likelihoods.native:DistancePriors",
        "cmb.planck_lite": "cosmofit.likelihoods.native:PlanckLite",
        "cmb.planck_lowe": "cosmofit.likelihoods.native:PlanckLowE",
        "cmb.planck_lensing": "cosmofit.likelihoods.native:PlanckLensing",
        "cmb.act_lensing": "cosmofit.likelihoods.native:ACTLensing",
        "external.h0": "cosmofit.likelihoods.native:H0",
        "external.bbn": "cosmofit.likelihoods.native:BBN",
        "external.tau": "cosmofit.likelihoods.native:Tau",
    },
    "sampler": {
        "evaluate": "cosmofit.samplers.evaluate:Evaluate",
        "minimize": "cosmofit.samplers.minimize:Minimize",
        "mcmc": "cosmofit.samplers.mcmc:MCMC",
        "emcee": "cosmofit.samplers.emcee:Emcee",
        "nested": "cosmofit.samplers.nested:Nested",
        "fisher": "cosmofit.samplers.fisher:Fisher",
        "profile": "cosmofit.samplers.profile:Profile",
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

    from cosmofit.stats.fitter import DATASET_REGISTRY

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
