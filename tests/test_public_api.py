"""
The public API, pinned.

A library that calls itself stable owes its users a surface that does
not move underneath them. Nothing else in this suite would notice a
name being renamed, moved between subpackages, or quietly dropped
from ``__all__`` -- every other test imports what it happens to need,
so it goes on passing as long as *some* path to the object exists.

So this module hard-codes the surface. The lists below are not
derived from the code; they are typed out, and a diff to one of them
is the point rather than a nuisance. Adding a name is a one-line
change here and is meant to be. Removing or renaming one is a
breaking change, and having to say so in this file is the reminder.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import typing

import pytest

import CosmoFit


# ============================================================
# The top-level surface
# ============================================================
#
# `from CosmoFit import X` for every X here is the contract almost
# every user relies on -- the README, the notebooks and the GUI all
# import this way.

TOP_LEVEL = {
    "__version__",
    # Cosmology
    "Cosmology",
    "CosmologyParameters",
    "ModelConfigurationError",
    "constants",
    "LCDM",
    "WCDM",
    "CPL",
    "JBP",
    "BA",
    "LogarithmicDE",
    "PEDE",
    "GEDE",
    "LsCDM",
    "GCG",
    "IDE",
    "RunningVacuum",
    "Cardassian",
    "HDE",
    "ADE",
    "RDE",
    "DGP",
    "FQExponential",
    "FRTLinear",
    "FRHuSawicki",
    "define_model",
    "model_from_expression",
    # Likelihoods
    "BaseLikelihood",
    "CCLikelihood",
    "DESILikelihood",
    "SDSSBAOLikelihood",
    "SDSSFullShapeLikelihood",
    "EBOSSELGLikelihood",
    "EBOSSELGFullShapeLikelihood",
    "EBOSSLyaLikelihood",
    "BAOLowZLikelihood",
    "PantheonLikelihood",
    "DESSN5YRLikelihood",
    "Union3Likelihood",
    "PlanckLikelihood",
    "PlanckLiteLikelihood",
    "PlanckLensingLikelihood",
    "PlanckLowEELikelihood",
    "ACTDR6LensingLikelihood",
    "H0Likelihood",
    "OmegaBLikelihood",
    "TauLikelihood",
    "FSigma8Likelihood",
    "S8Likelihood",
    "JointLikelihood",
    # Fitting / plotting
    "Fitter",
    "BaseSampler",
    "EnsembleSampler",
    "FitResult",
    "BestFitResult",
    "MCMCResult",
    "FitPlotter",
    # Saved chains
    "ChainFile",
    "StoredSampler",
    "open_chain",
    "chain_info",
    # Datasets
    "available_datasets",
    "available_versions",
    "dataset_reference",
}


def test_top_level_all_is_exactly_this():
    """
    The set, not merely a subset in either direction.

    A missing name is a break for somebody's script. An *extra* name
    is a promise nobody meant to make -- once it is in ``__all__`` it
    is public, and taking it back out is then itself a break.
    """

    assert set(CosmoFit.__all__) == TOP_LEVEL


def test_every_exported_name_actually_exists():
    """
    ``__all__`` is a list of strings and nothing validates it.

    A rename that misses this list leaves an entry pointing at
    nothing, and the only symptom is ``from CosmoFit import *``
    raising ``AttributeError`` -- which almost nobody writes, so it
    would be found by a user rather than here.
    """

    for name in CosmoFit.__all__:
        assert hasattr(CosmoFit, name), f"{name} is exported but missing"


def test_star_import_gives_exactly_the_public_names():

    namespace: dict = {}

    exec("from CosmoFit import *", namespace)  # noqa: S102

    namespace.pop("__builtins__", None)

    assert set(namespace) == set(CosmoFit.__all__)


def test_no_duplicate_entries():
    """
    A duplicate is harmless at runtime and a sign the list was edited
    in two places at once.
    """

    assert len(CosmoFit.__all__) == len(set(CosmoFit.__all__))


# ============================================================
# The subpackages
# ============================================================

SUBPACKAGES = [
    "CosmoFit.cosmology",
    "CosmoFit.cosmology.core",
    "CosmoFit.cosmology.models",
    "CosmoFit.cosmology.calculators",
    "CosmoFit.cosmology.numerics",
    "CosmoFit.likelihoods",
    "CosmoFit.stats",
    "CosmoFit.plots",
]


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_subpackage_exports_resolve(module_name):

    module = importlib.import_module(module_name)

    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name} is exported but missing"


# The layers of `cosmology` that are re-exported wholesale one level
# up. `ModelConfigurationError` and `GrowthCalculator` were both in a
# child's `__all__` and missing from the parent's for several
# releases -- reachable only by importing the private module path,
# while every one of their siblings was not.
COSMOLOGY_LAYERS = [
    "CosmoFit.cosmology.core",
    "CosmoFit.cosmology.models",
    "CosmoFit.cosmology.calculators",
    "CosmoFit.cosmology.numerics",
]


@pytest.mark.parametrize("module_name", COSMOLOGY_LAYERS)
def test_cosmology_re_exports_its_whole_layer(module_name):

    child = importlib.import_module(module_name)

    from CosmoFit import cosmology

    missing = set(child.__all__) - set(cosmology.__all__)

    assert not missing, (
        f"{module_name} exports {sorted(missing)}, which "
        f"CosmoFit.cosmology does not re-export"
    )


# ============================================================
# What importing the library is allowed to cost
# ============================================================

OPTIONAL_BACKENDS = ["sympy", "camb", "streamlit", "dynesty", "numba"]


def test_importing_the_library_does_not_import_the_optional_backends():
    """
    Every one of these is an extra, and the point of an extra is that
    somebody who did not install it can still ``import CosmoFit``.

    A subprocess rather than a check of ``sys.modules``: by the time
    this test runs, another test has almost certainly imported CAMB
    for its own reasons, and asking the current interpreter would
    then fail for a reason that has nothing to do with the library.
    """

    code = (
        "import sys; import CosmoFit; "
        f"leaked = [m for m in {OPTIONAL_BACKENDS!r} if m in sys.modules]; "
        "print(','.join(leaked))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    leaked = result.stdout.strip()

    assert not leaked, f"import CosmoFit pulled in: {leaked}"


# ============================================================
# The annotations on that surface
# ============================================================


def test_public_annotations_resolve():
    """
    Every module here uses ``from __future__ import annotations``, so
    annotations are strings and a wrong one is invisible until
    somebody calls ``get_type_hints`` -- which no test and no user
    ever did.

    This is what the package would have to survive before it could
    claim to be typed (a ``py.typed`` marker); it does not claim that
    yet, and this is the half of the claim that already holds.
    """

    unresolvable = []

    for name in CosmoFit.__all__:

        obj = getattr(CosmoFit, name)

        targets = []

        if inspect.isclass(obj):
            targets.append(obj)
            targets.extend(
                member
                for member in vars(obj).values()
                if inspect.isfunction(member)
            )

        elif inspect.isfunction(obj):
            targets.append(obj)

        for target in targets:
            try:
                typing.get_type_hints(target)
            except Exception as exc:  # noqa: BLE001
                unresolvable.append(
                    f"{name}: {getattr(target, '__qualname__', target)} "
                    f"({type(exc).__name__}: {exc})"
                )

    assert not unresolvable, "\n".join(unresolvable)


# ============================================================
# Version
# ============================================================


def test_version_is_a_release_number():
    """
    ``__version__`` falls back to ``0.0.0+unknown`` when the package
    metadata cannot be found, which is right for a bare source
    checkout and wrong for anything that got as far as running tests
    against an installed copy.
    """

    assert CosmoFit.__version__ != "0.0.0+unknown"

    parts = CosmoFit.__version__.split(".")

    assert len(parts) >= 2

    assert all(part.isdigit() for part in parts[:2])
