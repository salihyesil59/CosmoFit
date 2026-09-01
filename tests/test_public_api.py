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

import ast
import importlib
import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import CosmoFit
from CosmoFit import LCDM, Fitter


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
    "FTPowerLaw",
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


def _type_checking_namespace(module) -> dict:
    """
    The names a module imports under ``if TYPE_CHECKING``, imported
    for real.

    A type checker reads that block; the interpreter does not, which
    is the whole point of it -- ``matplotlib.figure`` costs 0.65 s to
    import, more than the rest of this library put together, and
    nobody who never draws a figure should pay it. So the annotation
    is correct for a checker and unresolvable at runtime, and a test
    that wants to verify it has to do what the checker does.
    """

    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):  # pragma: no cover - a builtin module
        return {}

    namespace = {}

    for node in ast.walk(ast.parse(source)):

        if not isinstance(node, ast.If):
            continue

        test = getattr(node.test, "id", None) or getattr(node.test, "attr", None)

        if test != "TYPE_CHECKING":
            continue

        for statement in node.body:

            if isinstance(statement, ast.ImportFrom):
                imported = importlib.import_module(statement.module)
                for alias in statement.names:
                    namespace[alias.asname or alias.name] = getattr(
                        imported, alias.name
                    )

            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    imported = importlib.import_module(alias.name)
                    namespace[alias.asname or alias.name.split(".")[0]] = (
                        imported if alias.asname
                        else importlib.import_module(alias.name.split(".")[0])
                    )

    return namespace


def _public_callables():
    """Every function and method reachable from ``CosmoFit.__all__``."""

    for name in CosmoFit.__all__:

        obj = getattr(CosmoFit, name)

        if inspect.isclass(obj):
            for member_name, member in vars(obj).items():
                if inspect.isfunction(member) and not member_name.startswith("_"):
                    yield f"{name}.{member_name}", member

        elif inspect.isfunction(obj):
            yield name, obj


def test_public_annotations_resolve():
    """
    Every module here uses ``from __future__ import annotations``, so
    annotations are strings and a wrong one is invisible until
    somebody calls ``get_type_hints`` -- which no test and no user
    ever did.

    Resolved against the ``TYPE_CHECKING`` names too, because that is
    what a type checker sees, and what the ``py.typed`` marker this
    package now ships is a promise about.
    """

    unresolvable = []

    for label, fn in _public_callables():

        module = inspect.getmodule(fn)

        namespace = {**vars(module), **_type_checking_namespace(module)}

        try:
            typing.get_type_hints(fn, globalns=namespace)
        except Exception as exc:  # noqa: BLE001
            unresolvable.append(f"{label} ({type(exc).__name__}: {exc})")

    assert not unresolvable, "\n".join(unresolvable)


def test_the_public_surface_is_completely_annotated():
    """
    What ``py.typed`` claims, asserted rather than assumed.

    Shipping the marker tells every downstream type checker to trust
    this package's annotations. A surface that is *partly* annotated
    is worse than one that is not: the gaps come back as ``Any``,
    which silences errors instead of finding them. So the bar is all
    of it, and this is what holds it there -- adding a public method
    without annotating it fails here.

    ``*args`` and ``**kwargs`` are exempt, as they conventionally
    are: every one on this surface is a pass-through to emcee,
    corner or dynesty, and naming a type for it would be inventing
    one.
    """

    variadic = (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )

    unannotated = []

    for label, fn in _public_callables():

        signature = inspect.signature(fn)

        if signature.return_annotation is inspect.Signature.empty:
            unannotated.append(f"{label}: no return annotation")

        for name, parameter in signature.parameters.items():

            if name == "self" or parameter.kind in variadic:
                continue

            if parameter.annotation is inspect.Parameter.empty:
                unannotated.append(f"{label}: parameter '{name}'")

    assert not unannotated, "\n".join(unannotated)


def test_declared_return_types_are_what_is_actually_returned():
    """
    The annotations checked above are checked for *resolving*, not
    for being true. A wrong one is worse than none once `py.typed`
    ships, because it is then something downstream type checkers
    repeat with confidence.

    Two of these were wrong when they were first written, and both
    were wrong the same way -- named after the object that lands on
    `fitter.result` rather than the one handed back:

    * `best_fit` returns scipy's `OptimizeResult`, while
      `BestFitResult` is what `fitter.result.best_fit` holds;
    * `run_mcmc` returns emcee's `EnsembleSampler`, while
      `MCMCResult` is what `fitter.result.mcmc` holds.

    Neither mypy nor the resolution test could catch that -- scipy
    and emcee both hand back `Any` -- so this calls them and looks.
    """

    import emcee
    from scipy.optimize import OptimizeResult

    from CosmoFit import CosmologyParameters  # noqa: F401  (import check)
    from CosmoFit.stats import fitter as fitter_module

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 70.0, "Omega_m": 0.3},
    )

    calls = [
        ("best_fit", lambda: fit.best_fit()),
        ("run_mcmc", lambda: fit.run_mcmc(
            nwalkers=8, nsteps=40, burnin=5, progress=False,
        )),
        ("summary", lambda: fit.summary()),
        ("flat_samples", lambda: fit.flat_samples()),
        ("convergence", lambda: fit.convergence()),
        ("samples_dict", lambda: fit.samples_dict()),
        ("chi2", lambda: fit.chi2()),
        ("chi2_breakdown", lambda: fit.chi2_breakdown()),
        ("chain_id", lambda: fit.chain_id()),
        ("fisher", lambda: fit.fisher()),
    ]

    namespace = {
        **vars(fitter_module),
        **_type_checking_namespace(fitter_module),
        "emcee": emcee,
        "OptimizeResult": OptimizeResult,
    }

    wrong = []

    for name, call in calls:

        declared = typing.get_type_hints(
            getattr(Fitter, name), globalns=namespace,
        )["return"]

        returned = call()

        if not isinstance(returned, declared):
            wrong.append(
                f"Fitter.{name}: declared {declared}, "
                f"returned {type(returned).__module__}."
                f"{type(returned).__name__}"
            )

    assert not wrong, "\n".join(wrong)


def test_the_py_typed_marker_is_shipped():
    """
    The marker is what makes any of the above visible downstream. It
    has to be in the installed package, not merely in the
    repository -- so this checks the imported package's own
    directory, and `pyproject.toml`'s `package-data` is what puts it
    there.
    """

    marker = Path(CosmoFit.__file__).parent / "py.typed"

    assert marker.exists(), (
        "py.typed is missing from the installed package -- "
        "check [tool.setuptools.package-data] in pyproject.toml"
    )


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
