"""
The ``cosmofit`` command.

::

    cosmofit run input.yaml [--output PREFIX] [--force | --resume] [--seed N]
    cosmofit list [theory | likelihood | sampler]
    cosmofit doc NAME
    cosmofit install [NAME | input.yaml ...] [--path DIR] [--force] [--list]
    cosmofit gui [streamlit options]

``run`` runs an input: its sampler, or -- for an input with a ``post``
block -- the reweighting of a finished run (:mod:`core.post`). Under
``mpirun -n 4 cosmofit run input.yaml`` the ``mcmc`` sampler shares its
chains between the four processes. What the run found is summarized on
the terminal; everything it found is in the output prefix.

``list`` names every component an input can use without an import path:
the built-in ones and those installed packages register under the
``cosmofit.theories``, ``cosmofit.likelihoods`` and
``cosmofit.samplers`` entry points.

``doc`` shows what one of them is: its description, the options it
takes, the parameters it brings with their default declarations, and
what it provides or requires -- the starting point for writing it into
an input.

``install`` downloads the data packages too large to ship with the
library (:mod:`cosmofit.install`): by name, or every one the
components of an input need; ``--list`` shows them and whether each is
installed.

``gui`` starts the graphical interface (:mod:`cosmofit.gui`); what
follows it goes to ``streamlit run``.

``python -m cosmofit`` is the same command.
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import textwrap

import numpy as np


__all__ = ["main"]


_KINDS = {"theory": "theory", "likelihood": "likelihood", "sampler": "sampler"}


# ============================================================
# run
# ============================================================

def _weighted_summary(names, samples, weights) -> list[str]:

    if len(samples) == 0 or weights.sum() <= 0:
        return ["  no samples"]

    mean = np.average(samples, axis=0, weights=weights)
    std = np.sqrt(np.average((samples - mean) ** 2, axis=0, weights=weights))

    width = max(len(n) for n in names)

    return [f"  {n:<{width}}  {m: .6g} +- {s:.3g}" for n, m, s in zip(names, mean, std)]


def _report(name: str, products: dict) -> list[str]:
    """What is worth printing of a sampler's products."""

    lines = []

    if "chi2" in products and isinstance(products["chi2"], dict) and "point" in products:

        for key, value in products["point"].items():
            lines.append(f"  {key} = {value:.6g}")

        for key, value in products["chi2"].items():
            lines.append(f"  chi2[{key}] = {value:.4f}")

    if "errors" in products:
        for key, value in products["errors"].items():
            point = products["point"][key]
            lines.append(f"  {key} = {point:.6g} +- {value:.3g}")

    if "delta_chi2" in products:

        lines.append(f"  {products['param']:>12}  delta_chi2")

        for value, delta in zip(products["values"], products["delta_chi2"]):
            lines.append(f"  {value:>12.6g}  {delta:.4f}")

    if "samples" in products and "names" in products:

        burn = "after burn-in " if name in ("mcmc", "emcee") else ""

        lines.append(f"  {len(products['samples'])} samples {burn}(weighted mean +- std):")
        lines.extend(_weighted_summary(
            products["names"], products["samples"], products["weights"],
        ))

    for key, label in (
        ("Rminus1", "R - 1"), ("tau", "autocorrelation time"),
        ("acceptance", "acceptance"), ("converged", "converged"),
        ("log_evidence", "ln Z"), ("ess_before", "effective samples before"),
        ("ess_after", "effective samples after"), ("n_rejected", "rejected"),
    ):

        if key in products:

            value = products[key]

            if key == "log_evidence":
                value = f"{value:.3f} +- {products['log_evidence_error']:.3f}"
            elif isinstance(value, float) and math.isfinite(value):
                value = f"{value:.4g}"

            lines.append(f"  {label}: {value}")

    return lines


def _run(args) -> int:

    from cosmofit.core import load_info, run
    from cosmofit.core.mpi import world

    info = load_info(args.input)

    if args.output is not None:
        info["output"] = args.output

    if args.force:
        info["force"] = True

    if args.resume:
        info["resume"] = True

    info, sampler = run(info, seed=args.seed)

    if world().rank != 0:
        return 0

    name = "post" if "post" in info else next(iter(info["sampler"]))

    products = sampler.products(skip=0.3) if name == "mcmc" else sampler.products()

    print(f"{name}: done.")

    for line in _report(name, products):
        print(line)

    if info.get("output"):
        prefix = info["output"] + (f".post.{info['post']['suffix']}" if name == "post" else "")
        print(f"Output: {prefix}.*")

    return 0


# ============================================================
# list and doc
# ============================================================

def _first_paragraph(doc: str | None) -> str:
    """The opening prose of a docstring, or "" if it opens on a section."""

    lines = (doc or "").strip().split("\n")

    # A numpydoc section opens with its name over a rule of dashes.
    if len(lines) > 1 and lines[1].strip() and set(lines[1].strip()) == {"-"}:
        return ""

    return "\n".join(lines).split("\n\n")[0].replace("\n", " ")


def _summary_line(kind: str, name: str, cls) -> str:

    from cosmofit.data.metadata import DATASETS

    if kind == "likelihood" and name in DATASETS:
        return f"(old name) prefer {DATASETS[name].likelihood}"

    module = sys.modules.get(cls.__module__)

    return (
        _first_paragraph(inspect.getdoc(cls))
        or _first_paragraph(getattr(module, "__doc__", None))
    )


def _list(args) -> int:

    from cosmofit.core.registry import builtin_names, resolve

    kinds = [args.kind] if args.kind else list(_KINDS)

    for kind in kinds:

        print(f"{kind}:")

        for name in builtin_names(kind):

            try:
                summary = _summary_line(kind, name, resolve(kind, name))
            except ValueError:
                summary = ""

            print(f"  {name:<26} {textwrap.shorten(summary, 70, placeholder='...')}")

    return 0


def _find(name: str):
    """``(kind, class)`` for a component name, whichever kind it is."""

    from cosmofit.core.registry import resolve

    found = []

    for kind in _KINDS:

        try:
            found.append((kind, resolve(kind, name)))
        except ValueError:
            continue

    if not found:
        raise SystemExit(
            f"cosmofit doc: nothing is called {name!r}; 'cosmofit list' names "
            f"every component."
        )

    return found[0]


def _doc(args) -> int:

    import yaml

    kind, cls = _find(args.name)

    print(f"{args.name}  ({kind}, {cls.__module__}:{cls.__qualname__})")
    print()

    doc = inspect.getdoc(cls)

    if doc:
        print(textwrap.indent(doc, "  "))
        print()

    if kind == "sampler":

        print("Options, with their defaults:")
        print(textwrap.indent(yaml.safe_dump(dict(cls.defaults), sort_keys=False), "  "))

        return 0

    try:
        component = cls({}, name=args.name)
    except Exception as error:  # noqa: BLE001 -- shown, not hidden
        print(f"(Built with no options it says: {error})")
        return 0

    from cosmofit.core.output import _clean

    params = component.get_default_params()

    if params:
        print("Parameters it brings, as declared by default:")
        print(textwrap.indent(yaml.safe_dump(_clean(params), sort_keys=False), "  "))

    required = getattr(component, "get_required_params", lambda: [])()

    if required:
        print(f"Parameters it needs: {', '.join(required)}")

    if kind == "theory":

        provides = component.get_can_provide()

        if provides:
            print(f"Provides: {', '.join(provides)}")

    requirements = component.get_requirements()

    if requirements:
        print(f"Requires: {', '.join(requirements)}")

    return 0


# ============================================================
# install
# ============================================================

def _install(args) -> int:

    from pathlib import Path

    from cosmofit import install as data

    root = data.data_path(args.path)

    if args.list:

        packages = data.available_packages()

        if not packages:
            print("No data packages are registered.")
            return 0

        print(f"Data packages (installed in {root}):")

        for name, package in packages.items():

            state = "installed" if data.is_installed(name, args.path) else "-"
            size = f"{package.size / 1e6:.0f} MB" if package.size else ""

            print(f"  {name:<26} {state:<10} {size:>8}  {package.description}")

        return 0

    if not args.names:
        raise SystemExit("cosmofit install: name a package or an input, or pass --list.")

    names = []

    for name in args.names:

        if name.endswith((".yaml", ".yml")) and Path(name).is_file():

            from cosmofit.core import load_info

            needed = data.packages_for(load_info(name))

            print(f"{name} needs: {', '.join(needed) or 'no data packages'}")
            names.extend(needed)

        else:
            names.append(name)

    if names:
        print(f"Installing in {root}:")
        data.install(names, path=args.path, force=args.force)

    return 0


# ============================================================

def _parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="cosmofit",
        description="Cosmological parameter estimation from a YAML input.",
    )

    parser.add_argument(
        "--debug", action="store_true", help="show the full traceback on an error",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an input (a sampler, or a post block)")
    run.add_argument("input", help="the input, a YAML file")
    run.add_argument("-o", "--output", help="output prefix, overriding the input's")
    run.add_argument("--seed", type=int, help="seed for the sampler's random draws")

    mode = run.add_mutually_exclusive_group()
    mode.add_argument("-f", "--force", action="store_true", help="delete earlier output first")
    mode.add_argument("-r", "--resume", action="store_true", help="continue earlier output")

    listing = commands.add_parser("list", help="name the components an input can use")
    listing.add_argument("kind", nargs="?", choices=list(_KINDS))

    doc = commands.add_parser("doc", help="describe one component")
    doc.add_argument("name")

    install = commands.add_parser(
        "install", help="download data packages, by name or for an input",
    )
    install.add_argument("names", nargs="*", metavar="NAME", help="a package, or an input YAML")
    install.add_argument("--path", help="where packages go (default: $COSMOFIT_PACKAGES_PATH "
                                        "or ~/.cosmofit/packages)")
    install.add_argument("-f", "--force", action="store_true", help="download again")
    install.add_argument("--list", action="store_true", help="list the packages")

    # What follows `gui` goes to `streamlit run` as it is; see main().
    commands.add_parser(
        "gui", help="start the graphical interface; other options go to 'streamlit run'",
    )

    return parser


def _gui(args) -> int:

    from cosmofit.gui import main as start

    return start(args.streamlit_args)


def main(argv=None) -> int:
    """Entry point of the ``cosmofit`` command; returns the exit status."""

    parser = _parser()

    # Options cosmofit does not know are Streamlit's, after `gui`, and
    # an error anywhere else. (argparse's REMAINDER would not take
    # them: it stops at the first option-like word in a subcommand.)
    args, unknown = parser.parse_known_args(argv)

    if unknown and args.command != "gui":
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    args.streamlit_args = unknown

    handler = {
        "run": _run, "list": _list, "doc": _doc, "install": _install, "gui": _gui,
    }[args.command]

    try:
        return handler(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:  # noqa: BLE001 -- reported, or re-raised with --debug

        if args.debug:
            raise

        print(f"cosmofit {args.command}: {type(error).__name__}: {error}", file=sys.stderr)

        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
