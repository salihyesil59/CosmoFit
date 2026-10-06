"""
Where a run writes what it found, in files other tools read.

An input's ``output: chains/run`` is a prefix. A run writes:

``chains/run.input.yaml``
    The input as given.
``chains/run.updated.yaml``
    The input with every default filled in: each component's parameters
    as the run used them, and the sampler's options.
``chains/run.1.txt``, ``chains/run.2.txt``, ...
    The chains, one file per chain, in getdist's format: a header line
    of names, then one row per distinct point -- ``weight``,
    ``minuslogpost``, the sampled and derived parameters, then
    ``minuslogprior`` and each likelihood's ``chi2__<name>``.
``chains/run.paramnames``, ``chains/run.ranges``
    For getdist: each column's name and LaTeX label (derived ones
    starred), and each sampled parameter's prior support.
``chains/run.covmat``
    The proposal covariance the sampler ended with, to start the next
    run from -- or, from ``fisher``, the inverse Fisher matrix.
``chains/run.minimum.txt``
    The best fit ``minimize`` (or ``fisher``, before it expands) found,
    as one row of a chain.
``chains/run.profile.txt``
    A profile likelihood: the fixed values, ``chi2`` minimized over the
    rest, and where the rest went.
``chains/run.evidence.yaml``
    Nested sampling's ``ln Z``, its error, and what it was computed with.

A run refuses to write over another's files unless told what to do with
them: ``force: true`` deletes them first, ``resume: true`` continues from
them -- and only if the input they were written with is this one.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np


__all__ = ["Output", "OutputError", "ChainFile"]


class OutputError(RuntimeError):
    """An output prefix in a state the run cannot use."""


def _yaml():

    import yaml

    return yaml


def _clean(obj):
    """An input as plain YAML: classes and functions by name."""

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]

    if isinstance(obj, type):
        return f"{obj.__module__}:{obj.__qualname__}"

    if callable(obj):
        return getattr(obj, "__qualname__", repr(obj))

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    return obj


#: Top-level blocks that do not change what a run computes, and so may
#: differ between a run and its resumption.
_RUN_CONTROL = {"force", "resume", "debug"}


class ChainFile:
    """
    One chain on disk, appended to as it grows.

    Parameters
    ----------
    path : Path
    columns : list[str]
        Column names after ``weight`` and ``minuslogpost``.
    """

    def __init__(self, path: Path, columns: list[str]):

        self.path = Path(path)
        self.columns = list(columns)

    @property
    def header(self) -> str:
        names = ["weight", "minuslogpost", *self.columns]
        return "# " + " ".join(f"{n:>16}" for n in names)

    def create(self) -> None:
        self.path.write_text(self.header + "\n", encoding="utf-8")

    def append(self, rows: np.ndarray) -> None:
        """Rows of ``weight, minuslogpost, *columns``."""

        rows = np.atleast_2d(np.asarray(rows, dtype=float))

        if rows.size == 0:
            return

        with open(self.path, "a", encoding="utf-8") as handle:
            np.savetxt(handle, rows, fmt="%.10e", delimiter=" ")

    def read(self) -> np.ndarray:
        """Every row so far, ``(n, 2 + len(columns))``."""

        with open(self.path, encoding="utf-8") as handle:
            first = handle.readline()

        names = first.lstrip("#").split()

        if names[2:] != self.columns:
            raise OutputError(
                f"{self.path} has columns {names}, not the ones this run "
                f"writes; it was written by a different input."
            )

        data = np.loadtxt(self.path, comments="#", ndmin=2)

        return data.reshape(-1, 2 + len(self.columns))


class Output:
    """
    Parameters
    ----------
    info : dict
        The validated input. Its ``output`` prefix, ``force`` and
        ``resume`` decide what happens here; without ``output`` nothing
        is written.
    """

    def __init__(self, info: dict):

        prefix = info.get("output")

        self.info = info
        self.force = bool(info.get("force", False))
        self.resume = bool(info.get("resume", False))

        if self.force and self.resume:
            raise OutputError("'force' and 'resume' contradict each other; pick one.")

        self.prefix = None if prefix in (None, "") else Path(str(prefix))

    @property
    def enabled(self) -> bool:
        return self.prefix is not None

    def path(self, suffix: str) -> Path:
        return self.prefix.parent / f"{self.prefix.name}{suffix}"

    def existing(self) -> list[Path]:
        """Every file this prefix has written before."""

        if not self.enabled or not self.prefix.parent.exists():
            return []

        name = self.prefix.name

        return sorted(
            p for p in self.prefix.parent.iterdir()
            if p.is_file() and p.name.startswith(name + ".")
            and p.name[len(name) + 1:].split(".")[0] in {
                "input", "updated", "paramnames", "ranges", "covmat", "progress",
                "minimum", "profile", "evidence",
            } | {str(i) for i in range(1, 1000)}
        )

    # ---------------------------------------------------------

    def prepare(self) -> bool:
        """
        Make the prefix ready to write. Returns whether this run resumes
        an earlier one.

        Raises
        ------
        OutputError
            If earlier output is there and neither ``force`` nor
            ``resume`` says what to do with it, or if ``resume`` finds
            output written by a different input.
        """

        if not self.enabled:
            return False

        self.prefix.parent.mkdir(parents=True, exist_ok=True)

        found = self.existing()

        if not found:
            self._write_input()
            return False

        if self.force:

            for path in found:
                path.unlink()

            self._write_input()

            return False

        if not self.resume:
            raise OutputError(
                f"Output with prefix '{self.prefix}' already exists "
                f"({', '.join(p.name for p in found[:4])}"
                f"{', ...' if len(found) > 4 else ''}). Set 'resume: true' to "
                f"continue it, or 'force: true' to delete it and start over."
            )

        self._check_same_input()

        return True

    def _comparable(self, info: dict) -> dict:
        return _clean({k: v for k, v in info.items() if k not in _RUN_CONTROL})

    def _write_input(self) -> None:

        self.path(".input.yaml").write_text(
            _yaml().safe_dump(_clean(self.info), sort_keys=False), encoding="utf-8",
        )

    def _check_same_input(self) -> None:

        path = self.path(".input.yaml")

        if not path.exists():
            raise OutputError(
                f"Cannot resume '{self.prefix}': its {path.name} is missing, so "
                f"there is no telling what input wrote it."
            )

        earlier = _yaml().safe_load(path.read_text(encoding="utf-8")) or {}

        now = self._comparable(self.info)
        before = self._comparable(earlier)

        # The sampler's own stopping rules may be loosened or tightened
        # on resumption; what it samples may not.
        for block in (now, before):
            for options in (block.get("sampler") or {}).values():
                if isinstance(options, dict):
                    for key in ("max_samples", "Rminus1_stop", "max_time"):
                        options.pop(key, None)

        if now != before:

            changed = sorted(
                key for key in set(now) | set(before) if now.get(key) != before.get(key)
            )

            raise OutputError(
                f"Cannot resume '{self.prefix}': it was written by a different "
                f"input (blocks that differ: {changed}). Use a new output "
                f"prefix, or 'force: true' to overwrite."
            )

    # ---------------------------------------------------------

    def write_updated(self, model, sampler_options: dict) -> None:
        """The input as run, every default filled in."""

        if not self.enabled:
            return

        updated = copy.deepcopy(_clean(self.info))

        updated["params"] = {
            name: _clean(spec.declaration())
            for name, spec in model.parameters.specs.items()
        }

        (name,) = updated.get("sampler") or {"sampler": None}
        updated["sampler"] = {name: _clean(sampler_options)}

        self.path(".updated.yaml").write_text(
            _yaml().safe_dump(updated, sort_keys=False), encoding="utf-8",
        )

    def write_getdist_metadata(self, model, columns: list[str]) -> None:
        """``.paramnames`` and ``.ranges``."""

        if not self.enabled:
            return

        specs = model.parameters.specs
        sampled = set(model.parameters.sampled)

        lines = []

        for name in columns:

            spec = specs.get(name)

            if spec is not None:
                label = spec.label
            elif name == "minuslogprior":
                label = r"-\log\pi"
            elif name.startswith("chi2__"):
                label = r"\chi^2_{\rm " + name[6:].replace("_", r"\_") + "}"
            else:
                label = name

            star = "" if name in sampled else "*"

            lines.append(f"{name}{star}\t{label}")

        self.path(".paramnames").write_text("\n".join(lines) + "\n", encoding="utf-8")

        ranges = []

        for name in model.parameters.sampled:

            low, high = specs[name].prior.support

            ranges.append(
                f"{name}\t{'N' if math.isinf(low) else repr(float(low))}"
                f"\t{'N' if math.isinf(high) else repr(float(high))}"
            )

        self.path(".ranges").write_text("\n".join(ranges) + "\n", encoding="utf-8")

    def chain(self, index: int, columns: list[str]) -> ChainFile:
        """The ``index``-th chain (1-based), as getdist numbers them."""

        return ChainFile(self.path(f".{index}.txt"), columns)

    def write_covmat(self, names: list[str], covariance: np.ndarray) -> None:

        if not self.enabled:
            return

        np.savetxt(
            self.path(".covmat"), covariance, fmt="%.10e",
            header=" ".join(names), comments="# ",
        )

    def write_table(self, suffix: str, names: list[str], rows) -> None:
        """Rows under a ``#``-header of names, as the chains are written."""

        if not self.enabled:
            return

        np.savetxt(
            self.path(suffix), np.atleast_2d(np.asarray(rows, dtype=float)),
            fmt="%.10e", header=" ".join(f"{n:>16}" for n in names), comments="# ",
        )

    def write_yaml(self, suffix: str, content: dict) -> None:

        if not self.enabled:
            return

        self.path(suffix).write_text(
            _yaml().safe_dump(_clean(content), sort_keys=False), encoding="utf-8",
        )

    def read_covmat(self):
        """``(names, covariance)`` from this prefix, or ``None``."""

        if not self.enabled or not self.path(".covmat").exists():
            return None

        return load_covmat(self.path(".covmat"))


def load_covmat(path) -> tuple[list[str], np.ndarray]:
    """A covariance file: a ``#``-header of names, then the matrix."""

    path = Path(path)

    with open(path, encoding="utf-8") as handle:
        names = handle.readline().lstrip("#").split()

    matrix = np.atleast_2d(np.loadtxt(path, comments="#"))

    if matrix.shape != (len(names), len(names)):
        raise OutputError(
            f"{path}: a {matrix.shape} matrix for {len(names)} names."
        )

    return names, matrix
