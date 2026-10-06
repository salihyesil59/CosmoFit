"""
The input: a dict, or the YAML file it was written as.

Top-level blocks::

    theory:      {name: options}      # what computes predictions
    likelihood:  {name: options}      # what compares them with data
    params:      {name: declaration}  # see core.parameters
    sampler:     {name: options}      # exactly one
    output:      path prefix          # optional (used by later phases)
    debug, force, resume              # optional flags

Options may be ``null`` (``desi:`` on its own line) for "defaults".
"""

from __future__ import annotations

import copy
from pathlib import Path


__all__ = ["load_info", "validate_info", "TOP_LEVEL_KEYS"]


TOP_LEVEL_KEYS = {
    "theory", "likelihood", "params", "sampler", "output",
    "debug", "force", "resume",
}


def load_info(source) -> dict:
    """
    An input as a dict: a dict (copied), a path to a ``.yaml``/``.yml``
    file, or YAML text.
    """

    if isinstance(source, dict):
        return copy.deepcopy(source)

    import yaml

    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source
        and source.endswith((".yaml", ".yml"))
    ):

        with open(source, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    elif isinstance(source, str):

        data = yaml.safe_load(source)

    else:

        raise TypeError(f"Cannot read an input from {type(source).__name__}.")

    if not isinstance(data, dict):
        raise ValueError("An input must be a mapping of blocks at the top level.")

    return data


def validate_info(info: dict) -> dict:
    """
    Check an input's shape and normalize it: every block a dict, every
    component's options a dict (``None`` becomes ``{}``).

    Raises
    ------
    ValueError
        Naming the first thing that is wrong.
    """

    unknown = set(info) - TOP_LEVEL_KEYS

    if unknown:
        raise ValueError(
            f"Unknown top-level block(s) {sorted(unknown)}; an input has "
            f"{sorted(TOP_LEVEL_KEYS)}."
        )

    out = dict(info)

    for block in ("theory", "likelihood", "sampler"):

        entries = out.get(block) or {}

        if isinstance(entries, (list, tuple)):
            entries = {name: None for name in entries}

        if not isinstance(entries, dict):
            raise ValueError(
                f"'{block}' must map names to options, got "
                f"{type(entries).__name__}."
            )

        normalized = {}

        for name, options in entries.items():

            if options is None:
                options = {}

            if not isinstance(options, dict):
                raise ValueError(
                    f"Options of {block} {name!r} must be a dict or empty."
                )

            normalized[str(name)] = dict(options)

        out[block] = normalized

    if not out["likelihood"]:
        raise ValueError("An input needs at least one likelihood.")

    if len(out["sampler"]) > 1:
        raise ValueError(
            f"Exactly one sampler, got {sorted(out['sampler'])}."
        )

    params = out.get("params") or {}

    if not isinstance(params, dict):
        raise ValueError("'params' must map names to declarations.")

    out["params"] = dict(params)

    return out
