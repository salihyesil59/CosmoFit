#!/usr/bin/env python3
"""
Collapse tqdm progress-bar spam in an executed notebook's outputs.

Why this exists
---------------
``jupyter nbconvert --execute`` has no real terminal to interpret
``\\r`` (carriage return), so every tqdm progress update is stored as
its own separate output entry in the .ipynb JSON instead of
overwriting one line. A 12000-step MCMC cell therefore lands ~6000
near-duplicate stream outputs in the file -- tens of MB of noise that
makes the notebook slow to open and unreadable in a diff.

This applies the terminal semantics nbconvert didn't: within each
line, only the text after the last ``\\r`` survives. The result is the
single clean ``100%|####| N/N [mm:ss<00:00, X it/s]`` line an
interactive Jupyter session would have shown.

This was previously done as a one-off hand edit of the notebook JSON,
which meant the spam came back the next time the notebook was
re-executed. Run this after every ``nbconvert --execute`` instead:

    python examples/tools/collapse_notebook_progress.py examples/*.ipynb

Outputs only -- source cells and computed values are never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def render_carriage_returns(text: str) -> str:
    """
    Apply terminal ``\\r`` semantics: on each line, everything before
    the last carriage return was overwritten and is discarded.
    """

    lines = text.split("\n")

    return "\n".join(line.split("\r")[-1] for line in lines)


def collapse_cell(cell) -> int:
    """
    Merge each run of consecutive same-stream outputs in ``cell`` into
    one, with carriage returns rendered. Returns how many output
    entries were removed.
    """

    outputs = cell.get("outputs") or []

    if not outputs:
        return 0

    merged: list = []

    for out in outputs:

        if out.get("output_type") != "stream":
            merged.append(out)
            continue

        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)

        prev = merged[-1] if merged else None

        if (
            prev is not None
            and prev.get("output_type") == "stream"
            and prev.get("name") == out.get("name")
        ):
            prev_text = prev.get("text", "")
            if isinstance(prev_text, list):
                prev_text = "".join(prev_text)
            prev["text"] = prev_text + text
        else:
            # A NotebookNode, not a plain dict: nbformat's writer
            # reaches for `output.output_type` as an attribute, which
            # a dict does not have.
            new = nbformat.NotebookNode(out)
            new["text"] = text
            merged.append(new)

    for out in merged:
        if out.get("output_type") == "stream":
            out["text"] = render_carriage_returns(out["text"])

    removed = len(outputs) - len(merged)
    cell["outputs"] = merged

    return removed


def collapse_notebook(path: Path) -> tuple[int, int, int]:
    """Rewrite ``path`` in place. Returns (removed, size_before, size_after)."""

    size_before = path.stat().st_size

    nb = nbformat.read(path, as_version=4)

    removed = sum(
        collapse_cell(cell)
        for cell in nb.cells
        if cell.cell_type == "code"
    )

    if removed:
        nbformat.write(nb, path)

    return removed, size_before, path.stat().st_size


def main(argv: list[str]) -> int:

    paths = [Path(a) for a in argv[1:]]

    if not paths:
        print(__doc__)
        print("error: no notebook given", file=sys.stderr)
        return 2

    for path in paths:

        if not path.exists():
            print(f"skip (not found): {path}", file=sys.stderr)
            continue

        removed, before, after = collapse_notebook(path)

        if removed:
            print(
                f"{path}: removed {removed} output entries, "
                f"{before / 1e6:.2f} MB -> {after / 1e6:.2f} MB"
            )
        else:
            print(f"{path}: nothing to collapse")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
