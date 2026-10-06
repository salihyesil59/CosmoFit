"""
The example inputs under ``examples/yaml``: every one must build and
evaluate, so that what the documentation points to runs.

Each is evaluated once at its ``ref`` centres, with its sampler swapped
for ``evaluate`` and its output dropped -- running the samplers
themselves is what the sampler tests are for. The post-processing
example is checked against the run it reweights.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import yaml

from cosmofit.core import load_info, run


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "yaml"

INPUTS = sorted(p for p in EXAMPLES.glob("*.yaml") if "post" not in yaml.safe_load(p.read_text()))
POSTS = sorted(p for p in EXAMPLES.glob("*.yaml") if p not in INPUTS)


def test_there_are_examples():
    assert len(INPUTS) >= 3 and POSTS


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.stem)
def test_an_example_input_evaluates(path):

    info = load_info(path)

    assert info["output"] == f"chains/{path.stem}"

    info["sampler"] = {"evaluate": None}
    del info["output"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sampler = run(info)

    products = sampler.products()

    assert products["finite"], products["rejected"]
    assert math.isfinite(products["logpost"])

    # Every derived parameter it declares is computed.
    declared = [n for n, d in info["params"].items() if isinstance(d, dict) and "derived" in d]

    for name in declared:
        assert math.isfinite(products["derived"][name]), name


@pytest.mark.parametrize("path", POSTS, ids=lambda p: p.stem)
def test_a_post_example_reweights_an_example_run(path):

    info = load_info(path)

    run_prefix = Path(info["output"])

    assert (EXAMPLES / f"{run_prefix.name}.yaml") in INPUTS
    assert info["post"]["suffix"]
