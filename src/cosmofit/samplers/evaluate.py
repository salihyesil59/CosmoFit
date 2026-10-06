"""
Evaluate the model at one point and report everything about it.

The first thing to run on a new input: it builds the model, evaluates
it once, and shows the log-prior, every likelihood's chi2 and every
derived parameter -- without sampling anything.
"""

from __future__ import annotations

import numpy as np

from .base import Sampler


__all__ = ["Evaluate"]


class Evaluate(Sampler):
    """
    Options
    -------
    override : dict, optional
        Values for sampled parameters. The rest are taken at their
        ``ref`` (or, failing that, their prior's center).
    """

    defaults = {"override": None}

    def initialize(self) -> None:

        override = dict(self.options["override"] or {})

        unknown = set(override) - set(self.model.sampled_params)

        if unknown:
            raise ValueError(
                f"override: {sorted(unknown)} are not sampled parameters "
                f"(those are {self.model.sampled_params})."
            )

        point = dict(zip(
            self.model.sampled_params, self.model.parameters.centers(),
        ))
        point.update({k: float(v) for k, v in override.items()})

        self.point = point
        self.result = None

    def run(self) -> None:

        self.result = self.model.logposterior(self.point)

    def products(self) -> dict:

        if self.result is None:
            raise RuntimeError("Call run() first.")

        r = self.result

        return {
            "point": dict(self.point),
            "logpost": r.logpost,
            "logprior": r.logprior,
            "loglikes": dict(r.loglikes),
            "chi2": r.chi2 if r.rejected is None else {},
            "derived": dict(r.derived),
            "rejected": r.rejected,
            "finite": bool(np.isfinite(r.logpost)),
        }
