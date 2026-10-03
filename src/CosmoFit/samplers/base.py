"""
What every sampler of the new core shares.
"""

from __future__ import annotations

import numpy as np


__all__ = ["Sampler"]


class Sampler:
    """
    Base sampler.

    Parameters
    ----------
    info : dict
        The sampler's options from the input.
    model : core.model.Model
        The model to sample.
    seed : int, optional
        Seeds ``rng``. An ``info["seed"]`` takes precedence.
    output : core.output.Output, optional
        Where to write; nothing is written without one.

    Notes
    -----
    Subclasses read their options in :meth:`initialize`, do the work
    in :meth:`run`, and return what they found from :meth:`products`.
    """

    #: Options this sampler takes, with defaults.
    defaults: dict = {}

    def __init__(self, info: dict | None, model, seed: int | None = None, output=None):

        from CosmoFit.core.output import Output

        info = dict(info or {})

        unknown = set(info) - set(self.defaults) - {"seed"}

        if unknown:
            raise ValueError(
                f"{type(self).__name__}: unknown option(s) {sorted(unknown)}; "
                f"it takes {sorted(self.defaults)}."
            )

        self.options = {**self.defaults, **info}
        self.model = model
        self.seed = info.get("seed", seed)
        self.rng = np.random.default_rng(self.seed)
        self.output = output if output is not None else Output({})

        self.initialize()

    def initialize(self) -> None:
        """Read options. Override."""

    def run(self) -> None:
        raise NotImplementedError

    def products(self) -> dict:
        raise NotImplementedError
