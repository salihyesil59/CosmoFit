"""
Running an input end to end.
"""

from __future__ import annotations

from .info import load_info, validate_info
from .model import Model
from .registry import resolve


__all__ = ["run"]


def run(info, seed: int | None = None):
    """
    Build the model an input describes, run its sampler, and return
    both.

    Parameters
    ----------
    info : dict, str or Path
        The input; it must have a ``sampler`` block.
    seed : int, optional
        For the sampler's random draws, unless the input sets one.

    Returns
    -------
    (dict, Sampler)
        The validated input and the sampler that ran -- its
        ``products()`` hold the result.
    """

    info = validate_info(load_info(info))

    if not info["sampler"]:
        raise ValueError(
            "run() needs a 'sampler' block (evaluate, minimize, ...). "
            "To build the model alone, use get_model()."
        )

    model = Model(info)

    (name, options), = info["sampler"].items()

    sampler_cls = resolve("sampler", name, options)

    options = {k: v for k, v in options.items() if k != "class"}

    sampler = sampler_cls(options, model, seed=seed)

    sampler.run()

    return info, sampler
