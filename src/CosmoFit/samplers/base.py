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

    A sampler that cannot pick up where an earlier run stopped leaves
    :attr:`resumable` false, and is refused ``resume: true``.
    """

    #: Options this sampler takes, with defaults.
    defaults: dict = {}

    #: Whether ``resume: true`` can continue an earlier run's output.
    resumable: bool = False

    #: Whether this sampler shares a run between processes under
    #: ``mpirun``. One that does not is refused there, rather than run
    #: once per process into the same output.
    parallel: bool = False

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

    # ---------------------------------------------------------
    # What every sampler writes a point as
    # ---------------------------------------------------------

    def _set_columns(self) -> None:
        """
        :attr:`names` (sampled), :attr:`derived_names`,
        :attr:`likelihood_names` and :attr:`columns`: a row's names after
        ``weight`` and ``minuslogpost``, as the chain files carry them.
        """

        parameters = self.model.parameters

        self.names = list(parameters.sampled)
        self.d = len(self.names)

        self.derived_names = [
            n for n in parameters.derived if not parameters.specs[n].drop
        ]
        self.likelihood_names = list(self.model.likelihoods)

        self.columns = [
            *self.names,
            *self.derived_names,
            "minuslogprior",
            *(f"chi2__{name.replace('.', '_')}" for name in self.likelihood_names),
        ]

    def _columns_of(self, x, result) -> np.ndarray:
        """``x`` and its evaluation, in :attr:`columns` order."""

        derived = [result.derived.get(n, np.nan) for n in self.derived_names]
        chi2 = [-2.0 * result.loglikes.get(n, np.nan) for n in self.likelihood_names]

        return np.concatenate([np.asarray(x, dtype=float), derived, [-result.logprior], chi2])

    def _prepare_output(self) -> bool:
        """
        Ready the output prefix; whether this run resumes an earlier one.
        """

        from CosmoFit.core.output import OutputError

        resumed = self.output.prepare()

        if resumed and not self.resumable:
            raise OutputError(
                f"{type(self).__name__} cannot resume an earlier run; use "
                f"'force: true' to start over, or a new output prefix."
            )

        return resumed
