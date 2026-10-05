"""
Importance reweighting: a finished run's samples, re-weighted for a
changed posterior -- a likelihood added or taken away -- and given new
derived parameters, without sampling again.

::

    output: chains/lcdm            # the run to start from
    post:
      suffix: bbn                  # writes chains/lcdm.post.bbn.*
      skip: 0.3                    # burn-in to drop, by weight per chain
      thin: 1
      add:
        likelihood: {external.bbn: }
        params: {S8: {derived: true}}
      remove:
        likelihood: [sn.pantheonplus]

Every kept sample is evaluated once on the new model, and its weight
multiplied by ``L_new / L_old``: the ratio of the new likelihood to the
one it was sampled from, which the chain's own ``chi2__`` columns
record. The priors of the sampled parameters cannot change, nor can
parameters be added to or taken from the sampled set -- the samples
say nothing about a direction that was never explored.

Reweighting is only as good as the overlap. When the new posterior is
much narrower than the old, or sits in its tail, a few samples carry
all the weight; ``products()`` reports the effective sample size
before and after, ``(sum w)^2 / sum w^2``, and a collapse is warned
about. Sample again rather than trust it.

The result is written like a run's own output, under the prefix
``<output>.post.<suffix>``: chains in getdist's format, one per input
chain, ``.paramnames``, ``.ranges`` and ``.updated.yaml`` (the new
input), which getdist reads as any other run.
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np

from .model import Model
from .output import Output, OutputError, _clean, _yaml


__all__ = ["Post", "PostError"]


class PostError(ValueError):
    """A post-processing input that cannot be applied to its run."""


#: Top-level blocks of a post-processing input.
_KEYS = {"output", "post", "force", "debug"}

#: What a ``post`` block takes.
_POST_KEYS = {"suffix", "skip", "thin", "add", "remove"}

#: Effective sample size, as a fraction of the input's, below which the
#: reweighting is warned about.
_ESS_WARN = 0.1


class Post:
    """
    Parameters
    ----------
    info : dict
        ``output`` (the run's prefix), ``post`` (see the module
        docstring) and optionally ``force``.

    Raises
    ------
    PostError
        Before anything is evaluated, for an input that cannot apply.
    """

    def __init__(self, info: dict):

        unknown = set(info) - _KEYS

        if unknown:
            raise PostError(
                f"A post-processing input takes {sorted(_KEYS)}; not "
                f"{sorted(unknown)}. The run's own input is read from its "
                f"output."
            )

        if not info.get("output"):
            raise PostError("'output' must name the run to post-process.")

        block = dict(info.get("post") or {})

        unknown = set(block) - _POST_KEYS

        if unknown:
            raise PostError(f"'post' takes {sorted(_POST_KEYS)}; not {sorted(unknown)}.")

        suffix = block.get("suffix")

        if not suffix or not isinstance(suffix, str) or "." in suffix or "/" in suffix:
            raise PostError("'post' needs a 'suffix': a plain name for the new output.")

        self.skip = float(block.get("skip", 0.0))
        self.thin = int(block.get("thin", 1))

        if not 0.0 <= self.skip < 1.0:
            raise PostError("'skip' is a fraction of each chain's weight, in [0, 1).")

        if self.thin < 1:
            raise PostError("'thin' must be at least 1.")

        self.info = info
        self.source = Output({"output": info["output"]})

        updated = self.source.path(".updated.yaml")

        if not updated.exists():
            raise PostError(
                f"{updated} is missing: post-processing starts from a run's "
                f"output, and reads what was sampled from that file."
            )

        self.before = _yaml().safe_load(updated.read_text(encoding="utf-8"))
        self.after = self._new_input(self.before, block)

        self.model = Model(self.after)

        sampled = [
            name for name, declaration in (self.before.get("params") or {}).items()
            if isinstance(declaration, dict) and declaration.get("prior") is not None
        ]

        if sorted(sampled) != sorted(self.model.sampled_params):
            raise PostError(
                f"The sampled parameters cannot change in post-processing: "
                f"the run sampled {sampled}, the new input samples "
                f"{self.model.sampled_params}."
            )

        self.output = Output({
            "output": f"{info['output']}.post.{suffix}",
            "force": info.get("force", False),
        })

        self.results = None

    # ---------------------------------------------------------

    @staticmethod
    def _without_sampler(info: dict) -> dict:
        return {
            k: copy.deepcopy(v) for k, v in info.items()
            if k in ("theory", "likelihood", "params")
        }

    def _new_input(self, before: dict, block: dict) -> dict:

        after = self._without_sampler(before)

        after.setdefault("theory", {})
        after.setdefault("likelihood", {})
        after.setdefault("params", {})

        remove = dict(block.get("remove") or {})
        add = dict(block.get("add") or {})

        for kind, entries in remove.items():

            if kind != "likelihood":
                raise PostError(
                    f"Only likelihoods can be removed, not {kind!r}: a "
                    f"theory or a sampled parameter is part of what was "
                    f"sampled."
                )

            for name in entries:

                if name not in after["likelihood"]:
                    raise PostError(
                        f"Cannot remove {name!r}: the run has likelihoods "
                        f"{sorted(after['likelihood'])}."
                    )

                del after["likelihood"][name]

        for kind, entries in add.items():

            if kind not in ("theory", "likelihood", "params"):
                raise PostError(f"'add' takes theory, likelihood and params; not {kind!r}.")

            for name, options in dict(entries or {}).items():

                if kind == "likelihood" and name in after["likelihood"]:
                    raise PostError(f"{name!r} is already a likelihood of the run.")

                if kind == "params":
                    self._check_added_param(name, options, after["params"])

                after[kind][name] = options

        if not after["likelihood"]:
            raise PostError("Removing every likelihood leaves nothing to weight by.")

        return after

    @staticmethod
    def _check_added_param(name, options, params) -> None:

        existing = params.get(name)

        if isinstance(existing, dict) and existing.get("prior") is not None:
            raise PostError(
                f"{name!r} is sampled; its declaration cannot change in "
                f"post-processing."
            )

        if isinstance(options, dict) and options.get("prior") is not None:
            raise PostError(
                f"Cannot add {name!r} as a sampled parameter: the samples "
                f"never explored it. Add it as derived or fixed, or sample "
                f"again."
            )

    # ---------------------------------------------------------

    def _chains(self) -> list:

        found = []
        index = 1

        while self.source.path(f".{index}.txt").exists():
            found.append(self.source.path(f".{index}.txt"))
            index += 1

        if not found:
            raise PostError(f"No chains found under '{self.source.prefix}'.")

        return found

    def run(self) -> None:

        if self.output.prepare():
            raise OutputError("Post-processing cannot resume; use 'force: true'.")

        model = self.model
        names = model.sampled_params

        from CosmoFit.samplers.base import ChainColumns

        layout = ChainColumns(model)

        logw_by_chain, rows_by_chain = [], []
        n_in = n_rejected = 0
        ess_before_w = []

        for path in self._chains():

            with open(path, encoding="utf-8") as handle:
                header = handle.readline().lstrip("#").split()

            data = np.loadtxt(path, comments="#", ndmin=2)

            if self.skip > 0.0 and len(data):
                w = data[:, 0]
                data = data[np.cumsum(w) > self.skip * w.sum()]

            data = data[:: self.thin]

            columns = {name: i for i, name in enumerate(header)}
            chi2 = [i for name, i in columns.items() if name.startswith("chi2__")]

            missing = [n for n in names if n not in columns]

            if missing or not chi2:
                raise PostError(
                    f"{path.name} has no column for {missing or 'any chi2__'}; "
                    f"it was not written by a run of this input."
                )

            logw, rows = [], []

            for row in data:

                n_in += 1
                ess_before_w.append(row[0])

                point = [row[columns[n]] for n in names]
                result = model.logposterior(point)

                if result.rejected is not None or not math.isfinite(result.logpost):
                    n_rejected += 1
                    continue

                old = -0.5 * float(np.sum(row[chi2]))

                logw.append(math.log(row[0]) + result.loglike - old)
                rows.append(np.concatenate([[-result.logpost], layout.of(point, result)]))

            logw_by_chain.append(np.array(logw))
            rows_by_chain.append(np.array(rows).reshape(-1, 1 + len(layout.columns)))

        if n_rejected == n_in:
            raise PostError(
                "The new model rejects every sample: the new posterior "
                "lies where the run never went."
            )

        top = max(lw.max() for lw in logw_by_chain if len(lw))

        chains = [
            np.column_stack([np.exp(lw - top), rows]) if len(lw)
            else np.empty((0, 2 + len(layout.columns)))
            for lw, rows in zip(logw_by_chain, rows_by_chain)
        ]

        weights = np.concatenate([c[:, 0] for c in chains])

        ess_before = _ess(np.array(ess_before_w))
        ess_after = _ess(weights)

        if ess_after < _ESS_WARN * ess_before:
            warnings.warn(
                f"Reweighting left an effective sample size of {ess_after:.0f}, "
                f"from {ess_before:.0f}: the new posterior overlaps the samples "
                f"poorly, and a few of them carry it. Sample it again rather "
                f"than trust these weights.",
                UserWarning, stacklevel=2,
            )

        self.results = {
            "chains": chains,
            "columns": ["weight", "minuslogpost", *layout.columns],
            "names": list(names),
            "n_samples": n_in,
            "n_rejected": n_rejected,
            "ess_before": ess_before,
            "ess_after": ess_after,
        }

        if self.output.enabled:

            for index, rows in enumerate(chains, start=1):
                chain = self.output.chain(index, layout.columns)
                chain.create()
                chain.append(rows)

            self.output.write_getdist_metadata(model, layout.columns)

            updated = _clean(copy.deepcopy(self.after))
            updated["params"] = {
                name: _clean(spec.declaration())
                for name, spec in model.parameters.specs.items()
            }
            updated["post"] = {"from": str(self.info["output"]), **_clean(self.info["post"])}

            self.output.write_yaml(".updated.yaml", updated)

    def products(self) -> dict:
        """
        ``samples`` and their new ``weights`` (largest 1), each chain's
        rows under ``chains`` with their ``columns``; ``n_samples`` read
        and ``n_rejected`` by the new model; the effective sample size
        ``ess_before`` and ``ess_after``.
        """

        if self.results is None:
            raise RuntimeError("Call run() first.")

        r = self.results
        d = len(r["names"])

        rows = (
            np.concatenate(r["chains"]) if r["chains"]
            else np.empty((0, len(r["columns"])))
        )

        return {
            **r,
            "samples": rows[:, 2: 2 + d],
            "weights": rows[:, 0],
        }


def _ess(weights: np.ndarray) -> float:

    total = weights.sum()

    return float(total * total / np.sum(weights * weights)) if total > 0 else 0.0
