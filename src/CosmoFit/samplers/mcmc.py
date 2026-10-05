"""
Adaptive Metropolis-Hastings, stopped by the Gelman-Rubin statistic.

Several chains run side by side, each a Gaussian random walk with the
optimal scaling ``(2.4 / sqrt(d))^2`` times a proposal covariance. That
covariance starts from a ``covmat`` file if given (the one the previous
run ended with, say), else from each parameter's ``proposal`` -- or its
``ref`` or prior width -- and is *learned*: once the chains roughly
agree, it is replaced by their pooled covariance, which is what makes a
random walk efficient on a correlated posterior.

Every ``learn_every`` steps the chains are compared. With the second
half of each chain, ``R - 1`` is the largest eigenvalue of
``W^-1 B``: the spread of the chains' means (``B``) in units of the
covariance within them (``W``), the multivariate form of Gelman &
Rubin's statistic (Brooks & Gelman 1998). Sampling stops once it falls
below ``Rminus1_stop``. A single chain is compared with itself, split
into four.

The chains are weighted: a point the walk stayed at for ``n`` steps is
one row of weight ``n``, as getdist reads them.

Fast and slow parameters
------------------------
A parameter that reaches only likelihoods -- a nuisance parameter
like ``MB`` or ``A_planck`` -- is *fast*: moving it leaves every
theory's inputs alone, so the theories answer from their caches and a
step costs a likelihood evaluation. One that changes a theory's inputs
is *slow*. :meth:`core.model.Model.slow_and_fast` makes the split.

When there are both, each step is a slow move followed by
``fast_steps`` fast ones. The proposal is blocked the way the
covariance is: with the parameters ordered slow first, its Cholesky
factor ``L`` gives a slow move ``L[:, slow] z`` -- which also shifts
the fast parameters along their correlation with the slow ones -- and
a fast move ``L[fast, fast] z``, a step in the fast parameters'
distribution *given* the slow ones. Each block is scaled by
``proposal_scale / sqrt(its dimension)``. Every fast move is an
ordinary Metropolis step, so the target is unchanged; the walk simply
spends its effort where effort is cheap.

``drag: true`` uses the fast moves differently, for Neal's (2005)
dragging: a slow proposal is not accepted or rejected on its own, but
after the fast parameters have been moved ``fast_steps - 1`` times
along a path of distributions interpolating between the current slow
point and the proposed one, so the fast parameters can adjust to the
new slow ones before the decision. On a posterior where the fast
parameters are strongly tied to the slow ones that raises the slow
acceptance; it costs two evaluations per fast step, and needs every
theory to cache at least two states.

``fast_steps`` defaults to ``(t_slow / t_fast) ** oversample_power``,
from timing the starting point -- 0.4, as in CosmoMC and cobaya. Give
it as a number for a run that repeats exactly under its seed.
"""

from __future__ import annotations

import math
import time

import numpy as np

from CosmoFit.core.output import load_covmat

from .base import Sampler


__all__ = ["MCMC", "gelman_rubin"]


def _weighted_mean_cov(x: np.ndarray, w: np.ndarray):

    mean = np.average(x, axis=0, weights=w)
    centred = x - mean
    cov = (w[:, None] * centred).T @ centred / w.sum()

    return mean, np.atleast_2d(cov)


def gelman_rubin(chains: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Multivariate ``R - 1`` of weighted chains ``[(x, w), ...]``: the
    largest eigenvalue of ``W^-1 B``, with ``W`` the mean covariance
    within chains and ``B`` the covariance of their means.
    """

    stats = [_weighted_mean_cov(x, w) for x, w in chains]

    means = np.array([m for m, _ in stats])
    W = np.mean([c for _, c in stats], axis=0)
    B = np.atleast_2d(np.cov(means, rowvar=False, ddof=1))

    try:
        L = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        return math.inf

    Linv = np.linalg.inv(L)

    return float(np.max(np.linalg.eigvalsh(Linv @ B @ Linv.T)))


class _Chain:
    """One random walk: its current point, and its rows so far."""

    def __init__(self, x, result, columns_of):

        self.x = np.asarray(x, dtype=float)
        self.result = result
        self.weight = 0
        self.rows: list[np.ndarray] = []
        self.unwritten = 0
        self.proposed = 0
        self.accepted = 0
        self._columns_of = columns_of

    def row(self) -> np.ndarray:
        return np.concatenate(
            [[self.weight, -self.result.logpost], self._columns_of(self.x, self.result)]
        )

    def leave(self) -> None:
        """Record the current point with the weight it has gathered."""

        if self.weight > 0:
            self.rows.append(self.row())
            self.unwritten += 1
            self.weight = 0


class MCMC(Sampler):
    """
    Options
    -------
    chains : int
        Chains run side by side; at least two to compare. Default 4.
    max_samples : int
        Steps per chain before giving up on convergence. Default 100000.
    Rminus1_stop : float
        Stop once ``R - 1`` is below this. Default 0.01.
    learn_proposal : bool
        Learn the proposal covariance from the chains. Default True.
    learn_proposal_Rminus1_max : float
        Learn only once ``R - 1`` is below this -- before then the chains
        have not found the posterior yet. Default 2.
    learn_every : int or str
        Steps per chain between checks; ``"40d"`` is 40 times the number
        of sampled parameters (the default).
    covmat : str or None
        A covariance file to start the proposal from (``# names`` header,
        then the matrix); parameters it lacks keep their own width.
    proposal_scale : float
        The walk's step in units of ``sqrt(covariance / d)``. Default 2.4.
    max_tries : int
        Consecutive rejected proposals before a chain is declared stuck.
        Default 10000.
    max_time : float or None
        Seconds before stopping, converged or not.
    oversample_power : float
        Fast moves per slow one, as a power of their cost ratio.
        Default 0.4; 0 makes one of each.
    fast_steps : int or None
        Fast moves per slow one, overriding ``oversample_power``.
    drag : bool
        Drag the fast parameters along each slow proposal instead of
        oversampling them. Default False.
    """

    resumable = True

    defaults = {
        "chains": 4,
        "max_samples": 100000,
        "Rminus1_stop": 0.01,
        "learn_proposal": True,
        "learn_proposal_Rminus1_max": 2.0,
        "learn_every": "40d",
        "covmat": None,
        "proposal_scale": 2.4,
        "max_tries": 10000,
        "max_time": None,
        "oversample_power": 0.4,
        "fast_steps": None,
        "drag": False,
    }

    def initialize(self) -> None:

        self._set_columns()

        if self.d == 0:
            raise ValueError("MCMC needs at least one sampled parameter.")

        self.n_chains = int(self.options["chains"])

        if self.n_chains < 1:
            raise ValueError("chains must be at least 1.")

        every = self.options["learn_every"]

        if isinstance(every, str) and every.endswith("d"):
            every = float(every[:-1] or 1) * self.d

        self.learn_every = max(1, int(every))

        self.covariance = self._initial_covariance()

        slow, fast = self.model.slow_and_fast()

        #: Sampled-parameter indices, slow first, ``n_slow`` of them
        #: slow. Blocking needs both kinds; otherwise one block.
        self.blocked = bool(slow) and bool(fast)
        self.order = [self.names.index(n) for n in slow + fast]
        self.n_slow = len(slow)
        self.fast_steps = 0

        self.Rminus1 = math.inf
        self.converged = False
        self.progress: list[dict] = []
        self.chains: list[_Chain] = []

    # ---------------------------------------------------------

    def _initial_covariance(self) -> np.ndarray:

        scales = self.model.parameters.scales()
        covariance = np.diag(scales ** 2)

        if self.options["covmat"] is not None:
            self._merge_covmat(covariance, *load_covmat(self.options["covmat"]))

        return covariance

    def _merge_covmat(self, covariance, names, matrix) -> None:
        """Overwrite the block of ``covariance`` a covmat file knows."""

        known = [n for n in names if n in self.names]

        if not known:
            return

        rows = [names.index(n) for n in known]
        cols = [self.names.index(n) for n in known]

        covariance[np.ix_(cols, cols)] = matrix[np.ix_(rows, rows)]

    def _evaluate(self, x):
        return self.model.logposterior(dict(zip(self.names, x)))

    # ---------------------------------------------------------

    def _start(self) -> None:

        resumed = self._prepare_output()

        if resumed:
            self._resume()
            return

        for index in range(self.n_chains):

            for _ in range(self.options["max_tries"]):

                x = self.model.parameters.sample_ref(self.rng)
                result = self._evaluate(x)

                if math.isfinite(result.logpost):
                    break

            else:
                raise RuntimeError(
                    f"No starting point with a finite posterior in "
                    f"{self.options['max_tries']} draws from the 'ref' "
                    f"distributions. Narrow 'ref' to where the posterior lives."
                )

            self.chains.append(_Chain(x, result, self._columns_of))

            if self.output.enabled:
                self.output.chain(index + 1, self.columns).create()

        if self.output.enabled:
            self.output.write_updated(self.model, self.options)
            self.output.write_getdist_metadata(self.model, self.columns)

    def _resume(self) -> None:

        saved = self.output.read_covmat()

        if saved is not None:
            self._merge_covmat(self.covariance, *saved)

        for index in range(self.n_chains):

            rows = self.output.chain(index + 1, self.columns).read()

            if len(rows) == 0:
                raise RuntimeError(
                    f"Cannot resume: chain {index + 1} has no samples yet."
                )

            x = rows[-1, 2: 2 + self.d]
            result = self._evaluate(x)

            chain = _Chain(x, result, self._columns_of)
            chain.rows = list(rows)

            self.chains.append(chain)

    # ---------------------------------------------------------

    def _halves(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The second half of each chain (by weight), as ``(x, w)``."""

        out = []

        for chain in self.chains:

            rows = np.array(chain.rows + ([chain.row()] if chain.weight else []))

            if len(rows) == 0:
                continue

            w = rows[:, 0]
            keep = np.cumsum(w) > 0.5 * w.sum()

            out.append((rows[keep, 2: 2 + self.d], w[keep]))

        if len(out) == 1:

            x, w = out[0]
            out = [(part, wp) for part, wp in zip(np.array_split(x, 4), np.array_split(w, 4))]

        return [(x, w) for x, w in out if len(x) > 1 and w.sum() > 0]

    def _check(self) -> None:

        halves = self._halves()

        if len(halves) < 2 or sum(len(x) for x, _ in halves) <= 2 * self.d:
            return

        self.Rminus1 = gelman_rubin(halves)

        if self.options["learn_proposal"] and (
            self.Rminus1 < self.options["learn_proposal_Rminus1_max"]
        ):

            x = np.concatenate([x for x, _ in halves])
            w = np.concatenate([w for _, w in halves])

            _, covariance = _weighted_mean_cov(x, w)

            if np.all(np.linalg.eigvalsh(covariance) > 0.0):
                self.covariance = covariance
                self.output.write_covmat(self.names, covariance)

        self.converged = self.Rminus1 < self.options["Rminus1_stop"]

    def _flush(self) -> None:

        if not self.output.enabled:
            return

        for index, chain in enumerate(self.chains):

            if chain.unwritten:
                new = np.array(chain.rows[-chain.unwritten:])
                self.output.chain(index + 1, self.columns).append(new)
                chain.unwritten = 0

    # ---------------------------------------------------------

    def _measure_fast_steps(self) -> int:
        """
        Fast moves per slow one: given, or ``(t_slow / t_fast) **
        oversample_power`` timed at the first chain's start.
        """

        if self.options["fast_steps"] is not None:

            steps = int(self.options["fast_steps"])

            if steps < 1:
                raise ValueError("fast_steps must be at least 1.")

            return steps

        x = self.chains[0].x
        scales = self.model.parameters.scales()

        def seconds(index):

            total = 0.0

            # Three distinct small moves, so none is a cache hit of
            # the one before.
            for k in (1, 2, 3):

                moved = x.copy()
                moved[index] += 1e-3 * k * scales[index]

                start = time.perf_counter()
                self._evaluate(moved)
                total += time.perf_counter() - start

            return total / 3

        # Fast first, from the start point freshly evaluated: timed
        # after the slow moves, the start's theory state has been
        # pushed out of the cache and the first fast move pays for it.
        self._evaluate(x)

        t_fast = seconds(self.order[-1])
        t_slow = seconds(self.order[0])

        ratio = t_slow / max(t_fast, 1e-9)

        return max(1, int(round(ratio ** self.options["oversample_power"])))

    def _proposal_factors(self):
        """
        ``(slow, fast)``: matrices taking standard normals to a move of
        every parameter. Unblocked, ``slow`` is the whole proposal and
        ``fast`` is None.
        """

        scale = self.options["proposal_scale"]

        if not self.blocked:
            return np.linalg.cholesky(self.covariance) * scale / math.sqrt(self.d), None

        order = self.order
        ns = self.n_slow
        nf = self.d - ns

        L = np.linalg.cholesky(self.covariance[np.ix_(order, order)])

        slow = np.zeros((self.d, ns))
        slow[order] = L[:, :ns] * scale / math.sqrt(ns)

        fast = np.zeros((self.d, nf))
        fast[order[ns:]] = L[ns:, ns:] * scale / math.sqrt(nf)

        return slow, fast

    def _accept(self, log_ratio: float) -> bool:
        return log_ratio >= 0.0 or self.rng.uniform() < math.exp(log_ratio)

    def _stay(self, index: int) -> None:

        self.stuck[index] += 1

        if self.stuck[index] >= self.options["max_tries"]:
            raise RuntimeError(
                f"Chain {index + 1} rejected {self.stuck[index]} proposals "
                f"in a row. The proposal is too wide for the posterior: "
                f"give smaller 'proposal' widths or a covmat."
            )

    def _move(self, index: int, chain: _Chain, x, result) -> None:

        chain.weight -= 1
        chain.leave()
        chain.x, chain.result = x, result
        chain.weight = 1
        chain.accepted += 1
        self.stuck[index] = 0

    def _metropolis(self, index: int, chain: _Chain, step) -> None:
        """One Metropolis step of ``chain`` by ``step``."""

        chain.proposed += 1
        chain.weight += 1

        proposal = chain.x + step
        result = self._evaluate(proposal)

        if math.isfinite(result.logpost) and self._accept(
            result.logpost - chain.result.logpost
        ):
            self._move(index, chain, proposal, result)
        else:
            self._stay(index)

    def _drag(self, index: int, chain: _Chain, step, fast) -> None:
        """
        A slow move by ``step``, decided after dragging the fast
        parameters (Neal 2005) through ``fast_steps - 1`` distributions
        interpolating between the current slow point and the proposed
        one. The end point carries the same fast moves as the start.

        With ``n = fast_steps`` the acceptance is that of Neal's
        construction, ``(1/n) sum_i [ln p(end_i) - ln p(start_i)]``
        over the ``n`` points the pair visits, each fast move being a
        Metropolis step on ``(1 - b) ln p(start) + b ln p(end)``, ``b =
        i/n``.
        """

        chain.proposed += 1
        chain.weight += 1

        start_x, start = chain.x, chain.result
        end_x = chain.x + step
        end = self._evaluate(end_x)

        if not math.isfinite(end.logpost):
            self._stay(index)
            return

        n = self.fast_steps
        log_weight = end.logpost - start.logpost

        for i in range(1, n):

            beta = i / n
            move = fast @ self.rng.standard_normal(fast.shape[1])

            trial_start = self._evaluate(start_x + move)
            trial_end = self._evaluate(end_x + move)

            if math.isfinite(trial_start.logpost) and math.isfinite(trial_end.logpost):

                new = (1 - beta) * trial_start.logpost + beta * trial_end.logpost
                old = (1 - beta) * start.logpost + beta * end.logpost

                if self._accept(new - old):
                    start_x, start = start_x + move, trial_start
                    end_x, end = end_x + move, trial_end

            log_weight += end.logpost - start.logpost

        if self._accept(log_weight / n):
            self._move(index, chain, end_x, end)
        else:
            self._stay(index)

    def run(self) -> None:

        self._start()

        if self.blocked:
            self.fast_steps = self._measure_fast_steps()

        start = time.monotonic()

        step = 0
        self.stuck = [0] * len(self.chains)

        while step < self.options["max_samples"]:

            slow, fast = self._proposal_factors()

            for _ in range(self.learn_every):

                step += 1

                for index, chain in enumerate(self.chains):

                    move = slow @ self.rng.standard_normal(slow.shape[1])

                    if fast is None:
                        self._metropolis(index, chain, move)

                    elif self.options["drag"]:
                        self._drag(index, chain, move, fast)

                    else:

                        self._metropolis(index, chain, move)

                        for _ in range(self.fast_steps):
                            self._metropolis(
                                index, chain, fast @ self.rng.standard_normal(fast.shape[1]),
                            )

            self._check()
            self._flush()
            self._record(step, time.monotonic() - start)

            if self.converged:
                break

            limit = self.options["max_time"]

            if limit is not None and time.monotonic() - start > limit:
                break

        for chain in self.chains:
            chain.leave()

        self._flush()

    def _record(self, step: int, elapsed: float) -> None:

        accepted = sum(c.accepted for c in self.chains)
        proposed = sum(c.proposed for c in self.chains)

        entry = {
            "steps": step,
            "acceptance": accepted / max(proposed, 1),
            "Rminus1": self.Rminus1,
            "seconds": elapsed,
        }

        self.progress.append(entry)

        if self.output.enabled:

            path = self.output.path(".progress")
            new = not path.exists()

            with open(path, "a", encoding="utf-8") as handle:
                if new:
                    handle.write("# steps acceptance Rminus1 seconds\n")
                handle.write(
                    f"{step} {entry['acceptance']:.4f} {self.Rminus1:.6g} {elapsed:.2f}\n"
                )

    # ---------------------------------------------------------

    def products(self, skip: float = 0.0) -> dict:
        """
        ``samples`` (sampled parameters), ``weights`` and ``minuslogpost``
        of every chain together, each chain's rows under ``chains``, the
        ``columns`` of those rows, the final ``Rminus1``, whether it
        ``converged``, the ``acceptance`` rate, the proposal ``covmat``,
        and the ``progress`` of every check.

        ``skip`` drops that fraction of each chain's weight from its start
        -- the burn-in, as getdist's ``ignore_rows`` does. The files keep
        every sample.
        """

        width = 2 + len(self.columns)
        chains = []

        for chain in self.chains:

            rows = np.array(chain.rows).reshape(-1, width)

            if skip > 0.0 and len(rows):
                w = rows[:, 0]
                rows = rows[np.cumsum(w) > skip * w.sum()]

            chains.append(rows)

        rows = np.concatenate(chains) if chains else np.empty((0, width))

        return {
            "samples": rows[:, 2: 2 + self.d],
            "weights": rows[:, 0],
            "minuslogpost": rows[:, 1],
            "chains": chains,
            "columns": ["weight", "minuslogpost", *self.columns],
            "names": list(self.names),
            "Rminus1": self.Rminus1,
            "converged": self.converged,
            "acceptance": self.progress[-1]["acceptance"] if self.progress else math.nan,
            "covmat": self.covariance.copy(),
            "progress": list(self.progress),
        }
