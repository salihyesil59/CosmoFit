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
    """

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
    }

    def initialize(self) -> None:

        parameters = self.model.parameters

        self.names = list(parameters.sampled)
        self.d = len(self.names)

        if self.d == 0:
            raise ValueError("MCMC needs at least one sampled parameter.")

        self.n_chains = int(self.options["chains"])

        if self.n_chains < 1:
            raise ValueError("chains must be at least 1.")

        every = self.options["learn_every"]

        if isinstance(every, str) and every.endswith("d"):
            every = float(every[:-1] or 1) * self.d

        self.learn_every = max(1, int(every))

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

        self.covariance = self._initial_covariance()

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

    def _columns_of(self, x, result) -> np.ndarray:

        derived = [result.derived.get(n, np.nan) for n in self.derived_names]
        chi2 = [-2.0 * result.loglikes.get(n, np.nan) for n in self.likelihood_names]

        return np.concatenate([x, derived, [-result.logprior], chi2])

    def _evaluate(self, x):
        return self.model.logposterior(dict(zip(self.names, x)))

    # ---------------------------------------------------------

    def _start(self) -> None:

        resumed = self.output.prepare()

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

    def run(self) -> None:

        self._start()

        start = time.monotonic()
        scale = self.options["proposal_scale"] / math.sqrt(self.d)

        step = 0
        stuck = [0] * len(self.chains)

        while step < self.options["max_samples"]:

            L = np.linalg.cholesky(self.covariance) * scale

            for _ in range(self.learn_every):

                step += 1

                for index, chain in enumerate(self.chains):

                    chain.proposed += 1
                    chain.weight += 1

                    proposal = chain.x + L @ self.rng.standard_normal(self.d)
                    result = self._evaluate(proposal)

                    log_ratio = result.logpost - chain.result.logpost

                    if math.isfinite(result.logpost) and (
                        log_ratio >= 0.0 or self.rng.uniform() < math.exp(log_ratio)
                    ):
                        chain.weight -= 1
                        chain.leave()
                        chain.x, chain.result = proposal, result
                        chain.weight = 1
                        chain.accepted += 1
                        stuck[index] = 0
                    else:
                        stuck[index] += 1

                    if stuck[index] >= self.options["max_tries"]:
                        raise RuntimeError(
                            f"Chain {index + 1} rejected {stuck[index]} proposals "
                            f"in a row. The proposal is too wide for the posterior: "
                            f"give smaller 'proposal' widths or a covmat."
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
