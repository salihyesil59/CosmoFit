"""
emcee's affine-invariant ensemble sampler, on the new core.

Walkers move together, each proposing along the line to another, so
the sampler is insensitive to linear correlations without learning a
covariance -- the alternative to adaptive Metropolis-Hastings when the
posterior is cheap and a proposal is not worth tuning.

The walkers are not independent chains, so convergence is not judged
by comparing them. It is judged by the integrated autocorrelation time
``tau`` (Goodman & Weare 2010; emcee's ``integrated_time``): sampling
stops once the run is ``tau_factor`` times longer than the largest
``tau``, and ``tau`` itself has stopped moving by more than
``tau_rtol`` between checks -- an estimate that is still growing is an
underestimate.

Every step of every walker is one row of weight 1 in ``.1.txt``, step
by step, so getdist's ``ignore_rows`` drops the first steps of all the
walkers together. The last ``walkers`` rows are where the walkers
stand, which is all ``resume: true`` needs.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .base import Sampler


__all__ = ["Emcee"]


class Emcee(Sampler):
    """
    Options
    -------
    walkers : int, optional
        Default ``max(32, 4 d)``; at least ``2 d``.
    max_samples : int
        Steps per walker before giving up on convergence. Default 20000.
    check_every : int
        Steps between convergence checks (and writes). Default 200.
    tau_factor : float
        Stop once the run is this many autocorrelation times long.
        Default 50.
    tau_rtol : float
        ...and ``tau`` changed by less than this since the last check.
        Default 0.01.
    moves : str
        ``"stretch"`` (the default) or ``"de"`` (differential
        evolution, for strongly correlated posteriors).
    max_tries : int
        Draws from ``ref`` allowed per walker for a finite start.
    max_time : float or None
        Seconds before stopping, converged or not.
    """

    resumable = True

    defaults = {
        "walkers": None,
        "max_samples": 20000,
        "check_every": 200,
        "tau_factor": 50.0,
        "tau_rtol": 0.01,
        "moves": "stretch",
        "max_tries": 1000,
        "max_time": None,
    }

    def initialize(self) -> None:

        self._set_columns()

        if self.d == 0:
            raise ValueError("emcee needs at least one sampled parameter.")

        walkers = self.options["walkers"]

        self.walkers = max(32, 4 * self.d) if walkers is None else int(walkers)

        if self.walkers < 2 * self.d:
            raise ValueError(
                f"emcee needs at least 2 d = {2 * self.d} walkers for {self.d} "
                f"parameters, not {self.walkers}."
            )

        if self.options["moves"] not in ("stretch", "de"):
            raise ValueError(f"moves is 'stretch' or 'de', not {self.options['moves']!r}.")

        self.rows: list[np.ndarray] = []
        self.tau = math.inf
        self.converged = False
        self.acceptance = math.nan
        self.progress: list[dict] = []

    # ---------------------------------------------------------

    def _log_prob(self, x):

        result = self.model.logposterior(dict(zip(self.names, x)), want_derived=self.want_derived)

        return result.logpost, self._columns_of(x, result)

    def _start(self) -> np.ndarray:

        if self._prepare_output():
            return self._resume()

        start = []

        for _ in range(self.walkers):

            for _ in range(self.options["max_tries"]):

                x = self.model.parameters.sample_ref(self.rng)

                if math.isfinite(self.model.logposterior(x, want_derived=False).logpost):
                    break

            else:
                raise RuntimeError(
                    f"No starting point with a finite posterior in "
                    f"{self.options['max_tries']} draws from the 'ref' "
                    f"distributions. Narrow 'ref' to where the posterior lives."
                )

            start.append(x)

        start = np.array(start)

        if np.linalg.matrix_rank(start - start.mean(axis=0)) < self.d:
            raise ValueError(
                "The walkers start in a subspace of the parameters, which an "
                "ensemble sampler can never leave. Give every sampled "
                "parameter a 'ref' distribution with a width, not a number."
            )

        if self.output.enabled:
            self.output.chain(1, self.columns).create()
            self.output.write_updated(self.model, self.options)
            self.output.write_getdist_metadata(self.model, self.columns)

        return start

    def _resume(self) -> np.ndarray:

        rows = self.output.chain(1, self.columns).read()

        if len(rows) == 0 or len(rows) % self.walkers:
            raise RuntimeError(
                f"Cannot resume: the chain has {len(rows)} rows, which is not a "
                f"whole number of steps of {self.walkers} walkers."
            )

        self.rows = [rows[i: i + self.walkers] for i in range(0, len(rows), self.walkers)]

        return rows[-self.walkers:, 2: 2 + self.d]

    # ---------------------------------------------------------

    def _tau(self, rows) -> float:

        import emcee

        chain = np.array([step[:, 2: 2 + self.d] for step in rows])

        return float(np.max(emcee.autocorr.integrated_time(chain, tol=0)))

    def _check(self) -> None:

        chain = np.array([step[:, 2: 2 + self.d] for step in self.rows])

        tau = self._tau(self.rows)

        # A walker that moved accepted its proposal; one that stayed
        # put rejected it. emcee keeps no count when it stores nothing.
        self.acceptance = float(np.mean(np.any(chain[1:] != chain[:-1], axis=2)))

        previous, self.tau = self.tau, tau

        self.converged = bool(
            len(self.rows) >= self.options["tau_factor"] * tau
            and abs(tau - previous) < self.options["tau_rtol"] * tau
        )

    def run(self) -> None:

        import emcee

        start = self._start()

        moves = emcee.moves.DEMove() if self.options["moves"] == "de" else None

        sampler = emcee.EnsembleSampler(
            self.walkers, self.d, self._log_prob, moves=moves,
            blobs_dtype=[("columns", float, (len(self.columns),))],
        )

        sampler.random_state = np.random.RandomState(int(self.rng.integers(2 ** 31))).get_state()

        clock = time.monotonic()
        state = start
        unwritten = 0

        # A resumed run is judged before it moves, as the mcmc sampler's
        # is: one that had converged is not sampled further.
        if self.rows:

            # `tau` must have stopped moving: compare with the estimate
            # one check earlier, as the run that wrote these rows did.
            earlier = len(self.rows) - self.options["check_every"]

            if earlier > 1:
                self.tau = self._tau(self.rows[:earlier])

            self._check()
            self._record(time.monotonic() - clock)

        while not self.converged and len(self.rows) < self.options["max_samples"]:

            n = min(self.options["check_every"], self.options["max_samples"] - len(self.rows))

            for step in sampler.sample(state, iterations=n, store=False):

                self.rows.append(np.column_stack([
                    np.ones(self.walkers), -step.log_prob, step.blobs["columns"],
                ]))

                unwritten += 1

            state = step

            self._check()

            if self.output.enabled:
                self.output.chain(1, self.columns).append(np.concatenate(self.rows[-unwritten:]))

            unwritten = 0

            self._record(time.monotonic() - clock)

            limit = self.options["max_time"]

            if limit is not None and time.monotonic() - clock > limit:
                break

    def _record(self, elapsed: float) -> None:

        entry = {
            "steps": len(self.rows),
            "acceptance": self.acceptance,
            "tau": self.tau,
            "seconds": elapsed,
        }

        self.progress.append(entry)

        if self.callback is not None:
            self.callback({
                **entry,
                "tau_factor": self.options["tau_factor"],
                "max_samples": self.options["max_samples"],
            })

        if self.output.enabled:

            path = self.output.path(".progress")
            new = not path.exists()

            with open(path, "a", encoding="utf-8") as handle:
                if new:
                    handle.write("# steps acceptance tau seconds\n")
                handle.write(
                    f"{entry['steps']} {self.acceptance:.4f} {self.tau:.6g} {elapsed:.2f}\n"
                )

    # ---------------------------------------------------------

    def products(self, skip: float | None = None) -> dict:
        """
        ``samples`` (sampled parameters) and ``weights`` of every walker
        together, the same rows under ``chains`` (one chain), their
        ``columns``, the autocorrelation time ``tau``, whether it
        ``converged``, the mean ``acceptance`` and the ``progress`` of
        every check.

        ``skip`` drops that fraction of the steps from the start; by
        default ``2 tau`` steps, the burn-in emcee's authors recommend.
        """

        steps = len(self.rows)

        if skip is None:
            first = min(steps, int(math.ceil(2 * self.tau))) if math.isfinite(self.tau) else 0
        else:
            first = int(skip * steps)

        width = 2 + len(self.columns)

        rows = np.concatenate(self.rows[first:]) if steps > first else np.empty((0, width))

        return {
            "samples": rows[:, 2: 2 + self.d],
            "weights": rows[:, 0],
            "minuslogpost": rows[:, 1],
            "chains": [rows],
            "columns": ["weight", "minuslogpost", *self.columns],
            "names": list(self.names),
            "tau": self.tau,
            "converged": self.converged,
            "acceptance": self.acceptance,
            "progress": list(self.progress),
        }
