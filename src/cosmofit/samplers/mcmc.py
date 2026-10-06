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

Several processes
-----------------
The chains can run in separate processes: ``processes: n`` starts them
here (see :mod:`core.mpi`), and a run under ``mpirun`` shares them
between the processes it was given. The processes meet only at each
check, where the first gathers every chain's second half, decides
``R - 1`` and the covariance, and hands both back.

Every chain draws from its own random stream, spawned from the run's
seed, so the split changes nothing: a run in four processes writes the
same chains, byte for byte, as in one. Starting a process rebuilds the
model and reloads its data, which takes seconds; it pays when an
evaluation is slow -- a Boltzmann code's, not a distance likelihood's.
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np

from cosmofit.core.mpi import Aborted, WorkerError, spawn, world
from cosmofit.core.output import load_covmat

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
    """One random walk: its current point, its rows so far, its own draws."""

    def __init__(self, index, x, result, columns_of, rng):

        self.index = index
        self.x = np.asarray(x, dtype=float)
        self.result = result
        self.rng = rng
        self.weight = 0
        self.rows: list[np.ndarray] = []
        self.unwritten = 0
        self.proposed = 0
        self.accepted = 0
        self.stuck = 0
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

    def half(self, d: int):
        """The second half of the chain (by weight), as ``(x, w)``."""

        rows = np.array(self.rows + ([self.row()] if self.weight else []))

        if len(rows) == 0:
            return None

        w = rows[:, 0]
        keep = np.cumsum(w) > 0.5 * w.sum()

        return rows[keep, 2: 2 + d], w[keep]


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
    processes : int
        Processes to run the chains in, started here; ignored under
        ``mpirun``, which decides that itself. Default 1.
    """

    resumable = True
    parallel = True

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
        "processes": 1,
    }

    def initialize(self) -> None:

        self._set_columns()

        if self.d == 0:
            raise ValueError("MCMC needs at least one sampled parameter.")

        self.n_chains = int(self.options["chains"])

        if self.n_chains < 1:
            raise ValueError("chains must be at least 1.")

        if int(self.options["processes"]) < 1:
            raise ValueError("processes must be at least 1.")

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

        #: The processes this run is shared between; set by :meth:`run`
        #: unless a process started by another sets it first.
        self.comm = None

        self.Rminus1 = math.inf
        self.converged = False
        self.resumed = False
        self.progress: list[dict] = []

        #: This process's chains.
        self.chains: list[_Chain] = []

        #: Every chain's rows, by index, once a run has ended -- on the
        #: first process only.
        self.rows_by_chain: list[np.ndarray] | None = None

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
    # Processes
    # ---------------------------------------------------------

    def _sync(self, error, payload, decide=lambda payloads: None):
        """
        Every process hands ``payload`` (and the error it hit, if any) to
        the first, which ``decide``s from all of them; every process
        returns the decision.

        If any process failed, every process raises instead: the one
        that failed its own error, the first a :class:`WorkerError`
        with the other's traceback, the rest :class:`Aborted`. So no
        process is left waiting on one that has stopped.
        """

        text = None if error is None else "".join(traceback.format_exception(error))

        gathered = self.comm.gather((text, payload))

        decision = None

        if self.comm.rank == 0:

            failed = [(rank, t) for rank, (t, _) in enumerate(gathered) if t is not None]

            if failed:
                decision = ("failed", failed[0])
            else:
                try:
                    decision = ("ok", decide([p for _, p in gathered]))
                except Exception as own:  # noqa: BLE001 -- shared, then raised
                    error = own
                    decision = ("failed", (0, "".join(traceback.format_exception(own))))

        decision = self.comm.bcast(decision)

        if decision[0] == "ok":
            return decision[1]

        if error is not None:
            raise error

        rank, text = decision[1]

        if self.comm.rank == 0:
            raise WorkerError(f"Process {rank + 1} of the run failed:\n{text}")

        raise Aborted(f"Process {rank + 1} of the run failed.")

    def run(self) -> None:

        spawned = None

        if self.comm is None:

            self.comm = world()

            processes = int(self.options["processes"])

            if self.comm.size == 1 and processes > 1:

                options = {k: v for k, v in self.options.items() if k != "processes"}

                self.comm = spawned = spawn(
                    min(processes, self.n_chains), _run_in_process,
                    (self.model.info, options),
                )

        try:
            self._run()
        finally:
            if spawned is not None:
                spawned.close()

    def _mine(self) -> list[int]:
        """Indices of the chains this process runs."""

        if self.n_chains < self.comm.size:
            raise ValueError(
                f"{self.comm.size} processes need at least as many chains; "
                f"set 'chains: {self.comm.size}' or more."
            )

        return list(range(self.comm.rank, self.n_chains, self.comm.size))

    # ---------------------------------------------------------
    # Starting
    # ---------------------------------------------------------

    def _start(self) -> None:
        """
        Output ready, every chain at its start, and the same random
        streams whichever process runs a chain: each chain draws from
        its own, spawned from the run's seed.
        """

        mine = self._mine()

        entropy = self.comm.bcast(
            (self.seed if self.seed is not None else np.random.SeedSequence().entropy)
            if self.comm.rank == 0 else None
        )

        streams = np.random.SeedSequence(entropy).spawn(self.n_chains)
        rngs = {i: np.random.default_rng(streams[i]) for i in mine}

        error = resumed = None

        if self.comm.rank == 0:
            try:
                resumed = self._prepare_output()
            except Exception as own:  # noqa: BLE001 -- shared by _sync
                error = own

        resumed = self._sync(error, None, lambda _: resumed)

        self.resumed = bool(resumed)

        error = None

        try:
            if resumed:
                self._resume(mine, rngs)
            else:
                self._fresh(mine, rngs)
        except Exception as own:  # noqa: BLE001 -- shared by _sync
            error = own

        self._sync(error, None)

        if not resumed and self.comm.rank == 0 and self.output.enabled:
            self.output.write_updated(self.model, self.options)
            self.output.write_getdist_metadata(self.model, self.columns)

        error = steps = None

        if self.comm.rank == 0 and self.blocked:
            try:
                steps = self._measure_fast_steps()
            except Exception as own:  # noqa: BLE001 -- shared by _sync
                error = own

        self.fast_steps = self._sync(error, None, lambda _: steps) or 0

    def _fresh(self, mine, rngs) -> None:

        for index in mine:

            rng = rngs[index]

            for _ in range(self.options["max_tries"]):

                x = self.model.parameters.sample_ref(rng)
                result = self._evaluate(x)

                if math.isfinite(result.logpost):
                    break

            else:
                raise RuntimeError(
                    f"No starting point with a finite posterior in "
                    f"{self.options['max_tries']} draws from the 'ref' "
                    f"distributions. Narrow 'ref' to where the posterior lives."
                )

            self.chains.append(_Chain(index, x, result, self._columns_of, rng))

            if self.output.enabled:
                self.output.chain(index + 1, self.columns).create()

    def _resume(self, mine, rngs) -> None:

        saved = self.output.read_covmat()

        if saved is not None:
            self._merge_covmat(self.covariance, *saved)

        for index in mine:

            rows = self.output.chain(index + 1, self.columns).read()

            if len(rows) == 0:
                raise RuntimeError(
                    f"Cannot resume: chain {index + 1} has no samples yet."
                )

            x = rows[-1, 2: 2 + self.d]
            result = self._evaluate(x)

            chain = _Chain(index, x, result, self._columns_of, rngs[index])
            chain.rows = list(rows)

            self.chains.append(chain)

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

    # ---------------------------------------------------------
    # Moving
    # ---------------------------------------------------------

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

    @staticmethod
    def _accept(chain: _Chain, log_ratio: float) -> bool:
        return log_ratio >= 0.0 or chain.rng.uniform() < math.exp(log_ratio)

    def _stay(self, chain: _Chain) -> None:

        chain.stuck += 1

        if chain.stuck >= self.options["max_tries"]:
            raise RuntimeError(
                f"Chain {chain.index + 1} rejected {chain.stuck} proposals "
                f"in a row. The proposal is too wide for the posterior: "
                f"give smaller 'proposal' widths or a covmat."
            )

    @staticmethod
    def _move(chain: _Chain, x, result) -> None:

        chain.weight -= 1
        chain.leave()
        chain.x, chain.result = x, result
        chain.weight = 1
        chain.accepted += 1
        chain.stuck = 0

    def _metropolis(self, chain: _Chain, step) -> None:
        """One Metropolis step of ``chain`` by ``step``."""

        chain.proposed += 1
        chain.weight += 1

        proposal = chain.x + step
        result = self._evaluate(proposal)

        if math.isfinite(result.logpost) and self._accept(
            chain, result.logpost - chain.result.logpost
        ):
            self._move(chain, proposal, result)
        else:
            self._stay(chain)

    def _drag(self, chain: _Chain, step, fast) -> None:
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
            self._stay(chain)
            return

        n = self.fast_steps
        log_weight = end.logpost - start.logpost

        for i in range(1, n):

            beta = i / n
            move = fast @ chain.rng.standard_normal(fast.shape[1])

            trial_start = self._evaluate(start_x + move)
            trial_end = self._evaluate(end_x + move)

            if math.isfinite(trial_start.logpost) and math.isfinite(trial_end.logpost):

                new = (1 - beta) * trial_start.logpost + beta * trial_end.logpost
                old = (1 - beta) * start.logpost + beta * end.logpost

                if self._accept(chain, new - old):
                    start_x, start = start_x + move, trial_start
                    end_x, end = end_x + move, trial_end

            log_weight += end.logpost - start.logpost

        if self._accept(chain, log_weight / n):
            self._move(chain, end_x, end)
        else:
            self._stay(chain)

    def _cycle(self, chain: _Chain, slow, fast) -> None:
        """One step of ``chain``: a slow move, and its fast ones."""

        move = slow @ chain.rng.standard_normal(slow.shape[1])

        if fast is None:
            self._metropolis(chain, move)

        elif self.options["drag"]:
            self._drag(chain, move, fast)

        else:

            self._metropolis(chain, move)

            for _ in range(self.fast_steps):
                self._metropolis(chain, fast @ chain.rng.standard_normal(fast.shape[1]))

    # ---------------------------------------------------------
    # Checking
    # ---------------------------------------------------------

    def _flush(self) -> None:

        if not self.output.enabled:
            return

        for chain in self.chains:

            if chain.unwritten:
                new = np.array(chain.rows[-chain.unwritten:])
                self.output.chain(chain.index + 1, self.columns).append(new)
                chain.unwritten = 0

    def _report(self) -> dict:
        """What this process tells the first at a check."""

        return {
            "halves": [(c.index, c.half(self.d)) for c in self.chains],
            "accepted": sum(c.accepted for c in self.chains),
            "proposed": sum(c.proposed for c in self.chains),
        }

    def _decide(self, reports, step, clock) -> dict:
        """
        On the first process: ``R - 1`` from every chain's second half,
        the covariance to go on with, and whether to stop.
        """

        halves = sorted(
            ((i, h) for report in reports for i, h in report["halves"] if h is not None),
            key=lambda item: item[0],
        )
        halves = [h for _, h in halves]

        if len(halves) == 1:

            x, w = halves[0]
            halves = [(part, wp) for part, wp in zip(np.array_split(x, 4), np.array_split(w, 4))]

        halves = [(x, w) for x, w in halves if len(x) > 1 and w.sum() > 0]

        Rminus1, covariance = self.Rminus1, self.covariance

        if len(halves) >= 2 and sum(len(x) for x, _ in halves) > 2 * self.d:

            Rminus1 = gelman_rubin(halves)

            if self.options["learn_proposal"] and (
                Rminus1 < self.options["learn_proposal_Rminus1_max"]
            ):

                x = np.concatenate([x for x, _ in halves])
                w = np.concatenate([w for _, w in halves])

                _, learned = _weighted_mean_cov(x, w)

                if np.all(np.linalg.eigvalsh(learned) > 0.0):
                    covariance = learned
                    self.output.write_covmat(self.names, covariance)

        converged = bool(Rminus1 < self.options["Rminus1_stop"])

        elapsed = time.monotonic() - clock
        limit = self.options["max_time"]

        accepted = sum(r["accepted"] for r in reports)
        proposed = sum(r["proposed"] for r in reports)

        entry = {
            "steps": step,
            "acceptance": accepted / max(proposed, 1),
            "Rminus1": Rminus1,
            "seconds": elapsed,
        }

        self._record(entry)

        if self.callback is not None:
            self.callback({
                **entry,
                "Rminus1_stop": self.options["Rminus1_stop"],
                "max_samples": self.options["max_samples"],
            })

        return {
            "Rminus1": Rminus1,
            "covariance": covariance,
            "converged": converged,
            "entry": entry,
            "stop": converged or (limit is not None and elapsed > limit),
        }

    def _record(self, entry: dict) -> None:

        if not self.output.enabled:
            return

        path = self.output.path(".progress")
        new = not path.exists()

        with open(path, "a", encoding="utf-8") as handle:
            if new:
                handle.write("# steps acceptance Rminus1 seconds\n")
            handle.write(
                f"{entry['steps']} {entry['acceptance']:.4f} "
                f"{entry['Rminus1']:.6g} {entry['seconds']:.2f}\n"
            )

    # ---------------------------------------------------------

    def _run(self) -> None:

        self._start()

        clock = time.monotonic()
        step = 0

        decision = {"stop": False}

        # A resumed run is judged before it moves: chains that already
        # meet the stopping rule are not sampled further, so opening a
        # finished run again costs nothing.
        if self.resumed:
            decision = self._check_now(None, step, clock)

        while not decision["stop"] and step < self.options["max_samples"]:

            slow, fast = self._proposal_factors()
            error = None

            try:

                for _ in range(self.learn_every):
                    for chain in self.chains:
                        self._cycle(chain, slow, fast)

                self._flush()

            except Exception as own:  # noqa: BLE001 -- shared by _sync
                error = own

            step += self.learn_every

            decision = self._check_now(error, step, clock)

        for chain in self.chains:
            chain.leave()

        self._flush()

        def collect(payloads):
            rows = dict(item for payload in payloads for item in payload)
            return [rows[i] for i in range(self.n_chains)]

        width = 2 + len(self.columns)

        self.rows_by_chain = self._sync(
            None, [(c.index, np.array(c.rows).reshape(-1, width)) for c in self.chains],
            collect,
        )

    def _check_now(self, error, step, clock) -> dict:
        """Compare the chains across processes, and take the decision."""

        decision = self._sync(
            error, self._report(),
            lambda reports: self._decide(reports, step, clock),
        )

        self.Rminus1 = decision["Rminus1"]
        self.covariance = decision["covariance"]
        self.converged = decision["converged"]
        self.progress.append(decision["entry"])

        return decision

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

        if self.rows_by_chain is not None:
            every = self.rows_by_chain
        else:
            every = [np.array(c.rows).reshape(-1, width) for c in self.chains]

        chains = []

        for rows in every:

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


def _run_in_process(comm, info, options) -> None:
    """A process started by :meth:`MCMC.run`: rebuild, and run its chains."""

    from cosmofit.core.model import Model
    from cosmofit.core.output import Output

    sampler = MCMC(options, Model(info), output=Output(info))
    sampler.comm = comm
    sampler._run()
