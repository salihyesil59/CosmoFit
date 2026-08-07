"""
High-level fitting interface.

``Fitter`` is the "few lines of code" entry point CosmoFit is
built around: pick a cosmological model, pick which datasets
to combine, pick which parameters are free, and get an MCMC
posterior, a best-fit point, and the usual diagnostic plots
back out.

Example
-------
>>> from CosmoFit import CPL, Fitter
>>>
>>> fit = Fitter(
...     model=CPL,
...     datasets=["cc", "desi", "pantheon"],
...     free_params=["H0", "Omega_m", "w0", "wa", "rd"],
...     initial={"H0": 67.4, "Omega_m": 0.315,
...              "w0": -1.0, "wa": 0.0, "rd": 147.1},
... )
>>> fit.run_mcmc(nwalkers=48, nsteps=6500, burnin=1000)
>>> fit.best_fit()
>>> fit.summary()
>>> fit.plots.corner()
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core.parameters import CosmologyParameters

from CosmoFit.likelihoods.cc import CCLikelihood
from CosmoFit.likelihoods.desi import DESILikelihood
from CosmoFit.likelihoods.sdss_bao import SDSSBAOLikelihood
from CosmoFit.likelihoods.pantheon import PantheonLikelihood
from CosmoFit.likelihoods.des_sn5yr import DESSN5YRLikelihood
from CosmoFit.likelihoods.planck import PlanckLikelihood
from CosmoFit.likelihoods.joint import JointLikelihood

from CosmoFit.plots import FitPlotter

from .priors import UniformPrior
from .posterior import LogPosterior
from .sampler import EnsembleSampler
from .results import FitResult, BestFitResult, MCMCResult


# ============================================================
# Dataset registry
# ============================================================

#: Maps a short dataset name (as passed in ``datasets=[...]``)
#: to the likelihood class that implements it.
DATASET_REGISTRY = {
    "cc": CCLikelihood,
    "desi": DESILikelihood,
    "sdss_bao": SDSSBAOLikelihood,
    "pantheon": PantheonLikelihood,
    "des_sn5yr": DESSN5YRLikelihood,
    "planck": PlanckLikelihood,
}


# ============================================================
# Multiprocessing worker plumbing (for Fitter.run_mcmc(n_processes=...))
# ============================================================
#
# emcee's own multiprocessing recipe (pass a `multiprocessing.Pool`
# straight to `EnsembleSampler`) re-pickles and sends the *entire*
# log-posterior object -- including every dataset's covariance
# matrix -- to a worker on every single ensemble step. For a dataset
# like Pantheon+ (a ~1600x1600 dense covariance), that's tens of MB
# per step, which comfortably outweighs the few milliseconds an
# actual likelihood evaluation takes -- i.e. naive multiprocessing
# here is *slower* than one process, not faster (measured directly:
# ~19ms to pickle a 4-dataset `LogPosterior` vs ~2ms to evaluate it).
#
# So instead, each worker builds its *own* `Fitter` once, from a
# small, cheap-to-pickle "recipe" (`Fitter._recipe()`: the model
# class, dataset names, free parameter names, and current parameter
# values -- no arrays), via the pool's `initializer`. Only a length-
# `ndim` float vector then crosses the process boundary per
# evaluation. This requires `model` to be picklable *by reference*
# (true for every built-in model, and for a module-level `Cosmology`
# subclass with `EXTRA_PARAMS` -- not for a `define_model()`/
# `model_from_expression()` class, which is built dynamically and
# has no stable import path; `Fitter.run_mcmc(n_processes=...)`
# checks for this and raises a clear error rather than an obscure
# pickling failure).

_worker_fitter = None


def _init_worker(recipe: dict) -> None:
    """
    Pool ``initializer``: build this worker process's own `Fitter`
    once (stored in a process-global, not returned -- `Pool`
    initializers can't have a return value collected).
    """

    global _worker_fitter

    try:
        import threadpoolctl
        # Undo whatever BLAS thread pool numpy set up before the
        # fork -- with N worker processes each also running a
        # multi-threaded BLAS, you oversubscribe the machine's
        # cores and multiprocessing can end up *slower* than one
        # process. Confirmed: without this, 8-process speedup on an
        # 8-core machine measured 1.4x; with it, 2.4x.
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    _worker_fitter = Fitter(**recipe)


def _worker_log_prob(theta):
    """Pool worker target: evaluate this worker's own `Fitter`."""

    return _worker_fitter.logpost(theta)


# ============================================================
# Fitter
# ============================================================

class Fitter:
    """
    Ties together a cosmological model, one or more datasets,
    a prior, and (optionally) an MCMC run and best-fit search.

    Parameters
    ----------
    model : type
        A ``Cosmology`` subclass, e.g. ``LCDM`` or ``CPL`` -- or a
        custom model built with :func:`cosmology.custom.define_model`
        (or a hand-written ``Cosmology`` subclass declaring
        ``EXTRA_PARAMS``), for testing a new model not built into
        CosmoFit.

    datasets : list[str]
        Which likelihoods to combine. Keys of
        :data:`DATASET_REGISTRY` (currently
        ``"cc"``, ``"desi"``, ``"sdss_bao"``, ``"pantheon"``,
        ``"des_sn5yr"``, ``"planck"``). Do not combine
        ``"desi"``/``"sdss_bao"`` or ``"pantheon"``/``"des_sn5yr"``
        in the same fit -- see the corresponding likelihood
        classes' docstrings for why.

    free_params : list[str]
        Names of the :class:`CosmologyParameters` fields that
        are allowed to vary (e.g.
        ``["H0", "Omega_m", "w0", "wa", "rd"]``).

    initial : dict[str, float]
        Starting values for every parameter in
        ``CosmologyParameters`` that you want to override
        (both free and fixed ones may be given here; anything
        not given falls back to the dataclass default).

    fixed : dict[str, float], optional
        Explicit values for parameters that are *not* being
        fit. Equivalent to putting them in ``initial`` and
        simply not listing them in ``free_params`` -- provided
        separately mainly for readability.

    bounds : dict[str, tuple[float, float]], optional
        Overrides for the default uniform-prior bounds
        (``model.PARAMS_CLASS.default_bounds()``, which is
        :data:`cosmology.core.parameters.DEFAULT_BOUNDS` for every
        built-in model).

    dataset_kwargs : dict[str, dict], optional
        Extra keyword arguments passed to the constructor of
        each likelihood, keyed by dataset name. E.g.
        ``{"pantheon": {"marginalize_MB": True}}``.
    """

    def __init__(
        self,
        model,
        datasets,
        free_params,
        initial,
        fixed=None,
        bounds=None,
        dataset_kwargs=None,
    ):

        self.model_cls = model
        self.free_params = list(free_params)
        self.dataset_names = list(datasets)

        # ------------------------------------------------------
        # Build the (shared, mutable) parameter object
        #
        # `model.PARAMS_CLASS` is `CosmologyParameters` for every
        # built-in model; a custom model (`cosmology.define_model`,
        # or a `Cosmology` subclass declaring `EXTRA_PARAMS`) gets
        # its own dynamically-built superset instead -- see
        # `Cosmology.__init_subclass__`.
        # ------------------------------------------------------

        params_cls = getattr(model, "PARAMS_CLASS", CosmologyParameters)
        self.params_cls = params_cls

        values = dict(params_cls.defaults())
        values.update(initial or {})
        values.update(fixed or {})

        missing = [
            name for name in params_cls.names()
            if name not in values
        ]
        if missing:
            raise ValueError(
                f"No initial/default value for parameter(s) "
                f"{missing}. Provide them via `initial=` or "
                f"`fixed=`."
            )

        self.initial = {name: values[name] for name in self.free_params}

        self.params = params_cls(
            **{name: values[name] for name in params_cls.names()}
        )

        self.cosmology = self.model_cls(self.params)

        # ------------------------------------------------------
        # Build likelihoods
        # ------------------------------------------------------

        dataset_kwargs = dataset_kwargs or {}
        self.dataset_kwargs = dataset_kwargs

        self.likelihoods = []

        for name in self.dataset_names:

            if name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset '{name}'. "
                    f"Available: {list(DATASET_REGISTRY)}"
                )

            cls = DATASET_REGISTRY[name]

            kwargs = dataset_kwargs.get(name, {})

            self.likelihoods.append(cls(self.cosmology, **kwargs))

        self.joint = JointLikelihood(*self.likelihoods)

        # ------------------------------------------------------
        # Prior + posterior
        # ------------------------------------------------------

        self.bounds = bounds

        merged_bounds = params_cls.default_bounds()
        if bounds:
            merged_bounds.update(bounds)

        self.prior = UniformPrior(self.free_params, merged_bounds)
        self.logpost = LogPosterior(self.joint, self.cosmology, self.prior)

        self.sampler = None
        self.burnin = 0
        self.best_fit_result = None

        # ------------------------------------------------------
        # Plots
        # ------------------------------------------------------

        self.plots = FitPlotter(self)

    # ============================================================
    # Basic properties
    # ============================================================

    @property
    def ndim(self) -> int:
        return len(self.free_params)

    @property
    def n_data(self) -> int:
        return self.joint.n_data

    @property
    def theta0(self) -> np.ndarray:
        return np.array([self.initial[n] for n in self.free_params])

    # ============================================================
    # chi2 / likelihood at a point (no MCMC needed)
    # ============================================================

    def chi2(self, theta=None) -> float:
        """
        Total chi2 at ``theta`` (defaults to the initial point).
        """

        if theta is None:
            theta = self.theta0

        return self.logpost.chi2(theta)

    def chi2_breakdown(self, theta=None) -> dict:
        """
        Per-dataset chi2 at ``theta`` (defaults to the initial
        point).
        """

        if theta is not None:
            self.logpost._apply(theta)

        return {lk.name: lk.chi2() for lk in self.likelihoods}

    # ============================================================
    # MCMC
    # ============================================================

    def run_mcmc(
        self,
        nwalkers: int = 48,
        nsteps: int = 6000,
        burnin: int = 1000,
        initial_scatter=None,
        seed: int = 42,
        progress: bool = True,
        moves=None,
        callback=None,
        n_processes: int = 1,
    ):
        """
        Run an ``emcee`` ensemble MCMC, via
        :class:`~stats.sampler.EnsembleSampler`.

        Parameters
        ----------
        nwalkers, nsteps : int
            Standard emcee settings. ``nwalkers`` should be at
            least ``2 * ndim``.

        burnin : int
            Number of initial steps discarded by
            :meth:`flat_samples` / :meth:`summary` by default.

        initial_scatter : dict[str, float], optional
            Per-parameter Gaussian scatter used to initialize
            the walkers around ``initial``. Defaults to 1% of
            each parameter's prior width.

        seed : int
            Seed for the walker initialization RNG.

        progress : bool
            Show an emcee progress bar.

        moves : emcee move or list of (move, weight), optional
            Passed through to ``emcee.EnsembleSampler`` (see
            :class:`~stats.sampler.EnsembleSampler`). Defaults to
            emcee's own default proposal (``StretchMove``).

        callback : callable(step, nsteps, elapsed), optional
            Called after every completed step -- see
            :meth:`~stats.sampler.EnsembleSampler.run`. Used by the
            GUI to drive a live progress bar.

        n_processes : int
            Evaluate walkers' log-posteriors across this many worker
            processes instead of one (default 1: no multiprocessing).
            Each worker builds its own `Fitter` once (see the
            `stats.fitter` module docstring for why -- naive
            `pool=multiprocessing.Pool()` is often *slower* here, not
            faster), so this needs `model` to be picklable by
            reference: every built-in model works; a model built
            with `cosmology.custom.define_model()` /
            `model_from_expression()` doesn't (raises `ValueError`
            rather than an obscure pickling failure -- use
            `n_processes=1` for those). Speedup is real but well
            under `n_processes`x in practice (~2.5x measured on 8
            processes for a 4-dataset CPL fit) -- per-step IPC has
            its own cost. Reliable on Linux/macOS; multiprocessing
            from a Windows notebook is fragile for reasons outside
            this library's control (the `spawn` start method needs
            worker functions re-importable from a real module, which
            an interactive notebook session isn't).

        Returns
        -------
        emcee.EnsembleSampler
        """

        backend = EnsembleSampler(moves=moves)

        with self._mcmc_pool(n_processes) as pool:

            # With a pool, `emcee` ships `logpost` itself to every
            # worker on every step -- `self.logpost` is the heavy,
            # data-carrying object multiprocessing here is built to
            # avoid sending (see the module docstring above
            # `_init_worker`). `_worker_log_prob` is a cheap
            # top-level reference to *this worker's own* `Fitter`
            # instead, built once by the pool's `initializer`.
            logpost = _worker_log_prob if pool is not None else self.logpost

            sampler = backend.run(
                logpost,
                self.prior,
                self.theta0,
                nwalkers=nwalkers,
                nsteps=nsteps,
                initial_scatter=initial_scatter,
                seed=seed,
                progress=progress,
                callback=callback,
                pool=pool,
            )

        self.sampler = sampler
        self.burnin = burnin

        return sampler

    # ------------------------------------------------------------

    def _recipe(self) -> dict:
        """
        This fitter's constructor arguments, cheap to pickle (no
        arrays -- just the model class, names, and current scalar
        parameter values) -- enough for a worker process to build an
        equivalent `Fitter` of its own. See `run_mcmc(n_processes=)`.
        """

        return dict(
            model=self.model_cls,
            datasets=self.dataset_names,
            free_params=self.free_params,
            initial=self.params.as_dict(),
            bounds=self.bounds,
            dataset_kwargs=self.dataset_kwargs,
        )

    # ------------------------------------------------------------

    def _mcmc_pool(self, n_processes: int):
        """
        A `multiprocessing.Pool` (as a context manager) for
        `run_mcmc(n_processes=...)`, or a no-op context manager
        yielding `None` for the default `n_processes=1` (so
        `run_mcmc` can always do ``with self._mcmc_pool(n) as
        pool:`` regardless).
        """

        from contextlib import nullcontext

        if n_processes <= 1:
            return nullcontext(None)

        import pickle
        import multiprocessing as mp

        recipe = self._recipe()

        try:
            pickle.dumps(recipe["model"])
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"n_processes={n_processes} needs `model` "
                f"({recipe['model']!r}) to be picklable by reference, "
                f"which every built-in CosmoFit model is, but a "
                f"dynamically-built `define_model()`/"
                f"`model_from_expression()` model isn't. Use "
                f"n_processes=1 for this model."
            ) from exc

        # Explicitly request "fork" rather than using whatever
        # `multiprocessing`'s *default* context is. That default
        # isn't "fork" everywhere: Python 3.14 changed it to
        # "forkserver" on Linux too (it already wasn't "fork" on
        # macOS/Windows), and "forkserver"/"spawn" need every
        # process-spawning call to happen behind an
        # `if __name__ == "__main__":` guard -- a plain script
        # without one crashes outright
        # ("RuntimeError: ... before bootstrapping phase"), and even
        # inside a Jupyter kernel (which sidesteps that crash) it
        # measured *no* speedup at all in practice, only overhead.
        # "fork" has neither problem and needs no such guard, and
        # remains available as an explicit context on Linux/macOS
        # even where it's no longer the default -- just not on
        # Windows, where it was never available.
        try:
            ctx = mp.get_context("fork")
        except ValueError as exc:
            raise ValueError(
                f"n_processes={n_processes} needs the 'fork' "
                f"multiprocessing start method, which isn't available "
                f"on this platform (e.g. Windows never has it). Use "
                f"n_processes=1 here."
            ) from exc

        return ctx.Pool(n_processes, initializer=_init_worker, initargs=(recipe,))

    # ------------------------------------------------------------

    def flat_samples(self, burnin=None) -> np.ndarray:

        if self.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        if burnin is None:
            burnin = self.burnin

        return self.sampler.get_chain(discard=burnin, flat=True)

    # ------------------------------------------------------------

    def convergence(self, burnin=None, tol: int = 50) -> dict:
        """
        MCMC convergence diagnostics, based on the integrated
        autocorrelation time tau of each free parameter's chain
        (``emcee``'s own recommended diagnostic -- see
        https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).

        A chain is considered trustworthy once it has run for at
        least ``tol`` (default 50, emcee's own default) times tau:
        shorter than that, the posterior (especially its width)
        is not yet reliable, no matter how good ``summary()``
        looks. This is *not* checked automatically anywhere else
        in ``Fitter`` -- call this explicitly after ``run_mcmc()``
        and inspect ``converged`` before trusting ``summary()`` /
        ``plots.corner()``.

        Parameters
        ----------
        burnin : int, optional
            Steps to discard before estimating tau. Defaults to
            ``self.burnin``.

        tol : int, optional
            Convergence threshold, in units of tau (chain length
            must exceed ``tol * tau`` for every parameter).

        Returns
        -------
        dict with keys:
            tau : dict[str, float]
                Autocorrelation time per free parameter.
            n_used : int
                Number of post-burnin steps the estimate is based on.
            n_effective : dict[str, float]
                Effective number of independent samples per
                parameter (n_used * nwalkers / tau).
            converged : bool
                Whether every parameter satisfies
                ``n_used >= tol * tau``.
        """

        if self.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        if burnin is None:
            burnin = self.burnin

        chain = self.sampler.get_chain(discard=burnin)
        n_used, nwalkers, _ = chain.shape

        # `quiet=True`: report the (unreliable) estimate instead of
        # raising when the chain is too short to trust it -- that
        # is exactly the situation `converged` below is meant to
        # flag to the caller.
        tau = self.sampler.get_autocorr_time(discard=0, quiet=True)

        tau_dict = dict(zip(self.free_params, map(float, tau)))

        n_effective = {
            name: float(n_used * nwalkers / t)
            for name, t in tau_dict.items()
        }

        converged = all(n_used >= tol * t for t in tau_dict.values())

        return {
            "tau": tau_dict,
            "n_used": int(n_used),
            "n_effective": n_effective,
            "converged": converged,
        }

    # ------------------------------------------------------------

    def samples_dict(self, burnin=None) -> dict:
        """
        Flat posterior samples as a dict of 1D arrays, keyed by
        parameter name.
        """

        flat = self.flat_samples(burnin=burnin)

        return {
            name: flat[:, i]
            for i, name in enumerate(self.free_params)
        }

    # ------------------------------------------------------------

    def summary(self, burnin=None) -> dict:
        """
        Posterior median +/- 68% interval for every free
        parameter.
        """

        flat = self.flat_samples(burnin=burnin)

        result = {}

        for i, name in enumerate(self.free_params):

            q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])

            result[name] = {
                "median": float(q50),
                "plus": float(q84 - q50),
                "minus": float(q50 - q16),
            }

        return result

    # ============================================================
    # Best fit
    # ============================================================

    def best_fit(self, x0=None, bounds=None):
        """
        Maximum-likelihood point via ``scipy.optimize.minimize``
        (L-BFGS-B), starting either from ``x0``, from the
        highest-posterior MCMC sample if available, or from
        ``initial``.
        """

        from scipy.optimize import minimize

        if x0 is None:

            if self.sampler is not None:

                flat = self.flat_samples()
                log_prob = self.sampler.get_log_prob(
                    discard=self.burnin, flat=True,
                )
                x0 = flat[np.argmax(log_prob)]

            else:

                x0 = self.theta0

        if bounds is None:
            bounds = list(zip(self.prior.lower, self.prior.upper))

        result = minimize(
            self.logpost.chi2,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        self.best_fit_result = result

        return result

    # ------------------------------------------------------------

    @property
    def best_fit_params(self) -> dict:

        if self.best_fit_result is None:
            raise RuntimeError("Call best_fit() first.")

        return dict(zip(self.free_params, self.best_fit_result.x))

    @property
    def best_fit_chi2(self) -> float:

        if self.best_fit_result is None:
            raise RuntimeError("Call best_fit() first.")

        return float(self.best_fit_result.fun)

    # ============================================================
    # Consolidated result
    # ============================================================

    @property
    def result(self) -> FitResult:
        """
        A single :class:`~stats.results.FitResult` snapshot of
        whatever this fitter currently has: the best-fit point
        (if :meth:`best_fit` has been called), the MCMC posterior
        (if :meth:`run_mcmc` has been called), or both.

        This is a read-only summary built on demand -- it does
        not replace :meth:`summary`, :meth:`convergence`,
        :attr:`best_fit_params` etc., which remain the way to get
        each piece individually. It exists for the "one object,
        printable in one line, saveable to JSON" case, e.g.::

            fit.run_mcmc(...)
            fit.best_fit()
            print(fit.result)
            fit.result.save_json("cpl_fit.json")
        """

        best_fit = None
        if self.best_fit_result is not None:
            best_fit = BestFitResult(
                params=self.best_fit_params,
                chi2=self.best_fit_chi2,
                ndim=self.ndim,
                n_data=self.n_data,
                success=bool(self.best_fit_result.success),
                message=str(self.best_fit_result.message),
            )

        mcmc = None
        if self.sampler is not None:
            mcmc = MCMCResult(
                summary=self.summary(),
                convergence=self.convergence(),
                nwalkers=self.sampler.nwalkers,
                nsteps=self.sampler.iteration,
                burnin=self.burnin,
                ndim=self.ndim,
                acceptance_fraction=float(
                    np.mean(self.sampler.acceptance_fraction)
                ),
            )

        return FitResult(
            model=self.model_cls.__name__,
            datasets=self.dataset_names,
            free_params=self.free_params,
            best_fit=best_fit,
            mcmc=mcmc,
        )

    # ------------------------------------------------------------

    def __repr__(self):

        return (
            f"Fitter(model={self.model_cls.__name__}, "
            f"datasets={self.dataset_names}, "
            f"free_params={self.free_params})"
        )
