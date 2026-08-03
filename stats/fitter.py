"""
High-level fitting interface.

``Fitter`` is the "few lines of code" entry point CosmoFit is
built around: pick a cosmological model, pick which datasets
to combine, pick which parameters are free, and get an MCMC
posterior, a best-fit point, and the usual diagnostic plots
back out.

Example
-------
>>> from cosmology import CPL
>>> from stats.fitter import Fitter
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

from cosmology.core.parameters import CosmologyParameters, DEFAULT_BOUNDS

from likelihoods.cc import CCLikelihood
from likelihoods.desi import DESILikelihood
from likelihoods.pantheon import PantheonLikelihood
from likelihoods.planck import PlanckLikelihood
from likelihoods.joint import JointLikelihood

from plots import FitPlotter

from .priors import UniformPrior
from .posterior import LogPosterior


# ============================================================
# Dataset registry
# ============================================================

#: Maps a short dataset name (as passed in ``datasets=[...]``)
#: to the likelihood class that implements it.
DATASET_REGISTRY = {
    "cc": CCLikelihood,
    "desi": DESILikelihood,
    "pantheon": PantheonLikelihood,
    "planck": PlanckLikelihood,
}


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
        A ``Cosmology`` subclass, e.g. ``LCDM`` or ``CPL``.

    datasets : list[str]
        Which likelihoods to combine. Keys of
        :data:`DATASET_REGISTRY` (currently
        ``"cc"``, ``"desi"``, ``"pantheon"``, ``"planck"``).

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
        (:data:`cosmology.core.parameters.DEFAULT_BOUNDS`).

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
        # ------------------------------------------------------

        values = dict(CosmologyParameters.defaults())
        values.update(initial or {})
        values.update(fixed or {})

        missing = [
            name for name in CosmologyParameters.names()
            if name not in values
        ]
        if missing:
            raise ValueError(
                f"No initial/default value for parameter(s) "
                f"{missing}. Provide them via `initial=` or "
                f"`fixed=`."
            )

        self.initial = {name: values[name] for name in self.free_params}

        self.params = CosmologyParameters(
            **{name: values[name] for name in CosmologyParameters.names()}
        )

        self.cosmology = self.model_cls(self.params)

        # ------------------------------------------------------
        # Build likelihoods
        # ------------------------------------------------------

        dataset_kwargs = dataset_kwargs or {}

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

        merged_bounds = dict(DEFAULT_BOUNDS)
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
    ):
        """
        Run an ``emcee`` ensemble MCMC.

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

        Returns
        -------
        emcee.EnsembleSampler
        """

        import emcee

        if initial_scatter is None:
            scatter = 0.01 * (self.prior.upper - self.prior.lower)
        else:
            scatter = np.array(
                [initial_scatter[n] for n in self.free_params],
                dtype=float,
            )

        rng = np.random.default_rng(seed)

        pos = self.theta0 + scatter * rng.normal(size=(nwalkers, self.ndim))

        # Keep the initial walker positions safely inside the prior.
        eps = 1e-6 * (self.prior.upper - self.prior.lower)
        pos = np.clip(pos, self.prior.lower + eps, self.prior.upper - eps)

        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.logpost)

        sampler.run_mcmc(pos, nsteps, progress=progress)

        self.sampler = sampler
        self.burnin = burnin

        return sampler

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

    # ------------------------------------------------------------

    def __repr__(self):

        return (
            f"Fitter(model={self.model_cls.__name__}, "
            f"datasets={self.dataset_names}, "
            f"free_params={self.free_params})"
        )
