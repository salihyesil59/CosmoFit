"""
Plotting for a fitted :class:`~stats.fitter.Fitter`.

``FitPlotter`` owns every figure CosmoFit knows how to produce --
MCMC diagnostics (chain, corner) as well as the standard
publication-style cosmology figures (Hubble diagram, H(z)
compilation, BAO distance plot, w(z) evolution, deceleration
parameter). It is attached to a ``Fitter`` instance as
``fitter.plots`` (see :class:`stats.fitter.Fitter`), the same
composition pattern the ``cosmology`` package uses for its
calculators (``cosmology.distance``, ``cosmology.background``, ...).

matplotlib and corner are only imported inside the methods that
need them, so building/fitting a model does not require either
package to be installed.
"""

from __future__ import annotations

import numpy as np

from likelihoods import (
    CCLikelihood,
    DESILikelihood,
    PantheonLikelihood,
    PlanckLikelihood,
)


# ============================================================
# Shared figure style
# ============================================================

#: Consistent colors reused across every figure below.
COLOR_DATA = "#2b2b2b"
COLOR_MODEL = "#d1495b"
COLOR_BAND = "#d1495b"
COLOR_REFERENCE = "#6c757d"


def _style_axes(ax) -> None:

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


# ============================================================
# FitPlotter
# ============================================================

class FitPlotter:
    """
    Plotting methods for a :class:`~stats.fitter.Fitter`.

    Parameters
    ----------
    fitter : stats.fitter.Fitter
        The (already constructed) fitter to plot. MCMC- and
        best-fit-dependent plots require ``run_mcmc()`` /
        ``best_fit()`` to have been called first; the
        data-vs-model figures (:meth:`hubble_diagram`, :meth:`hz`,
        :meth:`bao_distances`) work from ``fitter.cosmology`` as
        it currently stands (initial point) if neither has been
        run yet.
    """

    def __init__(self, fitter):

        self.fitter = fitter

    # ============================================================
    # Internal helpers
    # ============================================================

    def _find_likelihood(self, cls):
        """
        Return the first likelihood of type ``cls`` attached to
        the fitter, or ``None`` if that dataset was not included.
        """

        for lk in self.fitter.likelihoods:
            if isinstance(lk, cls):
                return lk

        return None

    # ------------------------------------------------------------

    def _require_likelihood(self, cls, dataset_name):

        lk = self._find_likelihood(cls)

        if lk is None:
            raise ValueError(
                f"'{dataset_name}' is not among this fitter's "
                f"datasets ({self.fitter.dataset_names}); nothing "
                f"to plot."
            )

        return lk

    # ------------------------------------------------------------

    def _reference_theta(self) -> np.ndarray:
        """
        The "best available" parameter point: the best-fit result
        if :meth:`Fitter.best_fit` has been run, else the
        posterior median if :meth:`Fitter.run_mcmc` has been run,
        else the initial point.
        """

        fitter = self.fitter

        if fitter.best_fit_result is not None:
            return np.asarray(fitter.best_fit_result.x, dtype=float)

        if fitter.sampler is not None:
            return np.median(fitter.flat_samples(), axis=0)

        return fitter.theta0

    # ------------------------------------------------------------

    def _evaluate(self, theta, func):
        """
        Apply ``theta`` to the shared cosmology and evaluate
        ``func(cosmology)``.
        """

        self.fitter.logpost._apply(theta)

        return np.asarray(func(self.fitter.cosmology))

    # ------------------------------------------------------------

    def _predictive_band(
        self,
        func,
        n_draws: int = 300,
        seed: int = 0,
        quantiles=(16, 50, 84),
    ):
        """
        Evaluate ``func(cosmology) -> ndarray`` at a reference
        parameter point, and -- if an MCMC chain is available --
        at ``n_draws`` random posterior samples, to produce a
        posterior-predictive uncertainty band. This is the
        standard way cosmology papers show a model curve against
        data: not a single line, but the range of curves the
        posterior actually allows.

        Restores the cosmology to the reference point before
        returning, so this has no visible side effect on
        ``fitter.cosmology``.

        Returns
        -------
        curve : ndarray
            ``func`` evaluated at the reference point.
        band : dict[int, ndarray] or None
            Percentile curves keyed by the requested quantiles,
            or ``None`` if no MCMC chain is available yet.
        """

        fitter = self.fitter
        reference = self._reference_theta()

        band = None

        if fitter.sampler is not None:

            flat = fitter.flat_samples()

            rng = np.random.default_rng(seed)
            idx = rng.choice(
                len(flat), size=min(n_draws, len(flat)), replace=False,
            )

            curves = np.array([
                self._evaluate(theta, func) for theta in flat[idx]
            ])

            band = {
                q: np.percentile(curves, q, axis=0)
                for q in quantiles
            }

        curve = self._evaluate(reference, func)

        return curve, band

    # ------------------------------------------------------------

    @staticmethod
    def _plot_band(ax, z, band, label="68% posterior"):

        if band is None:
            return

        ax.fill_between(
            z, band[16], band[84],
            color=COLOR_BAND, alpha=0.25, linewidth=0, label=label,
        )

    # ------------------------------------------------------------

    @staticmethod
    def _zero_crossings(z, y):
        """
        Redshifts at which ``y(z)`` changes sign, by linear
        interpolation between grid points.
        """

        sign_change = np.where(np.diff(np.sign(y)) != 0)[0]

        crossings = []

        for i in sign_change:
            z0, z1 = z[i], z[i + 1]
            y0, y1 = y[i], y[i + 1]
            crossings.append(z0 - y0 * (z1 - z0) / (y1 - y0))

        return crossings

    # ============================================================
    # MCMC diagnostics
    # ============================================================

    def chain(self, save_path=None):
        """
        Trace plot of every free parameter's walkers vs. MCMC step.
        """

        import matplotlib.pyplot as plt

        fitter = self.fitter

        if fitter.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        chain = fitter.sampler.get_chain()

        fig, axes = plt.subplots(
            fitter.ndim, 1, figsize=(12, 2.4 * fitter.ndim), sharex=True,
        )

        if fitter.ndim == 1:
            axes = [axes]

        for i, name in enumerate(fitter.free_params):
            axes[i].plot(chain[:, :, i], alpha=0.15, color=COLOR_MODEL)
            axes[i].set_ylabel(name)
            _style_axes(axes[i])

        axes[-1].set_xlabel("MCMC step")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def corner(self, burnin=None, save_path=None, **corner_kwargs):
        """
        Corner (triangle) plot of the free-parameter posterior.
        """

        import corner as corner_pkg

        flat = self.fitter.flat_samples(burnin=burnin)

        kwargs = dict(
            labels=self.fitter.free_params,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            smooth=1.0,
        )
        kwargs.update(corner_kwargs)

        fig = corner_pkg.corner(flat, **kwargs)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ============================================================
    # Data vs. model
    # ============================================================

    def hubble_diagram(self, save_path=None, n_draws=300, seed=0):
        """
        Supernova Hubble diagram: distance modulus mu vs. redshift,
        with a residuals sub-panel. The classic Pantheon-style
        figure.

        Data are shown at their calibrated apparent magnitude
        (``m_b_corr``); the model curve includes the analytically
        marginalized absolute-magnitude offset (or ``cosmology.MB``
        if it was fit explicitly) so both are on the same scale.
        """

        import matplotlib.pyplot as plt

        lk = self._require_likelihood(PantheonLikelihood, "pantheon")

        z_grid = np.linspace(
            1e-3, lk.data.z_hd.max() * 1.02, 300,
        )

        def offset():
            return lk.best_fit_offset() if lk.marginalize_MB else 0.0

        curve, band = self._predictive_band(
            lambda c: c.distance.mu(z_grid) + offset(),
            n_draws=n_draws, seed=seed,
        )

        # `_predictive_band` leaves the shared cosmology at the
        # reference point, so `lk.model()` here reflects it too.
        mu_data_model = lk.model() + offset()

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax_top.errorbar(
            lk.data.z_hd, lk.data.m_b_corr,
            yerr=lk.covariance.sigma,
            fmt="o", ms=2.5, elinewidth=0.6, alpha=0.5,
            color=COLOR_DATA, label=f"Pantheon+ ({lk.n_data})",
        )
        self._plot_band(ax_top, z_grid, band)
        ax_top.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")

        ax_top.set_xscale("log")
        ax_top.set_ylabel(r"$\mu = m_B - M_B$")
        ax_top.legend(frameon=False)
        _style_axes(ax_top)

        residual = lk.data.m_b_corr - mu_data_model
        ax_bot.errorbar(
            lk.data.z_hd, residual, yerr=lk.covariance.sigma,
            fmt="o", ms=2.5, elinewidth=0.6, alpha=0.5, color=COLOR_DATA,
        )
        ax_bot.axhline(0.0, color=COLOR_REFERENCE, lw=1.0, ls="--")
        ax_bot.set_ylabel(r"$\Delta\mu$")
        ax_bot.set_xlabel("Redshift $z$")
        _style_axes(ax_bot)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def hz(self, save_path=None, n_draws=300, seed=0):
        """
        Cosmic Chronometer H(z) diagram: expansion-rate
        measurements vs. redshift against the model H(z) curve.
        """

        import matplotlib.pyplot as plt

        lk = self._require_likelihood(CCLikelihood, "cc")

        z_grid = np.linspace(0.0, lk.data.z.max() * 1.05, 300)

        curve, band = self._predictive_band(
            lambda c: c.background.H(z_grid), n_draws=n_draws, seed=seed,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))

        ax.errorbar(
            lk.data.z, lk.data.H, yerr=lk.covariance.sigma,
            fmt="o", ms=4, elinewidth=0.8, color=COLOR_DATA,
            label=f"Cosmic Chronometers ({lk.n_data})",
        )
        self._plot_band(ax, z_grid, band)
        ax.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def bao_distances(self, save_path=None, n_draws=300, seed=0):
        """
        DESI BAO distance plot: one panel per observable type
        (D_M/r_d, D_H/r_d, D_V/r_d), each showing the tracers'
        measurements against the model curve -- the standard DESI
        / BOSS-style BAO summary figure.
        """

        import matplotlib.pyplot as plt

        from likelihoods.desi import MODEL_MAP

        lk = self._require_likelihood(DESILikelihood, "desi")

        observables = sorted(set(lk.data.observable.tolist()))

        z_grid = np.linspace(
            lk.data.z.min() * 0.9, lk.data.z.max() * 1.1, 300,
        )

        fig, axes = plt.subplots(
            1, len(observables), figsize=(5.0 * len(observables), 5), squeeze=False,
        )
        axes = axes[0]

        for ax, observable in zip(axes, observables):

            model_func = MODEL_MAP[observable]

            curve, band = self._predictive_band(
                lambda c, f=model_func: f(c, z_grid),
                n_draws=n_draws, seed=seed,
            )

            mask = lk.data.observable == observable

            ax.errorbar(
                lk.data.z[mask], lk.data.value[mask],
                yerr=lk.covariance.sigma[mask],
                fmt="o", ms=5, elinewidth=0.8, color=COLOR_DATA,
                label="DESI 2024",
            )
            self._plot_band(ax, z_grid, band)
            ax.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")

            ax.set_title(observable.replace("_", " "))
            ax.set_xlabel("Redshift $z$")
            ax.legend(frameon=False)
            _style_axes(ax)

        axes[0].set_ylabel("Distance / $r_d$")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def planck_residuals(self, save_path=None):
        """
        Standardized-residual ("pull") plot for the Planck
        distance-prior vector (R, l_A, omega_b_h2):

            pull_i = (data_i - model_i) / sigma_i

        With only 3 compressed data points a curve-vs-data figure
        isn't meaningful; a pull plot (common for compressed/
        summary-statistic likelihoods) is the standard way to show
        whether the current cosmology is consistent with each
        component, in units of its own uncertainty.
        """

        import matplotlib.pyplot as plt

        lk = self._require_likelihood(PlanckLikelihood, "planck")

        self._evaluate(self._reference_theta(), lambda c: 0.0)

        pull = lk.residuals() / lk.covariance.sigma

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.arange(len(pull))
        ax.axhline(0.0, color=COLOR_REFERENCE, lw=1.0, ls="--")
        ax.axhspan(-1, 1, color=COLOR_BAND, alpha=0.12, linewidth=0)
        ax.bar(x, pull, color=COLOR_MODEL, width=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(lk.data.labels)
        ax.set_ylabel(r"$(\mathrm{data} - \mathrm{model}) / \sigma$")
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ============================================================
    # Dark-energy / background diagnostics
    # ============================================================

    def w_of_z(self, z_max=2.5, save_path=None, n_draws=300, seed=0):
        """
        Dark-energy equation-of-state evolution w(z), with a
        posterior band and the LCDM reference line w = -1 -- the
        standard w0-wa dark-energy figure.

        Only meaningful for a model that defines w(z) (``CPL``,
        ``WCDM``); raises for ``LCDM``, since w = -1 identically
        there.
        """

        import matplotlib.pyplot as plt

        cosmology = self.fitter.cosmology

        if not hasattr(cosmology, "w"):
            raise NotImplementedError(
                f"{type(cosmology).__name__} does not define w(z) "
                f"(dark energy is a cosmological constant by "
                f"construction); w_of_z() is only meaningful for "
                f"dark-energy models such as CPL or WCDM."
            )

        z_grid = np.linspace(0.0, z_max, 300)

        curve, band = self._predictive_band(
            lambda c: c.w(z_grid), n_draws=n_draws, seed=seed,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))

        self._plot_band(ax, z_grid, band)
        ax.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")
        ax.axhline(
            -1.0, color=COLOR_REFERENCE, lw=1.2, ls="--", label=r"$\Lambda$CDM",
        )

        crossings = self._zero_crossings(z_grid, curve + 1.0)
        for zc in crossings:
            ax.axvline(zc, color=COLOR_REFERENCE, lw=0.8, ls=":")
            ax.annotate(
                rf"$z_c={zc:.2f}$", (zc, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 4), fontsize=8,
            )

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("$w(z)$")
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def deceleration(self, z_max=2.5, save_path=None, n_draws=300, seed=0):
        """
        Deceleration parameter q(z) = -1 + (1+z) E'(z)/E(z), with
        a posterior band, marking the deceleration/acceleration
        transition redshift z_t where q(z) = 0. Works for any
        model, since q(z) only needs E(z) and its derivative.
        """

        import matplotlib.pyplot as plt

        z_grid = np.linspace(0.0, z_max, 300)

        curve, band = self._predictive_band(
            lambda c: c.background.q(z_grid), n_draws=n_draws, seed=seed,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))

        self._plot_band(ax, z_grid, band)
        ax.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")
        ax.axhline(0.0, color=COLOR_REFERENCE, lw=1.0, ls="--")

        for zt in self._zero_crossings(z_grid, curve):
            ax.axvline(zt, color=COLOR_REFERENCE, lw=0.8, ls=":")
            ax.annotate(
                rf"$z_t={zt:.2f}$", (zt, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 4), fontsize=8,
            )

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("$q(z)$")
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig
