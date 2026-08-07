"""
Plotting for a fitted :class:`~stats.fitter.Fitter`.

``FitPlotter`` owns every figure CosmoFit knows how to produce --
MCMC diagnostics (chain, corner), the standard publication-style
cosmology figures for a single fit (Hubble diagram, H(z)
compilation, BAO distance plot, w(z) evolution, deceleration
parameter), and multi-model comparison versions of those same
figures (``compare_hz``, ``compare_deceleration``, ``compare_w_of_z``,
``compare_hubble_diagram``, ``compare_des_hubble_diagram``,
``compare_bao_distances``, ``compare_sdss_bao_distances``) that
overlay this fit's curve with one or more other models' curves on
the same data panel -- the standard "model A vs model B" figures
cosmology papers use. It is attached to a ``Fitter`` instance as
``fitter.plots`` (see :class:`stats.fitter.Fitter`), the same
composition pattern the ``cosmology`` package uses for its
calculators (``cosmology.distance``, ``cosmology.background``, ...).

matplotlib and corner are only imported inside the methods that
need them, so building/fitting a model does not require either
package to be installed.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.likelihoods import (
    CCLikelihood,
    DESILikelihood,
    SDSSBAOLikelihood,
    PantheonLikelihood,
    DESSN5YRLikelihood,
    PlanckLikelihood,
    FSigma8Likelihood,
)


# ============================================================
# Shared figure style
# ============================================================

#: Consistent colors reused across every figure below.
COLOR_DATA = "#2b2b2b"
COLOR_MODEL = "#d1495b"
COLOR_BAND = "#d1495b"
COLOR_REFERENCE = "#6c757d"

#: Colors cycled across models in the ``compare_*`` figures, in
#: order added (self first, then each of ``other_fits``). Chosen to
#: stay distinguishable from `COLOR_DATA`/`COLOR_REFERENCE` and from
#: each other; cycles if more models are compared than colors.
COMPARISON_PALETTE = [
    "#d1495b", "#2b6a99", "#3ca67a", "#e8a33d", "#7b52ab", "#4d4d4d",
]


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

    def _sn_hubble_diagram(
        self,
        lk,
        y_data,
        offset_fn,
        dataset_label,
        y_label,
        n_draws,
        seed,
        save_path,
    ):
        """
        Shared implementation behind :meth:`hubble_diagram`
        (Pantheon+) and :meth:`des_hubble_diagram` (DES-SN5YR):
        distance modulus vs. redshift with a residuals sub-panel,
        the classic SN Ia Hubble-diagram figure. Both datasets are
        SN-like likelihoods using
        :class:`~likelihoods.base.AnalyticOffsetMixin` -- only the
        observed y-values, the resulting offset, and a couple of
        labels differ.
        """

        import matplotlib.pyplot as plt

        z_grid = np.linspace(
            1e-3, lk.data.z_hd.max() * 1.02, 300,
        )

        curve, band = self._predictive_band(
            lambda c: c.distance.mu(z_grid) + offset_fn(),
            n_draws=n_draws, seed=seed,
        )

        # `_predictive_band` leaves the shared cosmology at the
        # reference point, so `lk.model()` here reflects it too.
        y_model = lk.model() + offset_fn()

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax_top.errorbar(
            lk.data.z_hd, y_data,
            yerr=lk.covariance.sigma,
            fmt="o", ms=2.5, elinewidth=0.6, alpha=0.5,
            color=COLOR_DATA, label=f"{dataset_label} ({lk.n_data})",
        )
        self._plot_band(ax_top, z_grid, band)
        ax_top.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")

        ax_top.set_xscale("log")
        ax_top.set_ylabel(y_label)
        ax_top.legend(frameon=False)
        _style_axes(ax_top)

        residual = y_data - y_model
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

    def hubble_diagram(self, save_path=None, n_draws=300, seed=0):
        """
        Pantheon+ Hubble diagram: distance modulus vs. redshift,
        with a residuals sub-panel.

        Data are shown at their calibrated apparent magnitude
        (``m_b_corr``); the model curve includes the analytically
        marginalized absolute-magnitude offset (or ``cosmology.MB``
        if it was fit explicitly) so both are on the same scale.
        """

        lk = self._require_likelihood(PantheonLikelihood, "pantheon")

        return self._sn_hubble_diagram(
            lk,
            y_data=lk.data.m_b_corr,
            offset_fn=lambda: lk.best_fit_offset() if lk.marginalize_MB else 0.0,
            dataset_label="Pantheon+",
            y_label=r"$\mu = m_B - M_B$",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def des_hubble_diagram(self, save_path=None, n_draws=300, seed=0):
        """
        DES-SN5YR Hubble diagram: distance modulus vs. redshift,
        with a residuals sub-panel.

        Unlike Pantheon+, DES-SN5YR's data vector (``mu``) is
        already a distance modulus (computed assuming a fiducial
        H0=70); the model curve includes the analytically
        marginalized zero-point offset (or ``cosmology.MB`` if it
        was fit explicitly) so both are on the same scale.
        """

        lk = self._require_likelihood(DESSN5YRLikelihood, "des_sn5yr")

        return self._sn_hubble_diagram(
            lk,
            y_data=lk.data.mu,
            offset_fn=lambda: lk.best_fit_offset() if lk.marginalize_offset else 0.0,
            dataset_label="DES-SN5YR",
            y_label=r"$\mu$",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

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

    def growth(self, save_path=None, n_draws=300, seed=0):
        """
        fsigma8(z) growth-rate diagram: RSD measurements (e.g. the
        "Gold-2018" compilation) against the model's fsigma8(z)
        curve (:meth:`~cosmology.calculators.background.BackgroundCalculator.fsigma8`).

        The plotted curve is the raw theory prediction, *not*
        Alcock-Paczynski-corrected the way
        :meth:`~likelihoods.fsigma8.FSigma8Likelihood.chi2` compares
        to each data point -- that correction depends on each
        survey's own fiducial H(z)*D_A(z), not a smooth function of
        z, so it has no single continuous curve to draw here.
        """

        import matplotlib.pyplot as plt

        lk = self._require_likelihood(FSigma8Likelihood, "fsigma8")

        z_grid = np.linspace(0.0, lk.data.z.max() * 1.05, 300)

        curve, band = self._predictive_band(
            lambda c: c.background.fsigma8(z_grid), n_draws=n_draws, seed=seed,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))

        ax.errorbar(
            lk.data.z, lk.data.fsigma8, yerr=lk.covariance.sigma,
            fmt="o", ms=4, elinewidth=0.8, color=COLOR_DATA,
            label=f"Growth rate ({lk.n_data})",
        )
        self._plot_band(ax, z_grid, band)
        ax.plot(z_grid, curve, color=COLOR_MODEL, lw=1.8, label="Model")

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel(r"$f\sigma_8(z)$")
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def _bao_distances(self, lk, dataset_label, n_draws, seed, save_path):
        """
        Shared implementation behind :meth:`bao_distances` (DESI)
        and :meth:`sdss_bao_distances` (SDSS): one panel per
        observable type (D_M/r_d, D_H/r_d, D_V/r_d), each showing
        the tracers' measurements against the model curve -- the
        standard DESI/BOSS-style BAO summary figure. Both datasets
        share the same (z, value, observable) structure and
        :data:`likelihoods.desi.MODEL_MAP`.
        """

        import matplotlib.pyplot as plt

        from CosmoFit.likelihoods.desi import MODEL_MAP

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
                label=dataset_label,
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

    def bao_distances(self, save_path=None, n_draws=300, seed=0):
        """
        DESI BAO distance plot: one panel per observable type
        (D_M/r_d, D_H/r_d, D_V/r_d), each showing the tracers'
        measurements against the model curve.
        """

        lk = self._require_likelihood(DESILikelihood, "desi")

        return self._bao_distances(
            lk, dataset_label="DESI 2024",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def sdss_bao_distances(self, save_path=None, n_draws=300, seed=0):
        """
        SDSS BAO distance plot (BOSS DR12 + eBOSS DR16 LRG/QSO):
        one panel per observable type (D_M/r_d, D_H/r_d), each
        showing the tracers' measurements against the model curve.
        """

        lk = self._require_likelihood(SDSSBAOLikelihood, "sdss_bao")

        return self._bao_distances(
            lk, dataset_label="SDSS (DR12+DR16)",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

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

    # ============================================================
    # Model comparison
    # ============================================================
    #
    # Multi-model versions of the single-fit figures above: this
    # fit's curve overlaid with one or more other fits' curves on
    # the same data panel -- the standard "model A vs model B"
    # comparison figure. Each `compare_*` method takes the same
    # `other_fits`/`labels` pair:
    #
    #   other_fits : Fitter, list[Fitter], or None
    #       None (default) auto-compares against a quick LCDM
    #       reference (best-fit only, no MCMC -- just enough for a
    #       curve) built from this fit's own datasets, unless this
    #       fit already *is* LCDM. A single Fitter compares against
    #       just that one. A list compares against all of them, for
    #       an arbitrary N-model figure.
    #   labels : list[str], optional
    #       One per model (this fit first, then `other_fits`, in
    #       order), defaulting to each model's class name.
    #
    # `other_fits` should generally share this fit's dataset(s) --
    # they're plotted against *this* fit's data points, and (for the
    # supernova/BAO comparisons) each fit's own likelihood is used
    # for its analytic offset, so a fit missing that dataset just
    # gets no offset correction rather than an error.

    def _lcdm_reference(self):
        """
        A quick best-fit-only (no MCMC) LCDM fit sharing this fit's
        datasets, for `other_fits=None`'s default comparison.
        """

        from CosmoFit.cosmology.models import LCDM
        from CosmoFit.stats.fitter import Fitter

        reference = Fitter(
            model=LCDM,
            datasets=self.fitter.dataset_names,
            free_params=["H0", "Omega_m"],
            initial=self.fitter.params.as_dict(),
        )
        reference.best_fit()

        return reference

    # ------------------------------------------------------------

    def _resolve_comparison(self, other_fits, labels):

        fits = [self.fitter]

        if other_fits is None:
            if self.fitter.model_cls.__name__ != "LCDM":
                fits.append(self._lcdm_reference())
        elif isinstance(other_fits, (list, tuple)):
            fits.extend(other_fits)
        else:
            fits.append(other_fits)

        if labels is None:
            labels = [f.model_cls.__name__ for f in fits]
        elif len(labels) != len(fits):
            raise ValueError(
                f"Got {len(labels)} label(s) for {len(fits)} model(s) "
                f"being compared -- labels must match one-to-one "
                f"(this fit first, then `other_fits`, including the "
                f"auto-added LCDM reference if `other_fits` was left "
                f"as None)."
            )

        colors = [
            COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)]
            for i in range(len(fits))
        ]

        return fits, list(labels), colors

    # ------------------------------------------------------------

    def _compare_data_curve(
        self, fits, labels, colors, z_grid, func,
        data_z, data_y, data_yerr, data_label,
        xlabel, ylabel, log_x=False,
        n_draws=300, seed=0, save_path=None,
    ):
        """
        Shared implementation behind every `compare_*` figure that
        has real data points (`compare_hz`, `compare_hubble_diagram`,
        ...): data + one model curve (with posterior band, if that
        fit has run `run_mcmc()`) per fit in `fits`.

        `func(fit, cosmology) -> ndarray` evaluates the quantity
        being plotted for one fit's cosmology -- taking `fit` too
        (not just `cosmology`) so callers needing that fit's own
        likelihood (e.g. for an analytic SN offset) can look it up.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))

        ax.errorbar(
            data_z, data_y, yerr=data_yerr,
            fmt="o", ms=3.5, elinewidth=0.7, alpha=0.5, color=COLOR_DATA,
            label=data_label,
        )

        for fit, label, color in zip(fits, labels, colors):

            curve, band = fit.plots._predictive_band(
                lambda c, f=fit: func(f, c), n_draws=n_draws, seed=seed,
            )

            if band is not None:
                ax.fill_between(
                    z_grid, band[16], band[84],
                    color=color, alpha=0.2, linewidth=0,
                )

            ax.plot(z_grid, curve, color=color, lw=1.8, label=label)

        if log_x:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def _compare_curve_only(
        self, fits, labels, colors, z_grid, func,
        xlabel, ylabel, zero_line=None, mark_crossings=False,
        n_draws=300, seed=0, save_path=None,
    ):
        """
        Shared implementation behind every `compare_*` figure with
        no data points, just theory curves (`compare_deceleration`,
        `compare_w_of_z`): one curve (with posterior band, if
        available) per fit in `fits`, optionally marking each
        curve's crossing of `zero_line`.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))

        if zero_line is not None:
            ax.axhline(zero_line, color=COLOR_REFERENCE, lw=1.0, ls="--")

        for fit, label, color in zip(fits, labels, colors):

            curve, band = fit.plots._predictive_band(
                func, n_draws=n_draws, seed=seed,
            )

            if band is not None:
                ax.fill_between(
                    z_grid, band[16], band[84],
                    color=color, alpha=0.15, linewidth=0,
                )

            ax.plot(z_grid, curve, color=color, lw=1.8, label=label)

            if mark_crossings:
                target = zero_line if zero_line is not None else 0.0
                for zc in self._zero_crossings(z_grid, curve - target):
                    ax.axvline(zc, color=color, lw=0.8, ls=":", alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        _style_axes(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def compare_hz(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        H(z) diagram (see :meth:`hz`) with this fit's curve overlaid
        with one or more other models' curves over the same Cosmic
        Chronometers data -- the standard model-vs-model H(z) figure.
        """

        lk = self._require_likelihood(CCLikelihood, "cc")
        fits, labels, colors = self._resolve_comparison(other_fits, labels)

        z_grid = np.linspace(0.0, lk.data.z.max() * 1.05, 300)

        return self._compare_data_curve(
            fits, labels, colors, z_grid,
            func=lambda f, c: c.background.H(z_grid),
            data_z=lk.data.z, data_y=lk.data.H, data_yerr=lk.covariance.sigma,
            data_label=f"Cosmic Chronometers ({lk.n_data})",
            xlabel="Redshift $z$", ylabel=r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def compare_growth(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        fsigma8(z) growth-rate diagram (see :meth:`growth`) with
        this fit's curve overlaid with one or more other models'
        curves over the same RSD data -- the standard model-vs-model
        growth-of-structure figure. Most useful for a modified-
        gravity model against an LCDM reference: unlike every other
        `compare_*` figure, the two curves can now visibly differ
        even when both share the exact same background (see
        `FRHuSawicki`).
        """

        lk = self._require_likelihood(FSigma8Likelihood, "fsigma8")
        fits, labels, colors = self._resolve_comparison(other_fits, labels)

        z_grid = np.linspace(0.0, lk.data.z.max() * 1.05, 300)

        return self._compare_data_curve(
            fits, labels, colors, z_grid,
            func=lambda f, c: c.background.fsigma8(z_grid),
            data_z=lk.data.z, data_y=lk.data.fsigma8, data_yerr=lk.covariance.sigma,
            data_label=f"Growth rate ({lk.n_data})",
            xlabel="Redshift $z$", ylabel=r"$f\sigma_8(z)$",
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def compare_deceleration(
        self, other_fits=None, labels=None, z_max=2.5, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        Deceleration parameter q(z) (see :meth:`deceleration`) for
        this fit and one or more other models on the same axes,
        each with its own deceleration/acceleration transition
        redshift z_t marked -- the standard model-vs-model transition
        -redshift comparison figure.
        """

        fits, labels, colors = self._resolve_comparison(other_fits, labels)
        z_grid = np.linspace(0.0, z_max, 300)

        return self._compare_curve_only(
            fits, labels, colors, z_grid,
            func=lambda c: c.background.q(z_grid),
            xlabel="Redshift $z$", ylabel="$q(z)$",
            zero_line=0.0, mark_crossings=True,
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def compare_w_of_z(
        self, other_fits=None, labels=None, z_max=2.5, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        Dark-energy equation of state w(z) (see :meth:`w_of_z`) for
        this fit and one or more other models on the same axes.
        Unlike the single-fit version, this doesn't raise for a
        model without its own w(z) (e.g. LCDM): it's shown as the
        constant w=-1 line, which is what "no w(z)" means in every
        model in this library.
        """

        fits, labels, colors = self._resolve_comparison(other_fits, labels)
        z_grid = np.linspace(0.0, z_max, 300)

        def _w(c):
            w = getattr(c, "w", None)
            return w(z_grid) if w is not None else np.full_like(z_grid, -1.0)

        return self._compare_curve_only(
            fits, labels, colors, z_grid, func=_w,
            xlabel="Redshift $z$", ylabel="$w(z)$",
            zero_line=-1.0, mark_crossings=True,
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def _compare_sn_hubble_diagram(
        self, lk_cls, dataset_name, offset_attr, y_attr,
        dataset_label, y_label, other_fits, labels,
        n_draws, seed, save_path,
    ):
        """
        Shared implementation behind :meth:`compare_hubble_diagram`
        (Pantheon+) and :meth:`compare_des_hubble_diagram`
        (DES-SN5YR). Each fit's own analytic magnitude/zero-point
        offset (if that fit includes this dataset and marginalizes
        over it) is used for its own curve -- offsets aren't shared
        across fits.
        """

        anchor_lk = self._require_likelihood(lk_cls, dataset_name)
        y_data = getattr(anchor_lk.data, y_attr)
        z_grid = np.linspace(1e-3, anchor_lk.data.z_hd.max() * 1.02, 300)

        fits, labels, colors = self._resolve_comparison(other_fits, labels)

        def _mu(fit, c):

            lk = fit.plots._find_likelihood(lk_cls)

            if lk is not None and getattr(lk, offset_attr, False):
                offset = lk.best_fit_offset()
            else:
                offset = 0.0

            return c.distance.mu(z_grid) + offset

        return self._compare_data_curve(
            fits, labels, colors, z_grid, func=_mu,
            data_z=anchor_lk.data.z_hd, data_y=y_data,
            data_yerr=anchor_lk.covariance.sigma,
            data_label=f"{dataset_label} ({anchor_lk.n_data})",
            xlabel="Redshift $z$", ylabel=y_label, log_x=True,
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def compare_hubble_diagram(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        Pantheon+ Hubble diagram (see :meth:`hubble_diagram`) with
        this fit's curve and one or more other models' curves over
        the same supernova data -- the standard model-vs-model
        distance-modulus comparison figure.
        """

        return self._compare_sn_hubble_diagram(
            PantheonLikelihood, "pantheon", "marginalize_MB", "m_b_corr",
            "Pantheon+", r"$\mu = m_B - M_B$",
            other_fits, labels, n_draws, seed, save_path,
        )

    # ------------------------------------------------------------

    def compare_des_hubble_diagram(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        DES-SN5YR Hubble diagram (see :meth:`des_hubble_diagram`)
        with this fit's curve and one or more other models' curves
        over the same supernova data.
        """

        return self._compare_sn_hubble_diagram(
            DESSN5YRLikelihood, "des_sn5yr", "marginalize_offset", "mu",
            "DES-SN5YR", r"$\mu$",
            other_fits, labels, n_draws, seed, save_path,
        )

    # ------------------------------------------------------------

    def _compare_bao_distances(
        self, lk_cls, dataset_name, dataset_label,
        other_fits, labels, n_draws, seed, save_path,
    ):
        """
        Shared implementation behind :meth:`compare_bao_distances`
        (DESI) and :meth:`compare_sdss_bao_distances` (SDSS): one
        panel per observable type, each with this fit's curve and
        one or more other models' curves over the same tracer data.
        """

        import matplotlib.pyplot as plt

        from CosmoFit.likelihoods.desi import MODEL_MAP

        anchor_lk = self._require_likelihood(lk_cls, dataset_name)
        observables = sorted(set(anchor_lk.data.observable.tolist()))

        z_grid = np.linspace(
            anchor_lk.data.z.min() * 0.9, anchor_lk.data.z.max() * 1.1, 300,
        )

        fits, labels, colors = self._resolve_comparison(other_fits, labels)

        fig, axes = plt.subplots(
            1, len(observables), figsize=(5.0 * len(observables), 5),
            squeeze=False,
        )
        axes = axes[0]

        for ax, observable in zip(axes, observables):

            model_func = MODEL_MAP[observable]
            mask = anchor_lk.data.observable == observable

            ax.errorbar(
                anchor_lk.data.z[mask], anchor_lk.data.value[mask],
                yerr=anchor_lk.covariance.sigma[mask],
                fmt="o", ms=5, elinewidth=0.8, color=COLOR_DATA,
                label=dataset_label,
            )

            for fit, label, color in zip(fits, labels, colors):

                curve, band = fit.plots._predictive_band(
                    lambda c, f=model_func: f(c, z_grid),
                    n_draws=n_draws, seed=seed,
                )

                if band is not None:
                    ax.fill_between(
                        z_grid, band[16], band[84],
                        color=color, alpha=0.2, linewidth=0,
                    )

                ax.plot(z_grid, curve, color=color, lw=1.8, label=label)

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

    def compare_bao_distances(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        DESI BAO distance plot (see :meth:`bao_distances`) with this
        fit's curves and one or more other models' curves over the
        same tracer data.
        """

        return self._compare_bao_distances(
            DESILikelihood, "desi", "DESI 2024",
            other_fits, labels, n_draws, seed, save_path,
        )

    # ------------------------------------------------------------

    def compare_sdss_bao_distances(
        self, other_fits=None, labels=None, save_path=None,
        n_draws=300, seed=0,
    ):
        """
        SDSS BAO distance plot (see :meth:`sdss_bao_distances`) with
        this fit's curves and one or more other models' curves over
        the same tracer data.
        """

        return self._compare_bao_distances(
            SDSSBAOLikelihood, "sdss_bao", "SDSS (DR12+DR16)",
            other_fits, labels, n_draws, seed, save_path,
        )
