"""
Plotting for a fitted :class:`~stats.fitter.Fitter`.

``FitPlotter`` owns every figure CosmoFit knows how to produce --
MCMC diagnostics (chain, corner), the standard publication-style
cosmology figures for a single fit (Hubble diagram, H(z)
compilation, BAO distance plot, w(z) evolution, deceleration
parameter, the (w0, wa) dark-energy plane), and multi-model
comparison versions of those same
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

#: Colors for the dark-energy regions of the (w0, wa) plane
#: (:meth:`FitPlotter.w0_wa_plane`). The two "always on one side of
#: w = -1" regions are shaded; the two quintom (crossing) regions
#: are left unshaded and carry only a label, which is how this
#: figure is drawn in the literature.
COLOR_PHANTOM = "#d9d9d9"
COLOR_QUINTESSENCE = "#c9edc9"
COLOR_QUINTESSENCE_TEXT = "#1a7a1a"
COLOR_QUINTOM_A = "#1f3fd1"
COLOR_QUINTOM_B = "#e02020"
COLOR_REGION_BOUNDARY = "#7b2d8e"

#: Contour color for a posterior in the (w0, wa) plane, and the
#: per-level fill opacities (outermost first). Reused by
#: `compare_w0_wa_plane` via `COMPARISON_PALETTE` instead.
COLOR_POSTERIOR = "#5b9bd5"
CONTOUR_ALPHAS = (0.30, 0.60)


def _style_axes(ax) -> None:

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def _title_formats(flat: np.ndarray) -> list[str]:
    """
    A number format per parameter for the titles above a corner
    plot's histograms, chosen so each one shows its uncertainty to
    two significant figures.

    ``corner``'s own default is a single ``".2f"`` for every
    parameter -- two decimal places regardless of scale. That works
    for ``H0`` (67.57 +0.75 -0.72) and destroys anything small:
    ``Omega_b`` (0.049 +/- 0.0011) is reported as
    "0.05 +0.00 -0.00", a title with no information in it. Since
    ``corner`` accepts a list of formats, size each one from that
    parameter's own posterior width instead, the way an uncertainty
    is quoted in print.

    Parameters
    ----------
    flat : ndarray, shape (n_samples, ndim)
        The flat posterior samples the plot is built from.

    Returns
    -------
    list of str
        One ``".Nf"`` spec per column of ``flat``.
    """

    formats = []

    for column in flat.T:

        q16, q50, q84 = np.percentile(column, [16, 50, 84])

        error = max(q84 - q50, q50 - q16)

        if error > 0 and np.isfinite(error):
            # e.g. error 0.0011 -> 4 decimals ("0.0011", 2 sig figs);
            # error 0.75 -> 2; error 25 -> 0. Capped so a pathological
            # (near-zero-width) posterior can't ask for 300 decimals.
            decimals = min(int(-np.floor(np.log10(error))) + 1, 8)
        else:
            decimals = 4

        formats.append(f".{max(decimals, 0)}f")

    return formats


def _robust_ylim(
    values, errors, include=(),
    cap_fraction=0.25, cap_multiple=10.0, margin=0.04,
):
    """
    Y-axis limits for a data panel, sized by the *measurements*
    rather than by their largest error bars.

    Matplotlib autoscales an ``errorbar`` to contain every whisker,
    which a handful of enormous uncertainties can hijack: DES-SN5YR
    ships 77 (of 1820) supernovae with ``mu_err`` between 5 and 468
    mag -- entries the survey effectively de-weights rather than
    removes. Those are real data and stay plotted, but letting them
    set the scale squeezes the actual Hubble diagram (35-45 mag)
    into a sliver of a +/-500 mag panel.

    So an error bar may only stretch the view so far. The threshold
    has to satisfy *both* of two conditions, because either alone
    misfires on a real dataset: a fraction of the data's own spread
    (which alone would clip the Gold-2018 fsigma8 points, whose
    largest errors are a sizeable part of a narrow range), and a
    multiple of the dataset's *typical* error (which alone would
    clip Cosmic Chronometers, where a few genuinely uncertain H(z)
    points sit well above the median). Taking the larger of the two
    only flags an error bar that is extreme by both measures.

    On every dataset CosmoFit ships this caps nothing at all --
    CC, DESI, Pantheon+, fsigma8 -- and reproduces matplotlib's own
    limits. Only DES-SN5YR's 81 de-weighted entries trip it.

    Parameters
    ----------
    values, errors : ndarray
        The plotted points and their 1-sigma uncertainties.

    include : iterable of float, optional
        Extra values that must stay in view -- e.g. the ends of the
        model curve, which may extend past the data.

    cap_fraction : float
        How far an error bar may stretch the limits, as a fraction
        of the data's own range.

    cap_multiple : float
        ...and as a multiple of the median error. The effective cap
        is the larger of the two.

    margin : float
        Padding added to each side, as a fraction of the range.

    Returns
    -------
    (low, high) : tuple of float
        The limits.
    oversized : ndarray of bool
        Which points' error bars exceed the cap. The caller is
        expected to say something about these rather than let them
        run off the panel silently.
    cap : float
        The threshold applied, in data units.
    """

    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    spread = float(values.max() - values.min())

    if not spread > 0:
        # Every point at the same value (a single measurement, say):
        # nothing to scale from, so let the errors define the view.
        cap = float(np.max(errors)) if errors.size else 0.0
    else:
        cap = max(
            cap_fraction * spread,
            cap_multiple * float(np.median(errors)),
        )

    effective = np.minimum(errors, cap)

    low = float((values - effective).min())
    high = float((values + effective).max())

    for extra in include:
        low = min(low, float(extra))
        high = max(high, float(extra))

    pad = margin * (high - low)

    return (low - pad, high + pad), errors > cap, cap


def _sn_model_label(symbol: str, marginalized: bool) -> str:
    """
    Legend entry for a supernova model curve, saying where its
    absolute-magnitude/zero-point normalization came from.

    Without this the curve is labelled just "Model" while sitting
    on an axis whose zero point was *fitted out* -- a reader has no
    way to tell whether ``symbol`` was a fitted parameter of the
    cosmology or an analytic nuisance the likelihood integrated
    over. Both are supported (see ``PantheonLikelihood``'s
    ``marginalize_MB``), and they mean different things about the
    figure, so the curve says which one it is.
    """

    if marginalized:
        return f"Model ({symbol} marginalized)"

    return f"Model (fitted {symbol})"


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

    def _param_labels(self) -> list[str]:
        """
        LaTeX labels for this fit's free parameters, in order --
        ``["$H_0$", "$\\Omega_m$", "$w_0$", ...]`` rather than the
        Python identifiers ``["H0", "Omega_m", "w0"]``.

        The labels come from the model's own parameter container
        (:meth:`~cosmology.core.parameters.CosmologyParameters.parameter_set`),
        so a custom model's ``extra_params={"beta": {"label": ...}}``
        is honoured here too, and a parameter with no declared
        label falls back to its name.
        """

        parameters = self.fitter.params_cls.parameter_set()

        return [
            parameters[name].label if name in parameters else name
            for name in self.fitter.free_params
        ]

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
        """
        Shade the 68% posterior-predictive band, and return the
        artist (or None if there is no chain yet) so a caller can
        place it in a legend of its own ordering.
        """

        if band is None:
            return None

        return ax.fill_between(
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
    # (w0, wa) plane helpers
    # ============================================================

    def _w0_wa_samples(self, burnin=None):
        """
        Posterior samples of ``(w0, wa)`` for this fit, with the
        checks the (w0, wa) plane needs: both must actually have
        been sampled, and the model's ``w(z)`` must tend to
        ``w0 + wa`` at high z -- that limit *is* the diagonal
        boundary the plane is divided by, so a model with a
        different asymptote would be classified against a line
        that doesn't mean what the axes say it does.
        """

        fitter = self.fitter

        if fitter.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        missing = [n for n in ("w0", "wa") if n not in fitter.free_params]

        if missing:
            raise ValueError(
                f"The (w0, wa) plane needs both w0 and wa to be free "
                f"parameters, but {missing} "
                f"{'is' if len(missing) == 1 else 'are'} not among "
                f"this fit's ({fitter.free_params}). Add "
                f"{'it' if len(missing) == 1 else 'them'} to "
                f"`free_params=` and re-run the MCMC."
            )

        cosmology = fitter.cosmology

        if not hasattr(cosmology, "w"):
            raise NotImplementedError(
                f"{type(cosmology).__name__} does not define w(z), so "
                f"there is no equation of state to classify. The "
                f"(w0, wa) plane applies to CPL-type models."
            )

        # Does w(z) really tend to w0 + wa at high z? Exactly, for
        # CPL and BA; not for JBP (which returns to w0), nor
        # necessarily for a custom model -- and there the diagonal
        # `wa = -1 - w0` boundary would split the plane along
        # something that isn't the model's own high-z limit.
        #
        # Tested on the functional form at a probe point where wa
        # visibly matters, not at the fitted values: every one of
        # these models has the same asymptote when wa happens to sit
        # near zero, so checking there could pass by accident.
        probe_w0, probe_wa = -0.9, -0.5
        reference = self._reference_theta()

        try:
            cosmology.params.update(w0=probe_w0, wa=probe_wa)
            cosmology.refresh()
            w_infinity = float(np.atleast_1d(cosmology.w(1e6))[0])
        finally:
            # w0 and wa are free here (checked above), so this puts
            # the shared cosmology back exactly as it was.
            fitter.logpost._apply(reference)

        if not np.isclose(w_infinity, probe_w0 + probe_wa, rtol=1e-3, atol=1e-3):
            raise ValueError(
                f"{type(cosmology).__name__}'s w(z) does not tend to "
                f"w0 + wa at high z: at the test point "
                f"(w0, wa) = ({probe_w0}, {probe_wa}) it tends to "
                f"{w_infinity:.4g} instead of "
                f"{probe_w0 + probe_wa:.4g}. The "
                f"phantom/quintessence/quintom regions of this figure "
                f"are bounded by w(z->inf) = -1, i.e. the line "
                f"wa = -1 - w0, which only classifies a model whose "
                f"high-z limit *is* w0 + wa (CPL, BA). Use "
                f"plots.w_of_z() for this model instead -- it shows "
                f"the same physics without assuming that limit."
            )

        samples = fitter.samples_dict(burnin=burnin)

        return samples["w0"], samples["wa"]

    # ------------------------------------------------------------

    @staticmethod
    def _posterior_density(x, y, bins=80, smooth=1.5):
        """
        Smoothed 2D histogram of a posterior, and the density
        thresholds enclosing the requested probability mass -- the
        two ingredients of a credible-region contour.

        A histogram rather than a kernel density estimate on
        purpose: a chain here is 10^5-10^6 samples, and evaluating
        a `gaussian_kde` over a grid that size costs
        (grid points x samples) operations -- minutes, for a
        picture indistinguishable from a binned-and-smoothed one.

        Returns
        -------
        x_centers, y_centers : ndarray
            Bin centers along each axis.
        density : ndarray, shape (len(y_centers), len(x_centers))
            Smoothed counts, oriented for ``contour``/``contourf``
            (rows are y, columns are x).
        """

        from scipy.ndimage import gaussian_filter

        counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

        if smooth:
            counts = gaussian_filter(counts, smooth)

        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])

        return x_centers, y_centers, counts.T

    # ------------------------------------------------------------

    @staticmethod
    def _credible_levels(density, levels=(0.68, 0.95)):
        """
        Density thresholds enclosing ``levels`` of the total
        posterior mass, ascending (outermost contour first as a
        *value*, i.e. lowest density last enclosed).

        These are 2D credible regions -- the smallest area holding
        68%/95% of the samples -- not "1 sigma"/"2 sigma" in the
        1D sense. In two dimensions the familiar 1D numbers enclose
        much less (39.3% and 86.5%), which is why the levels are
        stated as probabilities here rather than in sigma.
        """

        flat = np.sort(density.ravel())[::-1]

        cumulative = np.cumsum(flat)

        if cumulative[-1] <= 0:
            raise ValueError(
                "The posterior samples produced an empty density "
                "map -- nothing to draw contours from."
            )

        cumulative = cumulative / cumulative[-1]

        thresholds = [
            float(flat[min(int(np.searchsorted(cumulative, level)), flat.size - 1)])
            for level in levels
        ]

        # `contourf` needs strictly increasing levels. Two requested
        # probabilities can land on the same threshold when the chain
        # is short (or the bins coarse); nudge them apart rather than
        # letting matplotlib raise.
        thresholds = sorted(thresholds)

        for i in range(1, len(thresholds)):
            if thresholds[i] <= thresholds[i - 1]:
                thresholds[i] = np.nextafter(thresholds[i - 1], np.inf)

        return thresholds

    # ------------------------------------------------------------

    def _plot_credible_regions(
        self, ax, x, y, color, label=None,
        levels=(0.68, 0.95), bins=80, smooth=1.5,
    ):
        """
        Filled + outlined credible regions for one 2D posterior,
        drawn darkest-innermost, plus a legend proxy for ``label``.
        """

        from matplotlib.colors import to_rgba

        x_centers, y_centers, density = self._posterior_density(
            x, y, bins=bins, smooth=smooth,
        )

        thresholds = self._credible_levels(density, levels=levels)

        # One fill per band (outermost first), so the overlapping
        # regions read as increasing opacity toward the peak.
        n_bands = len(thresholds)
        alphas = [
            CONTOUR_ALPHAS[min(i, len(CONTOUR_ALPHAS) - 1)]
            if n_bands <= len(CONTOUR_ALPHAS)
            else 0.2 + 0.5 * i / max(n_bands - 1, 1)
            for i in range(n_bands)
        ]

        ax.contourf(
            x_centers, y_centers, density,
            levels=thresholds + [density.max() * 1.01],
            colors=[to_rgba(color, a) for a in alphas],
        )
        ax.contour(
            x_centers, y_centers, density,
            levels=thresholds,
            colors=[color], linewidths=0.8, alpha=0.9,
        )

        if label:
            # `contourf` produces no legend handle of its own, and a
            # bare `Patch` can't be added to an Axes (it has no path).
            # An empty filled polygon is a real artist with the right
            # face color, so `legend()` finds it like any other.
            ax.fill(
                [np.nan, np.nan], [np.nan, np.nan],
                color=to_rgba(color, alphas[-1]), label=label,
            )

    # ------------------------------------------------------------

    @staticmethod
    def _w0_wa_limits(w0_sets, wa_sets, w0_range=None, wa_range=None):
        """
        Axis limits for the (w0, wa) plane: wide enough for every
        posterior *and* for the LCDM point, since a plane that
        doesn't show (-1, 0) can't show what the fit is being
        compared against.
        """

        if w0_range is not None and wa_range is not None:
            return tuple(w0_range), tuple(wa_range)

        w0_all = np.concatenate([np.asarray(s) for s in w0_sets])
        wa_all = np.concatenate([np.asarray(s) for s in wa_sets])

        def span(values, anchor, pad_fraction=0.22, minimum=0.25):
            # 0.5/99.5 rather than min/max: a handful of stray
            # walker excursions shouldn't set the frame and squash
            # the contours into a corner of it.
            lo = min(float(np.percentile(values, 0.5)), anchor)
            hi = max(float(np.percentile(values, 99.5)), anchor)
            pad = max(pad_fraction * (hi - lo), minimum)
            return lo - pad, hi + pad

        return (
            tuple(w0_range) if w0_range is not None else span(w0_all, -1.0),
            tuple(wa_range) if wa_range is not None else span(wa_all, 0.0),
        )

    # ------------------------------------------------------------

    def _draw_w0_wa_background(self, ax, w0_lim, wa_lim, annotate_regions=True):
        """
        The physics behind the contours: the four dark-energy
        regions of the (w0, wa) plane, their boundaries, and the
        LCDM point.

        The classification is by where w(z) sits relative to -1 at
        the two ends of the model's history -- today (w0) and at
        high z (w0 + wa):

            phantom       w < -1 always
            quintessence  w > -1 always
            quintom-A     w > -1 in the past, w < -1 today
            quintom-B     w < -1 in the past, w > -1 today

        which the two lines w0 = -1 and wa = -1 - w0 cut the plane
        into. Crossing w = -1 at all (either quintom region) is the
        interesting case: no single canonical scalar field can do
        it, so a posterior sitting there points at something more
        than quintessence.
        """

        x0, x1 = w0_lim
        y0, y1 = wa_lim

        # Phantom: w0 < -1 (phantom today) and wa < -1 - w0
        # (still phantom at high z).
        xs = np.linspace(x0, min(-1.0, x1), 200)
        if xs.size > 1:
            upper = np.minimum(y1, -1.0 - xs)
            ax.fill_between(
                xs, y0, upper, where=upper > y0,
                color=COLOR_PHANTOM, linewidth=0, zorder=0,
            )

        # Quintessence: the mirror image -- above -1 at both ends.
        xs = np.linspace(max(-1.0, x0), x1, 200)
        if xs.size > 1:
            lower = np.maximum(y0, -1.0 - xs)
            ax.fill_between(
                xs, lower, y1, where=lower < y1,
                color=COLOR_QUINTESSENCE, linewidth=0, zorder=0,
            )

        ax.axvline(
            -1.0, color=COLOR_REGION_BOUNDARY, lw=1.3, zorder=1,
        )
        ax.plot(
            [x0, x1], [-1.0 - x0, -1.0 - x1],
            color=COLOR_REGION_BOUNDARY, lw=1.3, zorder=1,
        )

        # LCDM: w = -1 at every redshift, i.e. the one point both
        # boundaries pass through.
        ax.plot(
            -1.0, 0.0, marker="*", ms=13, color="black",
            markeredgewidth=0, zorder=6, clip_on=False,
        )
        ax.annotate(
            r"$\Lambda$CDM", (-1.0, 0.0),
            textcoords="offset points", xytext=(9, 9),
            fontsize=11, zorder=6,
        )

        # Label the diagonal *on* the diagonal (just below it),
        # three-quarters of the way across, where it can't be
        # confused with the vertical w0 = -1 boundary.
        x_label = x0 + 0.75 * (x1 - x0)
        ax.annotate(
            r"$w_a + w_0 = -1$", (x_label, -1.0 - x_label),
            textcoords="offset points", xytext=(0, -14),
            ha="center", va="top", fontsize=10,
            color=COLOR_REGION_BOUNDARY, zorder=6,
        )

        if not annotate_regions:
            return

        # Anchor each label in its own corner -- but only if that
        # corner really is in that region. Zoom in far enough (or
        # pass explicit ranges) and a corner can fall on the wrong
        # side of a boundary, where the label would be a plain
        # mislabelling of the plot.
        from CosmoFit.stats.cpl_diagnostics import classify_region

        pad_x = 0.03 * (x1 - x0)
        pad_y = 0.045 * (y1 - y0)

        regions = [
            ("Phantom", "#404040",
             (x0 + pad_x, y0 + pad_y), "left", "bottom"),
            ("Quintessence", COLOR_QUINTESSENCE_TEXT,
             (x1 - pad_x, y1 - pad_y), "right", "top"),
            ("Quintom-A", COLOR_QUINTOM_A,
             (x0 + pad_x, y1 - pad_y), "left", "top"),
            # Not in the corner: the legend lives there. A quarter
            # of the way up the panel is still unambiguously inside
            # this region for any frame wide enough to show it.
            ("Quintom-B", COLOR_QUINTOM_B,
             (x1 - pad_x, y0 + 0.25 * (y1 - y0)), "right", "center"),
        ]

        for name, color, (px, py), ha, va in regions:

            if classify_region(px, py) != name.lower():
                continue

            ax.text(
                px, py, name, color=color, fontsize=11, weight="bold",
                ha=ha, va=va, zorder=6,
            )

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

        for i, label in enumerate(self._param_labels()):
            axes[i].plot(chain[:, :, i], alpha=0.15, color=COLOR_MODEL)
            axes[i].set_ylabel(label)
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
            labels=self._param_labels(),
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=_title_formats(flat),
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
        residual_label,
        model_label,
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

        z = lk.data.z_hd
        sigma = lk.covariance.sigma
        residual = y_data - y_model

        # Scale the panel to the measurements, not to the largest
        # error bars -- see `_robust_ylim`. The model curve is kept
        # in view: it extends below the data at low z, where the
        # survey has no supernovae.
        ylim, oversized, cap = _robust_ylim(
            y_data, sigma, include=(curve.min(), curve.max()),
        )

        # A whisker taller than the panel is drawn as a full-height
        # line, and 81 of them (DES-SN5YR's de-weighted entries)
        # turn the figure into a picket fence. Those points stay --
        # dropping data to tidy a plot is not on -- but without the
        # bars that no longer carry readable information, called out
        # in the legend so nobody mistakes them for well-measured
        # supernovae.
        shown = ~oversized

        handles = []

        for ax, values in ((ax_top, y_data), (ax_bot, residual)):

            drawn = ax.errorbar(
                z[shown], values[shown], yerr=sigma[shown],
                fmt="o", ms=2.5, elinewidth=0.6, alpha=0.5,
                color=COLOR_DATA,
                label=f"{dataset_label} ({lk.n_data})",
            )

            if ax is ax_top:
                handles.append(drawn)

            if oversized.any():

                flagged, = ax.plot(
                    z[oversized], values[oversized],
                    "o", ms=3.5, alpha=0.55, color=COLOR_DATA,
                    markerfacecolor="none", linestyle="none",
                    label=(
                        f"{int(oversized.sum())} with "
                        rf"$\sigma > {cap:.1f}$ mag (bars omitted)"
                    ),
                )

                if ax is ax_top:
                    handles.append(flagged)

        band_artist = self._plot_band(ax_top, z_grid, band)

        if band_artist is not None:
            handles.append(band_artist)

        model_curve, = ax_top.plot(
            z_grid, curve, color=COLOR_MODEL, lw=1.8, label=model_label,
        )
        handles.append(model_curve)

        ax_top.set_xscale("log")
        ax_top.set_ylabel(y_label)
        ax_top.set_ylim(*ylim)
        # Explicit handles: matplotlib orders a legend by artist
        # type before creation order, which buries the data the
        # figure is about under the annotations about it.
        ax_top.legend(handles=handles, frameon=False)
        _style_axes(ax_top)

        ax_bot.axhline(0.0, color=COLOR_REFERENCE, lw=1.0, ls="--")
        ax_bot.set_ylabel(residual_label)
        ax_bot.set_xlabel("Redshift $z$")
        ax_bot.set_ylim(*_robust_ylim(residual, sigma, include=(0.0,))[0])
        _style_axes(ax_bot)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    def hubble_diagram(self, save_path=None, n_draws=300, seed=0):
        """
        Pantheon+ Hubble diagram: corrected apparent magnitude vs.
        redshift, with a residuals sub-panel.

        The y axis is the *apparent* magnitude ``m_B``
        (``data.m_b_corr``, ~11-27 mag), not the distance modulus:
        the model curve is ``mu(z)`` plus the absolute-magnitude
        offset, which brings it onto the data's scale. That offset
        is normally the analytically marginalized ``M_B`` (see
        ``PantheonLikelihood``'s ``marginalize_MB``), and the
        legend says so -- with ``M_B`` integrated out, no ``M_B``
        appears on the axis either.
        """

        lk = self._require_likelihood(PantheonLikelihood, "pantheon")

        return self._sn_hubble_diagram(
            lk,
            y_data=lk.data.m_b_corr,
            offset_fn=lambda: lk.best_fit_offset() if lk.marginalize_MB else 0.0,
            dataset_label="Pantheon+",
            y_label=r"$m_B$ [mag]",
            residual_label=r"$\Delta m_B$",
            model_label=_sn_model_label(r"$M_B$", lk.marginalize_MB),
            n_draws=n_draws, seed=seed, save_path=save_path,
        )

    # ------------------------------------------------------------

    def des_hubble_diagram(self, save_path=None, n_draws=300, seed=0):
        """
        DES-SN5YR Hubble diagram: distance modulus vs. redshift,
        with a residuals sub-panel.

        Unlike Pantheon+, DES-SN5YR's data vector (``mu``) is
        already a distance modulus (computed assuming a fiducial
        H0=70), so the y axis really is ``mu`` here; the model
        curve includes the analytically marginalized zero-point
        offset (or ``cosmology.MB`` if it was fit explicitly) so
        both are on the same scale, and the legend says which.

        This release ships 81 supernovae (of 1820) with ``mu_err``
        between 5 and 468 mag -- entries the survey de-weights
        rather than removes. They are plotted like any other point,
        but their error bars are left off and flagged in the
        legend: drawn, they are full-height lines that force the
        panel to +/-500 mag and bury the Hubble diagram itself. See
        :func:`_robust_ylim`.
        """

        lk = self._require_likelihood(DESSN5YRLikelihood, "des_sn5yr")

        return self._sn_hubble_diagram(
            lk,
            y_data=lk.data.mu,
            offset_fn=lambda: lk.best_fit_offset() if lk.marginalize_offset else 0.0,
            dataset_label="DES-SN5YR",
            y_label=r"$\mu$ [mag]",
            residual_label=r"$\Delta\mu$",
            model_label=_sn_model_label(
                "zero-point", lk.marginalize_offset,
            ),
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
            label=f"RSD $f\\sigma_8$ ({lk.n_data})",
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

        from CosmoFit.likelihoods.desi import MODEL_MAP, OBSERVABLE_LABELS

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

            ax.set_title(OBSERVABLE_LABELS.get(observable, observable))
            ax.set_xlabel("Redshift $z$")
            ax.legend(frameon=False)
            _style_axes(ax)

        # Each panel's title carries the full ratio; this states the
        # shared denominator once, in the same symbol the titles and
        # `MODEL_MAP`'s predictions use (r_d, not the data file's r_s
        # spelling for the same quantity).
        axes[0].set_ylabel(r"Distance $/\ r_d$")

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

        Panel titles and the y axis both say ``r_d``, which is what
        the predictions divide by. DESI's data file spells the same
        quantity ``rs`` in its ``observable`` column (and so do
        :data:`~likelihoods.desi.MODEL_MAP`'s keys) -- both mean the
        sound horizon at the drag epoch; see that map's comment.
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

        from CosmoFit.likelihoods.planck import OBSERVABLE_LABELS

        ax.set_xticks(x)
        ax.set_xticklabels([
            OBSERVABLE_LABELS.get(name, name) for name in lk.data.labels
        ])
        ax.set_ylabel(r"$(\mathrm{data} - \mathrm{model}) \,/\, \sigma$")
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

    def w0_wa_plane(
        self,
        burnin=None,
        levels=(0.68, 0.95),
        bins=80,
        smooth=1.5,
        w0_range=None,
        wa_range=None,
        annotate_regions=True,
        show_fractions=False,
        label=None,
        color=COLOR_POSTERIOR,
        save_path=None,
    ):
        """
        The (w0, wa) plane: this fit's 2D posterior contours on
        top of the four dark-energy regions -- phantom,
        quintessence, quintom-A, quintom-B -- with LCDM marked at
        (-1, 0).

        This is the headline figure of every recent
        evolving-dark-energy result (DESI DR2 and the papers
        responding to it): where the contours sit relative to
        (-1, 0) is the whole "is dark energy a cosmological
        constant?" question, and which region they sit *in* says
        what kind of dark energy it would have to be if not.
        The two boundaries are ``w0 = -1`` (the equation of state
        today) and ``wa = -1 - w0`` (its high-z limit
        ``w0 + wa``); see
        :func:`~stats.cpl_diagnostics.classify_region` for what
        each region means physically.

        Needs ``w0`` and ``wa`` to both be free parameters of a
        completed MCMC run, and a model whose ``w(z)`` really does
        tend to ``w0 + wa`` (CPL, BA) -- otherwise the diagonal
        boundary would not be the model's own high-z limit, and
        this raises rather than mislabel the regions.

        Parameters
        ----------
        burnin : int, optional
            Steps to discard. Defaults to the fitter's own.

        levels : tuple of float
            Credible *probabilities* to contour, as fractions of
            the posterior mass (default 68% and 95%). Note these
            are 2D regions: 68% here is not the 1D "1 sigma"
            interval, which in two dimensions encloses only 39%.

        bins, smooth : int, float
            Resolution of the density estimate behind the
            contours, and the Gaussian smoothing (in bins)
            applied to it. Raise ``bins`` for a long chain,
            lower ``smooth`` if the contours look over-rounded.

        w0_range, wa_range : tuple, optional
            Axis limits. Both default to a window sized around
            the posterior that always includes the LCDM point.

        annotate_regions : bool
            Label the four regions in the corners. A label is
            drawn only where its corner genuinely falls in that
            region, so a zoomed-in view silently drops the ones
            that no longer apply rather than mislabelling them.

        show_fractions : bool
            Add each region's posterior probability (from
            :func:`~stats.cpl_diagnostics.region_fractions`) under
            its label -- the same statement the figure makes, as
            a number.

        label : str, optional
            Legend entry for the contours. Defaults to this fit's
            dataset combination, e.g. ``"cc+desi+pantheon"``.

        color : str
            Contour color.

        Returns
        -------
        matplotlib.figure.Figure
        """

        import matplotlib.pyplot as plt

        w0, wa = self._w0_wa_samples(burnin=burnin)

        w0_lim, wa_lim = self._w0_wa_limits(
            [w0], [wa], w0_range=w0_range, wa_range=wa_range,
        )

        if label is None:
            # Deferred, and aliased: `stats.fitter` imports this
            # module, and `dataset_label` is also a parameter name in
            # the data-vs-model methods above.
            from CosmoFit.stats.fitter import dataset_label as _combo

            label = _combo(self.fitter.dataset_names)

        fig, ax = plt.subplots(figsize=(8, 5.5))

        self._draw_w0_wa_background(
            ax, w0_lim, wa_lim, annotate_regions=annotate_regions,
        )
        self._plot_credible_regions(
            ax, w0, wa, color=color, label=label,
            levels=levels, bins=bins, smooth=smooth,
        )

        if show_fractions:
            self._annotate_region_fractions(ax, w0, wa, w0_lim, wa_lim)

        self._finish_w0_wa_axes(ax, w0_lim, wa_lim)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------

    @staticmethod
    def _finish_w0_wa_axes(ax, w0_lim, wa_lim):

        ax.set_xlim(*w0_lim)
        ax.set_ylim(*wa_lim)
        ax.set_xlabel("$w_0$")
        ax.set_ylabel("$w_a$")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

        _style_axes(ax)

        # The region fills already carry the eye; a grid on top of
        # them reads as clutter, so keep it fainter than elsewhere.
        ax.grid(alpha=0.15, linewidth=0.5)

    # ------------------------------------------------------------

    def _annotate_region_fractions(self, ax, w0, wa, w0_lim, wa_lim):
        """
        Posterior probability of each region, printed just under
        that region's corner label.
        """

        from CosmoFit.stats.cpl_diagnostics import region_fractions

        fractions = region_fractions(w0, wa)

        x0, x1 = w0_lim
        y0, y1 = wa_lim
        pad_x = 0.03 * (x1 - x0)
        pad_y = 0.045 * (y1 - y0)
        line = 0.055 * (y1 - y0)

        placements = {
            "phantom": (x0 + pad_x, y0 + pad_y + line, "left", "bottom"),
            "quintessence": (x1 - pad_x, y1 - pad_y - line, "right", "top"),
            "quintom-a": (x0 + pad_x, y1 - pad_y - line, "left", "top"),
            "quintom-b": (x1 - pad_x, y0 + 0.25 * (y1 - y0) - line, "right", "center"),
        }

        for region, (px, py, ha, va) in placements.items():

            ax.text(
                px, py, f"{100 * fractions[region]:.1f}%",
                color="#404040", fontsize=9, ha=ha, va=va, zorder=6,
            )

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
            # `plot_label()`, not `__name__`: a legend should read
            # "$\Lambda$CDM" and "$w$CDM", not the ASCII spelling of
            # the Python class ("LCDM", "WCDM"). See
            # `Cosmology.MODEL_LABEL`.
            labels = [f.model_cls.plot_label() for f in fits]
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

        # Boolean-mask indexing below needs arrays, and callers pass
        # whatever their dataset holds.
        data_z = np.asarray(data_z, dtype=float)
        data_y = np.asarray(data_y, dtype=float)
        data_yerr = np.asarray(data_yerr, dtype=float)

        curves = []

        for fit, label, color in zip(fits, labels, colors):

            curve, band = fit.plots._predictive_band(
                lambda c, f=fit: func(f, c), n_draws=n_draws, seed=seed,
            )
            curves.append(curve)

            if band is not None:
                ax.fill_between(
                    z_grid, band[16], band[84],
                    color=color, alpha=0.2, linewidth=0,
                )

            ax.plot(z_grid, curve, color=color, lw=1.8, label=label)

        # Same treatment as the single-model panels: a few enormous
        # uncertainties must not set the scale, and the points they
        # belong to are kept, flagged, without their unreadable
        # whiskers. A no-op for every dataset whose errors are sane.
        ylim, oversized, cap = _robust_ylim(
            data_y, data_yerr,
            include=[c.min() for c in curves] + [c.max() for c in curves],
        )
        shown = ~oversized

        ax.errorbar(
            data_z[shown], data_y[shown], yerr=data_yerr[shown],
            fmt="o", ms=3.5, elinewidth=0.7, alpha=0.5, color=COLOR_DATA,
            label=data_label,
        )

        if oversized.any():
            ax.plot(
                data_z[oversized], data_y[oversized],
                "o", ms=4.5, alpha=0.55, color=COLOR_DATA,
                markerfacecolor="none", linestyle="none",
                label=f"{int(oversized.sum())} with large errors (bars omitted)",
            )

        ax.set_ylim(*ylim)

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
            data_label=f"RSD $f\\sigma_8$ ({lk.n_data})",
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

    def compare_w0_wa_plane(
        self,
        other_fits=None,
        labels=None,
        burnin=None,
        levels=(0.68, 0.95),
        bins=80,
        smooth=1.5,
        w0_range=None,
        wa_range=None,
        annotate_regions=True,
        save_path=None,
    ):
        """
        The (w0, wa) plane (see :meth:`w0_wa_plane`) with several
        posteriors overlaid -- the "what does adding this dataset
        do to w0-wa?" figure, one contour set per fit.

        Unlike the other ``compare_*`` methods, every fit here
        must have its own MCMC chain with ``w0`` and ``wa`` free:
        this figure *is* the posteriors, so there is no
        best-fit-only curve to fall back on. For that reason
        ``other_fits=None`` does not auto-build an LCDM
        reference either -- LCDM has no (w0, wa) posterior, it is
        the point at (-1, 0) already marked on the plot.

        Typically compared across dataset combinations rather
        than models::

            fit_desi.plots.compare_w0_wa_plane(
                other_fits=[fit_desi_sn, fit_desi_sn_cmb],
                labels=["DESI", "DESI+SN", "DESI+SN+CMB"],
            )

        Parameters
        ----------
        other_fits : Fitter or list[Fitter]
            The other fitted posteriors to overlay.

        labels : list[str], optional
            One per fit (this one first). Defaults to each fit's
            dataset combination rather than its model name, since
            these are usually the same model on different data.

        The remaining parameters are as in :meth:`w0_wa_plane`.
        """

        import matplotlib.pyplot as plt

        if other_fits is None:
            fits = [self.fitter]
        elif isinstance(other_fits, (list, tuple)):
            fits = [self.fitter, *other_fits]
        else:
            fits = [self.fitter, other_fits]

        if labels is None:
            from CosmoFit.stats.fitter import dataset_label as _combo

            labels = [_combo(f.dataset_names) for f in fits]
        elif len(labels) != len(fits):
            raise ValueError(
                f"Got {len(labels)} label(s) for {len(fits)} "
                f"posterior(s) being compared -- labels must match "
                f"one-to-one (this fit first, then `other_fits`)."
            )

        samples = [
            fit.plots._w0_wa_samples(burnin=burnin) for fit in fits
        ]

        w0_lim, wa_lim = self._w0_wa_limits(
            [w0 for w0, _ in samples],
            [wa for _, wa in samples],
            w0_range=w0_range, wa_range=wa_range,
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))

        self._draw_w0_wa_background(
            ax, w0_lim, wa_lim, annotate_regions=annotate_regions,
        )

        # Later fits drawn on top, but every fill is translucent,
        # so an overlap still shows both.
        for i, ((w0, wa), label) in enumerate(zip(samples, labels)):
            self._plot_credible_regions(
                ax, w0, wa,
                color=COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)],
                label=label, levels=levels, bins=bins, smooth=smooth,
            )

        self._finish_w0_wa_axes(ax, w0_lim, wa_lim)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

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
        apparent-magnitude comparison figure. Each model's own
        marginalized ``M_B`` sets its own curve's normalization.
        """

        return self._compare_sn_hubble_diagram(
            PantheonLikelihood, "pantheon", "marginalize_MB", "m_b_corr",
            "Pantheon+", r"$m_B$ [mag]",
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
            "DES-SN5YR", r"$\mu$ [mag]",
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

        from CosmoFit.likelihoods.desi import MODEL_MAP, OBSERVABLE_LABELS

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

            ax.set_title(OBSERVABLE_LABELS.get(observable, observable))
            ax.set_xlabel("Redshift $z$")
            ax.legend(frameon=False)
            _style_axes(ax)

        # Each panel's title carries the full ratio; this states the
        # shared denominator once, in the same symbol the titles and
        # `MODEL_MAP`'s predictions use (r_d, not the data file's r_s
        # spelling for the same quantity).
        axes[0].set_ylabel(r"Distance $/\ r_d$")

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
