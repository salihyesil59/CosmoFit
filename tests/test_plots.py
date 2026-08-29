"""
Every figure the library can draw, drawn.

`plots/plotter.py` is the largest module here -- 710 statements --
and until this file 16% of it had ever run. That is the worst place
in the library for a gap, because a plot fails *quietly*: an empty
axis, a curve drawn at the wrong scale, a band that silently
collapsed to a line. Nothing raises. The figure is saved, put in a
paper, and looked at.

So these are not only smoke tests. Each figure is asserted to be
*populated* -- axis labels set, data actually on the axes, a legend
where the method builds one -- because "it returned a Figure" is the
one thing a broken plot is still very good at.

The chains are deliberately tiny (10 walkers, 120 steps). Nothing
here is about inference; what matters is that a posterior exists, so
the predictive bands take their `sampler is not None` branch. The
whole module runs in a few seconds.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot anywhere)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from CosmoFit import CPL, LCDM, Fitter  # noqa: E402


INITIAL = {
    "H0": 70.0,
    "Omega_m": 0.3,
    "rd": 147.1,
    "Omega_b_h2": 0.0224,
    "sigma8": 0.81,
    "MB": -19.3,
}

NWALKERS = 10
NSTEPS = 120
BURNIN = 20


def _fit(model, datasets, free_params, mcmc=True):

    fitter = Fitter(
        model=model,
        datasets=datasets,
        free_params=free_params,
        initial=dict(INITIAL),
    )

    if mcmc:
        fitter.run_mcmc(
            nwalkers=NWALKERS, nsteps=NSTEPS, burnin=BURNIN, progress=False,
        )

    return fitter


# ============================================================
# Fixtures
# ============================================================
#
# Module-scoped: each one loads its datasets and samples a chain, and
# there is no reason to pay that per test.


@pytest.fixture(scope="module")
def broad_fit():
    """
    LCDM against as many figure-bearing datasets as may legally be
    combined.

    Not every dataset with a plot of its own is here, and that is the
    library's own rule rather than an oversight: the three supernova
    compilations overlap each other, and DESI overlaps SDSS. Putting
    them all in one fitter would work and would emit four warnings,
    which is a poor thing for the test suite to model.
    """

    return _fit(
        LCDM,
        ["cc", "desi", "pantheon", "planck", "bao_lowz"],
        ["H0", "Omega_m"],
    )


@pytest.fixture(scope="module")
def sdss_fit():
    """SDSS BAO, which cannot share a fitter with DESI."""

    return _fit(LCDM, ["sdss_bao"], ["H0", "Omega_m"], mcmc=False)


@pytest.fixture(scope="module")
def des_fit():
    """DES-SN5YR, which cannot share a fitter with Pantheon+."""

    return _fit(LCDM, ["des_sn5yr"], ["H0", "Omega_m"], mcmc=False)


@pytest.fixture(scope="module")
def union3_fit():
    """Union3, which cannot share a fitter with either of the others."""

    return _fit(LCDM, ["union3"], ["H0", "Omega_m"], mcmc=False)


@pytest.fixture(scope="module")
def cpl_fit():
    """CPL, for the figures that need a model with a w(z)."""

    return _fit(CPL, ["cc", "desi"], ["H0", "Omega_m", "w0", "wa"])


@pytest.fixture(scope="module")
def lcdm_reference_fit():
    """A second fit on the same data, for the compare_* figures."""

    return _fit(LCDM, ["cc", "desi"], ["H0", "Omega_m"])


@pytest.fixture(scope="module")
def growth_fit():

    return _fit(LCDM, ["fsigma8"], ["H0", "Omega_m", "sigma8"])


@pytest.fixture(scope="module")
def eboss_fit():
    """No chain: the tabulated figures draw the released surface."""

    return _fit(LCDM, ["eboss_lya", "eboss_elg"], ["H0", "Omega_m"], mcmc=False)


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Matplotlib keeps every figure alive until something closes it, and
    warns at 20. These tests make more than that.
    """

    yield

    plt.close("all")


# ============================================================
# What "populated" means
# ============================================================


def plotted_points(ax):
    """Total number of points on an axis, across every kind of artist."""

    n = sum(len(line.get_xdata()) for line in ax.lines)

    n += sum(len(collection.get_offsets()) for collection in ax.collections)

    n += len(ax.patches)

    n += len(ax.images)

    return n


def assert_is_a_real_figure(fig, min_points=2):
    """
    The assertion every test below shares.

    Deliberately about content rather than type: a method that
    returned an empty `plt.subplots()` would satisfy any check on the
    return value, and is exactly the failure worth catching.
    """

    assert isinstance(fig, matplotlib.figure.Figure)

    assert fig.axes, "figure has no axes"

    drawn = sum(plotted_points(ax) for ax in fig.axes)

    assert drawn >= min_points, f"figure has only {drawn} plotted points"

    # At least one axis has to say what it is showing. Not every
    # axis in a multi-panel figure carries a label -- a shared x
    # axis puts it on the bottom panel only.
    labelled = any(
        ax.get_xlabel() or ax.get_ylabel() or ax.get_title()
        for ax in fig.axes
    )

    assert labelled, "no axis on this figure is labelled"


# ============================================================
# Data vs. model
# ============================================================

BROAD_FIGURES = [
    "hz",
    "bao_distances",
    "lowz_bao_distances",
    "hubble_diagram",
    "planck_residuals",
    "deceleration",
]


@pytest.mark.parametrize("method", BROAD_FIGURES)
def test_data_versus_model_figures(broad_fit, method):

    assert_is_a_real_figure(getattr(broad_fit.plots, method)())


def test_sdss_bao_distances(sdss_fit):

    assert_is_a_real_figure(sdss_fit.plots.sdss_bao_distances())


def test_des_hubble_diagram(des_fit):

    assert_is_a_real_figure(des_fit.plots.des_hubble_diagram())


def test_union3_hubble_diagram(union3_fit):

    assert_is_a_real_figure(union3_fit.plots.union3_hubble_diagram())


def test_growth_figure(growth_fit):

    assert_is_a_real_figure(growth_fit.plots.growth())


@pytest.mark.parametrize("dataset", ["eboss_lya", "eboss_elg"])
def test_eboss_surface(eboss_fit, dataset):
    """
    The two released likelihood *surfaces*: a 2-D contour set and a
    1-D curve, which go down different branches of `eboss_surface`.
    """

    assert_is_a_real_figure(eboss_fit.plots.eboss_surface(dataset=dataset))


# ============================================================
# Posterior figures
# ============================================================


def test_chain_has_one_panel_per_free_parameter(broad_fit):

    fig = broad_fit.plots.chain()

    assert_is_a_real_figure(fig)

    assert len(fig.axes) == broad_fit.ndim

    # A trace plot is only useful if the walkers are on it.
    for ax in fig.axes:
        assert len(ax.lines) == NWALKERS

    assert fig.axes[-1].get_xlabel() == "MCMC step"


def test_corner_is_square_in_the_number_of_parameters(cpl_fit):

    fig = cpl_fit.plots.corner()

    assert isinstance(fig, matplotlib.figure.Figure)

    assert len(fig.axes) == cpl_fit.ndim ** 2


def test_chain_before_run_mcmc_says_so():

    fitter = _fit(LCDM, ["cc"], ["H0", "Omega_m"], mcmc=False)

    with pytest.raises(RuntimeError, match="run_mcmc"):
        fitter.plots.chain()


# ============================================================
# Dark-energy figures
# ============================================================


def test_w_of_z(cpl_fit):

    assert_is_a_real_figure(cpl_fit.plots.w_of_z())


def test_w0_wa_plane(cpl_fit):

    assert_is_a_real_figure(cpl_fit.plots.w0_wa_plane())


def test_w_of_z_is_refused_for_a_cosmological_constant(broad_fit):
    """
    LCDM has no w(z) to draw -- it is -1 by construction. Drawing a
    flat line at -1 and calling it a result would be worse than
    refusing.
    """

    with pytest.raises(NotImplementedError, match="w"):
        broad_fit.plots.w_of_z()


# ============================================================
# Model against model
# ============================================================

COMPARISON_FIGURES = [
    "compare_hz",
    "compare_deceleration",
]


@pytest.mark.parametrize("method", COMPARISON_FIGURES)
def test_comparison_figures(cpl_fit, lcdm_reference_fit, method):

    fig = getattr(cpl_fit.plots, method)(
        other_fits=[lcdm_reference_fit], labels=["CPL", "LCDM"],
    )

    assert_is_a_real_figure(fig)

    # Two models means at least two curves, whatever else is on the
    # axes -- a comparison that drew one of them would look fine.
    assert len(fig.axes[0].lines) >= 2


def test_compare_w_of_z(cpl_fit):

    fig = cpl_fit.plots.compare_w_of_z(other_fits=[cpl_fit], labels=["a", "b"])

    assert_is_a_real_figure(fig)


def test_compare_w0_wa_plane(cpl_fit):

    fig = cpl_fit.plots.compare_w0_wa_plane(
        other_fits=[cpl_fit], labels=["a", "b"],
    )

    assert_is_a_real_figure(fig)


def test_compare_growth(growth_fit):

    fig = growth_fit.plots.compare_growth(other_fits=[growth_fit])

    assert_is_a_real_figure(fig)


# ============================================================
# Asking for a figure whose dataset is not in the fit
# ============================================================


@pytest.mark.parametrize(
    "method, dataset",
    [
        ("hz", "cc"),
        ("bao_distances", "desi"),
        ("hubble_diagram", "pantheon"),
        ("growth", "fsigma8"),
        ("planck_residuals", "planck"),
    ],
)
def test_a_missing_dataset_names_itself(method, dataset):
    """
    The failure mode this guards is an `AttributeError` on `None`
    from somewhere deep in the plotting code, which tells the caller
    nothing about what they actually did wrong.
    """

    fitter = _fit(LCDM, ["union3"], ["H0", "Omega_m"], mcmc=False)

    with pytest.raises(ValueError, match=dataset):
        getattr(fitter.plots, method)()


# ============================================================
# save_path
# ============================================================


def test_save_path_writes_the_file(broad_fit, tmp_path):

    target = tmp_path / "hz.png"

    fig = broad_fit.plots.hz(save_path=str(target))

    assert isinstance(fig, matplotlib.figure.Figure)

    assert target.exists()

    assert target.stat().st_size > 1000


# ============================================================
# The band, which is the part that can vanish
# ============================================================


def test_the_posterior_band_appears_only_once_there_is_a_chain():
    """
    `_predictive_band` returns `None` for the band when no chain
    exists, and every caller has to cope with that. The figure is
    still drawn -- it just has no shading.
    """

    without = _fit(LCDM, ["cc"], ["H0", "Omega_m"], mcmc=False)

    fig = without.plots.hz()

    assert_is_a_real_figure(fig)

    assert not fig.axes[0].collections or all(
        collection.get_label() != "68% posterior"
        for collection in fig.axes[0].collections
    )

    with_chain = _fit(LCDM, ["cc"], ["H0", "Omega_m"])

    fig = with_chain.plots.hz()

    labels = [c.get_label() for c in fig.axes[0].collections]

    assert "68% posterior" in labels


def test_the_band_leaves_the_cosmology_where_it_found_it(broad_fit):
    """
    Drawing a band evaluates the model at hundreds of posterior
    samples. If it did not restore the reference point afterwards,
    every number read off the fitter after a plot would be from
    whichever random draw happened to be last -- silently.
    """

    before = np.asarray(
        [broad_fit.cosmology.H0, broad_fit.cosmology.Omega_m], dtype=float,
    )

    broad_fit.plots.hz(n_draws=25)

    after = np.asarray(
        [broad_fit.cosmology.H0, broad_fit.cosmology.Omega_m], dtype=float,
    )

    np.testing.assert_allclose(after, before, rtol=0, atol=0)


# ============================================================
# The rest of the compare_* family
# ============================================================


def test_compare_hubble_diagram(broad_fit, lcdm_reference_fit):

    fig = broad_fit.plots.compare_hubble_diagram(
        other_fits=[lcdm_reference_fit], labels=["a", "b"],
    )

    assert_is_a_real_figure(fig)


def test_compare_des_hubble_diagram(des_fit):

    fig = des_fit.plots.compare_des_hubble_diagram(other_fits=[des_fit])

    assert_is_a_real_figure(fig)


def test_compare_bao_distances(broad_fit, lcdm_reference_fit):

    fig = broad_fit.plots.compare_bao_distances(
        other_fits=[lcdm_reference_fit], labels=["a", "b"],
    )

    assert_is_a_real_figure(fig)


def test_compare_sdss_bao_distances(sdss_fit):

    fig = sdss_fit.plots.compare_sdss_bao_distances(other_fits=[sdss_fit])

    assert_is_a_real_figure(fig)


def test_comparison_with_no_other_fit_builds_an_lcdm_reference(cpl_fit):
    """
    `other_fits=None` is the documented default, and it is not a
    no-op: the plotter fits LCDM to the same datasets to compare
    against. That is the one branch here that runs an optimizer.
    """

    fig = cpl_fit.plots.compare_hz()

    assert_is_a_real_figure(fig)

    assert len(fig.axes[0].lines) >= 2

    labels = [line.get_label() for line in fig.axes[0].lines]

    assert any("CDM" in str(label) for label in labels), labels


# ============================================================
# The w0-wa region annotations
# ============================================================


def test_w0_wa_plane_can_print_the_region_probabilities(cpl_fit):
    """
    `show_fractions=True` routes through `stats.cpl_diagnostics`,
    which nothing else here exercises. The four fractions are a
    partition, so they have to sum to one however the contours fall.
    """

    from CosmoFit.stats.cpl_diagnostics import region_fractions

    fig = cpl_fit.plots.w0_wa_plane(show_fractions=True)

    assert_is_a_real_figure(fig)

    printed = [
        text.get_text()
        for text in fig.axes[0].texts
        if text.get_text().endswith("%")
    ]

    assert len(printed) == 4, printed

    w0, wa = cpl_fit.plots._w0_wa_samples()

    fractions = region_fractions(w0, wa)

    assert set(fractions) == {"phantom", "quintessence", "quintom-a", "quintom-b"}

    assert sum(fractions.values()) == pytest.approx(1.0)


def test_w0_wa_plane_without_the_region_shading(cpl_fit):

    fig = cpl_fit.plots.w0_wa_plane(annotate_regions=False)

    assert_is_a_real_figure(fig)


# ============================================================
# The CMB figures, which need a Boltzmann code
# ============================================================


def _has_camb():

    try:
        import camb  # noqa: F401

    except ImportError:
        return False

    return True


requires_camb = pytest.mark.skipif(
    not _has_camb(),
    reason="CAMB not installed (optional 'cmb' extra)",
)


@requires_camb
def test_cmb_spectra():
    """
    Three spectra, each with a residual panel under it, so six axes
    -- and the residual panels are the point of the figure, since at
    613 bandpowers the error bars are smaller than the markers.
    """

    fitter = _fit(
        LCDM,
        ["planck_lite"],
        ["H0", "Omega_m"],
        mcmc=False,
    )

    fig = fitter.plots.cmb_spectra()

    assert_is_a_real_figure(fig)

    assert len(fig.axes) == 6


@requires_camb
def test_cmb_lensing():

    fitter = _fit(
        LCDM,
        ["planck_lensing"],
        ["H0", "Omega_m"],
        mcmc=False,
    )

    assert_is_a_real_figure(fitter.plots.cmb_lensing())
