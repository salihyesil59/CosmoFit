"""
``Fitter.best_fit()`` and the failure it used to hide.

L-BFGS-B estimates gradients by finite differences with a step of
about 1.5e-8. For the analytic likelihoods -- closed-form ``E(z)``,
interpolated distances -- that is exactly right. For a likelihood
whose own numerical noise exceeds the ``chi2`` change such a step
produces, it is not: the estimated gradient is zero, and the
optimizer **returns the starting point while reporting success**.

That is the worst kind of bug. The caller gets their initial guess
back labelled "best fit", every downstream number is computed from
it, and nothing anywhere says so. On the full Planck CMB it cost 366
in ``chi2``.

The tests here reproduce it deterministically and cheaply, by
quantizing a chi2 that is otherwise perfectly smooth. Quantization is
not what CAMB does, but it has the property that matters -- a small
enough step changes nothing -- and it runs in milliseconds instead of
eight minutes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoFit import LCDM, Fitter


CHEAP = dict(
    model=LCDM,
    datasets=["cc", "desi"],
    free_params=["H0", "Omega_m"],
    initial={"H0": 70.0, "Omega_m": 0.28, "rd": 150.0},
)


def quantize(fitter, step):
    """
    Replace the fitter's chi2 with a quantized version of itself.

    Flat under any parameter change smaller than ``step``, and
    faithful above it -- which is the property that defeats a
    finite-difference gradient.
    """

    original = fitter.logpost.chi2

    def quantized(theta):
        return np.round(original(theta) / step) * step

    fitter.logpost.chi2 = quantized

    return original


# ============================================================
# The normal path is untouched
# ============================================================

def test_smooth_likelihood_converges_without_rescue():
    """
    A rescue that fired on the analytic likelihoods would be a
    regression: a larger step is not uniformly better, and on some
    of them it finds a slightly worse minimum.
    """

    fit = Fitter(**CHEAP)

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        fit.best_fit()

    assert not any("did not move" in str(w.message) for w in caught)

    assert fit.best_fit_chi2 < 50.0

    # It actually moved.
    assert abs(fit.best_fit_params["H0"] - 70.0) > 0.01


# ============================================================
# The failure, and the rescue
# ============================================================

def test_stall_is_detected_and_rescued():
    """
    With a chi2 that cannot respond to a 1.5e-8 step, the first
    attempt returns the starting point -- and the rescue has to
    notice and fix it.
    """

    fit = Fitter(**CHEAP)

    quantize(fit, 0.5)

    start = fit.theta0.copy()

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        fit.best_fit()

    assert any("did not move" in str(w.message) for w in caught), (
        [str(w.message) for w in caught]
    )

    # The rescue found something genuinely better than the start.
    moved = np.abs(fit.best_fit_result.x - start)

    assert moved.max() > 1e-3

    assert fit.best_fit_chi2 < fit.logpost.chi2(start)


def test_without_the_rescue_it_returns_the_starting_point():
    """
    The behaviour being fixed, pinned so the fix cannot quietly
    stop mattering: forcing the gradient method by hand reproduces
    the stall.
    """

    fit = Fitter(**CHEAP)

    quantize(fit, 0.5)

    start = fit.theta0.copy()

    # `eps=` suppresses the rescue -- the caller has taken control.
    result = fit.best_fit(eps=1.5e-8)

    np.testing.assert_allclose(result.x, start)

    # ...and SciPy reports success while doing it, which is why this
    # was invisible.
    assert result.success


def test_rescue_keeps_the_better_of_the_two_results():
    """
    The rescue must never make things worse. If Nelder-Mead somehow
    ends up above the first attempt, the first attempt is kept.
    """

    fit = Fitter(**CHEAP)

    quantize(fit, 0.5)

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        fit.best_fit()

    stalled_chi2 = fit.logpost.chi2(fit.theta0)

    assert fit.best_fit_chi2 <= stalled_chi2


# ============================================================
# The pieces
# ============================================================

def test_stall_detection_is_scale_free():
    """
    The parameters differ by three orders of magnitude, so "did it
    move?" has to be asked in units of each one's own prior width.
    An absolute threshold would call a 1e-4 change in ``Omega_b``
    stationary and a 1e-4 change in ``H0`` significant, when it is
    the other way round.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "Omega_b"],
        initial={"H0": 70.0, "Omega_m": 0.3, "Omega_b": 0.0493},
    )

    start = fit.theta0.copy()
    bounds = list(zip(fit.prior.lower, fit.prior.upper))

    class Result:
        def __init__(self, x):
            self.x = np.asarray(x, dtype=float)

    assert fit._optimizer_stalled(Result(start), start, bounds)

    # A change tiny in absolute terms but large for Omega_b's prior.
    nudged = start.copy()
    nudged[1] += 1e-3

    assert not fit._optimizer_stalled(Result(nudged), start, bounds)

    # A change large in absolute terms but tiny for H0's prior.
    nudged = start.copy()
    nudged[0] += 1e-6

    assert fit._optimizer_stalled(Result(nudged), start, bounds)


def test_explicit_method_disables_the_rescue():
    """
    Passing ``method=`` means the caller has chosen; second-guessing
    them would be surprising.
    """

    fit = Fitter(**CHEAP)

    quantize(fit, 0.5)

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        fit.best_fit(method="Nelder-Mead")

    assert not any("did not move" in str(w.message) for w in caught)


def test_prior_width_matches_the_bounds():

    fit = Fitter(**CHEAP)

    expected = (
        np.asarray(fit.prior.upper) - np.asarray(fit.prior.lower)
    )

    np.testing.assert_allclose(fit._prior_width(), expected)

    assert np.all(fit._prior_width() > 0)


# ============================================================
# Local minima, which the rescue cannot help with
# ============================================================

def double_well(fitter, tilt=1.5):
    """
    Replace the chi2 with a synthetic surface that has two minima.

    Deliberately not a deformation of a real likelihood: CC+DESI's
    chi2 rises by 750 between H0 = 68 and 78, so any bump big enough
    to compete is bigger than the physics. What is under test here is
    the optimizer machinery, so the surface is written down directly
    and its two basins are exact.

    In ``u = (H0 - 70) / 4`` and ``v = (Omega_m - 0.3) / 0.05``:

        chi2 = 10 (u^2 - 1)^2 + tilt * u + v^2

    Minima sit near ``u = -1`` (H0 = 66) and ``u = +1`` (H0 = 74),
    and ``tilt`` makes the second one worse -- so a run started in it
    converges cleanly, reports success, and is wrong.
    """

    i = fitter.free_params.index("H0")
    j = fitter.free_params.index("Omega_m")

    def surface(theta):

        u = (theta[i] - 70.0) / 4.0
        v = (theta[j] - 0.3) / 0.05

        return 10.0 * (u ** 2 - 1.0) ** 2 + tilt * u + v ** 2

    fitter.logpost.chi2 = surface

    return surface


def test_a_single_start_stays_in_the_basin_it_began_in():
    """
    The failure `restarts` exists for, shown before it is fixed: the
    optimizer converges, reports success, and returns the worse of
    the two minima.
    """

    fit = Fitter(**{**CHEAP, "initial": {**CHEAP["initial"], "H0": 74.0}})

    surface = double_well(fit)

    result = fit.best_fit()

    assert result.success

    # It went to the shallow basin near H0 = 74, not the deep one.
    assert fit.best_fit_params["H0"] > 71.0

    better = surface(np.array([66.0, 0.3]))

    assert better < fit.best_fit_chi2 - 1.0


def test_restarts_escape_a_local_minimum():
    """
    Same surface, same starting point, with restarts.
    """

    fit = Fitter(**{**CHEAP, "initial": {**CHEAP["initial"], "H0": 74.0}})

    double_well(fit)

    fit.best_fit(restarts=20, seed=0)

    assert fit.best_fit_params["H0"] < 68.0

    assert fit.best_fit_chi2 < 0.0


def test_restarts_are_reproducible():
    """
    A multi-start fit that gave a different answer each run would
    make every downstream number unreproducible.
    """

    results = []

    for _ in range(2):

        fit = Fitter(**CHEAP)

        fit.best_fit(restarts=5, seed=99)

        results.append(fit.best_fit_chi2)

    assert results[0] == results[1]


def test_restarts_never_make_the_result_worse():
    """
    A restart is kept only if it lands lower, so adding them cannot
    degrade an answer that was already right.
    """

    single = Fitter(**CHEAP)
    single.best_fit()

    multi = Fitter(**CHEAP)
    multi.best_fit(restarts=8, seed=3)

    assert multi.best_fit_chi2 <= single.best_fit_chi2 + 1e-9


def test_restarts_default_to_off():
    """
    Off by default: on a single-minimum posterior they only cost
    time, and every existing call must keep its behaviour.
    """

    plain = Fitter(**CHEAP)
    plain.best_fit()

    explicit = Fitter(**CHEAP)
    explicit.best_fit(restarts=0)

    np.testing.assert_allclose(plain.best_fit_result.x, explicit.best_fit_result.x)


def test_restarts_stay_inside_the_prior():
    """
    Starting points are drawn from the prior, so a restart can
    never begin somewhere the prior forbids.
    """

    fit = Fitter(**CHEAP)

    fit.best_fit(restarts=15, seed=11)

    lower = np.asarray(fit.prior.lower)
    upper = np.asarray(fit.prior.upper)

    assert np.all(fit.best_fit_result.x >= lower)
    assert np.all(fit.best_fit_result.x <= upper)


# ============================================================
# Non-finite parameters must never reach the cosmology
# ============================================================

def test_a_nan_parameter_vector_is_rejected_not_evaluated():
    """
    The crash that `restarts` uncovered, at its root.

    L-BFGS-B started where chi2 is already ``inf`` estimates a
    gradient of ``inf - inf = nan``, takes a nan search direction,
    and evaluates the objective at ``[nan, nan, nan]``. Writing that
    into the parameters builds an interpolation table full of nan,
    and SciPy's interpolator *raises* rather than returning
    anything -- from inside ``refresh()``, which used to sit outside
    the guard that was supposed to catch exactly this.
    """

    fit = Fitter(**CHEAP)

    nan_theta = np.full(len(fit.free_params), np.nan)

    assert fit.logpost.chi2(nan_theta) == np.inf

    assert fit.logpost.log_likelihood(nan_theta) == -np.inf


def test_an_optimizer_started_in_an_excluded_region_does_not_crash():
    """
    The whole point is that a *proposal* cannot break the library.
    Returning ``inf`` is a fine answer; raising is not.
    """

    from scipy.optimize import minimize

    fit = Fitter(
        model=LCDM,
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "Omega_b"],
        initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493},
        compute_rd=True,
    )

    lower = np.asarray(fit.prior.lower)
    upper = np.asarray(fit.prior.upper)

    # A corner of the prior, well away from anything physical.
    corner = lower + 0.98 * (upper - lower)

    result = minimize(
        fit.logpost.chi2,
        corner,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
    )

    assert np.isfinite(result.fun) or result.fun == np.inf


def test_restarts_only_start_where_the_likelihood_is_finite():
    """
    Prior-bounded is not the same as allowed. Starting an optimizer
    where chi2 is infinite buys nothing -- the objective is flat at
    infinity, so there is no direction to move in.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "Omega_b"],
        initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493},
        compute_rd=True,
    )

    rng = np.random.default_rng(0)

    points = fit._restart_points(rng, wanted=6)

    assert points

    for start in points:

        assert np.isfinite(fit.logpost.chi2(start))


def test_restart_draws_give_up_rather_than_spin():
    """
    A prior that is almost entirely excluded must not loop forever.
    Falling short is fine -- the first attempt's result stands.
    """

    fit = Fitter(**CHEAP)

    fit.logpost.chi2 = lambda theta: np.inf

    points = fit._restart_points(np.random.default_rng(0), wanted=4)

    assert points == []


def test_best_fit_leaves_the_cosmology_at_the_best_fit():
    """
    Reading a per-likelihood chi2 after the fit must describe the
    best fit, not whatever the optimizer evaluated last.

    Silent when wrong, and worse with `restarts`, where the last
    evaluation can be a random draw from the prior.
    """

    fit = Fitter(**CHEAP)

    fit.best_fit(restarts=6, seed=5)

    total = sum(lk.chi2() for lk in fit.likelihoods)

    assert total == pytest.approx(fit.best_fit_chi2, rel=1e-9)

    for name, value in fit.best_fit_params.items():

        assert getattr(fit.cosmology.params, name) == pytest.approx(value)
