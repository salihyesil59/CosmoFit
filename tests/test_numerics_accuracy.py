"""
The two numerical cores, against independent references.

Both were rewritten for speed, and both came out *more* accurate --
which is the part worth pinning, because a future "optimization"
that quietly loses precision would otherwise pass every other test
in the suite. Physical results here are far coarser than these
tolerances; the point is that the numerics are not what limits
them.

  chi(z) = Int dz/E   against scipy.integrate.quad
  growth D, f         against scipy.integrate.solve_ivp at rtol=1e-8

Neither reference is the implementation under test: `quad` is
adaptive Gauss-Kronrod, and `solve_ivp` is an adaptive Runge-Kutta
with dense output -- the two things the rewrites deliberately
stopped using.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.integrate import quad, solve_ivp

from CosmoFit import LCDM, CPL


def lcdm(**kwargs):

    return LCDM(
        LCDM.PARAMS_CLASS(H0=68.0, Omega_m=0.31, rd=147.1,
                          sigma8=0.811, **kwargs),
    )


# ============================================================
# The comoving-distance integral
# ============================================================

@pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_chi_matches_adaptive_quadrature(z):
    """
    The integrator knows ``chi'(z) = 1/E(z)`` exactly, so it uses a
    corrected trapezoid and a Hermite spline rather than a plain
    trapezoid and a shape-preserving one. On the default grid that
    took the maximum relative error from 4.5e-06 to 1.3e-10.

    Bounded here at 1e-8: comfortably inside what the method
    achieves, and comfortably outside what the old one could.
    """

    model = lcdm()

    reference, _ = quad(
        lambda zz: 1.0 / float(np.atleast_1d(model.E(zz))[0]),
        0.0, z, epsabs=1e-13, epsrel=1e-13,
    )

    computed = float(np.atleast_1d(model.integrator.chi(z))[0])

    assert computed == pytest.approx(reference, rel=1e-8)


def test_chi_is_accurate_for_an_evolving_dark_energy_model():
    """
    The same, where E(z) is not a closed-form LCDM -- CPL's dEdz
    is a different expression, and the corrected trapezoid uses it
    directly.
    """

    model = CPL(
        CPL.PARAMS_CLASS(H0=68.0, Omega_m=0.31, w0=-0.9, wa=-0.4,
                         rd=147.1),
    )

    for z in (0.5, 2.0, 5.0):

        reference, _ = quad(
            lambda zz: 1.0 / float(np.atleast_1d(model.E(zz))[0]),
            0.0, z, epsabs=1e-13, epsrel=1e-13,
        )

        assert float(np.atleast_1d(model.integrator.chi(z))[0]) == (
            pytest.approx(reference, rel=1e-8)
        )


def test_chi_derivative_is_exactly_one_over_E():
    """
    The property the Hermite construction is built on. It holds by
    construction at the nodes, so this checks *between* them, where
    a spline that estimated its own slopes would drift.
    """

    model = lcdm()

    z = np.linspace(0.05, 4.5, 97)

    derivative = model.integrator._chi.derivative()(z)

    np.testing.assert_allclose(
        derivative, 1.0 / model.E(z), rtol=1e-6,
    )


# ============================================================
# The growth ODE
# ============================================================

def reference_growth(model, z_values):
    """D(z)/D(0) and f(z) from an adaptive solver."""

    a_init = 1.0e-4

    def rhs(N, y):

        a = np.exp(N)
        zz = 1.0 / a - 1.0

        E = float(np.atleast_1d(model.E(zz))[0])
        dlnH = -(1.0 + zz) * float(np.atleast_1d(model.dEdz(zz))[0]) / E
        omega = model.Omega_m * (1.0 + zz) ** 3 / E ** 2

        return [
            y[1],
            -(2.0 + dlnH) * y[1]
            + 1.5 * omega * model.mu(a, k=0.1) * y[0],
        ]

    solution = solve_ivp(
        rhs, (np.log(a_init), 0.0), [a_init, a_init],
        dense_output=True, method="RK45", rtol=1e-10, atol=1e-12,
    )

    assert solution.success

    N = -np.log1p(np.asarray(z_values, dtype=float))

    y = solution.sol(N)
    y0 = solution.sol(0.0)

    return y[0] / y0[0], y[1] / y[0]


Z_GROWTH = np.array([0.02, 0.1, 0.38, 0.6, 1.0, 1.5, 1.944, 3.0])


def test_growth_matches_an_adaptive_solver():
    """
    Fixed-step RK4 on a precomputed grid, against adaptive RK45
    with dense output.

    The equation is linear, smooth and non-stiff, so adaptivity was
    only ever rediscovering a constant step -- at the cost of
    calling back into Python for every stage, each of which
    evaluated ``E(z)`` three times.
    """

    model = lcdm()

    D_ref, f_ref = reference_growth(model, Z_GROWTH)

    D = model.growth.D(Z_GROWTH)
    f = model.growth.growth_rate(Z_GROWTH)

    np.testing.assert_allclose(D, D_ref, rtol=1e-6)
    np.testing.assert_allclose(f, f_ref, rtol=1e-6)


def test_growth_matches_for_a_modified_gravity_model():
    """
    The same where ``mu(a, k) != 1``, so the source term varies
    along the grid rather than only through ``Omega_m(a)``.
    """

    from CosmoFit import FRHuSawicki

    model = FRHuSawicki(
        FRHuSawicki.PARAMS_CLASS(
            H0=68.0, Omega_m=0.31, rd=147.1, sigma8=0.811,
        ),
    )

    D_ref, f_ref = reference_growth(model, Z_GROWTH)

    np.testing.assert_allclose(
        model.growth.D(Z_GROWTH), D_ref, rtol=1e-5,
    )
    np.testing.assert_allclose(
        model.growth.growth_rate(Z_GROWTH), f_ref, rtol=1e-5,
    )


def test_growth_normalization_is_exact():

    model = lcdm()

    assert float(np.atleast_1d(model.growth.D(0.0))[0]) == (
        pytest.approx(1.0, abs=1e-12)
    )


def test_growth_is_rebuilt_when_parameters_change():
    """
    The solution is cached behind a dirty flag, and the rewrite
    moved what that flag guards. A stale cache would leave every
    growth prediction computed for the previous cosmology.
    """

    model = lcdm()

    before = float(np.atleast_1d(model.growth.D(1.0))[0])

    model.params.Omega_m = 0.40
    model.refresh()

    after = float(np.atleast_1d(model.growth.D(1.0))[0])

    assert after != before

    D_ref, _ = reference_growth(model, np.array([1.0]))

    assert after == pytest.approx(float(D_ref[0]), rel=1e-6)


# ============================================================
# The fast Hermite constructor
# ============================================================

def test_fast_hermite_matches_scipys_constructor():
    """
    ``hermite_spline`` writes the piecewise coefficients directly
    and hands them to ``PPoly.construct_fast``, skipping the
    validation and axis handling ``CubicHermiteSpline`` does. That
    is 43 microseconds to 13 on a 505-point grid, paid on every
    parameter change.

    Skipping the checks is only safe if the coefficients are
    identical, so this asserts exactly that rather than comparing
    evaluations at a few points.
    """

    from scipy.interpolate import CubicHermiteSpline

    from CosmoFit.cosmology.numerics.hermite import hermite_spline

    rng = np.random.default_rng(0)

    for n in (5, 60, 505):

        x = np.sort(rng.uniform(0.0, 10.0, n))
        x = np.linspace(0.0, 10.0, n) if len(np.unique(x)) < n else x

        y = np.cumsum(rng.uniform(0.1, 1.0, n))
        dydx = rng.uniform(-1.0, 1.0, n)

        fast = hermite_spline(x, y, dydx)
        reference = CubicHermiteSpline(x, y, dydx)

        np.testing.assert_allclose(fast.c, reference.c, rtol=1e-13)

        probe = np.linspace(x[0], x[-1], 97)

        np.testing.assert_allclose(fast(probe), reference(probe), rtol=1e-12)

        np.testing.assert_allclose(
            fast.derivative()(probe),
            reference.derivative()(probe),
            rtol=1e-12,
        )


def test_the_distance_table_still_refuses_to_extrapolate():
    """
    ``construct_fast`` defaults to extrapolating, and the distance
    table must not: beyond the redshift it was built for, a
    silently extrapolated cubic is worse than no answer. NaN is how
    that is signalled, and `DistanceIntegrator.extend` is how a
    caller asks for more.
    """

    model = lcdm()

    assert np.isfinite(model.integrator.chi(4.0))

    assert np.isnan(model.integrator.chi(50.0))


# ============================================================
# The optional compiled kernel
# ============================================================

@pytest.fixture
def without_numba(monkeypatch):
    """Force the NumPy stepping path regardless of what is installed."""

    from CosmoFit.cosmology.numerics import kernels

    monkeypatch.setattr(kernels, "HAVE_NUMBA", False)

    return kernels


def growth_at(z):

    model = lcdm()

    return (
        np.asarray(model.growth.D(z), dtype=float).copy(),
        np.asarray(model.growth.growth_rate(z), dtype=float).copy(),
    )


def test_the_two_stepping_paths_agree(without_numba):
    """
    The growth solve has two implementations: a compiled sequential
    loop when numba is installed, and a prefix product of 2x2 step
    matrices when it is not. Whichever one a given install runs, the
    answer has to be the same one.

    They agree to machine precision -- 8.9e-16 measured -- because
    they are the same RK4 with the same coefficients, composed two
    ways.
    """

    from CosmoFit.cosmology.numerics import kernels

    fallback_D, fallback_f = growth_at(Z_GROWTH)

    monkeypatched = kernels.HAVE_NUMBA

    assert monkeypatched is False

    # Restore whatever this install actually has and redo it.
    kernels.HAVE_NUMBA = _installed_numba()

    compiled_D, compiled_f = growth_at(Z_GROWTH)

    kernels.HAVE_NUMBA = monkeypatched

    np.testing.assert_allclose(compiled_D, fallback_D, rtol=1e-13)
    np.testing.assert_allclose(compiled_f, fallback_f, rtol=1e-13)


def _installed_numba():

    try:
        import numba  # noqa: F401
    except ImportError:
        return False

    return True


def test_the_fallback_path_is_correct_on_its_own(without_numba):
    """
    The NumPy path is not a second-class citizen -- it is what a
    plain `pip install cosmofit` runs. Checked against the adaptive
    solver directly, not only against the compiled kernel.
    """

    model = lcdm()

    D_ref, f_ref = reference_growth(model, Z_GROWTH)

    np.testing.assert_allclose(model.growth.D(Z_GROWTH), D_ref, rtol=1e-6)
    np.testing.assert_allclose(
        model.growth.growth_rate(Z_GROWTH), f_ref, rtol=1e-6,
    )


def test_the_reference_loop_agrees_with_both():
    """
    ``kernels._rk4_growth_python`` is the plain Python loop the
    compiled kernel is compiled *from*. It is kept as the readable
    statement of what both fast paths compute, so it has to still
    produce the same numbers.
    """

    from CosmoFit.cosmology.calculators import growth as growth_module
    from CosmoFit.cosmology.numerics import kernels

    model = lcdm()

    calculator = model.growth

    n = growth_module._N_STEPS
    N_init = np.log(growth_module._A_INIT)
    h = -N_init / n

    fine = N_init + 0.5 * h * np.arange(2 * n + 1)

    friction, source = calculator._coefficients(fine)

    args = (
        friction[0:-1:2], source[0:-1:2],
        friction[1::2], source[1::2],
        friction[2::2], source[2::2],
        h, growth_module._A_INIT,
    )

    D_loop, P_loop = kernels._rk4_growth_python(*args)

    D_scan, P_scan = calculator._step_by_prefix_product(
        *args[:6], h, n,
    )

    np.testing.assert_allclose(D_loop, D_scan, rtol=1e-11)
    np.testing.assert_allclose(P_loop, P_scan, rtol=1e-11)


# ============================================================
# Small integer powers
# ============================================================

def test_cube_matches_the_power_operator_to_one_ulp():
    """
    ``x * x * x`` rounds twice where ``x ** 3`` rounds once, so the
    two are allowed to differ in the last bit and nowhere else.
    NumPy does not special-case cubes -- it routes them through the
    general ``pow`` -- which is why the helper exists.
    """

    from CosmoFit.cosmology.numerics.powers import cube

    rng = np.random.default_rng(0)

    for scale in (1.0e-8, 1.0, 1.0e3):

        x = rng.uniform(0.5, 2.0, 4096) * scale

        difference = np.abs(cube(x) / x ** 3 - 1.0)

        assert difference.max() <= 3.0 * np.finfo(float).eps


def test_reciprocal_powers_match_the_negative_exponents():
    """
    The early-universe expansion rate needs 1/a^3 and 1/a^4 over a
    grid spanning eight decades; both were ``pow`` calls.
    """

    from CosmoFit.cosmology.numerics.powers import reciprocal_powers

    x = np.logspace(-11.0, 0.0, 4096)

    inverse, inverse_2, inverse_3, inverse_4 = reciprocal_powers(x)

    for computed, exponent in (
        (inverse, -1), (inverse_2, -2), (inverse_3, -3), (inverse_4, -4),
    ):

        np.testing.assert_allclose(
            computed, x ** float(exponent), rtol=1e-14,
        )


@pytest.mark.parametrize(
    "name",
    ["LCDM", "WCDM", "CPL", "JBP", "BA", "LogarithmicDE", "PEDE",
     "GEDE", "LsCDM", "GCG", "IDE", "RunningVacuum", "Cardassian",
     "DGP", "HDE"],
)
def test_every_model_still_closes_the_friedmann_equation(name):
    """
    The cube substitution touched seventeen files. ``E(0) = 1`` is
    the cheapest statement that every one of those edits landed on
    the right expression -- a cube written where a square belonged
    would break it immediately.
    """

    import CosmoFit

    cls = getattr(CosmoFit, name)

    model = cls(cls.PARAMS_CLASS(H0=68.0, Omega_m=0.31, rd=147.1))

    assert float(np.atleast_1d(model.E(0.0))[0]) == pytest.approx(
        1.0, abs=1e-10,
    )


# ============================================================
# Uniform-grid quadrature
# ============================================================

def test_simpson_uniform_matches_scipy():
    """
    Same rule, written out. The saving is not in the arithmetic --
    it is in not paying for arbitrary spacing that the callers'
    grids never have.
    """

    from scipy.integrate import simpson

    from CosmoFit.cosmology.numerics.quadrature import simpson_uniform

    rng = np.random.default_rng(0)

    for n in (5, 401, 1201):

        y = rng.standard_normal(n)

        assert simpson_uniform(y, 0.01) == pytest.approx(
            simpson(y, dx=0.01), rel=1e-13,
        )


def test_odd_grid_rounds_up_only():
    """
    Composite Simpson needs an even number of intervals. Rounding
    up never costs accuracy; rounding down could.
    """

    from CosmoFit.cosmology.numerics.quadrature import odd_grid

    assert odd_grid(400) == 401
    assert odd_grid(401) == 401
    assert odd_grid(8000) == 8001


def test_the_log_substituted_integrals_did_not_lose_accuracy():
    """
    The three integrals that moved from ``simpson(y, x=grid)`` to a
    uniform Simpson in the log variable. Two of them came out
    *more* accurate, because a uniform-grid rule on the substituted
    integrand is a better rule than a general one on the original:

      r_s(z*)   2.5e-07  ->  2.8e-09
      chi(z*)   4.1e-10  ->  4.1e-10   (unchanged; already uniform)
      r_d       3.2e-09  ->  3.4e-11

    Checked against the same code on a 40001-point grid, so this
    measures the quadrature and nothing else.
    """

    model = LCDM(
        LCDM.PARAMS_CLASS(H0=68.0, Omega_m=0.31, Omega_b=0.0493, m_nu=0.06),
    )

    recombination = model.recombination
    horizon = model.sound_horizon

    z_star = recombination.z_star()
    z_drag = horizon.z_drag()

    cases = (
        (recombination.sound_horizon, z_star, 3.0e-08),
        (recombination.chi_star, z_star, 3.0e-09),
        (horizon.sound_horizon, z_drag, 3.0e-10),
    )

    for integral, z, bound in cases:

        reference = integral(z, n_grid=40001)

        assert abs(integral(z) / reference - 1.0) < bound, (
            f"{integral.__name__}: {abs(integral(z) / reference - 1.0):.2e}"
        )
