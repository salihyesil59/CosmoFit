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
