"""
Growth when matter exchanges energy with the dark sector.

IDE and running vacuum change how matter dilutes, and the growth
solver read that through Omega_m(a) alone. But energy deposited
homogeneously into matter also dilutes the density contrast, and
that enters the perturbation equations twice more:

    delta' + psi delta + theta/(aH) = 0         (continuity)
    theta' + (2 + dlnH/dN) theta + (3/2) Omega_m mu delta = 0   (Euler)

with psi = Q / (H rho_m), geodesic matter and a non-clustering dark
sector. The solver works from the combined second-order equation;
these tests integrate the first-order system above directly, so the
algebra that combined them is what is being checked.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from CosmoFit import IDE, LCDM
from CosmoFit.cosmology.models.rvm import RunningVacuum


def _build(model, **extra):

    params = model.PARAMS_CLASS
    values = dict(params.defaults())
    values.update(H0=70.0, Omega_m=0.3, **extra)

    return model(params(**values))


def _first_order(cosmology):

    def coefficients(N):

        z = np.expm1(-N)
        e = float(cosmology.E(z))
        dlnH = -(1.0 + z) * float(cosmology.dEdz(z)) / e
        Om = float(cosmology.Omega_matter(z)) / e ** 2
        psi = float(cosmology.matter_exchange(z))

        return dlnH, Om, psi

    def rhs(N, y):

        delta, theta = y
        dlnH, Om, psi = coefficients(N)

        return [-psi * delta - theta, -(2.0 + dlnH) * theta - 1.5 * Om * delta]

    a0 = 1.0e-4
    psi0 = coefficients(np.log(a0))[2]

    solution = solve_ivp(
        rhs, [np.log(a0), 0.0], [a0, -a0 - psi0 * a0],
        rtol=1e-10, atol=1e-14, dense_output=True,
    )

    return solution, rhs


@pytest.mark.parametrize("model, extra", [
    (RunningVacuum, {"nu": 0.01}),
    (IDE, {"xi": 0.05, "w0": -0.9}),
])
def test_growth_matches_the_continuity_and_euler_system(model, extra):

    cosmology = _build(model, **extra)

    solution, rhs = _first_order(cosmology)

    D0 = solution.y[0, -1]

    for z in (0.0, 0.5, 1.0, 2.0):

        N = -np.log1p(z)
        delta, theta = solution.sol(N)
        slope = rhs(N, [delta, theta])[0]

        assert float(cosmology.growth.D(z)) == pytest.approx(
            delta / D0, rel=1e-6,
        )
        assert float(cosmology.growth.growth_rate(z)) == pytest.approx(
            slope / delta, rel=1e-6,
        )


def test_running_vacuum_exchange_is_three_nu():

    cosmology = _build(RunningVacuum, nu=2.0e-3)

    np.testing.assert_allclose(
        cosmology.matter_exchange(np.array([0.0, 1.0, 10.0])), 6.0e-3,
    )


def test_conserved_matter_has_no_exchange():

    assert _build(LCDM).matter_exchange(0.5) is None
    assert _build(IDE, xi=0.0).matter_exchange(0.5) == pytest.approx(0.0)
