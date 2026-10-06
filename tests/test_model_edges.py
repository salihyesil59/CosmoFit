"""
Three closed forms checked against the equations they solve.

* IDE at the resonance ``w0 + xi = 0``, where its closed form set the
  transfer term to zero (claiming continuity) and the true limit is
  ``-3 xi Omega_de0 (1+z)^3 ln(1+z)``; and next to it, where a huge
  ``C`` times a tiny bracket lost digits.
* Running vacuum with curvature, whose curvature term is
  ``Omega_k (1+z)^2 / (1 - 3 nu)`` and which feeds matter a
  curvature-like piece as well -- the closed form carried plain
  ``Omega_k (1+z)^2``.
* DGP with ``Omega_m + Omega_k >= 1``, where squaring away a sign
  produced ``E(0) != 1`` with no error.

Each is compared with a direct numerical integration of the model's
own continuity equations, not with a rearrangement of the same
formula.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cosmofit import DGP, IDE
from cosmofit.cosmology.models.rvm import RunningVacuum


def _build(model, **values):

    params = model.PARAMS_CLASS
    full = dict(params.defaults())
    full.update(H0=70.0, Omega_m=0.3)
    full.update(values)

    return model(params(**full))


# ============================================================
# IDE
# ============================================================

def _ide_by_integration(Omega_m, Omega_de0, Omega_k, w0, xi, z):
    """
    rho_de' = -3(1 + w0 + xi) rho_de,  rho_m' = -3 rho_m + 3 xi rho_de,
    integrated from today in N = ln a.
    """

    def rhs(N, y):
        rho_m, rho_de = y
        return [-3.0 * rho_m + 3.0 * xi * rho_de,
                -3.0 * (1.0 + w0 + xi) * rho_de]

    N_end = -np.log1p(z)
    sol = solve_ivp(rhs, [0.0, N_end], [Omega_m, Omega_de0],
                    rtol=1e-12, atol=1e-14)

    rho_m, rho_de = sol.y[:, -1]

    return np.sqrt(rho_m + rho_de + Omega_k * (1.0 + z) ** 2), rho_m


@pytest.mark.parametrize("w0", [-0.9, -0.1, -0.1 + 1e-7, -0.1 - 1e-7, -0.6])
def test_ide_matches_its_continuity_equations(w0):
    """
    xi = 0.1, so w0 = -0.1 is exactly the resonance and its two
    neighbours sit 1e-7 away from it.
    """

    xi = 0.1

    model = _build(IDE, w0=w0, xi=xi)

    for z in (0.5, 1.0, 3.0):

        E_ref, rho_m = _ide_by_integration(
            0.3, model.Omega_de0, 0.0, w0, xi, z,
        )

        assert float(model.E(z)) == pytest.approx(E_ref, rel=1e-9)
        assert float(model.Omega_matter(z)) == pytest.approx(rho_m, rel=1e-9)


def test_ide_dEdz_through_the_resonance():

    model = _build(IDE, w0=-0.1, xi=0.1)

    z = np.array([0.3, 1.0, 2.5])
    h = 1e-6

    numerical = (model.E(z + h) - model.E(z - h)) / (2 * h)

    np.testing.assert_allclose(model.dEdz(z), numerical, rtol=1e-7)


# ============================================================
# Running vacuum with curvature
# ============================================================

def _rvm_by_integration(Omega_m, Omega_k, nu, z):
    """
    Lambda = c + nu E^2 with E^2 = Omega_matter + Lambda + Omega_k a^-2,
    and d(Omega_matter)/dN = -3 Omega_matter - d(Lambda)/dN. Eliminating
    Lambda gives d(Omega_matter)/dN = -3(1-nu) Omega_matter
    + 2 nu Omega_k a^-2, and E^2 = (Omega_matter + c + Omega_k a^-2)
    / (1 - nu) with c fixed by E(0) = 1.
    """

    c = 1.0 - nu - Omega_m - Omega_k

    def rhs(N, y):
        return [-3.0 * (1.0 - nu) * y[0] + 2.0 * nu * Omega_k * np.exp(-2.0 * N)]

    N_end = -np.log1p(z)
    sol = solve_ivp(rhs, [0.0, N_end], [Omega_m], rtol=1e-12, atol=1e-14)

    matter = sol.y[0, -1]
    E2 = (matter + c + Omega_k * (1.0 + z) ** 2) / (1.0 - nu)

    return np.sqrt(E2), matter


@pytest.mark.parametrize("Omega_k", [0.0, -0.05, 0.08])
@pytest.mark.parametrize("nu", [0.0, 3e-3, -0.02])
def test_running_vacuum_matches_its_defining_equations(Omega_k, nu):

    model = _build(RunningVacuum, nu=nu, Omega_k=Omega_k)

    assert float(model.E(0.0)) == pytest.approx(1.0, abs=1e-12)

    for z in (0.5, 1.5, 4.0):

        E_ref, matter = _rvm_by_integration(0.3, Omega_k, nu, z)

        assert float(model.E(z)) == pytest.approx(E_ref, rel=1e-9)
        assert float(model.Omega_matter(z)) == pytest.approx(matter, rel=1e-9)


def test_running_vacuum_exchange_with_curvature():
    """
    ``matter_exchange`` must be what the matter continuity equation
    says: (dOmega_m/dN + 3 Omega_m) / Omega_m.
    """

    model = _build(RunningVacuum, nu=0.01, Omega_k=0.05)

    z = 1.2
    N = -np.log1p(z)
    h = 1e-6

    def matter(N):
        return float(model.Omega_matter(np.expm1(-N)))

    slope = (matter(N + h) - matter(N - h)) / (2 * h)

    expected = (slope + 3.0 * matter(N)) / matter(N)

    assert float(model.matter_exchange(z)) == pytest.approx(expected, rel=1e-6)


# ============================================================
# DGP
# ============================================================

def test_dgp_refuses_a_closed_matter_budget():

    with pytest.raises(ValueError, match="self-accelerating"):
        _build(DGP, Omega_m=0.9, Omega_k=0.2)


def test_dgp_still_closes_where_it_is_defined():

    model = _build(DGP, Omega_m=0.3, Omega_k=0.05)

    assert float(model.E(0.0)) == pytest.approx(1.0, abs=1e-12)
