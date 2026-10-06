"""
Linear growth across a jump in the expansion rate.

LsCDM's E(z) jumps at z_dagger. The growth solver stepped over the
jump on a fixed grid, which keeps dD/dN continuous. What is actually
continuous is H dD/dN -- the velocity d(delta)/dt, which a finite jump
in H cannot change in zero time -- so dD/dN, and with it f, must jump
by H_before/H_after. Keeping it continuous overestimated f*D below
the transition by 1.3% at z = 0 rising to 5.3% at z = 1.5, which is
the size of an RSD error bar.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cosmofit import LCDM, LsCDM


def _lscdm(z_dagger=1.8, Omega_m=0.3):

    params = LsCDM.PARAMS_CLASS
    values = dict(params.defaults())
    values.update(H0=70.0, Omega_m=Omega_m, z_dagger=z_dagger)

    return LsCDM(params(**values))


@pytest.mark.parametrize("z_dagger", [1.2, 1.8, 2.5])
def test_growth_rate_jumps_by_the_ratio_of_expansion_rates(z_dagger):

    model = _lscdm(z_dagger)

    above = float(model.growth.growth_rate(z_dagger + 1e-7))
    below = float(model.growth.growth_rate(z_dagger - 1e-7))

    h_before = float(model.E(z_dagger + 1e-9))
    h_after = float(model.E(z_dagger))

    assert below / above == pytest.approx(h_before / h_after, rel=1e-5)

    # And D itself does not jump.
    assert float(model.growth.D(z_dagger - 1e-7)) == pytest.approx(
        float(model.growth.D(z_dagger + 1e-7)), rel=1e-6,
    )


def test_matches_an_independent_piecewise_integration():
    """
    ``solve_ivp`` at tight tolerance, split at the jump, matched by
    continuity of ``H dD/dN``.
    """

    z_dagger = 1.8

    model = _lscdm(z_dagger)

    def E(z):
        return float(model.E(z))

    def rhs(N, y):

        z = np.expm1(-N)
        e = E(z)
        dlnH = -(1.0 + z) * float(model.dEdz(z)) / e
        Om = model.Omega_m * (1.0 + z) ** 3 / e ** 2

        return [y[1], -(2.0 + dlnH) * y[1] + 1.5 * Om * y[0]]

    a0 = 1.0e-4
    N_jump = -np.log1p(z_dagger)

    early = solve_ivp(rhs, [np.log(a0), N_jump - 1e-12], [a0, a0],
                      rtol=1e-11, atol=1e-14)

    D, P = early.y[:, -1]
    P *= E(z_dagger + 1e-9) / E(z_dagger)

    late = solve_ivp(rhs, [N_jump, 0.0], [D, P], rtol=1e-11, atol=1e-14,
                     dense_output=True)

    D0 = late.y[0, -1]

    for z in (0.0, 0.5, 1.0, 1.5):

        D_ref, P_ref = late.sol(-np.log1p(z))

        assert float(model.growth.D(z)) == pytest.approx(D_ref / D0, rel=1e-6)
        assert float(model.growth.growth_rate(z)) == pytest.approx(
            P_ref / D_ref, rel=1e-6,
        )


def test_smooth_models_report_no_jumps():

    model = LCDM(LCDM.PARAMS_CLASS(H0=70.0, Omega_m=0.3))

    assert model.background_jumps() == ()
