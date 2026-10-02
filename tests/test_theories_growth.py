"""
The native growth theory.

Without radiation it must be the old growth calculator, model by model
-- the equation and its numerics were carried over, not changed. With
radiation there is no old answer to compare with, so it is checked
against what the equation must do: grow as Meszaros's exact solution in
a matter-plus-radiation universe, rescale ``f`` by ``H_before/H_after``
across a jump in ``H``, and agree with the first-order continuity and
Euler equations it was derived from where matter exchanges energy.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import CosmoFit
from CosmoFit.core import ComponentError, Likelihood, get_model
from CosmoFit.theories import DarkSector, GrowthContext


class Probe(Likelihood):

    def get_requirements(self):
        return {name: None for name in (
            "growth_factor", "growth_rate", "sigma8_z", "fsigma8", "E", "expansion",
        )}

    def logp(self):
        return 0.0


BASE = {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "sigma8": 0.8}


def model(sector, radiation=True, growth=None, **params):

    m = get_model({
        "theory": {
            "background": {"dark_energy": sector, "radiation": radiation},
            "growth": growth,
        },
        "likelihood": {"probe": {"class": Probe}},
        "params": {**BASE, **params},
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = m.logposterior({})

    assert np.isfinite(point.logpost)

    return m, point


def old_model(name, Omega_k=0.0, **params):

    cls = getattr(CosmoFit, name)
    values = dict(cls.PARAMS_CLASS.defaults())
    values.update(H0=68.0, Omega_m=0.31, Omega_b=0.049, Omega_k=Omega_k, **params)

    return cls(cls.PARAMS_CLASS(**{n: values[n] for n in cls.PARAMS_CLASS.names()}))


Z = np.array([0.0, 0.1, 0.5, 1.0, 1.5, 1.79, 1.81, 2.5, 4.0])

# sector -> (old class, new params, old params, flat only)
OLD = {
    "lambda": ("LCDM", {}, {}, False),
    "cpl": ("CPL", {"w0": -0.8, "wa": -0.6}, {"w0": -0.8, "wa": -0.6}, False),
    "ide": ("IDE", {"w0": -0.9, "xi": 0.05}, {"w0": -0.9, "xi": 0.05}, False),
    "rvm": ("RunningVacuum", {"nu": 0.01}, {"nu": 0.01}, False),
    "lscdm": ("LsCDM", {"z_dagger": 1.8}, {"z_dagger": 1.8}, False),
    "dgp": ("DGP", {}, {}, False),
    "cardassian": ("Cardassian", {"n_card": 0.2, "q_card": 1.3},
                   {"n_card": 0.2, "q_card": 1.3}, False),
    "fq_exponential": ("FQExponential", {}, {}, True),
    "ft_power_law": ("FTPowerLaw", {"n_ft": 0.2}, {"n": 0.2}, True),
    "frt_linear": ("FRTLinear", {"beta": 0.05}, {"beta": 0.05}, False),
    "hu_sawicki": ("FRHuSawicki", {"f_R0": -1e-5, "n_hs": 1.0},
                   {"f_R0": -1e-5, "n": 1.0}, False),
}


# ============================================================
# Without radiation: the old growth calculator
# ============================================================

def _old_cases():

    for sector, (*_, flat) in OLD.items():
        for Omega_k in ((0.0,) if flat else (0.0, 0.05)):
            yield sector, Omega_k


@pytest.mark.parametrize("sector, Omega_k", list(_old_cases()))
def test_radiation_free_growth_is_the_old_calculator(sector, Omega_k):

    cls, new, old, _ = OLD[sector]

    m, _ = model(sector, radiation=False, Omega_k=Omega_k, **new)
    reference = old_model(cls, Omega_k, **old).growth

    # IDE's exchange rate is differenced with a different step.
    rtol = 2e-8 if sector == "ide" else 1e-9

    np.testing.assert_allclose(m.provider.get_growth_factor(Z), reference.D(Z), rtol=rtol)
    np.testing.assert_allclose(
        m.provider.get_growth_rate(Z), reference.growth_rate(Z), rtol=rtol,
    )


@pytest.mark.parametrize("cls", ["IDE", "LsCDM", "FRHuSawicki", "DGP"])
def test_wrapped_old_models_grow_through_their_own_hooks(cls):

    _, _, old, _ = next(v for v in OLD.values() if v[0] == cls)

    m, _ = model({"name": "legacy", "model": cls}, radiation=False, **old)
    reference = old_model(cls, **old).growth

    np.testing.assert_allclose(
        m.provider.get_growth_rate(Z), reference.growth_rate(Z), rtol=2e-8,
    )


# ============================================================
# With radiation: the physics of the equation
# ============================================================

class MatterRadiation(DarkSector):
    """Cold matter and radiation alone, ``E^2`` normalized to 1 today."""

    name = "matter_radiation"

    def solve(self, ctx):

        total = ctx.Omega_cb + ctx.rho_rel0

        return lambda z: (ctx.rho_cb(z) + ctx.rho_rel(z)) / total

    def clustering_matter(self, z, growth):

        ctx = growth.ctx

        return ctx.rho_cb(z) / (ctx.Omega_cb + ctx.rho_rel0)


def test_growth_in_matter_and_radiation_is_meszaros():
    """
    ``D proportional to 1 + 3y/2``, ``y = rho_m/rho_rel``, solves the
    equation exactly when only matter and (massless) radiation are
    present -- through equality, not just deep in either era.
    """

    m, _ = model(MatterRadiation, m_nu=0.0)

    ctx = m.provider.get_expansion()["context"]

    z = np.array([0.0, 1.0, 10.0, 1000.0, 3400.0, 1e4, 3e4])

    y = ctx.rho_cb(z) / ctx.rho_rel(z)

    # 5e-8 is the fixed-step RK4's own error: it falls 16-fold when the
    # steps are doubled.
    np.testing.assert_allclose(
        m.provider.get_growth_factor(z), (1.0 + 1.5 * y) / (1.0 + 1.5 * y[0]), rtol=1e-7,
    )
    np.testing.assert_allclose(
        m.provider.get_growth_rate(z), 1.5 * y / (1.0 + 1.5 * y), rtol=1e-8,
    )


def test_growth_rate_drops_by_the_jump_in_H():
    """``H dD/dN`` is continuous across the sign switch: ``D`` carries
    over, ``f`` is rescaled by ``H_before / H_after``."""

    m, _ = model("lscdm", z_dagger=1.8)

    above, below = 1.8 + 1e-7, 1.8 - 1e-7

    E = m.provider.get_E
    f = m.provider.get_growth_rate
    D = m.provider.get_growth_factor

    assert D(below) == pytest.approx(D(above), rel=1e-6)
    assert f(below) / f(above) == pytest.approx(E(above) / E(below), rel=1e-6)
    assert f(below) / f(above) < 0.95


def _first_order(m, N_start, z_out):
    """
    Integrate the continuity and Euler equations of the clustering
    matter directly,

        delta' = -u - psi delta,
        u'     = -(2 + dlnH/dN) u - 3/2 Omega_m mu delta,

    ``u = theta / (aH)``, with the exchange rate ``psi = dln rho_m/dN + 3``
    read off the matter density itself -- not the sector's
    ``matter_exchange`` -- from the growth theory's own ``D, D'`` at
    ``N_start``.
    """

    expansion = m.provider.get_expansion()

    sector, p = expansion["sector"], expansion["sector_params"]
    ctx, E2 = expansion["context"], expansion["E2"]

    step = 1e-5

    def lnE2(N):
        return math.log(float(E2(math.expm1(-N))))

    def dlnH(N):
        return 0.25 * (lnE2(N + step) - lnE2(N - step)) / step

    growth = GrowthContext(
        ctx=ctx, E2=E2,
        dlnH_dN=lambda z: dlnH(-math.log1p(float(z))),
    )

    def rho_m(N):
        return float(sector.clustering_matter(np.array(math.expm1(-N)), growth, **p))

    def rhs(N, y):

        delta, u = y

        psi = (math.log(rho_m(N + step)) - math.log(rho_m(N - step))) / (2 * step) + 3.0

        Om = rho_m(N) / float(E2(math.expm1(-N)))

        return [-u - psi * delta, -(2.0 + dlnH(N)) * u - 1.5 * Om * delta]

    z0 = math.expm1(-N_start)

    D0 = float(m.provider.get_growth_factor(z0))
    P0 = D0 * float(m.provider.get_growth_rate(z0))

    psi0 = (
        math.log(rho_m(N_start + step)) - math.log(rho_m(N_start - step))
    ) / (2 * step) + 3.0

    N_out = sorted(-np.log1p(z_out))

    solution = solve_ivp(
        rhs, (N_start, 0.0), [D0, -(P0 + psi0 * D0)],
        t_eval=N_out, rtol=1e-11, atol=1e-14, method="DOP853",
    )

    delta, u = solution.y

    return delta, -(u + np.array([
        (math.log(rho_m(N + step)) - math.log(rho_m(N - step))) / (2 * step) + 3.0
        for N in N_out
    ]) * delta)


@pytest.mark.parametrize("sector, params", [
    ("ide", {"w0": -0.9, "xi": 0.05}),
    ("rvm", {"nu": 0.01}),
    ("rvm", {"nu": 0.01, "Omega_k": 0.05}),
])
def test_exchange_terms_match_continuity_and_euler(sector, params):

    m, _ = model(sector, **params)

    z_out = np.array([0.0, 0.5, 1.0, 2.0, 5.0])

    delta, P = _first_order(m, math.log(1e-3), z_out)

    z_sorted = np.expm1(-np.array(sorted(-np.log1p(z_out))))

    np.testing.assert_allclose(m.provider.get_growth_factor(z_sorted), delta, rtol=1e-7)
    # f is small for IDE (0.2 today), where the two finite-differenced
    # exchange rates leave 1e-7.
    np.testing.assert_allclose(m.provider.get_growth_rate(z_sorted), P / delta, rtol=3e-7)


# ============================================================
# Hu-Sawicki
# ============================================================

def test_hu_sawicki_grows_faster_on_small_scales():

    lcdm, _ = model("lambda")
    f_lcdm = lcdm.provider.get_growth_rate(0.5)

    rates = [
        model("hu_sawicki", growth={"k": k}, f_R0=-1e-5, n_hs=1.0)[0]
        .provider.get_growth_rate(0.5)
        for k in (0.01, 0.1, 1.0)
    ]

    assert f_lcdm < rates[0] < rates[1] < rates[2]

    # f_R0 -> 0 is general relativity.
    gr, _ = model("hu_sawicki", f_R0=-1e-12, n_hs=1.0)

    assert gr.provider.get_growth_rate(0.5) == pytest.approx(f_lcdm, rel=1e-6)


# ============================================================
# Outputs
# ============================================================

def test_fsigma8_and_S8():

    m, point = model("lambda", S8={"derived": True})

    z = np.array([0.3, 0.8])

    D = m.provider.get_growth_factor(z)
    f = m.provider.get_growth_rate(z)

    np.testing.assert_allclose(m.provider.get_sigma8_z(z), 0.8 * D, rtol=1e-14)
    np.testing.assert_allclose(m.provider.get_fsigma8(z), 0.8 * f * D, rtol=1e-14)

    assert m.provider.get_growth_factor(0.0) == pytest.approx(1.0, rel=1e-14)
    assert point.derived["S8"] == pytest.approx(0.8 * math.sqrt(0.31 / 0.3), rel=1e-12)


def test_growth_rejects_unknown_options():

    with pytest.raises(ComponentError, match="unknown option"):
        model("lambda", growth={"kk": 0.1})
