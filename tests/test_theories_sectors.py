"""
The non-standard dark sectors, with and without radiation.

Without radiation each must be its old model -- the formulas were
generalized, not changed. With radiation there is no old model to
compare with, so each is checked against the physics that defines it,
by a route other than the one the code takes: the defining relation of
a holographic energy density differentiated numerically, the running
vacuum's continuity equations integrated directly, a modified Friedmann
equation's residual, a limit in which the model is LCDM.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

import cosmofit
from cosmofit.core import ComponentError, Likelihood, get_model
from cosmofit.theories.background import Background


def background(sector, radiation=True, **params):
    """A computed :class:`Background`, outside any model."""

    theory = Background({"dark_energy": sector, "radiation": radiation})

    values = {"H0": 68.0, "Omega_b": 0.049, "Omega_k": 0.0,
              "N_eff": 3.044, "m_nu": 0.06}

    if not theory.sector.derives_matter:
        values["Omega_m"] = 0.31

    values.update(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert theory.compute(values), "no solution"

    return theory


def E2(theory, z):
    return theory.current_state["E2"](np.asarray(z, dtype=float))


def ctx(theory):
    return theory.current_state["context"]


def dlog(f, N, step=1e-4):
    """``d ln f / dN`` by central difference, ``f`` a function of ``N``."""

    return (math.log(f(N + step)) - math.log(f(N - step))) / (2 * step)


def at(N):
    return math.expm1(-N)


Z = np.array([0.05, 0.5, 1.0, 2.0, 4.0])

# sector -> (old class, new params, old params, flat only)
OLD = {
    "lscdm": ("LsCDM", {"z_dagger": 1.8}, {"z_dagger": 1.8}, False),
    "gcg": ("GCG", {"A_gcg": 0.75, "alpha_gcg": 0.1}, {"A_s": 0.75, "alpha": 0.1}, False),
    "ide": ("IDE", {"w0": -0.9, "xi": 0.05}, {"w0": -0.9, "xi": 0.05}, False),
    "dgp": ("DGP", {}, {}, False),
    "cardassian": ("Cardassian", {"n_card": 0.2, "q_card": 1.3},
                   {"n_card": 0.2, "q_card": 1.3}, False),
    "fq_exponential": ("FQExponential", {}, {}, True),
    "ft_power_law": ("FTPowerLaw", {"n_ft": 0.2}, {"n": 0.2}, True),
    "frt_linear": ("FRTLinear", {"beta": 0.05}, {"beta": 0.05}, False),
    "hde": ("HDE", {"c_hde": 0.8}, {"c_hde": 0.8}, True),
    "rde": ("RDE", {"gamma_rde": 0.45}, {"gamma_rde": 0.45}, True),
    "rvm": ("RunningVacuum", {"nu": 0.01}, {"nu": 0.01}, False),
}


def old_model(name, Omega_k, **params):

    cls = getattr(cosmofit, name)
    values = dict(cls.PARAMS_CLASS.defaults())
    values.update(H0=68.0, Omega_m=0.31, Omega_b=0.049, Omega_k=Omega_k, **params)

    return cls(cls.PARAMS_CLASS(**{n: values[n] for n in cls.PARAMS_CLASS.names()}))


# ============================================================
# Without radiation: the old models
# ============================================================

def _old_cases():

    for sector, (_cls, _new, _old, flat) in OLD.items():
        for Omega_k in ((0.0,) if flat else (0.0, 0.05)):
            yield sector, Omega_k


@pytest.mark.parametrize("sector, Omega_k", list(_old_cases()))
def test_radiation_free_sector_is_the_old_model(sector, Omega_k):

    cls, new, old, _ = OLD[sector]

    theory = background(sector, radiation=False, Omega_k=Omega_k, **new)
    reference = old_model(cls, Omega_k, **old)

    np.testing.assert_allclose(theory.get_E(Z), reference.E(Z), rtol=1e-9)

    if sector != "lscdm":
        np.testing.assert_allclose(
            theory.get_DM(Z), reference.distance.DM(Z), rtol=1e-9,
        )


def test_interacting_dark_energy_through_its_resonance():

    for w0 in (-0.1, -0.1 + 1e-7, -0.1 - 1e-7):

        theory = background("ide", radiation=False, w0=w0, xi=0.1)
        reference = old_model("IDE", 0.0, w0=w0, xi=0.1)

        np.testing.assert_allclose(theory.get_E(Z), reference.E(Z), rtol=1e-12)


def test_agegraphic_matter_density_without_radiation():
    """
    ADE fixes Omega_m from n. The new solution starts deeper in the
    matter era (a = e^-30, not 1e-8), so the two agree only to the old
    solution's own accuracy.
    """

    theory = background("ade", radiation=False, n_ade=2.8)
    reference = old_model("ADE", 0.0, n_ade=2.8)

    assert theory.get_current_derived()["Omega_m"] == pytest.approx(reference.Omega_m, rel=1e-5)
    np.testing.assert_allclose(theory.get_E(Z), reference.E(Z), rtol=1e-5)


def test_sign_switching_distances_are_exact_across_the_jump():
    """
    E(z) jumps at z_dagger; the distance table is split there, where
    the old one smeared the jump over a grid cell (~1e-4).
    """

    theory = background("lscdm", radiation=False, z_dagger=1.8)

    def inverse(x):
        return 1.0 / float(theory.get_E(x))

    below = quad(inverse, 0.0, 1.8, epsrel=1e-13)[0]

    for z in (1.0, 1.81, 2.5, 4.5):

        if z <= 1.8:
            reference = quad(inverse, 0.0, z, epsrel=1e-13)[0]
        else:
            reference = below + quad(inverse, 1.8, z, epsrel=1e-13)[0]

        assert float(theory.get_comoving_distance(z)) * 68.0 / cosmofit.cosmology.core.constants.c == pytest.approx(reference, rel=1e-11)

    assert theory.get_background_jumps() == (1.8,)


# ============================================================
# With radiation: closure, and the physics that defines each
# ============================================================

RADIATION_CASES = {
    "lscdm": {"z_dagger": 1.8},
    "gcg": {"A_gcg": 0.75, "alpha_gcg": 0.1},
    "ide": {"w0": -0.9, "xi": 0.05},
    "dgp": {},
    "cardassian": {"n_card": 0.2, "q_card": 1.3},
    "fq_exponential": {},
    "ft_power_law": {"n_ft": 0.2},
    "frt_linear": {"beta": 0.05},
    "hde": {"c_hde": 0.8},
    "ade": {"n_ade": 2.8},
    "rde": {"gamma_rde": 0.45},
    "rvm": {"nu": 0.01},
}


@pytest.mark.parametrize("sector", sorted(RADIATION_CASES))
def test_every_sector_closes_with_radiation(sector):

    theory = background(sector, **RADIATION_CASES[sector])

    assert float(E2(theory, 0.0)) == pytest.approx(1.0, abs=1e-9)

    # Radiation dominates deep enough, for every one of them -- through
    # f(R,T)'s source, where each fluid counts as rho + beta (3 rho - p),
    # so radiation with 1 + 8 beta / 3.
    early = float(E2(theory, 1e5)) / float(ctx(theory).rho_std(1e5))

    expected = 1.0 + 8.0 * RADIATION_CASES[sector]["beta"] / 3.0 if sector == "frt_linear" else 1.0

    assert early == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("sector, params, lcdm_limit", [
    ("ft_power_law", {"n_ft": 0.0}, True),
    ("frt_linear", {"beta": 0.0}, True),
    ("cardassian", {"n_card": 0.0, "q_card": 1.0}, True),
    ("rvm", {"nu": 0.0}, True),
    ("ide", {"w0": -1.0, "xi": 0.0}, True),
])
def test_limits_that_are_lcdm_with_radiation(sector, params, lcdm_limit):

    theory = background(sector, **params)
    lcdm = background("lambda")

    z = np.array([0.1, 1.0, 10.0, 1100.0])

    np.testing.assert_allclose(E2(theory, z), E2(lcdm, z), rtol=1e-9)


def test_holographic_dark_energy_obeys_its_definition():
    """
    rho_de = 3 c^2 / L^2 with L the future event horizon gives
    d ln rho_de / d ln a = -2 + 2 sqrt(Omega_DE) / c, whatever else the
    universe holds. The code integrates a different equation -- one for
    Omega_DE, with the fluids' pressure in it -- so this checks that
    derivation.
    """

    c = 0.8

    theory = background("hde", c_hde=c)

    def rho_de(N):
        z = at(N)
        return float(E2(theory, z)) - float(ctx(theory).rho_std(z))

    for N in (-0.1, -1.0, -3.0, -8.0):

        z = at(N)
        omega = rho_de(N) / float(E2(theory, z))

        assert dlog(rho_de, N) == pytest.approx(-2.0 + 2.0 * math.sqrt(omega) / c, rel=1e-6)


def test_agegraphic_dark_energy_obeys_its_definition():
    """
    Omega_DE = n^2 / (eta H)^2, eta the conformal age -- checked
    against eta integrated from the solution's own E(z), the radiation
    era included.
    """

    n = 2.8

    theory = background("ade", n_ade=n)

    def integrand(log_a):
        a = math.exp(log_a)
        return 1.0 / (a * float(theory.get_E(1.0 / a - 1.0)))

    eta = quad(integrand, -29.9, 0.0, epsrel=1e-12, limit=1000)[0]

    # Radiation era below a = e^-29.9: H = sqrt(Omega_r) a^-2, eta = a/sqrt(Omega_r).
    a_min = math.exp(-29.9)
    eta += a_min / math.sqrt(float(ctx(theory).rho_rel(1.0 / a_min - 1.0)) * a_min ** 4)

    omega = 1.0 - float(ctx(theory).rho_std0)

    assert omega == pytest.approx(n ** 2 / eta ** 2, rel=1e-8)


def test_radiation_moves_the_agegraphic_matter_density():
    """
    n fixes Omega_m through the early attractor, and radiation changes
    the attractor (n^2 a^2, not n^2 a^2 / 4) -- so the prediction moves.
    """

    without = background("ade", radiation=False, n_ade=2.8).get_current_derived()["Omega_m"]
    with_radiation = background("ade", n_ade=2.8).get_current_derived()["Omega_m"]

    assert without == pytest.approx(0.2799, abs=2e-4)
    assert with_radiation == pytest.approx(0.2696, abs=2e-3)


def test_ricci_dark_energy_obeys_its_definition():
    """rho_de = gamma (Hdot + 2 H^2): Omega_de E^2 = gamma (E^2'/2 + 2 E^2)."""

    gamma = 0.45

    theory = background("rde", gamma_rde=gamma)

    def e2(N):
        return float(E2(theory, at(N)))

    for N in (-0.2, -1.5, -5.0):

        derivative = (e2(N + 1e-4) - e2(N - 1e-4)) / 2e-4
        rho_de = e2(N) - float(ctx(theory).rho_std(at(N)))

        assert rho_de == pytest.approx(gamma * (0.5 * derivative + 2.0 * e2(N)), rel=1e-6)


@pytest.mark.parametrize("Omega_k", [0.0, 0.04])
def test_running_vacuum_matches_its_continuity_equations(Omega_k):
    """
    Integrate rho_cb' = -3(1-nu) rho_cb - nu (rho_rel' - 2 Omega_k a^-2)
    directly from today, and build E^2 from the vacuum relation -- the
    code instead solves a single linear equation for E^2.
    """

    nu = 0.01

    theory = background("rvm", nu=nu, Omega_k=Omega_k)
    context = ctx(theory)

    c = (1.0 - nu) - context.Omega_cb - context.rho_rel0 - Omega_k

    def rho_rel_prime(N):
        z = at(N)
        return -3.0 * (context.rho_rel(z) + context.p_rel(z))

    def rhs(N, y):
        return [-3.0 * (1.0 - nu) * y[0]
                - nu * (float(rho_rel_prime(N)) - 2.0 * Omega_k * math.exp(-2.0 * N))]

    solution = solve_ivp(rhs, (0.0, -10.0), [context.Omega_cb],
                         rtol=1e-12, atol=1e-14, dense_output=True)

    for N in (-0.5, -2.0, -7.0):

        z = at(N)
        expected = (solution.sol(N)[0] + float(context.rho_rel(z)) + c
                    + Omega_k * math.exp(-2.0 * N)) / (1.0 - nu)

        assert float(E2(theory, z)) == pytest.approx(expected, rel=1e-8)


def test_fq_exponential_solves_its_friedmann_equation():

    theory = background("fq_exponential")
    context = ctx(theory)

    lam = 0.5 + float(np.real(
        __import__("scipy.special", fromlist=["lambertw"]).lambertw(
            -context.rho_std0 / (2 * math.sqrt(math.e)), k=0,
        )
    ))

    z = np.array([0.0, 0.7, 3.0, 1100.0])
    x = E2(theory, z)

    np.testing.assert_allclose(
        (x - 2 * lam) * np.exp(lam / x), context.rho_std(z), rtol=1e-12,
    )


def test_interacting_dark_energy_moves_energy_into_matter():
    """
    The energy moved into matter, T, obeys T' + 3 T = 3 xi rho_de -- and
    what E^2 holds beyond the standard fluids is exactly T plus the dark
    energy. The first is checked at low redshift, where T' + 3 T is not a
    small difference of large numbers that a finite difference would
    lose; the second, which is exact, at high redshift too.
    """

    xi, w0 = 0.05, -0.9

    theory = background("ide", w0=w0, xi=xi)
    context = ctx(theory)
    sector = theory.sector

    Omega_de = sector.dark_energy_today(context)

    def transfer(N):
        return float(sector.transfer(at(N), Omega_de, w0, xi))

    def rho_de(N):
        return Omega_de * float(sector.density(at(N), w0=w0, xi=xi))

    for N in (-0.1, -0.3, -0.6):

        slope = (transfer(N + 1e-4) - transfer(N - 1e-4)) / 2e-4

        assert slope + 3.0 * transfer(N) == pytest.approx(3.0 * xi * rho_de(N), rel=1e-7)

    for N in (-0.3, -2.5, -7.0):

        beyond = float(E2(theory, at(N)) - context.rho_std(at(N)) - context.curvature(at(N)))

        assert beyond == pytest.approx(transfer(N) + rho_de(N), rel=1e-12)


# ============================================================
# Assembly
# ============================================================

class Probe(Likelihood):

    def get_requirements(self):
        return {"DM": None}

    def logp(self):
        return 0.0


def model(sector, params, **options):

    return get_model({
        "theory": {"background": {"dark_energy": sector, **options}},
        "likelihood": {"probe": {"class": Probe}},
        "params": params,
    })


def test_flat_only_sectors_refuse_curvature():

    with pytest.raises(ComponentError, match="flat universe"):
        model("hde", {
            "H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "c_hde": 0.8,
            "Omega_k": {"prior": {"min": -0.1, "max": 0.1}},
        })


def test_agegraphic_does_not_take_a_matter_density():

    with pytest.raises(ComponentError, match="No theory or likelihood takes"):
        model("ade", {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "n_ade": 2.8})

    m = model("ade", {
        "H0": 68.0, "Omega_b": 0.049, "n_ade": 2.8,
        "Omega_m": {"derived": True},
    })

    assert m.logposterior({}).derived["Omega_m"] == pytest.approx(0.2696, abs=2e-3)


def test_wrapped_old_models_need_radiation_off():

    with pytest.raises(ComponentError, match="radiation: false"):
        model({"name": "legacy", "model": "CPL"}, {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049})

    m = model(
        {"name": "legacy", "model": "CPL"},
        {"H0": 68.0, "Omega_m": 0.31, "Omega_b": 0.049, "w0": -0.8, "wa": -0.5},
        radiation=False,
    )

    m.logposterior({})

    reference = old_model("CPL", 0.0, w0=-0.8, wa=-0.5)

    np.testing.assert_allclose(
        m.theories["background"].get_DM(Z), reference.distance.DM(Z), rtol=1e-9,
    )
