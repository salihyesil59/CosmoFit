"""
Physics checks on every cosmological model in the library.

These are not unit tests of Python plumbing -- they are the checks
that catch a transcription error in a Friedmann equation, which is
the failure mode that actually happens when adding models from
papers, and which produces a plausible-looking wrong answer rather
than a crash.

Three families of check:

1. **Friedmann closure.** ``E(z=0) = 1`` must hold identically, for
   flat and curved cases alike. Every model here is normalized so
   that its dark-energy (or effective) density at z=0 is whatever
   the closure requires. A sign error inside a closed-form solution
   -- the exact bug that was once caught in ``FQExponential`` --
   breaks this and nothing else.

2. **Derivative consistency.** ``dEdz`` must match a central finite
   difference of ``E``. Both are written out by hand for every
   model, so they are two independent transcriptions of the same
   physics; if they agree to 1e-6 the algebra is almost certainly
   right, and if they do not, one of them is wrong.

3. **Known limits.** Each generalization must collapse *exactly*
   onto the model it generalizes at the right parameter values.
   This is the strongest check available without a second
   implementation: it pins normalization constants that closure
   alone leaves free.
"""

from __future__ import annotations

import numpy as np
import pytest

import CosmoFit as C


#: Every model, by name. ``FRTLinear`` is excluded from the closure
#: check below (only) -- see `test_friedmann_closure`.
ALL_MODELS = [
    "LCDM", "WCDM", "CPL", "JBP", "BA", "LogarithmicDE",
    "PEDE", "GEDE", "LsCDM", "GCG", "IDE", "RunningVacuum",
    "Cardassian", "DGP", "HDE", "FQExponential", "FRTLinear",
    "FRHuSawicki",
]


def build(name, **kwargs):
    """
    Instantiate a model by name with a standard parameter set.
    """

    model = getattr(C, name)

    params = model.PARAMS_CLASS(

        H0=kwargs.pop("H0", 67.4),

        Omega_m=kwargs.pop("Omega_m", 0.315),

        **kwargs,

    )

    return model(params)


# ============================================================
# 1. Friedmann closure
# ============================================================

#: ``FRTLinear`` deliberately does *not* satisfy E(0) = 1 with its
#: default parameters: its ``Omega_m`` and ``Omega_L`` are
#: independent free parameters rather than being tied by a flatness
#: closure, which is how f(R,T) papers actually fit the model, and
#: which its own class docstring states explicitly. Excluded here
#: rather than silently passing a weakened assertion.
_NO_CLOSURE = {"FRTLinear"}

_CLOSURE_MODELS = [m for m in ALL_MODELS if m not in _NO_CLOSURE]

#: ``HDE`` is defined by the future event horizon, which is a
#: statement about causal structure rather than a term in the
#: Friedmann equation -- curvature changes what the holographic
#: bound is applied to, so a curved version is a different model.
#: It refuses ``Omega_k != 0`` outright; see
#: `test_hde_refuses_curvature`, which checks that rather than
#: leaving the limitation untested.
_FLAT_ONLY = {"HDE"}


def _closure_cases():

    for name in _CLOSURE_MODELS:

        for Omega_k in (0.0,) if name in _FLAT_ONLY else (0.0, -0.05, 0.05):

            yield name, Omega_k


@pytest.mark.parametrize("name,Omega_k", list(_closure_cases()))
def test_friedmann_closure(name, Omega_k):
    """
    E(z=0) must equal 1 exactly, flat or curved.
    """

    model = build(name, Omega_k=Omega_k)

    e0 = float(np.atleast_1d(model.E(0.0))[0])

    assert e0 == pytest.approx(1.0, abs=1e-10), (

        f"{name} violates the Friedmann constraint at z=0 "

        f"for Omega_k={Omega_k}: E(0) = {e0}"

    )


# ============================================================
# 2. dEdz against finite differences
# ============================================================

@pytest.mark.parametrize("name", ALL_MODELS)
def test_dEdz_matches_finite_difference(name):
    """
    The hand-written ``dEdz`` must agree with a numerical
    derivative of the hand-written ``E``.
    """

    model = build(name, w0=-0.9, wa=-0.3)

    z = np.array([0.1, 0.5, 1.0, 2.0, 3.0])

    h = 1.0e-5

    analytic = np.atleast_1d(model.dEdz(z))

    numeric = (

        np.atleast_1d(model.E(z + h))

        - np.atleast_1d(model.E(z - h))

    ) / (2.0 * h)

    np.testing.assert_allclose(

        analytic,

        numeric,

        rtol=1e-6,

        err_msg=f"{name}: dEdz disagrees with d/dz of E",

    )


# ============================================================
# 3. Known limits
# ============================================================

def _E(name, **kwargs):

    z = np.linspace(0.01, 3.0, 40)

    return np.asarray(build(name, **kwargs).E(z), dtype=float)


#: ``(description, model + parameters, model it must reduce to)``.
#: Every one of these is a statement made in a class docstring; the
#: test is what keeps those statements true.
REDUCTIONS = [

    (
        "GEDE(Delta -> 0) is LCDM",
        ("GEDE", {"Delta": 1e-12, "z_t": 0.0}),
        ("LCDM", {}),
    ),

    (
        "GEDE(Delta=1, z_t=0) is PEDE",
        ("GEDE", {"Delta": 1.0, "z_t": 0.0}),
        ("PEDE", {}),
    ),

    (
        "Cardassian(n=0, q=1) is LCDM",
        ("Cardassian", {"n_card": 0.0, "q_card": 1.0}),
        ("LCDM", {}),
    ),

    (
        "RunningVacuum(nu=0) is LCDM",
        ("RunningVacuum", {"nu": 0.0}),
        ("LCDM", {}),
    ),

    (
        "IDE(xi=0) is wCDM",
        ("IDE", {"xi": 0.0, "w0": -0.9}),
        ("WCDM", {"w0": -0.9}),
    ),

    (
        "LogarithmicDE(wa=0) is wCDM",
        ("LogarithmicDE", {"w0": -0.9, "wa": 0.0}),
        ("WCDM", {"w0": -0.9}),
    ),

    (
        "LsCDM below the transition is LCDM",
        ("LsCDM", {"z_dagger": 5.0}),
        ("LCDM", {}),
    ),

    (
        "CPL(wa=0) is wCDM",
        ("CPL", {"w0": -0.9, "wa": 0.0}),
        ("WCDM", {"w0": -0.9}),
    ),

]


@pytest.mark.parametrize(
    "description,general,special",
    REDUCTIONS,
    ids=[r[0] for r in REDUCTIONS],
)
def test_known_limits(description, general, special):
    """
    A generalization must reproduce its special case exactly, not
    approximately.
    """

    name_g, kwargs_g = general
    name_s, kwargs_s = special

    np.testing.assert_allclose(

        _E(name_g, **kwargs_g),

        _E(name_s, **kwargs_s),

        rtol=1e-9,

        err_msg=description,

    )


# ============================================================
# Published numerical signatures
# ============================================================

def test_pede_equation_of_state():
    """
    PEDE's ``w(0) = -1.145`` is the number its paper leads with,
    and it is a pure consequence of the functional form -- so it
    pins the transcription with no free parameter involved.
    """

    model = build("PEDE")

    assert float(model.w(0.0)) == pytest.approx(-1.145, abs=5e-4)

    # Asymptote deep in the past: -1 - 2/(3 ln 10).
    assert float(model.w(1e6)) == pytest.approx(

        -1.0 - 2.0 / (3.0 * np.log(10.0)),

        abs=1e-4,

    )


def test_dgp_crossover_and_growth_suppression():
    """
    DGP's ``Omega_rc`` is fixed by closure, and its ``mu`` must be
    below 1 today (weaker gravity) and approach 1 at high redshift
    (GR recovered well inside the crossover scale).
    """

    model = build("DGP", Omega_m=0.30)

    assert model.Omega_rc == pytest.approx((1.0 - 0.30) ** 2 / 4.0)

    mu_today = float(np.atleast_1d(model.mu(1.0))[0])

    assert 0.70 < mu_today < 0.75, mu_today

    mu_early = float(np.atleast_1d(model.mu(0.01))[0])

    assert mu_early == pytest.approx(1.0, abs=1e-3)


def test_dgp_growth_is_suppressed_relative_to_lcdm():
    """
    Weaker gravity must actually produce less structure -- the
    check that ``mu`` is wired into the growth ODE, not merely
    defined.
    """

    z = np.array([0.0, 0.5, 1.0])

    dgp = build("DGP").growth.fsigma8(z)
    lcdm = build("LCDM").growth.fsigma8(z)

    assert np.all(dgp < lcdm)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("RunningVacuum", {"nu": 0.05}),
        ("IDE", {"xi": 0.1, "w0": -1.0}),
    ],
)
def test_modified_matter_scaling_reaches_growth(name, kwargs):
    """
    Models that override ``Omega_matter`` must produce a different
    growth history from LCDM's -- and must match LCDM exactly at
    their GR limit.

    Before ``Cosmology.Omega_matter`` existed, the growth equation
    read ``Omega_m (1+z)^3`` off the parameters for every model, so
    a running vacuum or an interacting dark sector got LCDM's
    growth source term with its own ``E(z)``. That is internally
    inconsistent and produced no error at all.
    """

    z = np.array([0.5])

    lcdm = float(build("LCDM").growth.fsigma8(z)[0])

    limit = float(build(name).growth.fsigma8(z)[0])
    modified = float(build(name, **kwargs).growth.fsigma8(z)[0])

    assert limit == pytest.approx(lcdm, rel=1e-8)

    assert abs(modified - lcdm) > 1e-3


def test_lscdm_discontinuity_survives_the_distance_integrator():
    """
    LsCDM's ``E(z)`` genuinely jumps at ``z_dagger``. The
    trapezoidal interpolation grid smears that over one cell; this
    bounds the resulting error in ``chi(z)`` against a
    piecewise-exact quadrature that splits the integral at the
    transition.

    The class docstring claims "below 1e-4 relative". This is what
    makes that claim checkable rather than reassuring.
    """

    from scipy.integrate import quad

    z_dagger = 1.8

    model = build("LsCDM", z_dagger=z_dagger)

    def integrand(x):
        return 1.0 / float(model.E(x))

    for z in (1.0, 1.8, 2.5, 4.0):

        if z <= z_dagger:

            exact = quad(integrand, 0.0, z, limit=200)[0]

        else:

            exact = (

                quad(integrand, 0.0, z_dagger, limit=200)[0]

                + quad(integrand, z_dagger, z, limit=200)[0]

            )

        grid = float(model.integrator.chi(z))

        assert grid == pytest.approx(exact, rel=1e-4), (

            f"chi({z}) grid={grid} exact={exact}"

        )


# ============================================================
# HDE: the flat-only contract
# ============================================================

def test_hde_refuses_curvature():
    """
    Silently approximating would be the bad outcome: the ODE and
    the event-horizon integral both assume flatness, so a curved
    run would return numbers that look like a curved holographic
    model and are not one.
    """

    with pytest.raises(ValueError, match="flat"):

        build("HDE", Omega_k=0.05).E(0.0)


def test_hde_needs_a_positive_c():

    with pytest.raises(ValueError, match="c > 0"):

        build("HDE", c_hde=-0.5).E(0.0)
