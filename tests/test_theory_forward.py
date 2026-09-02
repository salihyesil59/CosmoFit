"""
The other direction: integrating a general ``f(R)`` forwards.

:func:`theory.curvature.integrate` goes backwards from today, which
is well conditioned only while the scalaron's oscillating mode does
not grow into the past. For the "disappearing cosmological constant"
family it does, and the backward path dies at ``z ~ 1.2`` however
carefully ``R_0`` is chosen. Forwards that mode decays instead.

The model used throughout is the arctan f(R) of arXiv:1601.07928,
written so that it saturates at ``-2 Lam``:

    f(R) = R - (4 Lam / pi) arctan(R / Rw)

which is LCDM at ``R >> Rw`` and loses the cosmological constant
below it -- and which the backward path cannot integrate at all.

Two things swap over relative to the backward build, and both are
tested here: ``R_0`` stops being a parameter and becomes derived,
and a ``closure`` becomes required, because ``E(0) = 1`` is now the
condition to satisfy rather than the point one starts from.

Cost
----
This is the slowest module in the suite, around twenty minutes,
because every model here has its closure shot for and each shot is
a scan of full background integrations. That is disproportionate
against a ~12 minute suite and is worth attacking, but not by
deleting the tests: the insensitivity check needs two models built
at different starting redshifts, and that comparison is the whole
justification for the analytic continuation.

What is actually at risk
------------------------
Forward integration cannot start deep enough for the growth ODE,
which wants ``E(z)`` out to ``z ~ 10^4``: at that curvature the
scalaron oscillates too fast to follow. So below the starting
redshift the history is continued analytically, on the grounds that
this family has returned to General Relativity there. That is an
assumption doing real work, so it gets two tests of its own -- that
the answer does not move when the starting redshift does, and that
the guard refuses when the assumption is false.
"""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("sympy", reason="theory.Action needs sympy")

import CosmoFit.theory.curvature as curvature  # noqa: E402
from CosmoFit.theory import Action  # noqa: E402


ARCTAN = "R - (4*Lam/pi)*atan(R/Rw)"

PARAMS = {
    "Lam": {"default": 2.25, "bounds": (1.0, 4.0)},
    "Rw": {"default": 1.0, "bounds": (0.05, 20.0)},
}

H0 = 70.0
OMEGA_M = 0.3

Z = np.array([0.0, 0.5, 1.0, 2.0])


def forward_model(name="ArctanForward", **overrides):

    spec = dict(
        params={k: dict(v) for k, v in PARAMS.items()},
        closure="Lam",
        background="forward",
    )
    spec.update(overrides)

    return Action(ARCTAN, **spec).build(name)


def built(model, **params):

    return model(
        model.PARAMS_CLASS(H0=H0, Omega_m=OMEGA_M, sigma8=0.81, **params)
    )


@pytest.fixture(scope="module")
def arctan():

    return forward_model(growth="quasi_static")


# ============================================================
# 1. It integrates at all, which backwards does not
# ============================================================


def test_the_backward_path_cannot_do_this_model():
    """
    The premise. If this ever starts passing, the forward path has
    stopped being necessary and this whole module is redundant --
    which is worth finding out from a test rather than by never
    asking.
    """

    with pytest.raises(ValueError, match="non-physical"):
        Action(
            ARCTAN, params={k: dict(v) for k, v in PARAMS.items()},
        ).build("ArctanBackward")


def test_forward_integration_lands_on_one_today(arctan):

    assert float(built(arctan).E(0.0)) == pytest.approx(1.0, abs=1e-9)


def test_the_curvature_rises_with_redshift_once_the_transient_has_gone(
    arctan,
):
    """
    The failure the backward path shows for this model is ``R``
    turning over -- the oscillating mode taking the solution off the
    attractor. Forwards it must not, *below the first few e-folds*.

    Just after the start it still can, and that is the mechanism
    rather than a defect: the initial conditions are LCDM's, not the
    theory's own, so they excite the scalaron, and what the forward
    direction buys is that the excitation decays. Measured here it
    is a few per cent at ``z ~ 15`` and gone entirely by ``z ~ 10``,
    from a start at ``z = 20``.
    """

    model = built(arctan)

    z = np.linspace(0.0, 10.0, 400)

    R = np.asarray(model.ricci(z), dtype=float)

    assert np.all(np.diff(R) > 0.0)


def test_the_starting_transient_is_confined_to_the_start_and_is_small(
    arctan,
):
    """
    The claim the whole direction rests on, as a measurement.

    ``R`` on the attractor rises monotonically with redshift, so any
    fall in it is the leftover oscillation and nothing else. Measure
    exactly that -- the largest relative *drop* between neighbouring
    samples, which is identically zero for a monotone curve.

    A first attempt measured the deviation from a straight line
    instead, which is not the same thing at all: ``R`` goes as
    ``(1+z)^3``, so that number is dominated by ordinary curvature
    and came out *larger* near today, where the curve bends most.
    """

    model = built(arctan)

    def worst_drop(lo, hi):
        z = np.linspace(lo, hi, 400)
        R = np.asarray(model.ricci(z), dtype=float)
        drops = -np.diff(R) / R[:-1]
        return float(np.max(np.maximum(drops, 0.0)))

    # Gone entirely over the range anything is fitted against.
    assert worst_drop(0.0, 10.0) == 0.0

    # Present near the start, where the initial conditions were
    # imposed rather than solved for. Measured at 4.3% over this
    # window; the bound is loose around that rather than tight,
    # because the number depends on how finely the oscillation is
    # sampled -- a coarser grid reports a smaller one for the same
    # solution, which is how a first version of this test came to
    # quote a part in a thousand.
    #
    # Four per cent at z ~ 15 is tolerable only because it is
    # *confined*: the assertion above puts it at exactly zero over
    # the range anything is fitted against, and
    # `test_the_answer_does_not_depend_on_where_the_integration_starts`
    # shows the observables do not move when the junction does.
    near_start = worst_drop(13.0, 18.0)

    assert near_start > 0.0

    assert near_start < 0.10


def test_it_returns_to_general_relativity_at_high_curvature(arctan):
    """
    Which is what makes the analytic continuation below the start
    legitimate, so it is asserted rather than assumed.
    """

    f_R, f_RR = built(arctan).scalaron(np.array([5.0, 10.0]))

    assert np.all(np.abs(np.atleast_1d(f_R) - 1.0) < 1e-3)
    assert np.all(np.atleast_1d(f_RR) < 1e-5)


# ============================================================
# 2. What the direction costs, and what it buys
# ============================================================


def test_r0_is_derived_rather_than_given(arctan):

    assert "R_0" not in arctan.EXTRA_PARAMS

    assert "R_0" not in arctan.PARAMS_CLASS.names()

    # and it is not LCDM's value, so it is really being solved for
    assert built(arctan).ricci_today() == pytest.approx(8.8877, abs=1e-3)


def test_the_closure_is_solved_not_taken_from_the_default(arctan):

    model = built(arctan)

    assert model.closure_value() != PARAMS["Lam"]["default"]

    assert model.closure_value() == pytest.approx(2.22254, abs=1e-4)


def test_forward_without_a_closure_is_refused():

    with pytest.raises(ValueError, match="needs closure"):
        Action(
            ARCTAN,
            params={k: dict(v) for k, v in PARAMS.items()},
            background="forward",
        ).build("NoClosure")


def test_backward_with_a_closure_is_still_refused():
    """
    The two arrangements are exclusive, and the message has to say
    which one the caller might have wanted.
    """

    with pytest.raises(ValueError, match="background='forward'"):
        Action(
            ARCTAN,
            params={k: dict(v) for k, v in PARAMS.items()},
            closure="Lam",
        ).build("Both")


def test_an_unknown_direction_is_refused():

    with pytest.raises(ValueError, match="background must be"):
        Action("R + alpha_fr*R**2",
               params={"alpha_fr": {"default": 1e-3}},
               background="sideways")


# ============================================================
# 3. The analytic continuation, which is the assumption
# ============================================================


def test_the_answer_does_not_depend_on_where_the_integration_starts(
    monkeypatch,
):
    """
    The load-bearing test of this module.

    Below the starting redshift the history is continued
    analytically rather than integrated. If that continuation were
    doing real work, moving the junction would move the answer.
    Moving it from z = 20 to z = 60 must not.

    `_Z_INIT` is read at call time rather than bound as a default
    argument precisely so that this test can move it. Bound as a
    default it would be unchangeable, and this test would pass while
    comparing a run against itself.
    """

    def observables(z_init):

        monkeypatch.setattr(curvature, "_Z_INIT", z_init)

        model = built(forward_model(f"Start{int(z_init)}",
                                    growth="quasi_static"))

        return (
            np.asarray(model.E(Z), dtype=float),
            np.asarray(model.background.fsigma8(Z), dtype=float),
            model.closure_value(),
        )

    E_a, fs_a, lam_a = observables(20.0)
    E_b, fs_b, lam_b = observables(60.0)

    assert lam_b == pytest.approx(lam_a, rel=1e-6)
    assert np.allclose(E_b, E_a, rtol=1e-6)
    assert np.allclose(fs_b, fs_a, rtol=1e-5)


def test_the_continuation_is_refused_where_it_would_be_invention(
    monkeypatch,
):
    """
    Started at z = 2, the model is still ~0.7% away from the
    matter-dominated expansion, so continuing it analytically below
    that would not be an approximation. The guard has to say so
    rather than splice.
    """

    monkeypatch.setattr(curvature, "_Z_INIT", 2.0)

    # Building is enough to trigger it: constructing any Cosmology
    # builds its distance integrator, which evaluates E(z) out to
    # z = 5 straight away -- past the junction, and so through the
    # continuation.
    with pytest.raises(ValueError, match="invention rather than"):
        forward_model("Shallow")
