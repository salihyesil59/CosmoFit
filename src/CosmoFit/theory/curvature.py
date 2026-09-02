"""
Fourth-order gravity: a general ``f(R)``.

:mod:`~theory.minisuperspace` reduces an action by integrating the
``addot`` in the Einstein-Hilbert term away by parts, which is
legitimate only while the Lagrangian is *linear* in it. A general
``f(R)`` is not, and the reduction refuses -- for good reason: the
term it would discard is not a total derivative, and dropping it
returns a different theory with nothing to show for it.

The Lagrange-multiplier route
-----------------------------
The standard way round is to stop treating ``R`` as shorthand for a
combination of ``a`` and its derivatives, and make it an
*independent* variable held to that combination by a multiplier:

    S = (1/2) integral dt { N a^3 f(R) - lambda (R - R_geom) }.

Varying ``lambda`` returns ``R = R_geom``; varying ``R`` gives
``lambda = N a^3 f'(R)``. Substituting that back,

    L = (1/2) N a^3 [ f(R) - f'(R) R + f'(R) R_geom ],

which is **linear in addot** -- because ``R_geom`` is, and it now
appears only multiplied by ``f'(R)``. So the ordinary reduction
applies again, at the cost of one extra dynamical variable. That
extra variable is the theory's fourth order, made visible.

The system
----------
What comes out is smaller than "fourth order" suggests. In e-folds,
with the state ``(H, R)``:

* varying ``R`` gives back ``R = 6(2H^2 + H dH/dN)``, i.e. an
  explicit ``dH/dN``;
* the Friedmann constraint is *linear* in ``dR/dN``, so it gives an
  explicit ``dR/dN``.

Two first-order equations. The price is where ``dR/dN`` divides by
``f''(R)``: as the theory approaches General Relativity that
vanishes, the equation stiffens, and the integration gets slower
without ever becoming wrong. That is measured in
``tests/test_theory_curvature.py`` rather than asserted.

Which f(R) this can and cannot do
---------------------------------
Backward integration from today is well conditioned only while the
scalaron stays light enough for its oscillating mode not to grow
into the past. That holds when ``f_RR`` is roughly constant --
``R + alpha R^2`` is the case measured below -- and fails for the
"disappearing cosmological constant" family, where ``f_RR`` falls
steeply with ``R``: Hu-Sawicki, Starobinsky 2007, Tsujikawa, and the
arctan models (arXiv:1601.07928, arXiv:1310.6915).

Measured on ``R - (4 Lam/pi) atan(R/Rw)`` with ``Rw = 1`` and ``Lam``
tuned so that ``dR/dN`` at ``N = 0`` matches an LCDM background
exactly: the solution reaches only ``z ~ 1.2`` before ``R`` turns
over and the oscillating mode takes it away from the attractor.
That is not the integrator's choice -- RK45, DOP853, Radau and BDF
all fail at the same point, and LSODA returns success with a
non-monotonic ``R`` and NaNs beyond ``z ~ 2.4``, which is worse. Nor
is it the initial condition being slightly off: scanning ``R_0`` from
8 to 15 never gets past ``z ~ 1.2``, because the required precision
in ``R_0`` grows exponentially with the redshift wanted.

The fix for that family is not a better solver. It is to integrate
*forwards* from deep in matter domination, where the attractor is an
attractor, or to impose the background and solve for the ``f`` that
produces it -- the designer construction that
:class:`~cosmology.models.fr.FRHuSawicki` uses, which is why that
model is written by hand and does not come through here.

Direction of integration
------------------------
Backwards, from today. That is the opposite of what
:mod:`~theory.fields` does, and the difference is not an
inconsistency -- it is what the two systems actually do.

A scalar field integrated backwards runs away: the Hubble friction
that damps it forwards becomes anti-friction, and the generic past
solution is kinetic-dominated. The ``f(R)`` scalaron does not. A
relative perturbation of ``1e-8`` in today's ``R`` changes ``E(z)``
by less than that out to ``z = 1100``, and loosening the
integrator's tolerance a hundredfold moves it by ``1e-8``. Both are
measured. So there is no shooting here and no early initial
condition: ``H = 1`` at ``a = 1`` holds by construction, and the
model's extra freedom is carried by ``R`` today.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from CosmoFit.theory.fields import History

from CosmoFit.cosmology.numerics.hermite import hermite_spline


#: Points per e-fold on the interpolating grid, as in
#: :mod:`~theory.fields`.
_POINTS_PER_EFOLD = 256

_MIN_POINTS = 128

_MARGIN = 0.35

#: An implicit method by default. Unlike the scalar-field system,
#: this one genuinely is stiff -- ``dR/dN`` carries ``1/f''(R)``,
#: which diverges in the General Relativity limit -- and an
#: explicit integrator answers that by taking very small steps.
_METHOD = "LSODA"

_RTOL = 1.0e-11
_ATOL = 1.0e-13


# ============================================================
# Symbolic assembly
# ============================================================

def is_higher_order(f: sp.Expr, scalar: sp.Symbol) -> bool:
    """
    Whether ``f`` is nonlinear in the geometry scalar, and so needs
    this module rather than the ordinary reduction.
    """

    return sp.simplify(sp.diff(f, scalar, 2)) != 0


# ------------------------------------------------------------

def multiplier_lagrangian(ms, f: sp.Expr, scalar: sp.Symbol) -> sp.Expr:
    """
    The gravitational Lagrangian of ``f(R)`` with ``R`` promoted to
    an independent variable, as set out in the module docstring.

    Parameters
    ----------
    ms : Minisuperspace
        Must carry a curvature variable (``ms.R``).

    f : sympy expression
        ``f`` in terms of ``scalar``.

    scalar : Symbol
        The symbol ``f`` is written in.
    """

    if ms.R is None:
        raise ValueError(
            "This minisuperspace has no curvature variable; build "
            "it with Minisuperspace(curvature=True)."
        )

    geometric, _ = ms.geometry_scalar("metric")

    f_of_R = f.subs(scalar, ms.R)
    f_prime = sp.diff(f, scalar).subs(scalar, ms.R)

    return ms.N * ms.a**3 * (
        f_of_R - f_prime * ms.R + f_prime * geometric
    ) / 2


# ============================================================
# The equations of motion
# ============================================================

class CurvatureSystem:
    """
    An ``f(R)`` background, compiled to a right-hand side in
    ``(H, R)`` over ``N = ln a``.

    Built from the two expressions the reduction produces: the
    equation that comes from varying ``R`` (explicit in ``dH/dN``)
    and the Friedmann constraint (linear in ``dR/dN``).
    """

    def __init__(self, constraint, curvature_equation, acceleration,
                 state, derivatives, args):

        a, H, R = state
        dH, dR = derivatives

        solution = sp.solve(
            [curvature_equation, constraint], [dH, dR], dict=True,
        )

        if len(solution) != 1:
            raise NotImplementedError(
                f"Could not solve this f(R) action's equations for "
                f"dH/dN and dR/dN ({len(solution)} solution(s)). "
                f"The system is linear in both for any f(R) with a "
                f"non-vanishing second derivative."
            )

        solution = solution[0]

        signature = (a, H, R, *args)

        self.rhs = sp.lambdify(
            signature, [solution[dH], solution[dR]], "numpy", cse=True,
        )

        # The independent check. Two of the three equations of
        # motion define the right-hand side above, so their
        # residuals are zero by construction and measure nothing --
        # unlike the scalar-field case, where the constraint is a
        # first integral that the integration never imposes.
        #
        # The third, from varying `a`, is the one left over. It
        # follows from the other two by the Bianchi identity, so it
        # holds on an exact solution and drifts on an approximate
        # one. It needs second derivatives, which come from
        # differentiating the solved system along N rather than
        # from anywhere new.
        def along_N(expr):
            return (
                a * sp.diff(expr, a)
                + solution[dH] * sp.diff(expr, H)
                + solution[dR] * sp.diff(expr, R)
            )

        self.residual = sp.lambdify(
            signature,
            acceleration.subs(
                {
                    sp.Symbol("d2H"): along_N(solution[dH]),
                    sp.Symbol("d2R"): along_N(solution[dR]),
                    dH: solution[dH],
                    dR: solution[dR],
                },
                simultaneous=True,
            ),
            "numpy",
            cse=True,
        )

        self.symbolic_constraint = constraint
        self.symbols = (a, H, R)


# ------------------------------------------------------------

def build_system(action):
    """
    Assemble the :class:`CurvatureSystem` of an ``f(R)``
    :class:`~theory.action.Action`.
    """

    from CosmoFit.theory.minisuperspace import (
        friedmann_constraint,
        fluid_lagrangian,
        reduce_order,
        Minisuperspace,
    )

    ms = Minisuperspace(curvature=True)

    L = multiplier_lagrangian(ms, action.gravity, action.scalar)

    L += fluid_lagrangian(
        ms,
        {
            fluid.w: 3 * action.namespace[fluid.parameter]
            for fluid in action.fluids
        },
    )

    L = reduce_order(L, ms)

    a = sp.Symbol("a", positive=True)
    H = sp.Symbol("H", positive=True)
    R = sp.Symbol("Ricci")

    dH, dR = sp.Symbol("dH"), sp.Symbol("dR")

    replacements = {
        sp.diff(ms.R, ms.t): H * dR,
        ms.R: R,
        sp.diff(ms.a, ms.t, 2): a * (H**2 + H * dH),
        sp.diff(ms.a, ms.t): a * H,
        ms.a: a,
    }

    def to_state(expr):

        expr = expr.subs(replacements, simultaneous=True)

        return sp.expand(
            expr.subs(ms.k, -action.namespace["Omega_k"])
        )

    constraint = to_state(friedmann_constraint(L, ms))

    # Second derivatives, written in terms of the state:
    #   addot   = a (H^2 + H H'),
    #   Rddot   = H (H' R' + H R'')
    # with ' meaning d/dN. `d2H` and `d2R` are placeholders the
    # system substitutes once it has solved for the first ones.
    second = {
        sp.diff(ms.R, ms.t, 2): H * (dH * dR + H * sp.Symbol("d2R")),
        sp.diff(ms.a, ms.t, 2): a * (H**2 + H * dH),
    }

    # Varying R returns the definition of R, which in these
    # variables is an explicit dH/dN. Taken from the reduced
    # Lagrangian rather than written out, so it stays whatever the
    # action actually implies.
    from CosmoFit.theory.minisuperspace import euler_lagrange

    curvature_equation = to_state(
        euler_lagrange(L, ms.R, ms.t).subs(
            {sp.diff(ms.N, ms.t): 0, ms.N: 1},
        ).doit()
    )

    acceleration = sp.expand(
        euler_lagrange(L, ms.a, ms.t)
        .subs({sp.diff(ms.N, ms.t): 0, ms.N: 1})
        .doit()
        .subs(second, simultaneous=True)
    )
    acceleration = to_state(acceleration)

    args = action.solver_arguments(
        set().union(
            constraint.free_symbols,
            curvature_equation.free_symbols,
            acceleration.free_symbols,
        )
        - {a, H, R, dH, dR, sp.Symbol("d2H"), sp.Symbol("d2R")}
    )

    return CurvatureSystem(
        constraint, curvature_equation, acceleration,
        (a, H, R), (dH, dR), args,
    ), args


# ============================================================
# Integration
# ============================================================

def integrate(system, values, R_today, N_lo, N_hi) -> History:
    """
    Integrate the ``f(R)`` background outwards from ``a = 1``, where
    ``H = 1`` by the definition of ``H0``.

    Parameters
    ----------
    system : CurvatureSystem

    values : tuple
        Parameter values, in the order ``system`` expects.

    R_today : float
        The Ricci scalar today, in units of ``H0^2``. This is the
        model's extra initial condition -- the freedom a
        fourth-order theory has and General Relativity does not.

    N_lo, N_hi : float
        Range of ``N = ln a`` to cover.
    """

    N_lo = min(N_lo, 0.0) - _MARGIN
    N_hi = max(N_hi, 0.0) + _MARGIN

    def rhs(N, y):
        return system.rhs(np.exp(N), *y, *values)

    branches = []

    for end in (N_lo, N_hi):

        if end == 0.0:
            continue

        n = max(_MIN_POINTS, int(_POINTS_PER_EFOLD * abs(end)))

        solution = solve_ivp(
            rhs, (0.0, end), [1.0, float(R_today)],
            method=_METHOD,
            t_eval=np.linspace(0.0, end, n),
            rtol=_RTOL, atol=_ATOL,
        )

        if not solution.success:
            raise RuntimeError(
                f"Could not integrate this f(R) background over "
                f"N = [0, {end:.3g}]: {solution.message} The "
                f"equation for dR/dN carries 1/f''(R), so a model "
                f"very close to General Relativity is stiff -- "
                f"slow rather than wrong -- and one where f''(R) "
                f"passes through zero has no solution there at all."
            )

        branches.append(solution)

    N = np.concatenate(
        [b.t[::-1] if b.t[-1] < 0 else b.t for b in branches]
    )
    Y = np.concatenate(
        [b.y[:, ::-1] if b.t[-1] < 0 else b.y for b in branches], axis=1,
    )

    N, unique = np.unique(N, return_index=True)
    Y = Y[:, unique]

    a = np.exp(N)

    derivatives = np.asarray(system.rhs(a, *Y, *values), dtype=float)

    H = Y[0]

    if not np.all(np.isfinite(H)) or np.any(H <= 0.0):
        raise RuntimeError(
            "This f(R) background became non-physical (H reached "
            "zero or stopped being finite) inside the requested "
            "redshift range."
        )

    residual = np.abs(
        np.asarray(system.residual(a, *Y, *values), dtype=float)
    )

    scale = np.maximum(np.abs(3.0 * H**2), 1.0e-300)

    return History(
        N[0], N[-1],
        [
            hermite_spline(N, Y[i], derivatives[i], extrapolate=False)
            for i in range(Y.shape[0])
        ],
        float(np.max(residual / scale)),
    )


# ============================================================
# Growth: the scale-dependent coupling
# ============================================================

def quasi_static_mu(f_R, f_RR, a, k):
    r"""
    ``G_eff(a,k)/G_N`` for ``f(R)`` gravity, quasi-static and
    sub-horizon:

        mu = (1/f_R) (1 + 4m)/(1 + 3m),    m = (k/a)^2 f_RR/f_R

    Perturbing the field equations, dropping time derivatives of the
    potentials against spatial ones, and keeping only the terms
    carrying ``k^2`` leaves this. It is the standard result -- the
    scalaron's Compton wavelength is what ``m`` measures, and the
    two limits are the familiar ones: ``m -> 0`` (scales far outside
    it) gives ``mu = 1/f_R``, and ``m -> infinity`` (far inside)
    gives ``mu = 4/(3 f_R)``, the extra third being the scalar
    fifth force.

    Parameters
    ----------
    f_R, f_RR : float or ndarray
        First and second derivatives of ``f`` with respect to
        ``R``, evaluated on the background at ``a``. ``f_R`` is
        dimensionless; ``f_RR`` is in units of ``1/H0^2``, matching
        the ``R`` this module integrates -- see the ``R_0``
        parameter, which is ``R_0/H0^2``.
    a : float or ndarray
        Scale factor.
    k : float or ndarray
        Comoving wavenumber [h/Mpc].

    Notes
    -----
    Units are the trap here, so they are spelled out. ``m`` is
    dimensionless: ``f_RR`` carries a length squared and ``(k/a)^2``
    an inverse length squared. Working in ``H0/c`` units, the
    physical wavenumber is ``Y = k (c/100) / a`` -- the ``h`` in
    ``k``'s units cancelling against ``H0 = 100h`` -- and with
    ``f_RR`` already in ``1/H0^2`` the product is simply
    ``m = Y^2 f_RR/f_R``. This is the same conversion
    :class:`~cosmology.models.fr.FRHuSawicki` makes, and
    ``tests/test_theory_growth.py`` holds the two against each
    other rather than trusting that they agree.

    This is the *linear* result. Chameleon screening is a
    non-linear effect and is not in it, so on scales and in
    environments where screening matters this overestimates the
    departure from GR.
    """

    from CosmoFit.cosmology.core import constants

    a = np.asarray(a, dtype=float)

    f_R = np.asarray(f_R, dtype=float)
    f_RR = np.asarray(f_RR, dtype=float)

    Y2 = (np.asarray(k, dtype=float) * (constants.c / 100.0) / a) ** 2

    m = Y2 * f_RR / f_R

    denominator = 1.0 + 3.0 * m

    if np.any(denominator <= 0.0) or np.any(f_R <= 0.0):
        raise ValueError(
            "This f(R) is not viable where mu was asked for: "
            "1 + 3m <= 0 or f_R <= 0. Both require f_RR < 0 or "
            "f_R < 0, which is a tachyonic scalaron or a ghost "
            "graviton -- see `viability_failures`. There is a "
            "finite number on the far side of that pole and it "
            "does not mean anything, so it is not returned."
        )

    return (1.0 + 4.0 * m) / (f_R * denominator)


# ============================================================
# Viability: whether the theory is one worth fitting
# ============================================================

#: Why an f(R) is rejected, keyed by the condition that failed.
VIABILITY_CONDITIONS = {
    "f_R": (
        "f_R > 0 -- the effective gravitational coupling. Where it "
        "is negative the graviton is a ghost and the theory has no "
        "stable vacuum; where it passes through zero the coupling "
        "diverges."
    ),
    "f_RR": (
        "f_RR >= 0 -- the scalaron's mass squared goes as 1/f_RR, "
        "so a negative f_RR makes it tachyonic. That is the "
        "Dolgov-Kawasaki instability, and its growth time is short "
        "enough that the background this model integrates would not "
        "survive to be observed. Zero is not flagged: it is the "
        "general-relativity limit, where the scalaron is simply "
        "absent."
    ),
}


def viability_failures(f_R, f_RR):
    """
    Which of the two standard ``f(R)`` conditions are violated.

    Returns the keys of :data:`VIABILITY_CONDITIONS` that fail
    anywhere in the arrays given -- empty when the theory is
    admissible over the range sampled.

    These are not stylistic preferences. ``f_R > 0`` keeps the
    graviton from being a ghost, and ``f_RR > 0`` keeps the scalaron
    from being tachyonic; a model violating either is not a
    cosmology whose parameters mean anything, however well it might
    fit. Checking them is cheap and the alternative is fitting a
    theory that has no business being fitted.
    """

    failures = []

    if np.any(np.asarray(f_R, dtype=float) <= 0.0):
        failures.append("f_R")

    # Strictly negative, not merely non-positive: f_RR = 0 is the
    # general-relativity limit and is perfectly well behaved --
    # m -> 0 and mu -> 1/f_R, with no pole anywhere near.
    if np.any(np.asarray(f_RR, dtype=float) < 0.0):
        failures.append("f_RR")

    return failures
