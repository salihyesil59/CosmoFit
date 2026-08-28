"""
Minisuperspace reduction: from an action to the Friedmann equation.

Every model in ``cosmology.models`` was written the other way
round -- somebody derived ``E(z)`` by hand from a Lagrangian and
typed the result in. This module does the derivation itself, so a
model can be specified the way it is specified in a paper (an
action and a metric) rather than the way it is specified in code
(an already-solved expansion history).

The method is the standard minisuperspace one. Write FLRW with an
explicit lapse ``N(t)``,

    ds^2 = -N(t)^2 dt^2 + a(t)^2 dSigma_k^2,

substitute it into the action to get a point-like Lagrangian
``L(N, Ndot, a, adot, addot, fields, fielddots)``, and vary. The
lapse is what makes this work: it is a non-dynamical gauge degree
of freedom, so varying it produces a *constraint* rather than an
evolution equation -- and that constraint, evaluated at the gauge
choice ``N = 1``, is exactly the Friedmann equation. Dropping the
lapse from the metric (writing ``N = 1`` from the start) loses it,
which is why it has to be carried through the whole reduction and
only set to 1 at the very end.

Because ``L`` contains ``addot`` for curvature-based gravity, the
reduction integrates by parts first, discarding the resulting
total time derivative -- legitimate exactly when ``L`` is *linear*
in ``addot``, which holds for General Relativity, a non-minimally
coupled ``F(phi) R``, and every ``f(T)`` / ``f(Q)`` (whose scalars
contain no ``addot`` at all). It fails for a general ``f(R)``,
which is genuinely fourth-order; :func:`reduce_order` raises
rather than silently discarding a term that is not a total
derivative, and :mod:`~theory.curvature` handles that case by
making ``R`` an independent variable first -- after which the
reduction here applies again unchanged.

Units
-----
Everything here is dimensionless: ``kappa = 8 pi G = 1``,
``H0 = 1``, time in units of ``1/H0``. In these units the GR
Friedmann equation is ``3 H^2 = rho``, so a fluid with today's
density parameter ``Omega_i`` has ``rho_i0 = 3 Omega_i``, and
spatial curvature enters as ``k = -Omega_k``.

Sign conventions
----------------
The three geometry scalars carry different signs in the
literature, and picking one at random silently flips the sense of
every modification. They are fixed here by one requirement: the
*undeformed* action ``f = R``, ``f = T`` or ``f = Q`` must
reproduce General Relativity exactly. That gives

    R = 6 [ addot/(a N^2) + adot^2/(a^2 N^2)
            - adot Ndot/(a N^3) + k/a^2 ],   L_g = +N a^3 f(R)/2

    T = -6 adot^2/(a^2 N^2)      (= -6 H^2),  L_g = +N a^3 f(T)/2

    Q = +6 adot^2/(a^2 N^2)      (= +6 H^2),  L_g = -N a^3 f(Q)/2

and ``test_theory.py`` asserts the GR limit for all three rather
than trusting the table.
"""

from __future__ import annotations

import sympy as sp


#: The three geometry sectors this module can reduce.
GEOMETRIES = ("metric", "teleparallel", "symmetric")

#: Symbol each sector's gravitational Lagrangian is a function of.
GEOMETRY_SCALAR = {
    "metric": "R",
    "teleparallel": "T",
    "symmetric": "Q",
}


# ============================================================
# Coordinates
# ============================================================

class Minisuperspace:
    """
    The symbols and functions of one FLRW minisuperspace.

    Holds the time coordinate, the lapse ``N(t)``, the scale
    factor ``a(t)``, the curvature constant ``k`` and any scalar
    fields, so the pieces below all talk about the same objects.

    Parameters
    ----------
    fields : sequence of str, optional
        Names of scalar fields living on this background. Each
        becomes a ``sympy.Function`` of ``t``.

    curvature : bool, optional
        Carry the Ricci scalar as an *independent* dynamical
        variable ``R(t)`` rather than as shorthand for a
        combination of ``a`` and its derivatives. That is what
        makes a general ``f(R)`` reducible -- see
        :mod:`~theory.curvature`.
    """

    def __init__(self, fields=(), curvature: bool = False):

        self.t = sp.Symbol("t", real=True)

        self.N = sp.Function("N", positive=True)(self.t)
        self.a = sp.Function("a", positive=True)(self.t)

        self.k = sp.Symbol("k", real=True)

        self.fields = {
            name: sp.Function(name, real=True)(self.t)
            for name in fields
        }

        self.R = sp.Function("R", real=True)(self.t) if curvature else None

    # ---------------------------------------------------------

    @property
    def coordinates(self):
        """
        The dynamical coordinates: ``a``, every field, and the
        curvature variable where there is one.
        """

        extra = () if self.R is None else (self.R,)

        return (self.a, *self.fields.values(), *extra)

    # ---------------------------------------------------------

    def d(self, expr, n: int = 1):
        """``n``-th time derivative of ``expr``."""

        return sp.diff(expr, self.t, n)

    # ---------------------------------------------------------

    def geometry_scalar(self, geometry: str):
        """
        The curvature / torsion / non-metricity scalar of this
        minisuperspace, in the sign convention fixed in the module
        docstring.

        Returns
        -------
        (expr, sign) : (sympy expression, int)
            ``expr`` is the scalar; ``sign`` multiplies the
            gravitational Lagrangian ``N a^3 f(scalar) / 2`` so
            that an undeformed ``f`` gives General Relativity.
        """

        if geometry not in GEOMETRIES:
            raise ValueError(
                f"Unknown geometry {geometry!r}. "
                f"Expected one of {GEOMETRIES}."
            )

        t, N, a, k = self.t, self.N, self.a, self.k

        adot = sp.diff(a, t)

        if geometry == "metric":

            R = 6 * (
                sp.diff(a, t, 2) / (a * N**2)
                + adot**2 / (a**2 * N**2)
                - adot * sp.diff(N, t) / (a * N**3)
                + k / a**2
            )

            return R, 1

        # Both teleparallel scalars are built from the same
        # first-derivative combination and differ only by sign;
        # the compensating sign on the Lagrangian is what keeps
        # `f = T` and `f = Q` both reducing to GR. Neither is
        # defined for a curved FLRW background in the gauges used
        # here, which `Action` checks before it gets this far.
        six_h2 = 6 * adot**2 / (a**2 * N**2)

        if geometry == "teleparallel":
            return -six_h2, 1

        return six_h2, -1


# ============================================================
# Building the point-like Lagrangian
# ============================================================

def fluid_lagrangian(ms: Minisuperspace, densities: dict) -> sp.Expr:
    """
    Minisuperspace Lagrangian of a set of perfect fluids.

    A fluid with constant equation of state ``w`` has
    ``rho(a) = rho_0 a^{-3(1+w)}``, and contributes
    ``-sqrt(-g) rho = -N a^3 rho(a) = -N rho_0 a^{-3w}``.
    Pressureless matter (``w = 0``) therefore contributes a term
    with no scale-factor dependence at all, and radiation
    (``w = 1/3``) one going as ``1/a`` -- the familiar forms.

    Parameters
    ----------
    densities : dict
        Maps ``w`` (as a sympy-compatible number) to today's
        density ``rho_0`` (in units where ``3 H0^2 = 1``, i.e.
        ``rho_0 = 3 Omega``).
    """

    N, a = ms.N, ms.a

    return -N * sum(
        rho0 * a ** (-3 * sp.nsimplify(w))
        for w, rho0 in densities.items()
    )


# ------------------------------------------------------------

def gravity_lagrangian(
    ms: Minisuperspace,
    geometry: str,
    f: sp.Expr,
    scalar_symbol: sp.Symbol,
) -> sp.Expr:
    """
    Minisuperspace Lagrangian of the gravitational sector,
    ``sign * N a^3 f(scalar) / 2`` with the scalar substituted for
    its FLRW expression (``kappa = 1``).

    ``f`` is given as an expression in ``scalar_symbol`` (plus any
    model parameters and fields); the substitution happens here so
    that everything downstream differentiates through ordinary
    sympy chain rules, with no abstract ``Function`` left to trip
    over.
    """

    scalar_expr, sign = ms.geometry_scalar(geometry)

    f_of_a = f.subs(scalar_symbol, scalar_expr)

    return sign * ms.N * ms.a**3 * f_of_a / 2


# ------------------------------------------------------------

def field_lagrangian(ms: Minisuperspace, lagrangians: dict) -> sp.Expr:
    """
    Minisuperspace Lagrangian of the scalar-field sector.

    Each field's Lagrangian density is given as an expression in
    the field itself and its kinetic scalar ``X``; on FLRW,

        X = -g^{mu nu} d_mu phi d_nu phi / 2 = phidot^2 / (2 N^2),

    so a canonical field is ``X - V(phi)`` and a k-essence one is
    any other function of ``(X, phi)``. The contribution to the
    point-like Lagrangian is ``sqrt(-g) L = N a^3 L(X, phi)``.

    Parameters
    ----------
    lagrangians : dict
        Maps field name -> ``(lagrangian_expr, X_symbol)``.
    """

    N, a, t = ms.N, ms.a, ms.t

    total = sp.Integer(0)

    for name, (expr, X) in lagrangians.items():

        phi = ms.fields[name]

        X_expr = sp.diff(phi, t) ** 2 / (2 * N**2)

        # The field arrives as a plain symbol, because that is how
        # it was written in the user's expression. It has to become
        # the *function of time* this minisuperspace carries before
        # anything is varied -- left as a symbol, a potential
        # V(phi) would differentiate to zero and the Klein-Gordon
        # equation would come out wrong, with the Friedmann
        # constraint still looking entirely reasonable.
        total += N * a**3 * expr.subs({X: X_expr, sp.Symbol(name): phi})

    return total


# ============================================================
# Reduction and variation
# ============================================================

def reduce_order(L: sp.Expr, ms: Minisuperspace) -> sp.Expr:
    """
    Remove second time derivatives from ``L`` by integrating by
    parts and discarding the total derivative.

    For a Lagrangian linear in ``qddot``, writing
    ``L = A qddot + B`` and dropping ``d(A qdot)/dt`` leaves
    ``L - A qddot - Adot qdot``, which has the same equations of
    motion. This is what turns the Einstein-Hilbert term (which
    contains ``addot``) into the familiar ``-3 a adot^2 / N``.

    Raises
    ------
    ValueError
        If ``L`` is *nonlinear* in some ``qddot`` -- the case of a
        general ``f(R)``, which is genuinely fourth-order. Then
        the discarded piece is not a total derivative and dropping
        it would quietly change the theory, so this refuses
        instead.
    """

    t = ms.t

    L = sp.expand(L)

    for q in ms.coordinates:

        qddot = sp.diff(q, t, 2)

        if sp.diff(L, qddot) == 0:
            continue

        if sp.simplify(sp.diff(L, qddot, 2)) != 0:
            raise ValueError(
                f"The Lagrangian is nonlinear in {qddot}, so it "
                f"cannot be reduced to second order by parts -- "
                f"the term that would be dropped is not a total "
                f"derivative.\n\n"
                f"A general f(R) reaches this and is handled: "
                f"`theory.curvature` promotes R to an independent "
                f"variable held to its geometric value by a "
                f"Lagrange multiplier, which makes the Lagrangian "
                f"linear in addot again, and `Action` routes there "
                f"automatically. Seeing this message means "
                f"something else produced the nonlinearity."
            )

        A = sp.diff(L, qddot)

        L = sp.expand(L - A * qddot - sp.diff(A, t) * sp.diff(q, t))

    return L


# ------------------------------------------------------------

def euler_lagrange(L: sp.Expr, q, t) -> sp.Expr:
    """Euler-Lagrange expression ``dL/dq - d/dt (dL/dqdot)``."""

    return sp.diff(L, q) - sp.diff(sp.diff(L, sp.diff(q, t)), t)


# ------------------------------------------------------------

def friedmann_constraint(L: sp.Expr, ms: Minisuperspace) -> sp.Expr:
    """
    The Friedmann equation: vary ``L`` with respect to the lapse,
    then fix the gauge ``N = 1``.

    Returned as an expression that vanishes on-shell (the equation
    is ``constraint == 0``). The overall normalization is whatever
    the variation produces -- only the zero set matters -- so a
    comparison against a textbook form should allow an arbitrary
    non-zero factor.
    """

    N, t = ms.N, ms.t

    varied = euler_lagrange(L, N, t)

    return sp.simplify(
        varied.subs({sp.diff(N, t): 0, N: 1}).doit()
    )


# ------------------------------------------------------------

def field_equations(L: sp.Expr, ms: Minisuperspace) -> dict:
    """
    Equation of motion for each scalar field, in the gauge
    ``N = 1``. Each vanishes on-shell.
    """

    N, t = ms.N, ms.t

    equations = {}

    for name, phi in ms.fields.items():

        eq = euler_lagrange(L, phi, t)

        equations[name] = sp.simplify(
            eq.subs({sp.diff(N, t): 0, N: 1}).doit()
        )

    return equations
