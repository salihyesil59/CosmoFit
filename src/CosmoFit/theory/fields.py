"""
Integrating an action that carries dynamical scalar fields.

:mod:`~theory.solve` handles actions whose Friedmann constraint
determines ``E(z)`` on its own -- ``f(T)``, ``f(Q)``, General
Relativity with a cosmological constant. Add a scalar field and
that stops being true: the constraint now involves the field and
its velocity, which have their own equation of motion, and the
expansion history has to be *integrated* rather than solved
pointwise.

The system
----------
In e-folds ``N = ln a``, with ``u = dphi/dN``, the state is
``(H, phi, u)`` and the equations come out of what
:mod:`~theory.minisuperspace` already derived, with no further
physics typed in:

* the Friedmann constraint ``C(a, H, phi, u) = 0``
* one Euler-Lagrange equation per field

The trick is that the constraint is a *first integral* of the
equations of motion -- so instead of separately deriving the
acceleration equation (varying ``a``), this differentiates the
constraint along the solution,

    dC/dN = a dC/da + (dC/dH) H' + sum_i [ (dC/dphi_i) u_i
                                           + (dC/du_i) u_i' ] = 0,

and solves that together with the field equations for the
unknowns ``H'`` and ``u_i'``. The two routes are equivalent by the
Bianchi identity, and this one reuses expressions already in hand.
It also leaves the constraint as an *independent* check on the
integration: it is imposed only in the initial conditions, so how
far it drifts along the solution measures the error.

Initial conditions, and why they are set early
----------------------------------------------
Each field contributes two parameters -- ``phi_i`` and ``dphi_i``,
its value and ``dphi/dN`` -- imposed at an early time
``z = z_init``, and the system is integrated *forwards* from
there.

Setting them at ``a = 1`` instead would be far more convenient:
``H = 1`` holds there by the definition of ``H0``, so the closure
condition ``E(0) = 1`` would be algebraic and no shooting would be
needed at all. It is also wrong. Integrating backwards from a
field at rest today, the Hubble friction ``-3u`` that damps the
field forwards in time becomes anti-friction, the generic past
solution is kinetic-dominated, and ``rho_phi`` grows as ``a^-6``:
for exponential quintessence with the potential normalized to
today's dark-energy density, that gives ``E(2.5) = 9.3`` where
Lambda-CDM gives 3.7. Nothing about the calculation fails -- it is
the correct past of those initial conditions, and those initial
conditions are not a universe.

So the field's state is specified where a quintessence model
actually specifies it, early and typically frozen
(``dphi_i = 0``), and ``E(0) = 1`` becomes a *shooting* condition
on whichever parameter ``Action(closure=...)`` names. Forwards is
also the numerically stable direction, since the friction term
that amplified error backwards now damps it.

The cost is that closure needs a handful of integrations rather
than one evaluation, and that ``E(z)`` is only defined out to
``z_init`` -- beyond which there is no history, and asking for one
is an error rather than an extrapolation.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from CosmoFit.cosmology.numerics.hermite import hermite_spline


#: Points per e-fold on the grid the history spline is built from.
#: The spline is cubic Hermite with derivatives taken from the
#: right-hand side rather than estimated, so this is about
#: resolving the field's dynamics, not about interpolation order.
_POINTS_PER_EFOLD = 256

#: Minimum grid points, for a history spanning very few e-folds.
_MIN_POINTS = 128

#: Margin in e-folds added whenever the integration range has to
#: be extended, so a sequence of slightly widening requests does
#: not re-integrate every time.
_MARGIN = 0.35

#: Tolerances for the history integration. Tight: this sets the
#: accuracy of every distance the model predicts, and the solve is
#: done once per parameter set rather than per redshift.
_RTOL = 1.0e-13
_ATOL = 1.0e-15

#: Integrator, chosen by measuring against a constant potential --
#: where the answer is exactly Lambda-CDM, so the error is known
#: rather than estimated. Compared at matched *accuracy* rather
#: than matched tolerance, which is the comparison that decides
#: anything: Radau reaches a constraint drift of 1e-11 in 79 ms,
#: and this reaches 7e-12 in 3.4 ms. The implicit methods only
#: looked competitive at equal `rtol`; an explicit high-order one
#: simply wants a tighter tolerance and is far cheaper per step.
_METHOD = "DOP853"


# ============================================================
# Symbolic assembly
# ============================================================

class FieldSystem:
    """
    The equations of motion of an action with dynamical fields,
    compiled to a right-hand side ``solve_ivp`` can integrate.

    Parameters
    ----------
    constraint : sympy expression
        The Friedmann constraint, already written in terms of the
        state symbols.

    equations : list of sympy expressions
        One Euler-Lagrange equation per field, in the same terms.

    state : tuple
        ``(a, H, phis, us)`` -- the symbols the above are written
        in.

    derivatives : tuple
        ``(dH, dus)`` -- the symbols standing for ``dH/dN`` and
        each ``du_i/dN``.

    args : tuple of Symbol
        Parameter symbols, in the order the compiled callables
        take them.
    """

    def __init__(
        self, constraint, equations, state, derivatives, args,
        densities=None,
    ):

        a, H, phis, us = state
        dH, dus = derivatives

        # The constraint differentiated along the solution: da/dN
        # is a, dphi_i/dN is u_i, and the remaining derivatives are
        # the unknowns.
        along = a * sp.diff(constraint, a) + sp.diff(constraint, H) * dH

        for phi, u, du in zip(phis, us, dus):
            along += sp.diff(constraint, phi) * u + sp.diff(constraint, u) * du

        unknowns = (dH, *dus)

        solution = sp.solve([along, *equations], unknowns, dict=True)

        if len(solution) != 1:
            raise NotImplementedError(
                f"Could not solve this action's equations of "
                f"motion for {[str(s) for s in unknowns]} "
                f"({len(solution)} solution(s) found). The system "
                f"is linear in those for a canonical or "
                f"non-minimally coupled scalar; a Lagrangian with "
                f"a more elaborate dependence on the kinetic term "
                f"can fail here."
            )

        solution = solution[0]

        signature = (a, H, *phis, *us, *args)

        self.rhs = sp.lambdify(
            signature,
            [solution[dH], *us, *[solution[du] for du in dus]],
            "numpy",
            cse=True,
        )

        self.constraint = sp.lambdify(signature, constraint, "numpy")

        # H at the initial time follows from the constraint once
        # the field's state there is given. Solved symbolically
        # when the constraint allows it (it is quadratic in H for
        # a canonical field), and left to a numerical solve
        # otherwise. Both roots are kept: declaring `H` positive
        # does not make sympy drop the negative branch, so which
        # one is physical is decided on the values.
        self.initial_H = self._compile_initial_H(
            constraint, a, H, phis, us, args,
        )

        self._constraint_slope = sp.lambdify(
            signature, sp.diff(constraint, H), "numpy",
        )

        #: Kept symbolically as well: the closure condition
        #: `E(0) = 1` is this same constraint at `a = 1, H = 1`,
        #: which is how a field action reuses `Action(closure=...)`
        #: unchanged.
        self.symbolic_constraint = constraint
        self.symbols = (a, H, tuple(phis), tuple(us))

        # Energy density and pressure of the field sector, when
        # the caller supplied them. Worth carrying separately: the
        # generic route to a dark-energy density -- subtracting
        # the fluids from E^2 -- cancels catastrophically once
        # matter dominates. At z = 2000 it is 2.4e9 minus 2.4e9 to
        # get 0.7, which loses nine digits and turns w(z) into
        # noise. Read off the Lagrangian instead, there is nothing
        # to cancel.
        if densities is None:
            self.rho = self.pressure = None

        else:
            rho, pressure = densities
            self.rho = sp.lambdify(signature, rho, "numpy", cse=True)
            self.pressure = sp.lambdify(signature, pressure, "numpy", cse=True)

        self.n_fields = len(phis)

    # ---------------------------------------------------------

    @staticmethod
    def _compile_initial_H(constraint, a, H, phis, us, args):

        try:
            roots = sp.solve(constraint, H, dict=False)

        except (NotImplementedError, sp.PolynomialError, TypeError):
            return None

        roots = [r for r in roots if not r.has(sp.I)]

        if not roots:
            return None

        signature = (a, *phis, *us, *args)

        return [sp.lambdify(signature, r, "numpy") for r in roots]

    # ---------------------------------------------------------

    def H_at(self, a, fields, velocities, values) -> float:
        """
        ``H`` at scale factor ``a`` for a given field state, from
        the Friedmann constraint.
        """

        if self.initial_H is not None:

            candidates = [
                float(root(a, *fields, *velocities, *values))
                for root in self.initial_H
            ]

            physical = [
                H for H in candidates if np.isfinite(H) and H > 0.0
            ]

            if len(physical) == 1:
                return physical[0]

            if not physical:
                raise RuntimeError(
                    f"The Friedmann constraint has no positive "
                    f"solution for H at z_init with these "
                    f"parameters (roots: {candidates}) -- the "
                    f"field's kinetic term has overtaken it, "
                    f"which is a statement about the model rather "
                    f"than a bug. Narrow the prior bounds."
                )

            raise RuntimeError(
                f"The Friedmann constraint has more than one "
                f"positive solution for H at z_init "
                f"({physical}), so which expansion history this "
                f"action means is ambiguous there."
            )

        # No symbolic solution: Newton on the constraint, from
        # what the fluids alone would give.
        H = 1.0 / (a * np.sqrt(a))

        for _ in range(80):

            f = float(self.constraint(a, H, *fields, *velocities, *values))
            df = float(
                self._constraint_slope(a, H, *fields, *velocities, *values)
            )

            if df == 0.0 or not np.isfinite(df):
                break

            step = f / df
            H -= step

            if abs(step) <= 1.0e-14 * max(abs(H), 1.0):
                return H

        raise RuntimeError(
            "Could not solve the Friedmann constraint for H at "
            "z_init with these parameters."
        )


# ------------------------------------------------------------

def build_system(action) -> tuple[FieldSystem, list]:
    """
    Assemble the :class:`FieldSystem` of an :class:`~theory.action.Action`.

    Everything here is substitution: the reduction and the
    variation already happened in
    :mod:`~theory.minisuperspace`. What changes is the independent
    variable, from coordinate time to e-folds, via

        adot = a H,   addot = a (H^2 + H H'),
        phidot = H u,  phiddot = H u H' + H^2 u'

    with ``'`` meaning ``d/dN``.
    """

    from CosmoFit.theory.minisuperspace import (
        field_equations,
        friedmann_constraint,
    )

    ms, L = action.lagrangian()

    a = sp.Symbol("a", positive=True)
    H = sp.Symbol("H", positive=True)
    dH = sp.Symbol("dH")

    phis, us, dus, replacements = [], [], [], {}

    for name in ms.fields:

        phi = ms.fields[name]

        phi_s = sp.Symbol(name)
        u = sp.Symbol(f"u_{name}")
        du = sp.Symbol(f"du_{name}")

        phis.append(phi_s)
        us.append(u)
        dus.append(du)

        replacements[sp.diff(phi, ms.t, 2)] = H * u * dH + H**2 * du
        replacements[sp.diff(phi, ms.t)] = H * u
        replacements[phi] = phi_s

    replacements[sp.diff(ms.a, ms.t, 2)] = a * (H**2 + H * dH)
    replacements[sp.diff(ms.a, ms.t)] = a * H
    replacements[ms.a] = a

    def to_state(expr):

        expr = expr.subs(replacements, simultaneous=True)

        return sp.expand(expr.subs(ms.k, -action.namespace["Omega_k"]))

    constraint = to_state(friedmann_constraint(L, ms))

    if constraint.has(dH) or any(constraint.has(du) for du in dus):
        raise NotImplementedError(
            "This action's Friedmann constraint is not first-order "
            "in the state variables, which the integration assumes."
        )

    equations = [
        to_state(eq) for eq in field_equations(L, ms).values()
    ]

    args = action.solver_arguments(
        set().union(
            constraint.free_symbols,
            *(eq.free_symbols for eq in equations),
        )
        - {a, H, dH, *phis, *us, *dus}
    )

    # Energy density and pressure of each field, from its own
    # Lagrangian: for any L(X, phi) the pressure *is* L and the
    # density is 2 X L_X - L. For a canonical X - V that gives the
    # familiar X + V and X - V.
    rho = sp.Integer(0)
    pressure = sp.Integer(0)

    for name, (expr, X) in action.field_lagrangians.items():

        phi_s = sp.Symbol(name)
        u = sp.Symbol(f"u_{name}")

        on_shell = {X: H**2 * u**2 / 2, phi_s: phi_s}

        L_field = expr.subs(on_shell)
        L_X = sp.diff(expr, X).subs(on_shell)

        pressure += L_field
        rho += 2 * (H**2 * u**2 / 2) * L_X - L_field

    return FieldSystem(
        constraint, equations, (a, H, phis, us), (dH, dus), args,
        densities=(rho, pressure),
    ), args


# ============================================================
# Integration
# ============================================================

class History:
    """
    A solved expansion history: ``H`` and ``dH/dN`` as functions of
    ``N = ln a``, over the range that has been integrated so far.

    Held as a cubic Hermite spline with derivatives taken from the
    equations of motion rather than estimated from the samples --
    the same construction the rest of the library uses for its
    distance integrals.
    """

    def __init__(self, N_lo, N_hi, splines, drift):

        self.N_lo = N_lo
        self.N_hi = N_hi

        #: One spline per state variable: H first, then each
        #: field, then each field velocity. The fields are kept
        #: because the dark-energy density and pressure are read
        #: off them rather than off E(z).
        self.splines = splines

        spline = splines[0]

        self.spline = spline

        # Not an independently interpolated dH/dN: the derivative
        # of the very spline `H` is read from. That keeps dE/dz
        # exactly consistent with the E it belongs to -- which is
        # what `background.q()` needs -- and avoids interpolating
        # dH/dN off second-order finite-difference slopes, which
        # measurably dominated the error when it was done that way.
        self.derivative_spline = spline.derivative()

        #: Largest relative drift of the Friedmann constraint over
        #: the solution. The constraint is imposed only in the
        #: initial conditions, so this is an independent measure of
        #: the integration error.
        self.drift = drift

    # ---------------------------------------------------------

    def covers(self, N_lo: float, N_hi: float) -> bool:

        return N_lo >= self.N_lo and N_hi <= self.N_hi

    # ---------------------------------------------------------

    def H(self, N):
        return self.spline(N)

    def dH_dN(self, N):
        return self.derivative_spline(N)

    def state(self, N):
        """Every state variable at ``N``, in the solver's order."""

        return [spline(N) for spline in self.splines]


# ------------------------------------------------------------

def _flow(system, values):
    """The right-hand side as ``solve_ivp`` wants it."""

    def rhs(N, y):
        return system.rhs(np.exp(N), *y, *values)

    return rhs


# ------------------------------------------------------------

def initial_state(system, values, fields, velocities, a_i):
    """
    The state vector ``[H, *phis, *us]`` at ``a = a_i``, with
    ``H`` taken from the Friedmann constraint.
    """

    H = system.H_at(a_i, fields, velocities, values)

    return np.array([H, *fields, *velocities], dtype=float)


# ------------------------------------------------------------

def expansion_today(system, values, fields, velocities, a_i) -> float:
    """
    ``H`` at ``a = 1`` from integrating forwards out of ``a_i``.

    This is the residual the closure condition drives to 1, so it
    runs without ``t_eval`` -- only the endpoint is wanted, and
    asking for samples along the way would pay for output nobody
    reads.
    """

    solution = solve_ivp(
        _flow(system, values),
        (np.log(a_i), 0.0),
        initial_state(system, values, fields, velocities, a_i),
        method=_METHOD,
        rtol=_RTOL,
        atol=_ATOL,
    )

    if not solution.success:
        raise RuntimeError(
            f"Could not integrate this action's equations of "
            f"motion up to the present: {solution.message}"
        )

    return float(solution.y[0, -1])


# ------------------------------------------------------------

def integrate(system, values, fields, velocities, a_i, N_hi) -> History:
    """
    Integrate the field system forwards from ``a_i`` and build the
    interpolating history.

    Parameters
    ----------
    system : FieldSystem

    values : tuple
        Parameter values, in the order ``system`` expects.

    fields, velocities : sequence of float
        The field state at ``a_i``.

    a_i : float
        Initial scale factor, ``1/(1 + z_init)``.

    N_hi : float
        Upper end of ``N = ln a`` to cover. Positive when the
        future is wanted (``w(z)`` and the deceleration parameter
        are plotted there).
    """

    N_lo = np.log(a_i)
    N_hi = max(N_hi, 0.0) + _MARGIN

    n = max(_MIN_POINTS, int(_POINTS_PER_EFOLD * (N_hi - N_lo)))

    solution = solve_ivp(
        _flow(system, values),
        (N_lo, N_hi),
        initial_state(system, values, fields, velocities, a_i),
        method=_METHOD,
        t_eval=np.linspace(N_lo, N_hi, n),
        rtol=_RTOL,
        atol=_ATOL,
    )

    if not solution.success:
        raise RuntimeError(
            f"Could not integrate this action's equations of "
            f"motion over N = [{N_lo:.3g}, {N_hi:.3g}]: "
            f"{solution.message} A scalar field can genuinely run "
            f"away for some parameter values -- the kinetic term "
            f"overtaking the constraint -- which is a statement "
            f"about the model, not a bug. Narrow the prior bounds."
        )

    N, Y = solution.t, solution.y

    a = np.exp(N)

    derivatives = np.asarray(system.rhs(a, *Y, *values), dtype=float)

    H = Y[0]

    if not np.all(np.isfinite(H)) or np.any(H <= 0.0):
        raise RuntimeError(
            "This action's expansion history became non-physical "
            "(H reached zero or stopped being finite) inside the "
            "requested redshift range."
        )

    residual = np.abs(
        np.asarray(system.constraint(a, *Y, *values), dtype=float)
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
