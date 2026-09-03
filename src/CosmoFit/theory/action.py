"""
User-facing action specification, and the compiler that turns one
into a fittable :class:`~cosmology.core.base.Cosmology`.

:mod:`~theory.minisuperspace` does the physics -- it reduces an
action on FLRW to a Friedmann constraint. This module wraps that
in something a user can actually write down, and closes the loop
by solving the constraint for ``E(z)`` and handing back a model
class that every existing dataset, likelihood, fitter and plot in
CosmoFit already knows how to use.

The distinction from :func:`cosmology.custom.define_model` is
where the derivation happens. ``define_model`` takes ``E(z)``,
which means the user has already done the variational calculus by
hand. :class:`Action` takes the action, and does that calculus.

Example
-------
Lambda-CDM, written as an action rather than as an answer:

>>> from CosmoFit.theory import Action
>>> LCDM = Action("R - 2*Lam", closure="Lam").build("LCDM_from_action")
>>> c = LCDM(H0=67.4, Omega_m=0.315)
>>> c.E(0.5)
1.2795...

The exponential ``f(Q)`` model of Anagnostopoulos, Basilakos &
Saridakis, whose Friedmann equation this rederives:

>>> FQ = Action(
...     "Q * exp(lam * Q0 / Q)",
...     geometry="symmetric",
...     params={"lam": {"default": 0.3}},
...     closure="lam",
...     growth="quasi_static",
... ).build("FQ_from_action")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from CosmoFit.cosmology.core.base import Cosmology
from CosmoFit.cosmology.core.errors import ModelConfigurationError
from CosmoFit.cosmology.core.parameters import CosmologyParameters
from CosmoFit.cosmology.core.utils import (
    TELEPARALLEL_CONDITIONS,
    coupling_from_derivative,
    teleparallel_failures,
)

from CosmoFit.theory.curvature import (
    VIABILITY_CONDITIONS,
    build_system as build_curvature_system,
    integrate as integrate_curvature,
    integrate_forward as integrate_curvature_forward,
    is_higher_order,
    quasi_static_mu,
    screening_margin,
    viability_failures,
)

from CosmoFit.theory.fields import expansion_today, integrate

from CosmoFit.theory.solve import (
    closed_form,
    compile_constraint,
    dE2_dz,
    solve_E2,
)

from CosmoFit.theory.minisuperspace import (
    GEOMETRIES,
    GEOMETRY_SCALAR,
    Minisuperspace,
    field_lagrangian,
    fluid_lagrangian,
    friedmann_constraint,
    gravity_lagrangian,
    reduce_order,
)


# ============================================================
# Fluids
# ============================================================

@dataclass(frozen=True)
class Fluid:
    """
    A perfect fluid with constant equation of state.

    Parameters
    ----------
    w : float
        Equation of state ``p / rho``. ``0`` for pressureless
        matter, ``1/3`` for radiation.

    parameter : str
        Name of the cosmological parameter carrying this fluid's
        density parameter today. Its value enters the action as
        ``rho_0 = 3 Omega`` (see the units note in
        :mod:`~theory.minisuperspace`).
    """

    w: float
    parameter: str


#: Fluids available by name in ``Action(fluids=...)``.
STANDARD_FLUIDS = {
    "matter": Fluid(w=0.0, parameter="Omega_m"),
    "radiation": Fluid(w=1.0 / 3.0, parameter="Omega_r"),
}


# ============================================================
# Symbol namespace
# ============================================================

#: Functions a gravity/field expression may call.
#:
#: The inverse trigonometric and hyperbolic ones are here because
#: published models use them: the arctan f(R) family
#: (arXiv:1601.07928, arXiv:1310.6915) and the arcsin one
#: (arXiv:1507.04927) cannot be written without them, and leaving
#: them out silently pushed such an action into a name error rather
#: than a physics answer.
_SAFE_FUNCTIONS = {
    name: getattr(sp, name)
    for name in (
        "sqrt", "exp", "log", "sin", "cos", "tan",
        "sinh", "cosh", "tanh", "Abs", "sign",
        "asin", "acos", "atan",
        "asinh", "acosh", "atanh",
    )
}


#: Mathematical constants an expression may name. Only `pi`: it is
#: needed to normalise a bounded correction (an arctan saturates at
#: pi/2, so writing the saturated value costs a `pi`), and writing
#: `3.14159...` instead would be both uglier and less exact. Sympy's
#: other one-letter constants are deliberately left out -- `E` and
#: `S` in particular read as parameter names and would shadow them.
_SAFE_CONSTANTS = {"pi": sp.pi}


#: Identifiers in an expression string: a name not preceded by a
#: word character or a dot, so the `e` of `1e-5` and the `b` of
#: `a.b` are not mistaken for one.
_NAME = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


def _names(expr: str) -> set:
    """Every identifier mentioned in an expression string."""

    return set(_NAME.findall(expr))


def _sympify(expr: str, namespace: dict) -> sp.Expr:
    """
    Parse a user expression string with only ``namespace`` and
    :data:`_SAFE_FUNCTIONS` in scope.

    Like :func:`cosmology.custom._compile_expression`, this is a
    convenience for trusted local use, not a hardened sandbox --
    but it is markedly narrower than ``eval``, since sympy never
    executes the string as Python.
    """

    local = dict(_SAFE_FUNCTIONS)
    local.update(_SAFE_CONSTANTS)
    local.update(namespace)

    # Checked *before* parsing. sympify falls back to its own
    # global namespace for anything it does not recognize, where
    # plenty of ordinary-looking names (`beta`, `gamma`, `E`, `S`)
    # are special functions -- so an undeclared parameter would
    # otherwise become a sympy object and fail much later with a
    # type error that says nothing about the real mistake.
    unknown = sorted(
        _names(expr) - set(local)
    )

    if unknown:
        raise ValueError(
            f"Unknown name(s) in expression {expr!r}: "
            f"{unknown}. Available: "
            f"{sorted(namespace)} plus "
            f"{sorted(_SAFE_FUNCTIONS)} and "
            f"{sorted(_SAFE_CONSTANTS)}. Declare any new model "
            f"parameter in Action(params=...)."
        )

    return sp.sympify(expr, locals=local)


# ============================================================
# Action
# ============================================================

class Action:
    """
    A gravitational action on an FLRW background, and the
    machinery to turn it into a fittable model.

    Parameters
    ----------
    gravity : str
        The gravitational Lagrangian ``f``, as an expression in
        the geometry scalar (``R``, ``T`` or ``Q``), the
        cosmological parameters, and any parameters declared in
        ``params``. ``R0``/``T0``/``Q0`` are available as the
        scalar's value today for a matter-free de Sitter
        normalization (``T0 = -6``, ``Q0 = 6`` in ``H0 = 1``
        units), which is how the ``f(Q)`` literature writes its
        models.

        Written in the sign convention of
        :mod:`~theory.minisuperspace`: an undeformed ``f`` (just
        ``"R"``, ``"T"`` or ``"Q"``) is exactly General Relativity.

    geometry : {"metric", "teleparallel", "symmetric"}, optional
        Which geometric formulation the action is built on --
        curvature, torsion, or non-metricity. Inferred from which
        scalar symbol appears in ``gravity`` when only one does.

    fluids : sequence, optional
        Matter content, as names from :data:`STANDARD_FLUIDS` or
        :class:`Fluid` instances. Defaults to pressureless matter
        alone.

    params : dict, optional
        Model parameters beyond the standard set, in the same
        ``{name: {"default": ..., "bounds": ..., "label": ...}}``
        form :func:`cosmology.custom.define_model` takes.

    closure : str, optional
        Name of the one parameter fixed by requiring ``E(0) = 1``
        rather than fit. Every consistent model needs this
        condition satisfied somehow: in Lambda-CDM it is what
        makes ``Omega_de0 = 1 - Omega_m - Omega_k``, and in the
        exponential ``f(Q)`` model it is what makes ``lam`` a
        function of ``Omega_m`` rather than a free parameter. If
        the action satisfies it identically, leave this unset;
        :meth:`build` checks and complains either way.

    growth : {"gr", "quasi_static"}, optional
        How the linear growth of structure responds to the
        modification, i.e. what ``mu = G_eff/G_N`` the generated
        model reports.

        ``"gr"`` (default) leaves ``mu = 1``: correct whenever the
        action modifies the background only in the sense that the
        extra gravitational degrees of freedom do not propagate
        into sub-horizon clustering -- and the safe default,
        because it is what every dark-energy model assumes.

        ``"quasi_static"`` asks for the sub-horizon quasi-static
        result, which depends on what the action modified:

        * a deformed geometry scalar (``f(T)``, ``f(Q)``) gives
          ``mu = 1/f'``, what this library's hand-written
          ``FQExponential`` and ``FRTLinear`` already use;
        * a field coupled to the curvature gives the scalar-tensor
          result of Boisseau, Esposito-Farese, Polarski &
          Starobinsky (2000),
          ``mu = (2F + 4 F_phi^2) / (F (2F + 3 F_phi^2))`` with
          ``F = df/dR``, evaluated on the field's own solution so
          that it moves as the field rolls;
        * a general ``f(R)`` -- anything non-linear in ``R`` --
          gives the *scale-dependent*
          ``mu = (1/f_R)(1 + 4m)/(1 + 3m)`` with
          ``m = (k/a)^2 f_RR/f_R``, since the scalaron has a
          Compton wavelength and there is no scale-free answer to
          give. See :func:`~theory.curvature.quasi_static_mu`; the
          resulting ``mu(a, k)`` takes ``k`` in h/Mpc.

        Either way it is an *additional* physical assumption on top
        of the action -- a statement about perturbations, which a
        background action does not by itself determine -- so it
        must be asked for explicitly. Asking for it where nothing
        is modified is an error rather than a no-op.

    fields : dict, optional
        Scalar fields, mapping name -> Lagrangian density written
        in terms of the field and its kinetic scalar ``X``
        (``"X - V0*exp(-lam*phi)"`` for exponential quintessence,
        or any other ``L(X, phi)`` for k-essence). More than one is
        fine; each gets its own equation of motion.

        A field's name is also in scope in ``gravity``, which is
        how scalar-tensor gravity is written --
        ``"(1 + xi*phi**2)*R"`` couples the field to curvature
        rather than adding it on top of General Relativity, and
        brings the ``3 H dF/dt`` term into the Friedmann equation.

        A field's expansion history is *integrated* rather than
        solved pointwise -- see :mod:`~theory.fields` -- and it
        adds two parameters, ``<name>_i`` and ``d<name>_i``, its
        value and ``dphi/dN`` at ``z_init``. Because those are
        given early rather than today, ``E(0) = 1`` becomes a
        shooting condition, so ``closure`` is required.

    z_init : float, optional
        Redshift at which a field's initial conditions are set,
        and the earliest redshift the resulting model can be
        evaluated at. Default 3000, which covers recombination.
        Ignored by an action with no fields.
    """

    def __init__(
        self,
        gravity: str,
        *,
        geometry: str | None = None,
        fluids=("matter",),
        params: dict | None = None,
        closure: str | None = None,
        growth: str = "gr",
        background: str = "backward",
        fields: dict | None = None,
        z_init: float = 3000.0,
    ):

        self.params = dict(params or {})
        self.fields = dict(fields or {})
        self.closure = closure
        self.growth = growth
        self.background = background
        self.z_init = float(z_init)

        self.fluids = tuple(
            STANDARD_FLUIDS[f] if isinstance(f, str) else f
            for f in fluids
        )

        if background not in ("backward", "forward"):
            raise ValueError(
                f"background must be 'backward' or 'forward', got "
                f"{background!r}. It says which way a fourth-order "
                f"f(R) is integrated, and the right answer is a "
                f"property of the theory -- see "
                f"`theory.curvature.integrate_forward`."
            )

        if growth not in ("gr", "quasi_static"):
            raise ValueError(
                f"growth must be 'gr' or 'quasi_static', "
                f"got {growth!r}."
            )

        # Each dynamical field needs its state at z_init: the
        # field value and dphi/dN there. Declared automatically,
        # because they are not optional -- a field with no initial
        # condition is not a model -- but overridable, since a
        # default and prior bounds for them are exactly what a
        # user may want to set. The default dphi_i = 0 is a frozen
        # field, which is what Hubble friction does to one at
        # early times and how a thawing model is normally posed.
        for name in self.fields:

            self.params.setdefault(f"{name}_i", {"default": 0.0})
            self.params.setdefault(f"d{name}_i", {"default": 0.0})

        # The closure parameter is not fit -- it is solved for --
        # so requiring it to be declared like a free parameter
        # would be noise. Declaring it anyway is still meaningful:
        # its default is where the numerical closure solve starts,
        # and so which branch it lands on when the condition is
        # transcendental.
        if (
            closure is not None
            and closure not in self.params
            and closure not in CosmologyParameters.names()
        ):
            self.params[closure] = {"default": 0.0}

        self.geometry = self._resolve_geometry(gravity, geometry)

        self.scalar = sp.Symbol(GEOMETRY_SCALAR[self.geometry])

        self._namespace = self._build_namespace()

        self.gravity = _sympify(gravity, self._namespace)

        # A general f(R) is fourth-order: R stops being shorthand for
        # a combination of `a` and its derivatives and becomes a
        # variable in its own right, with its own value today. That
        # is a degree of freedom the theory has and General
        # Relativity does not, so it is a real parameter -- declared
        # here because it is not optional, and overridable because
        # its default is only a guess.
        #
        # 9.3 is the Lambda-CDM value at Omega_m = 0.3, in units of
        # H0^2: R = 6 (2 H^2 + Hdot) = 6 (2 - 3 Omega_m / 2) H0^2.
        # Only backwards. Forwards, the curvature today is where the
        # integration lands rather than where it starts, so it is
        # derived and offering it as a parameter would be a lie.
        if self.is_fourth_order and self.background == "backward":

            self.params.setdefault(
                "R_0",
                {
                    "default": 9.3,
                    "bounds": (0.0, 40.0),
                    "label": r"$R_0/H_0^2$",
                },
            )

            self._namespace = self._build_namespace()

        self._field_lagrangians = {
            name: (
                _sympify(
                    expr,
                    {
                        **self._namespace,
                        name: sp.Symbol(name),
                        "X": sp.Symbol("X"),
                    },
                ),
                sp.Symbol("X"),
            )
            for name, expr in self.fields.items()
        }

        self._check_closure_name()

        self._curvature = None

        self._constraint = None
        self._system = None

    # ---------------------------------------------------------

    @staticmethod
    def _resolve_geometry(gravity: str, geometry: str | None) -> str:
        """
        Work out which geometric formulation ``gravity`` is
        written in, from the scalar symbol it mentions.
        """

        if geometry is not None:

            if geometry not in GEOMETRIES:
                raise ValueError(
                    f"geometry must be one of {GEOMETRIES}, "
                    f"got {geometry!r}."
                )

            return geometry

        # Scanned as text rather than parsed: sympify resolves an
        # unrecognized name against sympy's own global namespace
        # (`beta` is a special function there), which turns a
        # user's typo into a confusing type error long before the
        # check in `_sympify` that exists to catch it.
        names = _names(gravity)

        present = [
            g for g, s in GEOMETRY_SCALAR.items() if s in names
        ]

        if len(present) == 1:
            return present[0]

        if not present:
            raise ValueError(
                f"The gravity expression {gravity!r} mentions none "
                f"of the geometry scalars "
                f"{sorted(GEOMETRY_SCALAR.values())}, so there is "
                f"nothing to vary. A pure cosmological constant is "
                f"'R - 2*Lam', not '-2*Lam'."
            )

        raise ValueError(
            f"The gravity expression {gravity!r} mentions more "
            f"than one geometry scalar "
            f"({sorted(GEOMETRY_SCALAR[g] for g in present)}). "
            f"These are different formulations of gravity, not "
            f"terms to be added: pass geometry= explicitly if one "
            f"of them is meant as a parameter name."
        )

    # ---------------------------------------------------------

    def _build_namespace(self) -> dict:
        """
        Every symbol a user expression may mention: the geometry
        scalar and its value today, the standard cosmological
        parameters, and this action's own extra parameters.
        """

        namespace = {
            name: sp.Symbol(name)
            for name in CosmologyParameters.names()
        }

        for name in self.params:

            if name in namespace:
                raise ValueError(
                    f"Parameter {name!r} collides with a standard "
                    f"cosmological parameter. Pick another name."
                )

            namespace[name] = sp.Symbol(name)

        for fluid in self.fluids:
            namespace.setdefault(
                fluid.parameter, sp.Symbol(fluid.parameter)
            )

        # A field's own name, so the gravitational sector can
        # couple to it: `F(phi) R` is scalar-tensor gravity, not a
        # scalar on top of General Relativity, and writing it is
        # the only way to say so.
        for name in self.fields:

            if name in GEOMETRY_SCALAR.values():
                raise ValueError(
                    f"A field cannot be called {name!r}: that is "
                    f"the geometry scalar of the "
                    f"{[g for g, s in GEOMETRY_SCALAR.items() if s == name][0]}"
                    f" sector, and the two would be the same symbol "
                    f"in every expression. Rename the field."
                )

            namespace[name] = sp.Symbol(name)

        scalar_name = GEOMETRY_SCALAR[self.geometry]

        namespace[scalar_name] = self.scalar

        # The scalar's present-day value, in H0 = 1 units. The
        # torsion and non-metricity scalars are 6 H^2 up to sign,
        # so today they are just +-6; the Ricci scalar also
        # involves addot, which is not known before the model is
        # solved, so it is deliberately absent rather than wrong.
        if self.geometry == "teleparallel":
            namespace["T0"] = sp.Integer(-6)

        elif self.geometry == "symmetric":
            namespace["Q0"] = sp.Integer(6)

        return namespace

    # ---------------------------------------------------------

    def _check_closure_name(self) -> None:

        if self.closure is None:
            return

        known = set(self._namespace) | set(CosmologyParameters.names())

        if self.closure not in known:
            raise ValueError(
                f"closure={self.closure!r} is not a parameter of "
                f"this action. Known: {sorted(self.params)} "
                f"(declared) plus the standard set."
            )

    # ---------------------------------------------------------
    # Symbolic layer
    # ---------------------------------------------------------

    @property
    def field_lagrangians(self) -> dict:
        """
        Each field's Lagrangian density, as
        ``{name: (expression, X_symbol)}``.
        """

        return dict(self._field_lagrangians)

    # ---------------------------------------------------------

    def lagrangian(self) -> tuple[Minisuperspace, sp.Expr]:
        """
        The reduced point-like Lagrangian of this action, together
        with the minisuperspace it lives on.
        """

        ms = Minisuperspace(tuple(self.fields))

        # A field appearing in the gravitational sector arrives as
        # a plain symbol, and has to become the function of time
        # this minisuperspace carries before anything is varied --
        # for the same reason `field_lagrangian` does it: left as a
        # symbol, F(phi) would differentiate to zero and the
        # non-minimal coupling would silently disappear while the
        # constraint still looked reasonable.
        gravity = self.gravity.subs(
            {sp.Symbol(name): field for name, field in ms.fields.items()},
            simultaneous=True,
        )

        L = gravity_lagrangian(
            ms, self.geometry, gravity, self.scalar,
        )

        L += fluid_lagrangian(
            ms,
            {
                fluid.w: 3 * self._namespace[fluid.parameter]
                for fluid in self.fluids
            },
        )

        if self._field_lagrangians:
            L += field_lagrangian(ms, self._field_lagrangians)

        return ms, reduce_order(L, ms)

    # ---------------------------------------------------------

    def constraint(self):
        """
        The Friedmann constraint of this action, as an expression
        in ``E2`` (the squared dimensionless Hubble rate) and
        ``z``, vanishing on-shell.

        Returns
        -------
        (expr, E2, z) : (sympy expression, Symbol, Symbol)
        """

        if self._constraint is not None:
            return self._constraint

        ms, L = self.lagrangian()

        C = friedmann_constraint(L, ms)

        E2 = sp.Symbol("E2", positive=True)
        z = sp.Symbol("z", nonnegative=True)

        # adot = a H, and H = E in H0 = 1 units. For an action
        # with no dynamical fields the constraint depends on adot
        # only quadratically, so this leaves a function of E2
        # alone -- which `_require_E2_only` verifies rather than
        # assumes.
        C = C.subs(sp.diff(ms.a, ms.t), ms.a * sp.sqrt(E2))

        C = C.subs(ms.k, -self._namespace["Omega_k"])

        C = sp.simplify(C.subs(ms.a, 1 / (1 + z)))

        self._constraint = (sp.together(C), E2, z)

        return self._constraint

    # ---------------------------------------------------------

    def field_equations(self) -> dict:
        """
        Equation of motion for each scalar field in this action
        (each vanishing on-shell), in the gauge ``N = 1``.

        Provided for inspection -- :meth:`build` integrates this
        system rather than returning it, so this is the way to see
        what was actually derived.
        """

        ms, L = self.lagrangian()

        from CosmoFit.theory.minisuperspace import field_equations

        return field_equations(L, ms)

    # ---------------------------------------------------------
    # Closure
    # ---------------------------------------------------------

    def closure_equation(self):
        """
        The condition ``E(0) = 1``, as an expression in the model
        parameters that must vanish.

        Every model has to satisfy this -- ``H(z=0)`` is ``H0`` by
        definition of ``H0``. What differs is whether the action
        satisfies it identically (nothing to do), or fixes one
        parameter in terms of the others.
        """

        if self.fields:
            raise NotImplementedError(
                "An action with dynamical fields has no algebraic "
                "closure equation. Its field state is set at "
                "z_init and the history integrated forwards, so "
                "E(0) = 1 is a shooting condition on the closure "
                "parameter rather than something that can be "
                "written down and solved -- see CosmoFit.theory."
                "fields for why the state is not set at a = 1 "
                "instead."
            )

        C, E2, z = self.constraint()

        return sp.simplify(C.subs({E2: 1, z: 0}))

    # ---------------------------------------------------------

    def _closure_solution(self):
        """
        Solve :meth:`closure_equation` for the ``closure``
        parameter.

        Returns a sympy expression for it in terms of the other
        parameters, or ``None`` when the equation is
        transcendental in it and has to be solved numerically at
        evaluation time.
        """

        equation = sp.numer(sp.together(self.closure_equation()))

        symbol = self._namespace[self.closure]

        try:
            solutions = sp.solve(equation, symbol, dict=False)

        except (NotImplementedError, sp.PolynomialError, TypeError):
            return None

        solutions = [s for s in solutions if not s.has(sp.I)]

        return solutions[0] if len(solutions) == 1 else None

    # ---------------------------------------------------------
    # Compilation
    # ---------------------------------------------------------

    def build(self, name: str, label: str | None = None) -> type:
        """
        Compile this action into a
        :class:`~cosmology.core.base.Cosmology` subclass.

        The returned class is an ordinary CosmoFit model: pass it
        to :class:`~stats.fitter.Fitter` and every dataset,
        likelihood, sampler and plot works on it unchanged.

        An action carrying dynamical scalar fields is integrated
        rather than solved pointwise -- see
        :mod:`~theory.fields` -- and gains two parameters per
        field, ``<name>0`` and ``d<name>0``, its value and
        ``dphi/dN`` at ``a = 1``.

        Parameters
        ----------
        name : str
            Model name, used for the class and for
            ``MODEL_NAME``.

        label : str, optional
            LaTeX name for figures, as in
            :func:`cosmology.custom.define_model`.
        """

        if self.fields:
            return self._build_with_fields(name, label)

        if self.is_fourth_order:
            return self._build_fourth_order(name, label)

        C, E2, z = self.constraint()

        args = self.solver_arguments(C.free_symbols - {E2, z})

        functions = compile_constraint(C, E2, z, args)

        direct = closed_form(C, E2, z, args)

        closure = self._compile_closure(args)

        mu = self._compile_mu(args)

        extra = {
            key: spec for key, spec in self.params.items()
            if key != self.closure
        }

        attrs = {
            "MODEL_NAME": name,
            "MODEL_LABEL": label,
            "EXTRA_PARAMS": extra,
            "ACTION": self,
            "_ARGS": tuple(s.name for s in args),
            "_FUNCTIONS": staticmethod(functions),
            "_DIRECT": staticmethod(direct) if direct else None,
            "_CLOSURE": staticmethod(closure) if closure else None,
            "_FLUIDS": self.fluids,
            "E": _make_E(),
            "dEdz": _make_dEdz(),
            "Omega_de": _make_Omega_de(),
            "w": _make_w(),
        }

        if self.closure in CosmologyParameters.names():
            attrs["DERIVED_PARAMS"] = frozenset({self.closure})

        if mu is not None:
            attrs["mu"] = _make_mu(mu)
            attrs["coupling"] = _make_coupling(mu)
            attrs["viability"] = _make_teleparallel_viability()

        model = type(name, (_ActionModel,), attrs)

        self._verify(model)

        return model

    # ---------------------------------------------------------

    def field_system(self):
        """
        The compiled equations of motion of this action, for the
        case where it carries dynamical fields.

        Returns ``(system, args)`` -- see
        :func:`theory.fields.build_system`. Cached: assembling it
        means solving a linear system symbolically.
        """

        if self._system is None:

            from CosmoFit.theory.fields import build_system

            self._system = build_system(self)

        return self._system

    # ---------------------------------------------------------

    def curvature_system(self):
        """
        The compiled equations of motion of a general ``f(R)``
        action. Cached: assembling it means a symbolic solve.
        """

        if self._curvature is None:
            self._curvature = build_curvature_system(self)

        return self._curvature

    # ---------------------------------------------------------

    def _build_fourth_order(self, name: str, label: str | None) -> type:
        """
        Compile a general ``f(R)`` action, whose expansion history
        has to be integrated -- see :mod:`~theory.curvature`.
        """

        if self.closure is not None and self.background == "backward":
            raise ValueError(
                f"closure={self.closure!r} was given, but a general "
                f"f(R) action integrated backwards needs no closure "
                f"condition: H = 1 at a = 1 holds by construction, "
                f"because the history is integrated outwards from "
                f"there. The freedom a closure would have fixed is "
                f"carried by R_0, the Ricci scalar today, which is "
                f"a parameter. Pass background='forward' if you "
                f"want the other arrangement, where R_0 is derived "
                f"and the closure is shot for."
            )

        if self.background == "forward":
            return self._build_fourth_order_forward(name, label)

        system, args = self.curvature_system()

        extra = dict(self.params)

        attrs = {
            "MODEL_NAME": name,
            "MODEL_LABEL": label,
            "EXTRA_PARAMS": extra,
            "ACTION": self,
            "_ARGS": tuple(s.name for s in args),
            "_SYSTEM": system,
            "_FLUIDS": self.fluids,
            "E": _make_curvature_E(),
            "dEdz": _make_curvature_dEdz(),
            "Omega_de": _make_Omega_de(),
            "w": _make_w(),
        }

        if self.growth == "quasi_static":

            f_R, f_RR, mu_args = self._compile_fourth_order_mu()

            attrs.update({
                "_F_R": staticmethod(f_R),
                "_F_RR": staticmethod(f_RR),
                "_MU_ARGS": mu_args,
                "mu": _make_curvature_mu(),
                "scalaron": _make_scalaron(),
                "viability": _make_viability(),
                "screening": _make_screening(),
            })

        model = type(name, (_CurvatureModel,), attrs)

        self._verify(model)

        return model

    # ---------------------------------------------------------

    def _build_fourth_order_forward(self, name: str, label: str | None) -> type:
        """
        Compile a general ``f(R)`` integrated *forwards*, from deep
        in matter domination to today -- see
        :func:`~theory.curvature.integrate_forward` for when that is
        the direction a theory needs.

        Two things swap over relative to the backward build. ``R_0``
        is no longer a parameter, because the curvature today is
        where the integration lands rather than where it starts. And
        a closure parameter becomes *required*, because ``E(0) = 1``
        is no longer automatic; it is the condition the closure is
        shot for, exactly as in the second-order case.
        """

        if self.closure is None:
            raise ValueError(
                "background='forward' needs closure=<parameter "
                "name>. Integrating towards today rather than away "
                "from it means E(0) = 1 is a condition to satisfy "
                "rather than a starting point, and the closure "
                "names the parameter that satisfies it. In exchange "
                "R_0 is not a parameter here: on the attractor the "
                "curvature today is not free."
            )

        system, args = self.curvature_system()

        arg_names = tuple(a.name for a in args)

        if self.closure not in arg_names:
            raise ValueError(
                f"closure={self.closure!r} does not appear in the "
                f"background equations ({list(arg_names)}), so "
                f"shooting on it could not change E(0)."
            )

        extra = dict(self.params)

        attrs = {
            "MODEL_NAME": name,
            "MODEL_LABEL": label,
            "EXTRA_PARAMS": extra,
            "ACTION": self,
            "_ARGS": arg_names,
            "_SYSTEM": system,
            "_FLUIDS": self.fluids,
            "_CLOSURE": self.closure,
            "_CLOSURE_BRACKET": self._closure_bracket(),
            "E": _make_curvature_E(),
            "dEdz": _make_curvature_dEdz(),
            "Omega_de": _make_Omega_de(),
            "w": _make_w(),
        }

        if self.growth == "quasi_static":

            f_R, f_RR, mu_args = self._compile_fourth_order_mu()

            attrs.update({
                "_F_R": staticmethod(f_R),
                "_F_RR": staticmethod(f_RR),
                "_MU_ARGS": mu_args,
                "mu": _make_curvature_mu(),
                "scalaron": _make_scalaron(),
                "viability": _make_viability(),
                "screening": _make_screening(),
            })

        model = type(name, (_ForwardCurvatureModel,), attrs)

        self._verify(model)

        return model

    # ---------------------------------------------------------

    def _closure_bracket(self):
        """
        Where to look for the closure value. The declared bounds if
        there are any, since a parameter's bounds are the modeller's
        own statement of where it is meaningful; otherwise a wide
        interval around the default.
        """

        spec = self.params.get(self.closure, {})

        bounds = spec.get("bounds")

        if bounds is not None:
            return (float(bounds[0]), float(bounds[1]))

        default = float(spec.get("default", 1.0))

        span = max(abs(default), 1.0) * 10.0

        return (default - span, default + span)

    # ---------------------------------------------------------

    def _build_with_fields(self, name: str, label: str | None) -> type:
        """
        Compile an action with dynamical scalar fields, whose
        expansion history has to be integrated.
        """

        system, args = self.field_system()

        compiled_mu = self._compile_scalar_tensor_mu()

        mu = None if compiled_mu is None else compiled_mu

        arg_names = tuple(s.name for s in args)
        field_names = tuple(self.fields)

        if self.closure is None:
            raise ValueError(
                "An action with dynamical fields needs "
                "closure=<parameter name>. Its field state is set "
                "at z_init and the expansion history integrated "
                "forwards, so nothing makes H(0) come out as H0 "
                "on its own -- one parameter has to be solved for "
                "by requiring it. For quintessence that is "
                "normally the potential's overall scale."
            )

        closure = _ShootingClosure(
            system=system,
            arg_names=arg_names,
            field_names=field_names,
            parameter=self.closure,
            start=float(
                self.params.get(self.closure, {}).get("default", 1.0)
            ),
            a_i=1.0 / (1.0 + self.z_init),
        )

        extra = {
            key: spec for key, spec in self.params.items()
            if key != self.closure
        }

        attrs = {
            "MODEL_NAME": name,
            "MODEL_LABEL": label,
            "EXTRA_PARAMS": extra,
            "ACTION": self,
            "_ARGS": tuple(s.name for s in args),
            "_SYSTEM": system,
            "_CLOSURE": closure,
            "_FLUIDS": self.fluids,
            "_FIELDS": field_names,
            "_A_INIT": 1.0 / (1.0 + self.z_init),
            "_Z_INIT": self.z_init,
            "_NON_MINIMAL": self.is_non_minimal,
            "E": _make_field_E(),
            "dEdz": _make_field_dEdz(),
            "Omega_de": _make_field_Omega_de(),
            "w": _make_field_w(),
        }

        if self.closure in CosmologyParameters.names():
            attrs["DERIVED_PARAMS"] = frozenset({self.closure})

        if mu is not None:
            attrs["mu"] = _make_field_mu(*mu)

        model = type(name, (_FieldModel,), attrs)

        self._verify(model)

        return model

    # ---------------------------------------------------------

    @property
    def coupling(self) -> sp.Expr:
        """
        ``F``, the coefficient of ``R/2`` in this action -- which is
        ``d f / d R``. Constant for a minimally coupled field,
        and a function of the field for scalar-tensor gravity.
        """

        return sp.diff(self.gravity, self.scalar)

    # ---------------------------------------------------------

    @property
    def is_non_minimal(self) -> bool:
        """
        Whether the gravitational sector couples to a field --
        scalar-tensor gravity, where the field sets the strength of
        gravity rather than sitting on top of it.
        """

        return any(
            sp.Symbol(name) in self.coupling.free_symbols
            for name in self.fields
        )

    # ---------------------------------------------------------

    def _compile_scalar_tensor_mu(self):
        """
        ``mu = G_eff/G_N`` in the sub-horizon quasi-static limit of
        scalar-tensor gravity, when it has been asked for.

        For ``S = (1/2) integral F(phi) R - (1/2)(d phi)^2 - V``,
        Boisseau, Esposito-Farese, Polarski & Starobinsky (2000)
        give

            G_eff = (1 / 8 pi F) (2F + 4 F_phi^2) / (2F + 3F_phi^2)

        so that ``mu`` is ``1`` exactly when ``F = 1`` and the
        coupling is constant. The extra ``F_phi^2`` terms are the
        scalar's own fifth force; the ``1/F`` in front is the
        rescaling of Newton's constant itself.

        Like every other ``quasi_static`` in this module it is a
        statement about perturbations, which a background action
        does not by itself determine, so it is opt-in.
        """

        if self.growth == "gr":
            return None

        if self.geometry != "metric":
            raise NotImplementedError(
                "growth='quasi_static' with scalar fields is the "
                "scalar-tensor result, which is a statement about "
                "the metric sector. In the teleparallel sectors "
                "mu = 1/f' describes the modified gravity alone "
                "and says nothing about a field alongside it."
            )

        if not self.is_non_minimal:
            raise NotImplementedError(
                "growth='quasi_static' has nothing to correct "
                "here: this action's fields are minimally coupled, "
                "so gravity is unmodified and mu = 1 exactly -- "
                "which is what growth='gr' already gives. Couple a "
                "field to the curvature, as in "
                "'(1 + xi*phi**2)*R', for it to mean anything."
            )

        F = self.coupling

        derivatives = [
            sp.diff(F, sp.Symbol(name)) for name in self.fields
        ]

        squared = sum(d**2 for d in derivatives)

        mu = (2 * F + 4 * squared) / (F * (2 * F + 3 * squared))

        system, args = self.field_system()

        fields = [sp.Symbol(name) for name in self.fields]

        # `F` as well as `mu`. With `s = sum (dF/dphi)^2 >= 0` by
        # construction, `F > 0` makes both `2F + 4s` and
        # `F(2F + 3s)` strictly positive, and is the only way they
        # can be: the pole sits at `2F + 3s = 0`, which needs
        # `F = -3s/2 <= 0`. So one number decides whether this
        # expression means anything, and it is worth returning.
        return (
            sp.lambdify((*fields, *args), mu, "numpy"),
            sp.lambdify((*fields, *args), F, "numpy"),
        )

    # ---------------------------------------------------------

    @property
    def is_fourth_order(self) -> bool:
        """
        Whether this action is a general ``f(R)`` -- nonlinear in
        the Ricci scalar, and so fourth-order in the metric.

        Those need the Lagrange-multiplier reduction of
        :mod:`~theory.curvature`; everything else goes through the
        ordinary one.
        """

        return (
            self.geometry == "metric"
            and is_higher_order(self.gravity, self.scalar)
        )

    # ---------------------------------------------------------

    @property
    def namespace(self) -> dict:
        """
        Every symbol a user expression of this action may mention.
        """

        return dict(self._namespace)

    # ---------------------------------------------------------

    def solver_arguments(self, symbols) -> tuple:
        """
        The parameter symbols a compiled expression needs, in a
        fixed order -- and a check that every one of them is
        actually a parameter the generated model will carry.
        """

        free = sorted(symbols, key=lambda s: s.name)

        known = set(CosmologyParameters.names()) | set(self.params)

        missing = [s.name for s in free if s.name not in known]

        if missing:
            raise ValueError(
                f"The constraint depends on {missing}, which "
                f"is neither a standard cosmological parameter "
                f"nor declared in Action(params=...). A "
                f"non-standard fluid needs its density parameter "
                f"declared there -- e.g. radiation needs "
                f"params={{'Omega_r': {{'default': 9e-5}}}}."
            )

        return tuple(free)

    # ---------------------------------------------------------

    def _compile_closure(self, args):
        """
        Build the callable that evaluates the closure parameter,
        or ``None`` if this action closes identically.
        """

        if self.closure is None:
            return None

        symbol = self._namespace[self.closure]

        # Taken from the closure equation rather than from the
        # constraint's own arguments: a field action's closure
        # condition also involves that field's state today, which
        # the constraint (written in state variables) does not
        # carry as a parameter.
        others = tuple(
            sorted(
                self.closure_equation().free_symbols - {symbol},
                key=lambda s: s.name,
            )
        )

        start = float(
            self.params.get(self.closure, {}).get("default", 0.0)
        )

        solution = self._closure_solution()

        if solution is not None:

            explicit = sp.lambdify(others, solution, "numpy")

            return _ClosureSolver(
                names=tuple(s.name for s in others),
                explicit=explicit,
                residual=None,
                parameter=self.closure,
                start=start,
            )

        # Transcendental in the closure parameter (the exponential
        # f(Q) case, whose hand-written counterpart needs a
        # Lambert W): solve it numerically, from the value the
        # user gave as a starting point.
        equation = sp.numer(sp.together(self.closure_equation()))

        return _ClosureSolver(
            names=tuple(s.name for s in others),
            explicit=None,
            residual=(
                sp.lambdify((symbol, *others), equation, "numpy"),
                sp.lambdify(
                    (symbol, *others), sp.diff(equation, symbol), "numpy",
                ),
            ),
            parameter=self.closure,
            start=start,
        )

    # ---------------------------------------------------------

    def _compile_fourth_order_mu(self):
        """
        Compile ``f_R`` and ``f_RR`` for the quasi-static ``mu`` of
        a general ``f(R)`` -- see
        :func:`~theory.curvature.quasi_static_mu` for the formula
        and for what the units have to be.

        Both are functions of ``R`` and of whatever parameters the
        action carries, and ``R`` comes from the integrated
        background rather than from a closed form, which is why
        this is compiled here and evaluated per call.
        """

        R = self.scalar

        f_R = sp.diff(self.gravity, R)
        f_RR = sp.diff(f_R, R)

        free = (f_R.free_symbols | f_RR.free_symbols) - {R}

        args = self.solver_arguments(free)

        return (
            sp.lambdify((R, *args), f_R, "numpy"),
            sp.lambdify((R, *args), f_RR, "numpy"),
            tuple(a.name for a in args),
        )

    # ---------------------------------------------------------

    def _compile_mu(self, args):
        """
        Compile ``mu = G_eff/G_N`` for ``growth="quasi_static"``.

        The sub-horizon quasi-static limit of ``f(T)`` and
        ``f(Q)`` gravity gives ``mu = 1/f'``, the derivative taken
        with respect to the geometry scalar and evaluated on the
        background -- where the scalar is ``-6 E^2`` (torsion) or
        ``+6 E^2`` (non-metricity).
        """

        if self.growth == "gr":
            return None

        if self.geometry == "metric":
            raise NotImplementedError(
                "growth='quasi_static' gives mu = 1/f' in the "
                "teleparallel and symmetric-teleparallel sectors, "
                "and this metric action is linear in R -- which is "
                "general relativity plus whatever else is in it, "
                "where mu = 1 and growth='gr' is the right answer "
                "rather than an approximation. A metric action "
                "that is *non-linear* in R does support "
                "growth='quasi_static': it is fourth-order, and "
                "takes the scale-dependent mu(a, k) of "
                "`theory.curvature.quasi_static_mu`."
            )

        E2 = sp.Symbol("E2", positive=True)

        sign = -6 if self.geometry == "teleparallel" else 6

        f_prime = sp.diff(self.gravity, self.scalar).subs(
            self.scalar, sign * E2,
        )

        # `f'`, not `1/f'`. Taking the reciprocal here would hide the
        # two ways it stops meaning anything -- a pole at `f' = 0`
        # and repulsive gravity at `f' < 0` -- behind a number that
        # looks perfectly ordinary. The reciprocal is taken in
        # `cosmology.core.utils.coupling_from_derivative`, which
        # refuses both.
        return sp.lambdify((E2, *args), f_prime, "numpy")

    # ---------------------------------------------------------

    def _verify(self, model) -> None:
        """
        Check the compiled model against the one condition every
        expansion history must satisfy, ``E(0) = 1``, at the
        model's own default parameters.

        An action that neither closes identically nor declares a
        ``closure`` parameter produces an ``E(z)`` that is simply
        not normalized to ``H0`` -- every distance it predicts is
        then wrong by a constant factor, silently. This is the
        check that turns that into an error.
        """

        try:
            E0 = float(
                model(model.PARAMS_CLASS(H0=70.0, Omega_m=0.3)).E(0.0)
            )

        except RuntimeError as exc:
            raise ValueError(
                f"This action has no real solution at its default "
                f"parameters, so it cannot be compiled into a "
                f"model. Adjust the defaults in params=. "
                f"({exc})"
            ) from None

        if abs(E0 - 1.0) > 1.0e-8:
            raise ValueError(
                f"E(0) = {E0:.6g}, not 1: this action does not "
                f"satisfy H(0) = H0 at its default parameters. "
                f"Every distance it predicts would be wrong by "
                f"that factor. Pass closure=<parameter name> to "
                f"name the parameter fixed by that condition -- "
                f"in Lambda-CDM, 'R - 2*Lam' with closure='Lam' "
                f"is what makes Lam = 3 (1 - Omega_m - Omega_k)."
            )


# ============================================================
# The generated model
# ============================================================

#: How many redshift grids a model keeps solved `E2` seeds for.
#: A likelihood evaluation touches under a dozen distinct grids,
#: and each entry only ever seeds a solve, so this bounds memory
#: without bounding correctness.
_E2_CACHE_SIZE = 16

class _ClosureSolver:
    """
    Evaluates the parameter fixed by ``E(0) = 1``.

    Two cases, decided once at compile time. When the closure
    equation is solvable for the parameter, ``explicit`` is a
    lambdified formula and nothing is iterated. When it is
    transcendental in it -- the exponential ``f(Q)`` case, whose
    hand-written counterpart in this library inverts a Lambert
    ``W`` -- Newton's method runs on the residual instead, from
    the parameter's declared default.

    Starting from that default is also how the branch is chosen:
    a transcendental closure condition can have several roots
    (Lambert ``W`` has two real branches), and the one reached is
    the one nearest where the user said the parameter lives.
    """

    def __init__(self, names, explicit, residual, parameter, start=0.0):

        self.names = names
        self.explicit = explicit
        self.residual = residual
        self.parameter = parameter
        self.start = start

    # ---------------------------------------------------------

    def value(self, params: dict) -> float:

        others = tuple(float(params[name]) for name in self.names)

        if self.explicit is not None:

            try:
                value = float(self.explicit(*others))

            except ZeroDivisionError:
                value = np.inf

            if not np.isfinite(value):
                raise RuntimeError(
                    f"The condition E(0) = 1 does not fix "
                    f"{self.parameter!r} at these parameter "
                    f"values: the closure formula is singular "
                    f"there. That is a degeneracy of the model, "
                    f"not a numerical failure -- the parameter "
                    f"has dropped out of the Friedmann equation, "
                    f"so nothing can be solved for it. Exclude "
                    f"that point with the prior bounds."
                )

            return value

        f, df = self.residual

        x = float(params.get(self.parameter, self.start))

        for _ in range(60):

            fx = float(f(x, *others))
            dfx = float(df(x, *others))

            if dfx == 0.0 or not np.isfinite(dfx):
                break

            step = fx / dfx
            x -= step

            if abs(step) <= 1.0e-14 * max(abs(x), 1.0):
                return x

        raise RuntimeError(
            f"Could not solve the closure condition E(0) = 1 for "
            f"{self.parameter!r} at these parameter values. The "
            f"model may have no consistent normalization here."
        )


# ------------------------------------------------------------

class _ActionModel(Cosmology):
    """
    Base of every class :meth:`Action.build` produces.

    Carries the compiled constraint and the parameter bookkeeping;
    the ``E``/``dEdz``/``Omega_de``/``mu`` implementations are
    installed per model by the ``_make_*`` factories below, since
    each closes over its own compiled callables.
    """

    #: The `Action` this model was compiled from, kept so a fitted
    #: model can still be asked what it was derived from.
    ACTION = None

    _ARGS = ()
    _FUNCTIONS = None
    _DIRECT = None
    _CLOSURE = None
    _FLUIDS = ()
    _MU = None

    # ---------------------------------------------------------

    def _parameter_values(self) -> tuple:
        """
        Current parameter values in the order the compiled
        constraint expects, with the closure parameter replaced by
        its solved value.
        """

        values = self.params.as_dict()

        closure = self._CLOSURE

        if closure is None:
            return tuple(float(values[name]) for name in self._ARGS)

        key = tuple(float(values[name]) for name in closure.names)

        cached = getattr(self, "_closure_cache", None)

        if cached is None or cached[0] != key:
            cached = (key, closure.value(values))
            self._closure_cache = cached

        values = dict(values)
        values[closure.parameter] = cached[1]

        return tuple(float(values[name]) for name in self._ARGS)

    # ---------------------------------------------------------

    def closure_value(self) -> float:
        """
        The value of the parameter fixed by ``E(0) = 1`` at the
        current parameters -- ``Lam`` for an action written as
        ``R - 2*Lam``, ``lam`` for the exponential ``f(Q)`` model.

        Raises ``AttributeError`` for an action that closes
        identically and so has no such parameter.
        """

        if self._CLOSURE is None:
            raise AttributeError(
                f"{type(self).__name__} was built from an action "
                f"that satisfies E(0) = 1 identically, so no "
                f"parameter is fixed by it."
            )

        self._parameter_values()

        return self._closure_cache[1]

    # ---------------------------------------------------------

    def _E2(self, z):

        z = np.asarray(z, dtype=float)

        values = self._parameter_values()

        if self._DIRECT is not None:

            E2 = np.asarray(self._DIRECT(z, *values), dtype=float)

            return np.broadcast_to(E2, z.shape) if E2.shape != z.shape else E2

        # A fit calls this thousands of times, moving the
        # parameters a little each time, so the previous answer is
        # an excellent starting point and skips the continuation
        # walk entirely. It is only ever a seed: `solve_E2` checks
        # the residual and falls back to the full walk from z = 0
        # if the seed led somewhere wrong, so a stale cache costs
        # time and never accuracy.
        #
        # Several *different* grids are in play within a single
        # likelihood evaluation -- the distance integrator's own,
        # each dataset's redshifts, and a handful of single-point
        # calls -- so this keys on the grid rather than holding
        # one. Keying on the bytes makes a hit exact; a miss just
        # means the slow path.
        cache = self.__dict__.setdefault("_E2_cache", {})

        key = (z.shape, z.tobytes())

        E2 = solve_E2(
            z, self._FUNCTIONS, values, seed=cache.get(key),
        )

        if len(cache) >= _E2_CACHE_SIZE:
            cache.pop(next(iter(cache)))

        cache[key] = E2

        return E2


# ------------------------------------------------------------

def _make_E():

    def E(self, z):

        return np.sqrt(self._E2(z))

    E.__doc__ = (
        "Dimensionless Hubble rate, from solving this model's "
        "Friedmann constraint (see ``ACTION``)."
    )

    return E


def _make_dEdz():

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        E2 = self._E2(z)

        derivative = dE2_dz(
            z, E2, self._FUNCTIONS, self._parameter_values(),
        )

        return derivative / (2.0 * np.sqrt(E2))

    dEdz.__doc__ = (
        "Derivative of E(z), by implicit differentiation of the "
        "Friedmann constraint -- exact, not finite-differenced."
    )

    return dEdz


def _make_Omega_de():

    def Omega_de(self, z):

        z = np.asarray(z, dtype=float)

        total = self._E2(z)

        for fluid in self._FLUIDS:

            total = total - getattr(self, fluid.parameter) * (
                1.0 + z
            ) ** (3.0 * (1.0 + fluid.w))

        return total - self.Omega_k * (1.0 + z) ** 2

    Omega_de.__doc__ = (
        "Effective dark-energy density, in units of today's "
        "critical density: whatever is left of E(z)^2 once the "
        "explicit fluids and curvature are removed.\n\n"
        "For a modified-gravity action nothing in the theory is "
        "actually a dark-energy fluid -- the acceleration comes "
        "from the modification itself. This is the standard way "
        "of presenting such a model's expansion history against "
        "a dark-energy one, and matches how ``FQExponential`` "
        "reports its own."
    )

    return Omega_de


def _make_coupling(compiled):

    def coupling(self, z):
        """
        ``f'`` on this model's own background -- the quantity whose
        reciprocal is ``mu``, and whose sign decides whether that
        reciprocal means anything.
        """

        E2 = self._E2(np.asarray(z, dtype=float))

        return np.asarray(
            compiled(E2, *self._parameter_values()), dtype=float,
        )

    return coupling


def _make_teleparallel_viability():

    def viability(self, z=None):
        """
        Whether this model is one worth fitting, checked along its
        own background.

        Returns ``{"ok": bool, "failed": [...], "reasons": [...]}``
        with the single condition ``f' > 0``, the counterpart of the
        metric sector's. Sampled rather than proved, on a grid over
        ``z``.

        Where ``mu`` raises, this reports: a caller asking for a
        number must not get a meaningless one, and a caller asking
        whether the model is admissible wants an answer rather than
        an exception.
        """

        if z is None:
            z = np.concatenate([
                np.linspace(0.0, 3.0, 240),
                np.geomspace(3.0, 1100.0, 240),
            ])

        failed = teleparallel_failures(self.coupling(z))

        return {
            "ok": not failed,
            "failed": failed,
            "reasons": [TELEPARALLEL_CONDITIONS[k] for k in failed],
        }

    return viability


def _make_mu(compiled):

    def mu(self, a, k=None):

        a = np.asarray(a, dtype=float)

        E2 = self._E2(1.0 / a - 1.0)

        f_prime = np.asarray(
            compiled(E2, *self._parameter_values()), dtype=float,
        )

        return coupling_from_derivative(f_prime, model=type(self).__name__)

    mu.__doc__ = (
        "Effective gravitational coupling G_eff/G_N = 1/f', the "
        "sub-horizon quasi-static limit of this model's "
        "gravitational sector (see ``Action(growth=...)``). Raises "
        "where f' <= 0, since 1/f' is then either singular or "
        "negative -- see "
        "``cosmology.core.utils.coupling_from_derivative``."
    )

    return mu


# ============================================================
# The generated model, with fields
# ============================================================

class _FieldModel(Cosmology):
    """
    Base of every class :meth:`Action.build` produces for an
    action carrying dynamical scalar fields.

    The difference from :class:`_ActionModel` is that ``E(z)``
    cannot be solved redshift by redshift: the field has its own
    equation of motion, so the whole history is integrated at once
    and interpolated. That makes the *parameters* the unit of work
    rather than the redshift grid -- one integration serves every
    ``E(z)`` call until a parameter moves.
    """

    ACTION = None

    _ARGS = ()
    _SYSTEM = None
    _CLOSURE = None
    _FLUIDS = ()
    _FIELDS = ()

    # ---------------------------------------------------------

    def _resolved_params(self) -> dict:
        """
        Current parameters, with the one fixed by ``E(0) = 1``
        replaced by its solved value.
        """

        values = dict(self.params.as_dict())

        closure = self._CLOSURE

        if closure is None:
            return values

        key = tuple(float(values[name]) for name in closure.names)

        cached = getattr(self, "_closure_cache", None)

        if cached is None or cached[0] != key:
            cached = (key, closure.value(values))
            self._closure_cache = cached

        values[closure.parameter] = cached[1]

        return values

    # ---------------------------------------------------------

    def closure_value(self) -> float:
        """
        The value of the parameter fixed by ``E(0) = 1`` at the
        current parameters.
        """

        if self._CLOSURE is None:
            raise AttributeError(
                f"{type(self).__name__} was built from an action "
                f"that satisfies E(0) = 1 identically, so no "
                f"parameter is fixed by it."
            )

        return self._resolved_params()[self._CLOSURE.parameter]

    # ---------------------------------------------------------

    def history(self, z=0.0):
        """
        The solved expansion history covering ``z``, integrating
        it if the cached one does not reach that far.

        Exposed because its ``drift`` -- how far the Friedmann
        constraint moved along the solution, which the integration
        never imposes after the initial conditions -- is the
        honest measure of how well this model is solved.
        """

        N = -np.log1p(np.asarray(z, dtype=float))

        return self._history_for(float(np.min(N)), float(np.max(N)))

    # ---------------------------------------------------------

    def _E2(self, z):
        """``E(z)^2``, to match :class:`_ActionModel`'s interface."""

        return self.E(z) ** 2

    # ---------------------------------------------------------

    def _field_density(self, z):
        """
        ``(rho, p)`` of the field sector at ``z``, in the units of
        :mod:`~theory.minisuperspace` (where a fluid with density
        parameter ``Omega`` has ``rho_0 = 3 Omega``).
        """

        N = -np.log1p(np.asarray(z, dtype=float))

        history = self._history_for(float(np.min(N)), float(np.max(N)))

        values = self._resolved_params()

        args = tuple(float(values[name]) for name in self._ARGS)

        state = history.state(N)

        a = np.exp(N)

        return (
            np.asarray(self._SYSTEM.rho(a, *state, *args), dtype=float),
            np.asarray(
                self._SYSTEM.pressure(a, *state, *args), dtype=float,
            ),
        )

    # ---------------------------------------------------------

    def _history_for(self, N_lo: float, N_hi: float):

        values = self._resolved_params()

        args = tuple(float(values[name]) for name in self._ARGS)

        fields = tuple(
            float(values[f"{name}_i"]) for name in self._FIELDS
        )
        velocities = tuple(
            float(values[f"d{name}_i"]) for name in self._FIELDS
        )

        if N_lo < np.log(self._A_INIT) - 1.0e-12:
            raise ModelConfigurationError(
                f"E(z) was asked for at z = "
                f"{np.expm1(-N_lo):.4g}, beyond this model's "
                f"initial redshift z_init = {self._Z_INIT:.4g}. "
                f"There is no history before the field's state is "
                f"given, and extrapolating one would be inventing "
                f"the early universe rather than solving for it. "
                f"Build the model with a larger "
                f"Action(z_init=...)."
            )

        key = (args, fields, velocities)

        cached = getattr(self, "_history_cache", None)

        if cached is not None and cached[0] == key:

            if cached[1].covers(N_lo, N_hi):
                return cached[1]

            # Same parameters, further into the future: integrate
            # the union, so a sequence of widening requests does
            # not keep discarding work.
            N_hi = max(N_hi, cached[1].N_hi)

        history = integrate(
            self._SYSTEM, args, fields, velocities, self._A_INIT, N_hi,
        )

        self._history_cache = (key, history)

        return history


# ------------------------------------------------------------

def _make_w():

    def w(self, z):

        z = np.asarray(z, dtype=float)

        E = self.E(z)

        rho = self.Omega_de(z)

        drho = (
            2.0 * E * self.dEdz(z)
            - 3.0 * self.Omega_m * (1.0 + z) ** 2
            - 2.0 * self.Omega_k * (1.0 + z)
        )

        for fluid in self._FLUIDS:

            if fluid.parameter == "Omega_m":
                continue

            drho = drho - getattr(self, fluid.parameter) * (
                3.0 * (1.0 + fluid.w)
            ) * (1.0 + z) ** (3.0 * (1.0 + fluid.w) - 1.0)

        return -1.0 + (1.0 + z) * drho / (3.0 * rho)

    w.__doc__ = (
        "Effective dark-energy equation of state.\n\n"
        "Read off the background: w = -1 + (1+z) "
        "dln(rho_de)/dz / 3, which is what conservation of the "
        "effective dark-energy density means. That is the only "
        "definition available for a modified gravitational "
        "sector, where there is no dark-energy fluid to take a "
        "pressure of -- and it is exactly how such a model is "
        "compared against a dark-energy one.\n\n"
        "It loses precision once matter dominates, because "
        "rho_de is then a small difference of large numbers "
        "(2.4e9 minus 2.4e9 to get 0.7, at z = 2000). Trust it "
        "where dark energy is a comparable share of the budget, "
        "which is everywhere it means anything. An action with "
        "scalar fields does not use this: there the density and "
        "pressure come straight from the Lagrangian."
    )

    return w


def _make_field_Omega_de():

    def Omega_de(self, z):

        return self._field_density(z)[0] / 3.0

    Omega_de.__doc__ = (
        "Dark-energy density in units of today's critical "
        "density, read off the field's own Lagrangian rather "
        "than by subtracting the fluids from E(z)^2.\n\n"
        "The subtraction is what the fieldless models here do, "
        "and it is fine while dark energy is a comparable "
        "fraction of the budget. It stops being fine early: at "
        "z = 2000 it is 2.4e9 minus 2.4e9 to get 0.7, which "
        "throws away nine digits. There is nothing to cancel in "
        "rho = 2 X L_X - L."
    )

    return Omega_de


def _make_field_w():

    def w(self, z):

        rho, pressure = self._field_density(z)

        return pressure / rho

    w.__doc__ = (
        "Dark-energy equation of state, ``p/rho`` with both taken "
        "from the field's Lagrangian: the pressure of any "
        "``L(X, phi)`` is ``L`` itself and the density is "
        "``2 X L_X - L``. Exact at every redshift, where reading "
        "it off the background instead loses precision as soon "
        "as matter dominates."
    )

    return w


def _make_field_E():

    def E(self, z):

        N = -np.log1p(np.asarray(z, dtype=float))

        history = self._history_for(float(np.min(N)), float(np.max(N)))

        return history.H(N)

    E.__doc__ = (
        "Dimensionless Hubble rate, interpolated from this "
        "model's integrated expansion history (see ``ACTION``)."
    )

    return E


def _make_field_dEdz():

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        N = -np.log1p(z)

        history = self._history_for(float(np.min(N)), float(np.max(N)))

        # E is H in these units and N = -ln(1+z), so the chain rule
        # is one factor: dE/dz = (dH/dN) (dN/dz) = -(dH/dN)/(1+z).
        return -history.dH_dN(N) / (1.0 + z)

    dEdz.__doc__ = (
        "Derivative of E(z), from the same equations of motion "
        "that produced the history -- not finite-differenced."
    )

    return dEdz


# ------------------------------------------------------------

class _ShootingClosure:
    """
    Solves ``E(0) = 1`` for an action with dynamical fields.

    For a fieldless action that condition is the Friedmann
    constraint at ``a = 1``, and :class:`_ClosureSolver` either
    inverts it or runs Newton on it. Here there is no such
    equation: the field's state is given at ``z_init`` and the
    history integrated forwards, so ``H(0)`` is only known after
    integrating. This drives it to 1 with a secant iteration, each
    step costing one integration up to the present.

    Which root it finds is set by the closure parameter's declared
    default, the same convention :class:`_ClosureSolver` uses for
    a transcendental condition.
    """

    #: Relative accuracy required of H(0) = 1. Well inside the
    #: integrator's own error, and reached in a handful of steps
    #: because H(0) responds smoothly and monotonically to a
    #: potential scale.
    TOLERANCE = 1.0e-12

    MAX_STEPS = 60

    #: How many times a secant step is halved back towards the
    #: last working point before the solve is abandoned.
    BACKTRACKS = 40

    def __init__(self, system, arg_names, field_names, parameter, start, a_i):

        self.system = system
        self.arg_names = arg_names
        self.field_names = field_names
        self.parameter = parameter
        self.start = start
        self.a_i = a_i

        #: Last solved value and the secant slope dx/df that
        #: found it, used to seed the next solve. A fit moves the
        #: parameters a little at a time, so the previous answer
        #: is usually within one step -- and carrying the slope
        #: too makes that first step a real one rather than a
        #: blind offset used only to get a second point. The same
        #: warm start `Fitter.profile` uses. Only ever a seed: the
        #: residual is checked every time, so a stale one costs
        #: iterations and never accuracy.
        self._seed = None
        self._slope = None

        #: Every parameter H(0) depends on -- what a cached value
        #: has to be keyed on.
        self.names = tuple(
            sorted(
                (
                    set(arg_names)
                    | {f"{n}_i" for n in field_names}
                    | {f"d{n}_i" for n in field_names}
                )
                - {parameter}
            )
        )

    # ---------------------------------------------------------

    def _residual(self, x: float, params: dict) -> float:

        trial = dict(params)
        trial[self.parameter] = x

        values = tuple(float(trial[name]) for name in self.arg_names)

        fields = [float(trial[f"{n}_i"]) for n in self.field_names]
        velocities = [float(trial[f"d{n}_i"]) for n in self.field_names]

        return expansion_today(
            self.system, values, fields, velocities, self.a_i,
        ) - 1.0

    # ---------------------------------------------------------

    def _remember(self, x0: float, f0: float, x1: float, f1: float) -> None:
        """Keep the converged value and the local slope dx/df."""

        self._seed = x1

        if f1 != f0 and np.isfinite(x1 - x0):
            self._slope = (x1 - x0) / (f1 - f0)

    # ---------------------------------------------------------

    def _stepped_residual(self, x_from: float, x_to: float, params: dict):
        """
        The residual at ``x_to``, backtracking towards ``x_from``
        if the integration cannot survive there.

        A secant step can overshoot into parameter values where
        this model has no expansion history at all -- a scalar
        field runs away and the integrator stalls. That is a real
        boundary rather than a failure, so the step is halved back
        towards the last point that worked instead of giving up on
        the solve.
        """

        for _ in range(self.BACKTRACKS):

            try:
                return x_to, self._residual(x_to, params)

            except (RuntimeError, FloatingPointError, ValueError):
                x_to = 0.5 * (x_to + x_from)

        raise RuntimeError(
            f"Solving E(0) = 1 for {self.parameter!r} left the "
            f"region where this action has an expansion history "
            f"at all, and backtracking did not recover. The model "
            f"may not reach the requested Omega_m from these "
            f"initial conditions -- for exponential quintessence "
            f"that happens once the potential is steep enough "
            f"that the scaling attractor fixes the dark-energy "
            f"fraction below it."
        )

    # ---------------------------------------------------------

    def value(self, params: dict) -> float:

        x0 = float(
            self._seed
            if self._seed is not None
            else params.get(self.parameter, self.start)
        )

        x0, f0 = self._stepped_residual(
            float(params.get(self.parameter, self.start)), x0, params,
        )

        if abs(f0) <= self.TOLERANCE:
            self._seed = x0
            return x0

        # The second point for the secant. With a remembered
        # slope this is already a Newton step; without one it is
        # an offset big enough to move H(0) measurably and small
        # enough not to leave the branch.
        guess = (
            x0 - f0 * self._slope
            if self._slope is not None
            else x0 * 1.01 + 1.0e-3
        )

        if not np.isfinite(guess) or guess == x0:
            guess = x0 * 1.01 + 1.0e-3

        x1, f1 = self._stepped_residual(x0, guess, params)

        for _ in range(self.MAX_STEPS):

            if f1 == f0:
                break

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

            if not np.isfinite(x2):
                break

            x0, f0 = x1, f1
            x1, f1 = self._stepped_residual(x0, x2, params)

            if abs(f1) <= self.TOLERANCE:
                self._remember(x0, f0, x1, f1)
                return x1

        raise RuntimeError(
            f"Could not solve E(0) = 1 for {self.parameter!r} at "
            f"these parameter values: H(0) came out "
            f"{f1 + 1.0:.6g} rather than 1. The requested "
            f"expansion history may not be reachable from this "
            f"model -- narrow the prior bounds, or move the "
            f"parameter's default nearer the intended branch."
        )


# ============================================================
# The generated model, fourth-order
# ============================================================

class _CurvatureModel(Cosmology):
    """
    Base of every class :meth:`Action.build` produces for a general
    ``f(R)``.

    The state is ``(H, R)`` over ``N = ln a``, integrated outwards
    from ``a = 1`` -- where ``H = 1`` by the definition of ``H0``,
    so nothing has to be shot for. See :mod:`~theory.curvature` for
    why backwards is safe here and is not in
    :mod:`~theory.fields`.
    """

    ACTION = None

    _ARGS = ()
    _SYSTEM = None
    _FLUIDS = ()

    #: Wavenumber [h/Mpc] used when `mu` is called without one --
    #: the same value `FRHuSawicki` picks, so the two f(R) routes
    #: answer the same question when asked the short way. It sits
    #: in the middle of the linear regime, which is where a
    #: quasi-static mu is meant to be used.
    _DEFAULT_K = 0.1

    # ---------------------------------------------------------

    def _E2(self, z):
        return self.E(z) ** 2

    # ---------------------------------------------------------

    def history(self, z=0.0):
        """
        The solved background covering ``z``. Its ``drift`` is how
        far the Friedmann constraint moved along the solution,
        which the integration never imposes after the initial
        conditions -- an independent measure of the error.
        """

        N = -np.log1p(np.asarray(z, dtype=float))

        return self._history_for(float(np.min(N)), float(np.max(N)))

    # ---------------------------------------------------------

    def ricci(self, z=0.0):
        """
        The Ricci scalar in units of ``H0^2``, along the solution.
        """

        N = -np.log1p(np.asarray(z, dtype=float))

        return self._history_for(
            float(np.min(N)), float(np.max(N)),
        ).state(N)[1]

    # ---------------------------------------------------------

    def _history_for(self, N_lo: float, N_hi: float):

        values = self.params.as_dict()

        args = tuple(float(values[name]) for name in self._ARGS)

        key = (args, float(values["R_0"]))

        cache = getattr(self, "_history_cache", None)

        if cache is not None and cache[0] == key:

            if cache[1].covers(N_lo, N_hi):
                return cache[1]

            N_lo = min(N_lo, cache[1].N_lo)
            N_hi = max(N_hi, cache[1].N_hi)

        history = integrate_curvature(
            self._SYSTEM, args, float(values["R_0"]), N_lo, N_hi,
        )

        self._history_cache = (key, history)

        return history


# ------------------------------------------------------------

class _ForwardCurvatureModel(_CurvatureModel):
    """
    A general ``f(R)`` integrated forwards, from deep in matter
    domination to today.

    The state and the equations are the backward model's; what
    differs is where the integration starts and what has to be
    solved for. See
    :func:`~theory.curvature.integrate_forward` for why a theory
    might need this direction, and
    :meth:`~theory.action.Action._build_fourth_order_forward` for
    what it costs.
    """

    _CLOSURE = None
    _CLOSURE_BRACKET = (0.0, 10.0)

    #: How close E(0) must come to 1 for the shot to count. The
    #: build's own check is at 1e-8, so this has to be tighter.
    _CLOSURE_TOL = 1.0e-11

    #: Secant steps allowed when polishing at full tolerance.
    _REFINE_STEPS = 5

    # ---------------------------------------------------------

    def closure_value(self) -> float:
        """
        The closure parameter's value, solved so that ``E(0) = 1``.
        """

        return self._solved_closure()

    # ---------------------------------------------------------

    def ricci_today(self) -> float:
        """
        ``R_0`` in units of ``H0^2`` -- derived here rather than
        given, which is the whole point of this direction.
        """

        return float(self.ricci(0.0))

    # ---------------------------------------------------------

    def _args_with(self, values, closure_value):

        return tuple(
            float(closure_value) if name == self._CLOSURE
            else float(values[name])
            for name in self._ARGS
        )

    # ---------------------------------------------------------

    def _E0_for(self, values, closure_value, N_hi, loose=True):

        args = self._args_with(values, closure_value)

        # Loose tolerance while searching, on purpose: a shot is not
        # the answer. See `theory.curvature._SHOOT_RTOL` -- the root
        # moves by ~1e-8 and each evaluation costs six times less.
        # The search is followed by a refinement at full tolerance,
        # because 1e-8 is exactly the size of the E(0) = 1 check the
        # build then applies, and a loose root fails it.
        from CosmoFit.theory.curvature import _SHOOT_ATOL, _SHOOT_RTOL

        kwargs = (
            {} if not loose
            else {"rtol": _SHOOT_RTOL, "atol": _SHOOT_ATOL}
        )

        history = integrate_curvature_forward(
            self._SYSTEM, args, float(values["Omega_m"]), 0.0, N_hi,
            **kwargs,
        )

        return float(history.H(0.0))

    # ---------------------------------------------------------

    def _solved_closure(self, N_hi=0.0):

        values = self.params.as_dict()

        key = tuple(
            float(values[name])
            for name in self._ARGS if name != self._CLOSURE
        ) + (float(values["Omega_m"]),)

        cached = getattr(self, "_closure_cache", None)

        if cached is not None and cached[0] == key:
            return cached[1]

        lo, hi = self._CLOSURE_BRACKET

        def miss(x):
            try:
                return self._E0_for(values, x, N_hi) - 1.0
            except RuntimeError:
                return np.nan

        # Warm start. Each `miss` is a background integration and the
        # scan below needs up to seventeen of them, which profiling
        # put at 89% of a likelihood evaluation. Along a chain the
        # parameters move a little at a time, so the previous answer
        # is nearly right.
        #
        # A secant iteration rather than a bracket around it: fixed
        # widths were tried first and essentially always failed to
        # bracket, falling through to the full scan and buying
        # nothing -- 26 evaluations per call, measured. The secant
        # needs no bracket and converges in about three.
        previous = getattr(self, "_closure_last", None)

        solution = None

        if previous is not None:
            solution = _secant(miss, previous, lo, hi, tol=1.0e-8)

        if solution is None:
            solution = _bracketed_root(miss, lo, hi)

        if solution is not None:
            solution = self._refine_closure(values, solution, N_hi)

        if solution is None:
            raise RuntimeError(
                f"Could not find a value of {self._CLOSURE!r} in "
                f"[{lo:.4g}, {hi:.4g}] that gives E(0) = 1 for this "
                f"f(R). Widen that parameter's bounds, or check "
                f"that the action can reproduce today's expansion "
                f"rate at all."
            )

        self._closure_cache = (key, solution)
        self._closure_last = solution

        return solution

    # ---------------------------------------------------------

    def _refine_closure(self, values, start, N_hi):
        """
        Polish a loosely-found closure at full tolerance.

        A secant iteration rather than another bracketed solve: the
        residual is smooth and very nearly linear this close to the
        root, so two or three evaluations get there, and each one
        costs six times a search evaluation.
        """

        step = max(abs(start), 1.0) * 1.0e-6

        x0, x1 = start, start + step

        f0 = self._E0_for(values, x0, N_hi, loose=False) - 1.0

        if abs(f0) <= self._CLOSURE_TOL:
            return x0

        f1 = self._E0_for(values, x1, N_hi, loose=False) - 1.0

        for _ in range(self._REFINE_STEPS):

            if f1 == f0:
                break

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

            x0, f0 = x1, f1

            x1 = x2

            f1 = self._E0_for(values, x1, N_hi, loose=False) - 1.0

            if abs(f1) <= self._CLOSURE_TOL:
                break

        return x1 if abs(f1) < abs(f0) else x0

    # ---------------------------------------------------------

    def _history_for(self, N_lo: float, N_hi: float):

        values = self.params.as_dict()

        closure_value = self._solved_closure()

        args = self._args_with(values, closure_value)

        key = args + (float(values["Omega_m"]),)

        cache = getattr(self, "_history_cache", None)

        if cache is not None and cache[0] == key:

            if cache[1].covers(N_lo, N_hi):
                return cache[1]

            N_lo = min(N_lo, cache[1].N_lo)
            N_hi = max(N_hi, cache[1].N_hi)

        history = integrate_curvature_forward(
            self._SYSTEM, args, float(values["Omega_m"]), N_lo, N_hi,
        )

        self._history_cache = (key, history)

        return history


# ------------------------------------------------------------

def _secant(f, start, lo, hi, tol, steps=6):
    """
    A secant iteration from a point already believed close.

    Returns the root, or ``None`` if it wandered outside
    ``[lo, hi]``, stalled, or hit a point the integrator could not
    do -- in which case the caller falls back to scanning.
    """

    step = max(abs(start), 1.0) * 1.0e-4

    x0 = float(start)
    x1 = float(min(hi, max(lo, start + step)))

    f0 = f(x0)

    if not np.isfinite(f0):
        return None

    if abs(f0) <= tol:
        return x0

    f1 = f(x1)

    for _ in range(steps):

        if not np.isfinite(f1) or f1 == f0:
            return None

        if abs(f1) <= tol:
            return x1

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if not (lo <= x2 <= hi):
            return None

        x0, f0 = x1, f1

        x1 = float(x2)

        f1 = f(x1)

    return x1 if np.isfinite(f1) and abs(f1) <= tol else None


def _bracketed_root(f, lo, hi, samples=17):
    """
    A sign change on a coarse scan, then Brent on that sub-interval.

    The scan is there because the closure's declared bounds are a
    statement about where the parameter is meaningful, not about
    where the background integrates: parts of the interval can
    return NaN, and handing that straight to a root finder gets an
    exception rather than an answer.
    """

    xs = np.linspace(lo, hi, samples)

    # Evaluated one at a time and stopped at the first sign change
    # rather than sampled all at once: each of these is a full
    # background integration, and the scan is the expensive half of
    # building a forward model.
    previous = float(f(xs[0]))

    for i in range(len(xs) - 1):

        current = float(f(xs[i + 1]))

        if np.isfinite(previous) and np.isfinite(current):

            if previous == 0.0:
                return float(xs[i])

            if previous * current <= 0.0:
                return float(brentq(f, xs[i], xs[i + 1], xtol=1.0e-13))

        previous = current

    return None


def _make_curvature_E():

    def E(self, z):

        N = -np.log1p(np.asarray(z, dtype=float))

        return self._history_for(
            float(np.min(N)), float(np.max(N)),
        ).H(N)

    E.__doc__ = (
        "Dimensionless Hubble rate, interpolated from this f(R) "
        "model's integrated background (see ``ACTION``)."
    )

    return E


def _make_scalaron():

    def scalaron(self, z=0.0):
        """
        ``(f_R, f_RR)`` along this model's own background, at the
        redshifts given. ``f_R`` is dimensionless; ``f_RR`` is in
        units of ``1/H0^2``, matching the ``R`` this model
        integrates.
        """

        values = self.params.as_dict()

        args = tuple(float(values[name]) for name in self._MU_ARGS)

        R = self.ricci(z)

        return self._F_R(R, *args), self._F_RR(R, *args)

    return scalaron


def _make_screening():

    def screening(self, bound=None):
        """
        Whether local tests have already excluded this f(R).

        Returns ``{"ok", "deviation", "bound"}``, where ``deviation``
        is ``|f_R(0) - 1|`` -- the fractional departure of the
        gravitational coupling today, which unscreened would appear
        as a fifth force.

        Deliberately **not** part of :meth:`viability`. That method
        answers "is this a consistent theory"; this one answers "has
        it already been ruled out", and a model can pass the first
        while failing the second. Merging them into one boolean
        would lose exactly the distinction a reader needs.

        See :func:`theory.curvature.screening_margin` for what the
        bound is and, more importantly, for what it does not settle:
        chameleon screening is non-linear, so failing here means
        "excluded unless screening rescues it".
        """

        from CosmoFit.theory.curvature import SOLAR_SYSTEM_BOUND

        if bound is None:
            bound = SOLAR_SYSTEM_BOUND

        f_R, _ = self.scalaron(0.0)

        deviation, ok = screening_margin(f_R, bound)

        return {"ok": ok, "deviation": deviation, "bound": float(bound)}

    return screening


def _make_viability():

    def viability(self, z=None):
        """
        Whether this f(R) is a theory worth fitting, checked along
        its own background.

        Returns ``{"ok": bool, "failed": [...], "reasons": [...]}``.
        The two conditions are ``f_R > 0`` (no graviton ghost) and
        ``f_RR > 0`` (no tachyonic scalaron -- the Dolgov-Kawasaki
        instability). See
        :data:`theory.curvature.VIABILITY_CONDITIONS`.

        This is a *sampled* statement, not a proof: the conditions
        are evaluated on a grid over ``z``, so a violation confined
        between samples is missed. It samples densely enough that
        anything smooth would have to be contrived to hide, and the
        default range is the one observations cover.
        """

        if z is None:
            z = np.concatenate([
                np.linspace(0.0, 3.0, 240),
                np.geomspace(3.0, 1100.0, 240),
            ])

        f_R, f_RR = self.scalaron(z)

        failed = viability_failures(f_R, f_RR)

        return {
            "ok": not failed,
            "failed": failed,
            "reasons": [VIABILITY_CONDITIONS[name] for name in failed],
        }

    return viability


def _make_curvature_mu():

    def mu(self, a, k=None):

        if k is None:
            k = self._DEFAULT_K

        a = np.asarray(a, dtype=float)

        values = self.params.as_dict()

        args = tuple(float(values[name]) for name in self._MU_ARGS)

        R = self.ricci(1.0 / a - 1.0)

        return quasi_static_mu(
            self._F_R(R, *args), self._F_RR(R, *args), a, k,
        )

    mu.__doc__ = (
        "Effective gravitational coupling G_eff(a,k)/G_N in the "
        "quasi-static, sub-horizon limit, with R taken from this "
        "model's own integrated background. Scale-dependent: `k` is "
        "in h/Mpc. See `theory.curvature.quasi_static_mu`."
    )

    return mu


def _make_curvature_dEdz():

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        N = -np.log1p(z)

        history = self._history_for(float(np.min(N)), float(np.max(N)))

        return -history.dH_dN(N) / (1.0 + z)

    dEdz.__doc__ = (
        "Derivative of E(z), from the same equations of motion "
        "that produced the background -- not finite-differenced."
    )

    return dEdz


def _make_field_mu(compiled, coupling):

    def mu(self, a, k=None):

        a = np.asarray(a, dtype=float)

        N = np.log(a)

        history = self._history_for(float(np.min(N)), float(np.max(N)))

        state = history.state(N)

        fields = state[1:1 + len(self._FIELDS)]

        values = self._resolved_params()

        args = tuple(float(values[name]) for name in self._ARGS)

        F = np.asarray(coupling(*fields, *args), dtype=float)

        if np.any(~np.isfinite(F)) or np.any(F <= 0.0):
            raise ValueError(
                f"{type(self).__name__} has a non-minimal coupling "
                f"F = {float(np.nanmin(F)):.6g} somewhere in the "
                f"requested range. F is the effective Newton "
                f"constant's reciprocal, so F <= 0 is a ghost, and "
                f"the scalar-tensor mu is then negative or "
                f"singular -- 'mu = -2' and 'mu = inf' are both "
                f"reachable values of the formula. F > 0 is the "
                f"condition for it to mean anything."
            )

        return np.asarray(compiled(*fields, *args), dtype=float)

    mu.__doc__ = (
        "Effective gravitational coupling G_eff/G_N in the "
        "sub-horizon quasi-static limit of scalar-tensor gravity "
        "(see ``Action(growth=...)``). Evaluated on the field's own "
        "solution, so it varies with time as the field rolls. "
        "Raises where the non-minimal coupling F <= 0, since the "
        "expression is then negative or singular."
    )

    return mu
