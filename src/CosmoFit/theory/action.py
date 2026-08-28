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

from CosmoFit.cosmology.core.base import Cosmology
from CosmoFit.cosmology.core.parameters import CosmologyParameters

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
_SAFE_FUNCTIONS = {
    name: getattr(sp, name)
    for name in (
        "sqrt", "exp", "log", "sin", "cos", "tan",
        "sinh", "cosh", "tanh", "Abs", "sign",
    )
}


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
            f"{sorted(_SAFE_FUNCTIONS)}. Declare any new model "
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

        ``"quasi_static"`` sets ``mu = 1/f'`` (derivative with
        respect to the geometry scalar), the standard sub-horizon
        quasi-static result for ``f(T)`` and ``f(Q)`` gravity and
        what this library's hand-written ``FQExponential`` and
        ``FRTLinear`` already use. It is an *additional* physical
        assumption on top of the action -- a statement about
        perturbations, which a background action does not by
        itself determine -- so it must be asked for explicitly.

    fields : dict, optional
        Scalar fields, mapping name -> Lagrangian density written
        in terms of the field and its kinetic scalar ``X``
        (``"X - V0*exp(-lam*phi)"`` for exponential quintessence).
        The symbolic reduction handles these -- see
        :meth:`field_equations` -- but :meth:`build` does not yet
        solve the resulting coupled ODE system, and says so.
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
        fields: dict | None = None,
    ):

        self.params = dict(params or {})
        self.fields = dict(fields or {})
        self.closure = closure
        self.growth = growth

        self.fluids = tuple(
            STANDARD_FLUIDS[f] if isinstance(f, str) else f
            for f in fluids
        )

        if growth not in ("gr", "quasi_static"):
            raise ValueError(
                f"growth must be 'gr' or 'quasi_static', "
                f"got {growth!r}."
            )

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

        self._constraint = None

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

    def lagrangian(self) -> tuple[Minisuperspace, sp.Expr]:
        """
        The reduced point-like Lagrangian of this action, together
        with the minisuperspace it lives on.
        """

        ms = Minisuperspace(tuple(self.fields))

        L = gravity_lagrangian(
            ms, self.geometry, self.gravity, self.scalar,
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

        Provided for inspection: the symbolic reduction handles
        fields, while :meth:`build` does not yet integrate the
        coupled system they produce.
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

        Parameters
        ----------
        name : str
            Model name, used for the class and for
            ``MODEL_NAME``.

        label : str, optional
            LaTeX name for figures, as in
            :func:`cosmology.custom.define_model`.

        Raises
        ------
        NotImplementedError
            If the action carries scalar fields. Their equations
            of motion are derived correctly (see
            :meth:`field_equations`) but not yet integrated.
        """

        if self.fields:
            raise NotImplementedError(
                "Actions with scalar fields reduce correctly -- "
                "see Action.field_equations() -- but building a "
                "model from one needs the coupled field/Friedmann "
                "system integrated, which is not implemented yet. "
                "Actions built only from the geometry scalar and "
                "perfect fluids (f(T), f(Q), GR with a "
                "cosmological constant) do build."
            )

        C, E2, z = self.constraint()

        args = self._solver_arguments(C, E2, z)

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
        }

        if self.closure in CosmologyParameters.names():
            attrs["DERIVED_PARAMS"] = frozenset({self.closure})

        if mu is not None:
            attrs["mu"] = _make_mu(mu)

        model = type(name, (_ActionModel,), attrs)

        self._verify(model)

        return model

    # ---------------------------------------------------------

    def _solver_arguments(self, C, E2, z) -> tuple:
        """
        The parameter symbols the compiled constraint needs, in a
        fixed order -- and a check that every one of them is
        actually a parameter the generated model will carry.
        """

        free = sorted(
            (C.free_symbols - {E2, z}), key=lambda s: s.name,
        )

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

        others = tuple(s for s in args if s != symbol)

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
                "growth='quasi_static' is implemented for the "
                "teleparallel and symmetric-teleparallel sectors, "
                "where mu = 1/f' is a settled sub-horizon result. "
                "In the metric sector it is f(R) gravity, whose "
                "mu is scale-dependent (a Compton wavelength "
                "enters) -- and whose action this module cannot "
                "reduce in the first place."
            )

        E2 = sp.Symbol("E2", positive=True)

        sign = -6 if self.geometry == "teleparallel" else 6

        f_prime = sp.diff(self.gravity, self.scalar).subs(
            self.scalar, sign * E2,
        )

        return sp.lambdify((E2, *args), 1 / f_prime, "numpy")

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


def _make_mu(compiled):

    def mu(self, a, k=None):

        a = np.asarray(a, dtype=float)

        E2 = self._E2(1.0 / a - 1.0)

        return np.asarray(
            compiled(E2, *self._parameter_values()), dtype=float,
        )

    mu.__doc__ = (
        "Effective gravitational coupling G_eff/G_N = 1/f', the "
        "sub-horizon quasi-static limit of this model's "
        "gravitational sector (see ``Action(growth=...)``)."
    )

    return mu
