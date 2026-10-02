"""
Parameters: what is sampled, what is fixed, what is computed.

The old fitting path had one global dataclass holding every parameter
of every model, a list of names to free, and a uniform box per name.
Here each parameter is declared by itself, in one of four roles:

sampled
    Has a prior; the sampler moves it. ``{"prior": {...}}``, with an
    optional ``ref`` (where to start), ``proposal`` (a step scale) and
    ``latex`` label.

fixed
    A number. ``H0: 67.4``, or ``{"value": 67.4}``.

dependent
    An input computed from other parameters before anything is
    evaluated -- a reparametrization. ``Omega_b: "lambda omega_b, H0:
    omega_b / (H0 / 100) ** 2"`` lets a fit sample ``omega_b`` (with
    ``drop: True``, so no component is handed it) and give the
    cosmology the ``Omega_b`` it expects.

derived
    An output, recorded with each point and never fed back: ``{"derived":
    True}`` for a value a component computes, or a lambda of other
    parameters and derived values.

A lambda may be a Python callable or a string ``"lambda a, b: ..."``,
evaluated with ``np`` in scope. Its argument names are the parameters
it reads, so the dependency is declared by the signature itself.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .priors import Prior, make_prior


__all__ = [
    "ParamSpec",
    "ParameterSet",
    "parse_param",
]


#: Keys a parameter's dict may carry.
_KNOWN_KEYS = {"prior", "value", "ref", "proposal", "latex", "derived", "drop"}


def _as_function(obj, what: str) -> Callable:
    """
    A callable from a callable or a ``"lambda ..."`` string.
    """

    if callable(obj):
        return obj

    if isinstance(obj, str) and obj.strip().startswith("lambda"):

        try:
            return eval(obj.strip(), {"np": np, "numpy": np, "math": math})
        except SyntaxError as error:
            raise ValueError(f"{what}: not a valid lambda ({error}).") from None

    raise ValueError(f"{what}: expected a number, a callable or a lambda string.")


def _arguments(function: Callable) -> list[str]:
    """
    The parameter names a function reads, from its signature.
    """

    return [
        name for name, p in inspect.signature(function).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]


@dataclass
class ParamSpec:
    """
    One parameter's declaration. Build it with :func:`parse_param`.
    """

    name: str
    prior: Prior | None = None
    value: float | Callable | None = None
    ref: Prior | float | None = None
    proposal: float | None = None
    latex: str | None = None
    derived: bool | Callable = False
    drop: bool = False

    # Filled from `value`/`derived` when they are functions.
    arguments: list = field(default_factory=list)

    @property
    def role(self) -> str:
        """``"sampled"``, ``"fixed"``, ``"dependent"`` or ``"derived"``."""

        if self.derived is not False:
            return "derived"

        if self.prior is not None:
            return "sampled"

        if callable(self.value):
            return "dependent"

        return "fixed"

    @property
    def label(self) -> str:
        return self.latex or self.name


def parse_param(name: str, spec: Any) -> ParamSpec:
    """
    Read one entry of an input's ``params`` block.

    Raises
    ------
    ValueError
        For an entry that does not make a parameter, saying why.
    """

    if isinstance(spec, ParamSpec):
        return spec

    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return ParamSpec(name=name, value=float(spec))

    if callable(spec) or isinstance(spec, str):

        function = _as_function(spec, f"Parameter {name!r}")

        return ParamSpec(name=name, value=function, arguments=_arguments(function))

    if not isinstance(spec, dict):
        raise ValueError(
            f"Parameter {name!r}: expected a number, a lambda or a dict, "
            f"got {spec!r}."
        )

    unknown = set(spec) - _KNOWN_KEYS

    if unknown:
        raise ValueError(
            f"Parameter {name!r}: unknown key(s) {sorted(unknown)}; a "
            f"parameter takes {sorted(_KNOWN_KEYS)}."
        )

    has_prior = spec.get("prior") is not None
    has_value = spec.get("value") is not None
    derived = spec.get("derived", False)

    if sum([has_prior, has_value, derived is not False]) > 1:
        raise ValueError(
            f"Parameter {name!r} is given more than one of prior, value and "
            f"derived. It is sampled, fixed or derived -- one of them."
        )

    out = ParamSpec(
        name=name,
        latex=spec.get("latex"),
        proposal=spec.get("proposal"),
        drop=bool(spec.get("drop", False)),
    )

    if has_prior:

        out.prior = make_prior(spec["prior"])

        ref = spec.get("ref")

        if ref is not None:
            out.ref = (
                float(ref) if isinstance(ref, (int, float)) else make_prior(ref)
            )

    elif has_value:

        value = spec["value"]

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.value = float(value)
        else:
            out.value = _as_function(value, f"Parameter {name!r}")
            out.arguments = _arguments(out.value)

    elif derived is not False:

        if derived is True:
            out.derived = True
        else:
            out.derived = _as_function(derived, f"Derived parameter {name!r}")
            out.arguments = _arguments(out.derived)

    else:

        raise ValueError(
            f"Parameter {name!r} has neither a prior, a value nor "
            f"'derived'; nothing says what it is."
        )

    if out.drop and out.role != "sampled":
        raise ValueError(
            f"Parameter {name!r}: only a sampled parameter can be dropped."
        )

    return out


class ParameterSet:
    """
    Every parameter of a model, by role, and the operations a sampler
    needs on them.

    Parameters
    ----------
    specs : dict[str, ParamSpec | Any]
        Parsed (or parseable) declarations, keyed by name.
    """

    def __init__(self, specs: dict):

        self.specs = {name: parse_param(name, s) for name, s in specs.items()}

        self.sampled = [n for n, s in self.specs.items() if s.role == "sampled"]
        self.fixed = {
            n: s.value for n, s in self.specs.items() if s.role == "fixed"
        }
        self.dependent = [n for n, s in self.specs.items() if s.role == "dependent"]
        self.derived = [n for n, s in self.specs.items() if s.role == "derived"]

        self._dependent_order = self._order_dependents()

    # ---------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------

    def _order_dependents(self) -> list[str]:
        """
        Dependent parameters in an order where each one's arguments are
        already known when it is computed.
        """

        known = set(self.sampled) | set(self.fixed)

        pending = list(self.dependent)
        order = []

        while pending:

            ready = [
                n for n in pending if set(self.specs[n].arguments) <= known
            ]

            if not ready:

                missing = {
                    n: sorted(set(self.specs[n].arguments) - known
                              - set(pending))
                    for n in pending
                }

                cyclic = [n for n in pending if not missing[n]]

                if cyclic:
                    raise ValueError(
                        f"Dependent parameters {cyclic} depend on each "
                        f"other in a cycle."
                    )

                raise ValueError(
                    "Dependent parameters read names that are neither "
                    "sampled, fixed nor dependent: "
                    + "; ".join(f"{n} needs {m}" for n, m in missing.items() if m)
                )

            order.extend(ready)
            known.update(ready)
            pending = [n for n in pending if n not in ready]

        return order

    @property
    def inputs(self) -> list[str]:
        """
        Names handed to components: sampled (unless dropped), fixed and
        dependent.
        """

        return [
            n for n in self.specs
            if self.specs[n].role in ("sampled", "fixed", "dependent")
            and not self.specs[n].drop
        ]

    def as_dict(self, point) -> dict:
        """
        ``point`` -- a dict, or an array in :attr:`sampled` order -- as a
        ``{name: value}`` dict of the sampled parameters.
        """

        if isinstance(point, dict):

            missing = [n for n in self.sampled if n not in point]

            if missing:
                raise ValueError(f"Point is missing sampled parameter(s) {missing}.")

            return {n: float(point[n]) for n in self.sampled}

        values = np.asarray(point, dtype=float).ravel()

        if values.size != len(self.sampled):
            raise ValueError(
                f"Expected {len(self.sampled)} values {self.sampled}, got "
                f"{values.size}."
            )

        return dict(zip(self.sampled, map(float, values)))

    def all_values(self, point) -> dict:
        """
        Sampled, fixed and dependent values together, dependents
        computed.
        """

        values = self.as_dict(point)
        values.update(self.fixed)

        for name in self._dependent_order:

            spec = self.specs[name]

            values[name] = float(
                spec.value(**{a: values[a] for a in spec.arguments})
            )

        return values

    def to_input(self, point) -> dict:
        """
        The values components receive: :meth:`all_values` without the
        dropped parameters.
        """

        values = self.all_values(point)

        return {n: values[n] for n in self.inputs}

    # ---------------------------------------------------------
    # Prior
    # ---------------------------------------------------------

    def logprior(self, point) -> float:
        """
        Sum of the sampled parameters' normalized log-priors.
        """

        values = self.as_dict(point)

        total = 0.0

        for name in self.sampled:

            total += self.specs[name].prior.logpdf(values[name])

            if total == -math.inf:
                return -math.inf

        return total

    def bounds(self) -> list[tuple[float, float]]:
        """Prior support of each sampled parameter, in order."""

        return [self.specs[n].prior.support for n in self.sampled]

    def centers(self) -> np.ndarray:
        """Each sampled parameter's ``ref`` value, or prior center."""

        out = []

        for name in self.sampled:

            spec = self.specs[name]

            if isinstance(spec.ref, (int, float)):
                out.append(float(spec.ref))
            elif isinstance(spec.ref, Prior):
                out.append(spec.ref.center)
            else:
                out.append(spec.prior.center)

        return np.array(out)

    def scales(self) -> np.ndarray:
        """
        A typical step per sampled parameter: ``proposal`` if given,
        else the ``ref`` width, else the prior's.
        """

        out = []

        for name in self.sampled:

            spec = self.specs[name]

            if spec.proposal:
                out.append(float(spec.proposal))
            elif isinstance(spec.ref, Prior):
                out.append(spec.ref.scale)
            else:
                out.append(spec.prior.scale)

        return np.array(out)

    def sample_ref(self, rng: np.random.Generator, tries: int = 1000) -> np.ndarray:
        """
        A starting point: drawn from each ``ref`` (or the prior), kept
        inside the prior.
        """

        for _ in range(tries):

            point = []

            for name in self.sampled:

                spec = self.specs[name]

                if isinstance(spec.ref, (int, float)):
                    point.append(float(spec.ref))
                else:
                    point.append((spec.ref or spec.prior).sample(rng))

            if np.isfinite(self.logprior(point)):
                return np.array(point)

        raise RuntimeError(
            "Could not draw a starting point inside the prior from the "
            "'ref' distributions."
        )

    # ---------------------------------------------------------
    # Derived
    # ---------------------------------------------------------

    def compute_derived(self, values: dict, provided: dict) -> dict:
        """
        Every derived parameter: from ``provided`` (component outputs)
        when declared ``derived: True``, else from its lambda over
        ``values`` and what has been derived so far.
        """

        out = {}
        known = dict(values)
        known.update(provided)

        pending = list(self.derived)

        while pending:

            progressed = False

            for name in list(pending):

                spec = self.specs[name]

                if spec.derived is True:

                    out[name] = float(provided.get(name, math.nan))

                elif set(spec.arguments) <= set(known):

                    out[name] = float(
                        spec.derived(**{a: known[a] for a in spec.arguments})
                    )

                else:
                    continue

                known[name] = out[name]
                pending.remove(name)
                progressed = True

            if not progressed:

                for name in pending:
                    out[name] = math.nan

                break

        return out
