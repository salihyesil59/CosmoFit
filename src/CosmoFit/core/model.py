"""
A model: theories, likelihoods and parameters, assembled from an input,
evaluated as one log-posterior.

::

    model = get_model({
        "theory": {"legacy_cosmology": {"model": "LCDM"}},
        "likelihood": {"cc": None, "desi": {"version": "desi2025"}},
        "params": {
            "H0": {"prior": {"min": 50, "max": 90}},
            "Omega_m": {"prior": {"min": 0.1, "max": 0.6}},
            "rd": 147.1,
        },
    })

    result = model.logposterior({"H0": 68.0, "Omega_m": 0.31})
    result.logpost, result.loglikes, result.chi2

Assembly checks everything that can be checked before a single point
is evaluated, and fails with a message naming the problem: a
requirement no theory provides or two theories both claim, theories
that require each other in a cycle, an input parameter no component
takes (a misspelled name, usually), a derived parameter nothing can
compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .component import ComponentError, Likelihood, Provider, Theory
from .info import load_info, validate_info
from .parameters import ParameterSet
from .registry import resolve


__all__ = ["Model", "PosteriorPoint", "get_model"]


#: Exceptions that mean "no prediction at this point" rather than "the
#: code is broken": an unphysical E(z)^2 < 0, an overflow, a solver
#: refusing. The point gets -inf, and is counted.
_REJECTIONS = (ValueError, FloatingPointError, ArithmeticError, RuntimeError)


@dataclass
class PosteriorPoint:
    """
    One point's evaluation.

    Attributes
    ----------
    logpost : float
        ``logprior + sum(loglikes)``.
    logprior : float
        Normalized log-prior of the sampled parameters.
    loglikes : dict[str, float]
        Per likelihood.
    derived : dict[str, float]
    rejected : str or None
        Why the point was rejected, if it was.
    """

    logpost: float
    logprior: float
    loglikes: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    rejected: str | None = None

    @property
    def loglike(self) -> float:
        return float(sum(self.loglikes.values())) if self.loglikes else -math.inf

    @property
    def chi2(self) -> dict:
        """``-2 ln L`` per likelihood, plus ``"total"``."""

        out = {name: -2.0 * value for name, value in self.loglikes.items()}
        out["total"] = -2.0 * self.loglike

        return out


class Model:
    """
    Parameters
    ----------
    info : dict, str or Path
        The input; see :mod:`core.info`. A ``sampler`` block, if any, is
        ignored here.
    """

    def __init__(self, info):

        self.info = validate_info(load_info(info))

        self.theories: dict[str, Theory] = {}
        self.likelihoods: dict[str, Likelihood] = {}

        for name, options in self.info["theory"].items():
            self.theories[name] = self._build("theory", name, options)

        for name, options in self.info["likelihood"].items():
            self.likelihoods[name] = self._build("likelihood", name, options)

        self.parameters = ParameterSet(self._merged_params())

        # Requirements first: a missing theory explains everything
        # downstream of it (its parameters then go unused too).
        self._resolve_requirements()
        self._assign_parameters()

        for component in self.components:
            component.initialize_complete()

        self._check_derived()

        for component in self.components:
            component.check_model(self)

        #: Rejected points, by reason.
        self.rejections: dict[str, int] = {}

    # ---------------------------------------------------------
    # Assembly
    # ---------------------------------------------------------

    @staticmethod
    def _build(kind, name, options):

        cls = resolve(kind, name, options)

        expected = Theory if kind == "theory" else Likelihood

        if not (isinstance(cls, type) and issubclass(cls, expected)):
            raise ComponentError(
                f"{kind} {name!r} resolved to {cls!r}, which is not a "
                f"{expected.__name__}."
            )

        options = {k: v for k, v in options.items() if k != "class"}

        return cls(options, name=name)

    @property
    def components(self) -> list:
        return [*self.theories.values(), *self.likelihoods.values()]

    def _merged_params(self) -> dict:
        """
        Components' own parameter defaults, overridden by the input.
        """

        merged = {}

        for component in self.components:
            merged.update(type(component).params)

        merged.update(self.info["params"])

        return merged

    def _assign_parameters(self) -> None:
        """
        Decide which component receives which input parameter, and refuse
        an input no component takes.
        """

        self._inputs_of = {c.name: [] for c in self.components}

        unused = []

        for name in self.parameters.inputs:

            takers = [c for c in self.components if c.accepts(name)]

            if not takers:
                unused.append(name)

            for component in takers:
                self._inputs_of[component.name].append(name)

        if unused:
            raise ComponentError(
                f"No theory or likelihood takes the parameter(s) {unused}. "
                f"Check the spelling; a parameter used only to compute "
                f"others is declared with 'drop: True'."
            )

    def _resolve_requirements(self) -> None:
        """
        Match every requirement to the one theory that provides it, and
        order theories so each is computed after those it requires.
        """

        def provider_of(quantity, asking):

            offering = [
                t for t in self.theories.values()
                if quantity in t.get_can_provide() and t is not asking
            ]

            if not offering:
                raise ComponentError(
                    f"{asking.name} requires {quantity!r}, and no theory "
                    f"provides it. Theories here provide: "
                    + str({t.name: sorted(t.get_can_provide())
                           for t in self.theories.values()})
                )

            if len(offering) > 1:
                raise ComponentError(
                    f"{quantity!r}, required by {asking.name}, is provided "
                    f"by more than one theory: {[t.name for t in offering]}."
                )

            return offering[0]

        providers = {}
        depends_on = {name: set() for name in self.theories}

        for component in self.components:

            for quantity, options in component.get_requirements().items():

                theory = provider_of(quantity, component)

                theory.must_provide(**{quantity: options})

                providers[quantity] = theory

                if isinstance(component, Theory):
                    depends_on[component.name].add(theory.name)

        # Kahn's algorithm: theories with nothing left to wait for first.
        order = []
        remaining = {n: set(d) for n, d in depends_on.items()}

        while remaining:

            ready = sorted(n for n, d in remaining.items() if not d)

            if not ready:
                raise ComponentError(
                    f"Theories {sorted(remaining)} require each other in "
                    f"a cycle."
                )

            order.extend(ready)

            for name in ready:
                del remaining[name]

            for deps in remaining.values():
                deps.difference_update(ready)

        self.theory_order = order

        self.provider = Provider(providers)

        for component in self.components:
            component.initialize_with_provider(self.provider)

    def _check_derived(self) -> None:

        available = set(self.parameters.inputs) | set(self.parameters.derived)

        for component in self.components:
            available.update(component.get_derived_params())

        for name in self.parameters.derived:

            spec = self.parameters.specs[name]

            if spec.derived is True and not any(
                name in c.get_derived_params() for c in self.components
            ):
                raise ComponentError(
                    f"Derived parameter {name!r} is declared 'derived: True', "
                    f"but no component outputs it."
                )

            missing = set(spec.arguments) - available - set(
                self.parameters.specs
            )

            if missing:
                raise ComponentError(
                    f"Derived parameter {name!r} reads {sorted(missing)}, "
                    f"which nothing provides."
                )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    @property
    def sampled_params(self) -> list[str]:
        return list(self.parameters.sampled)

    def _reject(self, reason: str, logprior: float) -> PosteriorPoint:

        self.rejections[reason] = self.rejections.get(reason, 0) + 1

        return PosteriorPoint(
            logpost=-math.inf, logprior=logprior, rejected=reason,
        )

    def logposterior(self, point, want_derived: bool = True) -> PosteriorPoint:
        """
        Evaluate the model at ``point`` (a dict, or an array in
        :attr:`sampled_params` order).

        A point outside the prior is not evaluated. A theory or
        likelihood that cannot produce a prediction there gives
        ``-inf`` with the reason in ``rejected``, counted in
        :attr:`rejections`.
        """

        logprior = self.parameters.logprior(point)

        if not math.isfinite(logprior):
            return PosteriorPoint(
                logpost=-math.inf, logprior=logprior, rejected="prior",
            )

        values = self.parameters.all_values(point)

        provided = {}

        for name in self.theory_order:

            theory = self.theories[name]

            inputs = {p: values[p] for p in self._inputs_of[name]}

            try:
                ok = theory.compute(inputs, want_derived)
            except NotImplementedError:
                raise
            except _REJECTIONS as error:
                return self._reject(
                    f"{name}: {type(error).__name__}", logprior,
                )

            if not ok:
                return self._reject(f"{name}: no solution", logprior)

            provided.update(theory.get_current_derived())

        loglikes = {}

        for name, likelihood in self.likelihoods.items():

            inputs = {p: values[p] for p in self._inputs_of[name]}

            try:
                value = float(likelihood.logp(**inputs))
            except NotImplementedError:
                raise
            except _REJECTIONS as error:
                return self._reject(
                    f"{name}: {type(error).__name__}", logprior,
                )

            if not math.isfinite(value):
                return self._reject(f"{name}: non-finite", logprior)

            loglikes[name] = value

        derived = (
            self.parameters.compute_derived(values, provided)
            if want_derived else {}
        )

        return PosteriorPoint(
            logpost=logprior + sum(loglikes.values()),
            logprior=logprior,
            loglikes=loglikes,
            derived=derived,
        )

    def loglike(self, point) -> float:
        """``sum(ln L)`` at ``point``, ``-inf`` if rejected."""

        result = self.logposterior(point, want_derived=False)

        return result.loglike if result.rejected is None else -math.inf

    def logprior(self, point) -> float:
        return self.parameters.logprior(point)

    def __call__(self, point) -> float:
        """The log-posterior as a plain float, for samplers."""

        return self.logposterior(point, want_derived=False).logpost

    def __repr__(self) -> str:

        return (
            f"Model(theories={list(self.theories)}, "
            f"likelihoods={list(self.likelihoods)}, "
            f"sampled={self.sampled_params})"
        )


def get_model(info) -> Model:
    """Build a :class:`Model` from an input."""

    return Model(info)

