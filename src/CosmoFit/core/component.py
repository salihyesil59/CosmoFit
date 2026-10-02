"""
Theories, likelihoods, and how they find each other.

A **likelihood** says what it needs -- ``get_requirements()`` returns
``{"H": {"z": [...]}, "rd": None}`` -- and reads it back through
``self.provider`` when asked for a log-likelihood. It never touches a
cosmology object or a parameter container.

A **theory** says what it can provide (``get_can_provide()``), is told
what is wanted of it (``must_provide(**requirements)``), and computes it
in ``calculate(state, want_derived, **params)`` into a fresh ``state``
dict. Results are read back through ``get_result(name, **kwargs)`` or a
``get_<name>`` method.

:class:`~core.model.Model` does the wiring: it matches each requirement
to the one theory that provides it, orders theories so that a theory's
own requirements are computed before it is, and hands every component
the parameters it accepts.

**Caching is by parameter value.** A theory keeps its last few states
keyed on the exact input values it was computed at, so asking again at
the same point costs nothing, and a likelihood can never read a result
computed at different parameters than the ones it was given. The old
fitting path depended instead on every caller remembering to call
``refresh()`` after changing a parameter -- and the one that did not
(``stats.derived``) evaluated every sample on a stale background.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


__all__ = [
    "Component",
    "Theory",
    "Likelihood",
    "Provider",
    "ComponentError",
]


class ComponentError(ValueError):
    """
    A model that cannot be assembled: a requirement nobody provides,
    a parameter nobody uses, a cycle between theories.
    """


class Component:
    """
    Base of theories and likelihoods.

    Parameters
    ----------
    info : dict, optional
        This component's options from the input.
    name : str, optional
        The name it was listed under; defaults to the class name.

    Class attributes
    ----------------
    params : dict
        Parameters this component owns, with default declarations --
        a likelihood's nuisance parameters and their priors, say. The
        input's ``params`` block overrides them.
    """

    #: Parameters this component owns, ``{name: declaration}``.
    params: dict = {}

    def __init__(self, info: dict | None = None, name: str | None = None):

        self.info = dict(info or {})
        self.name = name or type(self).__name__
        self.provider = None

        self.initialize()

    # ---------------------------------------------------------

    def initialize(self) -> None:
        """Set up from ``self.info``. Override; called by ``__init__``."""

    def initialize_with_provider(self, provider: "Provider") -> None:
        """
        Called once every component exists and requirements are
        resolved, with the provider this component reads through.
        """

        self.provider = provider

    def initialize_complete(self) -> None:
        """
        Called once every component has its provider. For setup that
        must wait for *other* components to be ready.
        """

    def check_model(self, model) -> None:
        """
        Called last, with the assembled :class:`~core.model.Model`, for
        checks that need to see the whole of it -- which datasets are
        combined, which parameters are sampled. Warn or raise here.
        """

    # ---------------------------------------------------------

    def accepts(self, name: str) -> bool:
        """
        Whether this component takes the input parameter ``name``.
        Defaults to the names in :attr:`params`.
        """

        return name in type(self).params

    def get_requirements(self) -> dict:
        """
        What this component needs from theories:
        ``{quantity: options or None}``.
        """

        return {}

    def get_derived_params(self) -> list[str]:
        """Names of the derived parameters this component can output."""

        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class Theory(Component):
    """
    A component that computes quantities for others.

    Subclasses implement :meth:`get_can_provide`, :meth:`calculate`,
    and either :meth:`get_result` or ``get_<name>`` methods.
    """

    #: How many past states to keep. A theory wrapping a single
    #: mutable object can only hold one, and says so by setting 1.
    cache_size: int = 3

    def __init__(self, info: dict | None = None, name: str | None = None):

        self._cache: OrderedDict = OrderedDict()
        self.current_state: dict | None = None
        self.requested: dict = {}

        #: Calls of :meth:`calculate`, as opposed to cache hits.
        self.n_calculations = 0

        super().__init__(info, name)

    # ---------------------------------------------------------

    def get_can_provide(self) -> list[str]:
        """Quantities this theory can compute."""

        return []

    def must_provide(self, **requirements) -> None:
        """
        Record what is wanted of this theory. Called once per requiring
        component; the default merges the requests by name.
        """

        for name, options in requirements.items():
            self.requested.setdefault(name, []).append(options)

    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool | None:
        """
        Compute everything requested at ``params`` into ``state``.
        Return ``False`` if the parameters admit no solution (the point
        is then rejected, not an error).
        """

        raise NotImplementedError

    # ---------------------------------------------------------

    def compute(self, params: dict, want_derived: bool = True) -> bool:
        """
        Make :attr:`current_state` the state at ``params``, from the
        cache if it holds one. Returns ``False`` if :meth:`calculate`
        rejected the point.
        """

        key = tuple(sorted(params.items()))

        if key in self._cache:

            self._cache.move_to_end(key)
            self.current_state = self._cache[key]

            return self.current_state is not None

        state = {"params": dict(params), "derived": {}}

        self.n_calculations += 1

        ok = self.calculate(state, want_derived, **params)

        stored = None if ok is False else state

        self._cache[key] = stored

        while len(self._cache) > max(1, self.cache_size):
            self._cache.popitem(last=False)

        self.current_state = stored

        return stored is not None

    def get_result(self, name: str, **kwargs) -> Any:
        """
        A computed quantity from the current state. The default looks
        for a ``get_<name>`` method, then for ``state[name]``.
        """

        method = getattr(self, f"get_{name}", None)

        if method is not None:
            return method(**kwargs)

        if self.current_state is None or name not in self.current_state:
            raise KeyError(f"{self.name} has no result {name!r} at this point.")

        return self.current_state[name]

    def get_current_derived(self) -> dict:

        if self.current_state is None:
            return {}

        return dict(self.current_state.get("derived", {}))

    def clear_cache(self) -> None:

        self._cache.clear()
        self.current_state = None


class Likelihood(Component):
    """
    A component that returns a log-likelihood.

    Subclasses implement :meth:`logp`, reading theory results through
    ``self.provider``.
    """

    def logp(self, **params) -> float:
        """
        ``ln L`` at the current point. ``params`` are this likelihood's
        own input parameters (its nuisance parameters, say).
        """

        raise NotImplementedError


class Provider:
    """
    What a component reads theory results through.

    ``provider.get_result("H", z=...)`` asks the theory that provides
    ``"H"``; ``provider.get_H(z=...)`` is the same thing.
    """

    def __init__(self, providers: dict[str, Theory]):

        self._providers = dict(providers)

    def get_result(self, name: str, **kwargs) -> Any:

        try:
            theory = self._providers[name]
        except KeyError:
            raise KeyError(
                f"Nothing provides {name!r}. Only quantities a component "
                f"listed in get_requirements() can be read."
            ) from None

        return theory.get_result(name, **kwargs)

    def __getattr__(self, attribute: str):

        if attribute.startswith("get_"):

            name = attribute[4:]

            def getter(**kwargs):
                return self.get_result(name, **kwargs)

            return getter

        raise AttributeError(attribute)

    @property
    def provides(self) -> dict[str, str]:
        """``{quantity: theory name}``."""

        return {q: t.name for q, t in self._providers.items()}
