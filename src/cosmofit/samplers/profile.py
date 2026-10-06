"""
A profile likelihood: one parameter held at each of a list of values,
``chi2`` minimized over all the others.

The honest tool where Wilks' theorem does not apply, and how a
parameter against a boundary should be reported -- a marginal
posterior smooths over a cliff that a profile shows (``LsCDM``'s
``z_dagger`` has a 28-unit one).

It maximizes the *likelihood* by default (``ignore_prior: true``): a
profile is a frequentist object, and the prior's support only bounds
the search.

Neighbouring values differ by one small step in one parameter, so
their optima are close; ``warm_start`` starts each from the last one's
solution, which costs a fraction of a cold start. Turn it off where the
surface may have a discontinuity the walk could be trapped on the wrong
side of, or give ``starts`` above one.

With an ``output`` prefix the profile is written to ``.profile.txt``:
the value, ``chi2``, ``delta_chi2`` from the profile's minimum, then the
other parameters where the minimization left them.
"""

from __future__ import annotations

import numpy as np

from .minimize import Minimize


__all__ = ["Profile"]


class Profile(Minimize):
    """
    Options
    -------
    param : str
        The sampled parameter to profile.
    values : list or dict
        Values to hold it at: a list, or ``{min, max, n}`` for ``n``
        evenly spaced ones.
    warm_start : bool
        Start each value from the previous one's optimum. Default True.
    method, starts, max_evals, tol
        As for ``minimize``.
    ignore_prior : bool
        Default True: profile the likelihood.
    """

    defaults = {
        **Minimize.defaults,
        "ignore_prior": True,
        "param": None,
        "values": None,
        "warm_start": True,
    }

    def initialize(self) -> None:

        super().initialize()

        name = self.options["param"]

        if name not in self.names:
            raise ValueError(
                f"profile: 'param' must be one of the sampled parameters "
                f"{self.names}, not {name!r}."
            )

        if self.d < 2:
            raise ValueError(
                f"Profiling '{name}' needs at least one other sampled "
                f"parameter to minimize over."
            )

        self.param = name
        self.index = self.names.index(name)
        self.values = _values(self.options["values"])

        low, high = self.model.parameters.bounds()[self.index]

        outside = [v for v in self.values if not low <= v <= high]

        if outside:
            raise ValueError(
                f"profile: {outside} lie outside the prior of '{name}' "
                f"([{low}, {high}]); the posterior is zero there."
            )

        self.points: list[np.ndarray] = []
        self.minima: list = []

    def run(self) -> None:

        self._prepare_output()

        first = None

        for value in self.values:

            self.fixed = {self.index: float(value)}

            self._search(first)

            theta = self._to_theta(self.best.x)

            self.points.append(theta)
            self.minima.append(self.best)

            if self.options["warm_start"]:
                first = theta

        self.fixed = {}

        if self.output.enabled:

            self.output.write_updated(self.model, self.options)

            products = self.products()
            others = [n for n in self.names if n != self.param]

            self.output.write_table(
                ".profile.txt",
                [self.param, "chi2", "delta_chi2", *others],
                np.column_stack([
                    products["values"], products["chi2"], products["delta_chi2"],
                    *(products["params"][n] for n in others),
                ]),
            )

    def products(self) -> dict:
        """
        ``param``, ``values``, ``chi2`` (``-2`` times the maximized
        log-likelihood, or log-posterior with ``ignore_prior: false``),
        ``delta_chi2`` from the smallest, ``params`` (each sampled
        parameter's value along the profile), and ``success`` per value.
        """

        if not self.points:
            raise RuntimeError("Call run() first.")

        chi2 = 2.0 * np.array([m.fun for m in self.minima])
        points = np.array(self.points)

        return {
            "param": self.param,
            "values": np.array(self.values),
            "chi2": chi2,
            "delta_chi2": chi2 - chi2.min(),
            "params": {n: points[:, i] for i, n in enumerate(self.names)},
            "success": [bool(m.success) for m in self.minima],
            "maximized": "likelihood" if self.options["ignore_prior"] else "posterior",
        }


def _values(spec) -> list[float]:

    if isinstance(spec, dict):

        unknown = set(spec) - {"min", "max", "n"}

        if unknown or not {"min", "max", "n"} <= set(spec):
            raise ValueError(
                f"profile: 'values' as a dict takes exactly min, max and n; "
                f"got {sorted(spec)}."
            )

        return [float(v) for v in np.linspace(spec["min"], spec["max"], int(spec["n"]))]

    if spec is None or np.ndim(spec) != 1 or len(spec) == 0:
        raise ValueError("profile: give 'values', a list or {min, max, n}.")

    return [float(v) for v in spec]
