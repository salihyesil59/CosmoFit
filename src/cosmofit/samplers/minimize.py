"""
The maximum of the posterior (or of the likelihood), found by
optimization.

Two things here that the old ``Fitter.best_fit`` did not do:

**It works in scaled coordinates.** Each parameter is optimized as
``(theta - center) / scale``, with the scale its proposal width, its
``ref`` width or its prior's -- so ``H0 ~ 70`` and ``omega_b ~ 0.02``
both move by order one per unit step. A finite-difference gradient and
an initial simplex then mean the same thing for every parameter.

**It maximizes the posterior by default.** With a Gaussian prior on a
parameter, the maximum-likelihood point and the MAP differ, and they
are different questions. ``ignore_prior: true`` asks the
maximum-likelihood one (the prior's support still bounds the search).

With an ``output`` prefix the best point is written to
``.minimum.txt``, as one row in the chains' format.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Sampler


__all__ = ["Minimize"]


#: Methods that follow a (finite-difference) gradient, and are polished.
_GRADIENT = {"L-BFGS-B", "TNC", "SLSQP", "BFGS", "CG", "trust-constr"}

#: A finite stand-in for -inf, so an optimizer that steps outside the
#: allowed region sees a wall rather than a NaN gradient.
_WALL = 1.0e30


class Minimize(Sampler):
    """
    Options
    -------
    method : str
        Any ``scipy.optimize.minimize`` method; default ``"L-BFGS-B"``.
        ``"Nelder-Mead"`` and ``"Powell"`` need no gradient.
    starts : int
        Independent starting points: the first at the ``ref`` centers,
        the rest drawn from ``ref`` (or the prior). The best is kept.
    ignore_prior : bool
        Maximize the likelihood instead of the posterior.
    max_evals : int, optional
        Evaluation budget per start (``maxfun``/``maxfev``).
    tol : float, optional
        Passed to ``scipy.optimize.minimize``.
    polish : bool
        After a gradient-based method, go on from where it stopped with
        Nelder-Mead and keep the better of the two. Default True.
    """

    defaults = {
        "method": "L-BFGS-B",
        "starts": 1,
        "ignore_prior": False,
        "max_evals": None,
        "tol": None,
        "polish": True,
    }

    def initialize(self) -> None:

        self._set_columns()

        parameters = self.model.parameters

        self.center = parameters.centers()
        self.scale = parameters.scales()

        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("Every sampled parameter needs a positive scale.")

        self.starts = int(self.options["starts"])

        if self.starts < 1:
            raise ValueError("starts must be at least 1.")

        #: Sampled parameters held at a value, ``{index: value}``; the
        #: search runs over the others. Empty here; a profile fills it.
        self.fixed: dict[int, float] = {}

        self.runs = []
        self.best = None

    # ---------------------------------------------------------

    @property
    def _free(self) -> list[int]:
        return [i for i in range(self.d) if i not in self.fixed]

    def _to_theta(self, x):

        theta = self.center.copy()

        free = self._free

        theta[free] = self.center[free] + self.scale[free] * np.asarray(x, dtype=float)

        for index, value in self.fixed.items():
            theta[index] = value

        return theta

    def _to_x(self, theta):

        free = self._free

        return (np.asarray(theta, dtype=float)[free] - self.center[free]) / self.scale[free]

    def _objective(self, x) -> float:

        theta = self._to_theta(x)

        result = self.model.logposterior(theta, want_derived=False)

        if result.rejected is not None:
            return _WALL

        value = result.loglike if self.options["ignore_prior"] else result.logpost

        return -value if math.isfinite(value) else _WALL

    def _bounds(self):

        out = []

        bounds = self.model.parameters.bounds()

        for i in self._free:

            (low, high), c, s = bounds[i], self.center[i], self.scale[i]

            out.append((
                None if not math.isfinite(low) else (low - c) / s,
                None if not math.isfinite(high) else (high - c) / s,
            ))

        return out

    # ---------------------------------------------------------

    def _minimize_from(self, theta0):
        """
        One optimization over the free parameters, from ``theta0``, then
        polished (see :attr:`polish`).

        Why the polish: a gradient-based method differentiates ``chi2``
        by finite differences, and stops when a step no longer reduces
        it by a relative ``~1e-9``. On a likelihood whose ``chi2`` is
        ~1500 and not quite smooth at that level, it can stop well short
        -- profiling CPL's ``w0`` on CC + DESI + Pantheon+, L-BFGS-B
        reported convergence 8 units of ``chi2`` above the minimum
        Nelder-Mead and Powell both found, and the profile came out
        with a kink a marginal posterior would never show.
        """

        result = self._optimize(theta0, self.options["method"])

        if self.options["polish"] and self.options["method"] in _GRADIENT:

            polished = self._optimize(self._to_theta(result.x), "Nelder-Mead")

            polished.nfev += result.nfev

            if polished.fun < result.fun:
                return polished

            result.nfev = polished.nfev

        return result

    def _optimize(self, theta0, method):

        from scipy.optimize import minimize

        options = {}

        if self.options["max_evals"]:

            key = "maxfun" if method in ("L-BFGS-B", "TNC") else "maxfev"
            options[key] = int(self.options["max_evals"])

        x0 = self._to_x(theta0)

        if method == "Nelder-Mead":

            # A simplex one unit -- one typical step -- wide in every
            # direction, which scaled coordinates make meaningful for
            # every parameter at once.
            options["initial_simplex"] = np.vstack(
                [x0] + [x0 + np.eye(len(x0))[i] for i in range(len(x0))]
            )

        return minimize(
            self._objective, x0, method=method, bounds=self._bounds(),
            tol=self.options["tol"], options=options or None,
        )

    def _search(self, first=None) -> None:
        """
        :attr:`starts` optimizations; the first from ``first`` (default
        the ``ref`` centers), the rest from draws of ``ref``.
        """

        self.runs = []
        self.best = None

        for index in range(self.starts):

            if index == 0:
                theta0 = self.center if first is None else np.asarray(first, dtype=float)
            else:
                theta0 = self.model.parameters.sample_ref(self.rng)

            result = self._minimize_from(theta0)

            self.runs.append(result)

            if self.best is None or result.fun < self.best.fun:
                self.best = result

    def run(self) -> None:

        self._prepare_output()

        self._search()

        if self.output.enabled:

            self.output.write_updated(self.model, self.options)
            self._write_minimum()

    def _write_minimum(self) -> None:

        theta = self._to_theta(self.best.x)
        result = self.model.logposterior(theta)

        row = np.concatenate([[1.0, -result.logpost], self._columns_of(theta, result)])

        self.output.write_table(
            ".minimum.txt", ["weight", "minuslogpost", *self.columns], row,
        )

    # ---------------------------------------------------------

    def products(self) -> dict:

        if self.best is None:
            raise RuntimeError("Call run() first.")

        theta = self._to_theta(self.best.x)

        point = dict(zip(self.model.sampled_params, map(float, theta)))

        evaluation = self.model.logposterior(theta)

        return {
            "point": point,
            "logpost": evaluation.logpost,
            "logprior": evaluation.logprior,
            "loglikes": dict(evaluation.loglikes),
            "chi2": evaluation.chi2,
            "derived": dict(evaluation.derived),
            "success": bool(self.best.success),
            "message": str(self.best.message),
            "n_evaluations": int(sum(r.nfev for r in self.runs)),
            "all_minima": sorted(float(r.fun) for r in self.runs),
            "maximized": "likelihood" if self.options["ignore_prior"] else "posterior",
        }
