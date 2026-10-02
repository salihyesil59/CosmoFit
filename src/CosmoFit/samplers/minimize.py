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
"""

from __future__ import annotations

import math

import numpy as np

from .base import Sampler


__all__ = ["Minimize"]


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
    """

    defaults = {
        "method": "L-BFGS-B",
        "starts": 1,
        "ignore_prior": False,
        "max_evals": None,
        "tol": None,
    }

    def initialize(self) -> None:

        parameters = self.model.parameters

        self.center = parameters.centers()
        self.scale = parameters.scales()

        if np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("Every sampled parameter needs a positive scale.")

        self.starts = int(self.options["starts"])

        if self.starts < 1:
            raise ValueError("starts must be at least 1.")

        self.runs = []
        self.best = None

    # ---------------------------------------------------------

    def _to_theta(self, x):
        return self.center + self.scale * np.asarray(x, dtype=float)

    def _to_x(self, theta):
        return (np.asarray(theta, dtype=float) - self.center) / self.scale

    def _objective(self, x) -> float:

        theta = self._to_theta(x)

        result = self.model.logposterior(theta, want_derived=False)

        if result.rejected is not None:
            return _WALL

        value = result.loglike if self.options["ignore_prior"] else result.logpost

        return -value if math.isfinite(value) else _WALL

    def _bounds(self):

        out = []

        for (low, high), c, s in zip(
            self.model.parameters.bounds(), self.center, self.scale,
        ):
            out.append((
                None if not math.isfinite(low) else (low - c) / s,
                None if not math.isfinite(high) else (high - c) / s,
            ))

        return out

    # ---------------------------------------------------------

    def run(self) -> None:

        from scipy.optimize import minimize

        method = self.options["method"]

        options = {}

        if self.options["max_evals"]:

            key = "maxfun" if method in ("L-BFGS-B", "TNC") else "maxfev"
            options[key] = int(self.options["max_evals"])

        bounds = self._bounds()

        for index in range(self.starts):

            theta0 = (
                self.center if index == 0
                else self.model.parameters.sample_ref(self.rng)
            )

            x0 = self._to_x(theta0)

            if method == "Nelder-Mead":

                # A simplex one unit -- one typical step -- wide in
                # every direction, which scaled coordinates make
                # meaningful for every parameter at once.
                options["initial_simplex"] = np.vstack(
                    [x0] + [x0 + np.eye(len(x0))[i] for i in range(len(x0))]
                )

            result = minimize(
                self._objective, x0, method=method, bounds=bounds,
                tol=self.options["tol"], options=options or None,
            )

            self.runs.append(result)

            if self.best is None or result.fun < self.best.fun:
                self.best = result

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
