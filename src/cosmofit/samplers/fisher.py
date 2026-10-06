"""
The Fisher matrix: the curvature of ``chi2`` at the best fit, and its
inverse as a Gaussian approximation to the posterior.

``F_ij = (1/2) d^2 chi2 / d theta_i d theta_j``, by central differences
-- a forward difference would inherit a first-derivative error, and at
the minimum the first derivative is what is supposed to vanish. It
costs ``~2 d^2`` evaluations where an MCMC costs thousands per
parameter, which is the reason to have it for a likelihood whose every
evaluation runs a Boltzmann code. It is only an approximation: good for
near-elliptical contours, poor for a parameter against a prior edge or
a posterior with a plateau.

What the old ``Fitter.fisher`` got wrong, and this does not:

**The step is chosen by the likelihood, not by the prior.** A step of
1e-3 of the prior width is far too small for a likelihood with
numerical noise (CAMB's is enough to swamp it) and needlessly
arbitrary for the rest. Here each parameter's step is found so that it
moves ``chi2`` by about one -- one conditional standard deviation --
starting from the parameter's scale and correcting by the measured
curvature. That is large against any noise and small enough for the
quadratic approximation where one holds at all. A fixed ``step`` (in
units of each parameter's scale) is still available.

**The step is tested.** The diagonal is recomputed at half the steps;
an entry that moves by more than 10% means the quadratic approximation
does not hold there, and is warned about.

**The result is checked before it is inverted.** A matrix that is not
positive definite is not the curvature at a minimum -- the point is not
the best fit, or a direction is flat. The worst direction is named, the
pseudo-inverse is returned, and the error of a parameter it leaves
without a positive variance is NaN. Only a positive-definite inverse is
written to ``.covmat``, where an MCMC can start from it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .base import Sampler
from .minimize import Minimize


__all__ = ["Fisher"]


#: chi2 change the automatic step aims for along each axis, and the
#: range it accepts without refining further.
_TARGET = 1.0
_ACCEPT = (0.3, 3.0)

#: Refinements of each step before settling for what it has.
_REFINE = 8

#: Half-step change in a diagonal entry that is warned about.
_STEP_TOL = 0.1


class Fisher(Sampler):
    """
    Options
    -------
    point : dict, optional
        Where to expand, every sampled parameter by name. Without it the
        best fit is found first, by ``minimize`` with the options under
        ``minimize``.
    minimize : dict, optional
        Options for that minimization (``method``, ``starts``, ...).
    ignore_prior : bool
        Expand ``-2 ln L`` rather than ``-2 ln posterior``. Default False.
    step : "auto" or float
        ``"auto"`` (the default) finds each parameter's step from the
        likelihood; a number is a fixed step in units of each
        parameter's scale (``proposal``, ``ref`` width or prior width).
    check_step : bool
        Recompute the diagonal at half the steps and warn if it moves.
        Default True; ``2 d`` more evaluations.
    """

    defaults = {
        "point": None,
        "minimize": None,
        "ignore_prior": False,
        "step": "auto",
        "check_step": True,
    }

    def initialize(self) -> None:

        self._set_columns()

        if self.d == 0:
            raise ValueError("Fisher needs at least one sampled parameter.")

        self.scale = self.model.parameters.scales()

        point = self.options["point"]

        if point is not None:

            missing = set(self.names) - set(point)
            unknown = set(point) - set(self.names)

            if missing or unknown:
                raise ValueError(
                    f"fisher: 'point' must give exactly the sampled parameters "
                    f"{self.names}"
                    + (f"; missing {sorted(missing)}" if missing else "")
                    + (f"; not sampled {sorted(unknown)}" if unknown else "")
                    + "."
                )

        step = self.options["step"]

        if step != "auto" and not (isinstance(step, (int, float)) and step > 0):
            raise ValueError(f"fisher: 'step' is 'auto' or a positive number, not {step!r}.")

        self.minimizer = None
        self.result = None
        self.n_evaluations = 0

    # ---------------------------------------------------------

    def _chi2(self, theta) -> float:

        self.n_evaluations += 1

        result = self.model.logposterior(theta, want_derived=False)

        value = result.loglike if self.options["ignore_prior"] else result.logpost

        return -2.0 * value if result.rejected is None else math.inf

    def _expansion_point(self) -> np.ndarray:

        if self.options["point"] is not None:
            return np.array([float(self.options["point"][n]) for n in self.names])

        options = {"ignore_prior": self.options["ignore_prior"], **(self.options["minimize"] or {})}

        self.minimizer = Minimize(options, self.model, seed=self.seed)
        self.minimizer._search()

        return self.minimizer._to_theta(self.minimizer.best.x)

    def _room(self, theta) -> np.ndarray:
        """How far each parameter can step either way inside its prior."""

        room = []

        for i, (low, high) in enumerate(self.model.parameters.bounds()):

            space = min(theta[i] - low, high - theta[i])

            if not space > 0.0:
                raise ValueError(
                    f"fisher: '{self.names[i]}' = {theta[i]:g} sits on its prior "
                    f"edge, where the posterior has no curvature to measure. "
                    f"A Fisher matrix is the wrong tool for it; profile it, "
                    f"or sample."
                )

            room.append(0.99 * space)

        return np.array(room)

    def _axis(self, theta, i, h, centre) -> float:
        """``chi2`` change half-way between one step up and one down."""

        shift = np.zeros(self.d)
        shift[i] = h

        return 0.5 * (self._chi2(theta + shift) + self._chi2(theta - shift)) - centre

    def _steps(self, theta, centre, room) -> np.ndarray:

        steps = np.minimum(self.scale, room)

        if self.options["step"] != "auto":
            return np.minimum(float(self.options["step"]) * self.scale, room)

        for i in range(self.d):

            h = steps[i]

            for _ in range(_REFINE):

                delta = self._axis(theta, i, h, centre)

                if _ACCEPT[0] <= delta <= _ACCEPT[1]:
                    break

                if not math.isfinite(delta):
                    h /= 4.0
                elif delta <= 0.0:
                    h *= 4.0
                else:
                    h *= math.sqrt(_TARGET / delta)

                h = min(h, room[i])

            steps[i] = h

        return steps

    def _matrix(self, theta, centre, steps) -> np.ndarray:

        F = np.zeros((self.d, self.d))

        for i in range(self.d):
            F[i, i] = self._axis(theta, i, steps[i], centre) / steps[i] ** 2

        for i in range(self.d):
            for j in range(i + 1, self.d):

                si, sj = np.zeros(self.d), np.zeros(self.d)
                si[i], sj[j] = steps[i], steps[j]

                mixed = (
                    self._chi2(theta + si + sj) - self._chi2(theta + si - sj)
                    - self._chi2(theta - si + sj) + self._chi2(theta - si - sj)
                )

                F[i, j] = F[j, i] = mixed / (8.0 * steps[i] * steps[j])

        return F

    # ---------------------------------------------------------

    def run(self) -> None:

        self._prepare_output()

        theta = self._expansion_point()
        centre = self._chi2(theta)

        if not math.isfinite(centre):
            raise ValueError(
                f"fisher: the posterior is zero at the expansion point "
                f"{dict(zip(self.names, theta))}."
            )

        room = self._room(theta)
        steps = self._steps(theta, centre, room)

        F = self._matrix(theta, centre, steps)

        if not np.all(np.isfinite(F)):
            bad = sorted({self.names[i] for i, j in zip(*np.where(~np.isfinite(F))) for i in (i, j)})
            raise ValueError(
                f"fisher: chi2 is not finite a step away from the expansion "
                f"point along {bad}; the likelihood is undefined there."
            )

        unstable = []

        if self.options["check_step"]:

            for i in range(self.d):

                half = self._axis(theta, i, steps[i] / 2.0, centre) / (steps[i] / 2.0) ** 2

                # No curvature at either step is a flat direction, which
                # the positive-definiteness check below names.
                if half == F[i, i] == 0.0:
                    continue

                if not abs(half - F[i, i]) < _STEP_TOL * abs(F[i, i]):
                    unstable.append(self.names[i])

            if unstable:
                warnings.warn(
                    f"The Fisher matrix depends on the step for {unstable}: "
                    f"halving it moved those diagonal entries by more than "
                    f"{_STEP_TOL:.0%}. The posterior is not quadratic over a "
                    f"step there, or the likelihood is noisy on that scale.",
                    UserWarning, stacklevel=2,
                )

        self.result = self._invert(theta, centre, steps, F, unstable)

        if self.output.enabled:

            self.output.write_updated(self.model, self.options)

            if self.minimizer is not None:
                self.minimizer.output = self.output
                self.minimizer._write_minimum()

            if self.result["positive_definite"]:
                self.output.write_covmat(self.names, self.result["covariance"])

    def _invert(self, theta, centre, steps, F, unstable) -> dict:

        # Decomposed in units of each parameter's scale, where the
        # entries are of order one and a flat direction stands out.
        s = self.scale
        scaled = F * np.outer(s, s)

        eigenvalues, eigenvectors = np.linalg.eigh(scaled)

        positive_definite = bool(eigenvalues.min() > 0.0)

        if positive_definite:

            covariance = np.linalg.inv(scaled)
            errors = np.sqrt(np.diag(covariance))

        else:

            worst = eigenvectors[:, int(np.argmin(eigenvalues))]

            direction = ", ".join(
                f"{w:+.2f} {name}" for w, name in zip(worst, self.names) if abs(w) > 0.1
            )

            warnings.warn(
                f"The Fisher matrix is not positive definite (smallest "
                f"eigenvalue {eigenvalues.min():.3g}, along {direction}, in "
                f"units of each parameter's scale). This is not the curvature "
                f"at a minimum: the point is not the best fit, or that "
                f"direction is unconstrained. The pseudo-inverse is returned "
                f"and errors without a positive variance are NaN.",
                UserWarning, stacklevel=3,
            )

            covariance = np.linalg.pinv(scaled)
            variances = np.diag(covariance)
            errors = np.where(variances > 0.0, np.sqrt(np.abs(variances)), np.nan)

        return {
            "names": list(self.names),
            "point": dict(zip(self.names, map(float, theta))),
            "chi2": float(centre),
            "matrix": F,
            "covariance": covariance * np.outer(s, s),
            "errors": dict(zip(self.names, map(float, errors * s))),
            "steps": dict(zip(self.names, map(float, steps))),
            "positive_definite": positive_definite,
            "unstable_steps": unstable,
            "n_evaluations": self.n_evaluations,
        }

    def products(self) -> dict:
        """
        ``point`` and the ``chi2`` there, the Fisher ``matrix`` and its
        inverse ``covariance`` (in ``names`` order, parameter units),
        ``errors``, the ``steps`` used, whether the matrix was
        ``positive_definite``, the parameters whose step test failed
        (``unstable_steps``) and ``n_evaluations``.
        """

        if self.result is None:
            raise RuntimeError("Call run() first.")

        return dict(self.result)
