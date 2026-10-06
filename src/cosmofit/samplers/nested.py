"""
Nested sampling with dynesty: the Bayesian evidence, and posterior
samples on the way.

The tool for comparisons a likelihood-ratio test cannot make --
``LsCDM`` against ``LCDM``, reached only as ``z_dagger -> infinity``,
or ``DGP``, which is not nested at all. Read :mod:`stats.evidence` on
prior sensitivity before quoting a Bayes factor: ``ln Z`` moves with
the prior's width, so every prior here enters the evidence, and a
comparison is only as meaningful as the priors are.

Every prior kind is supported, through its inverse CDF: the unit cube
is mapped parameter by parameter, so a Gaussian prior is a Gaussian
prior here too and not the box around it.

The dead points are written to ``.1.txt`` with their posterior weights
-- getdist reads unequal weights as they are -- and ``ln Z`` with its
error to ``.evidence.yaml``. A run cannot be resumed.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Sampler


__all__ = ["Nested"]


#: What a point with no prediction scores. dynesty needs a finite
#: value everywhere inside the prior; this is far below any reachable
#: likelihood, so it changes no result and only keeps the sampler able
#: to move.
_FLOOR = -1.0e300


class Nested(Sampler):
    """
    Options
    -------
    nlive : int
        Live points; the evidence error goes roughly as
        ``sqrt(information / nlive)``. Default 500.
    dlogz : float
        Stop once the estimated remaining evidence is below this.
        Default 0.05.
    bound, sample : str
        dynesty's bounding and sampling methods. Defaults ``"multi"``
        and ``"auto"``.
    max_calls : int, optional
        Likelihood evaluations before stopping.
    """

    defaults = {
        "nlive": 500,
        "dlogz": 0.05,
        "bound": "multi",
        "sample": "auto",
        "max_calls": None,
    }

    def initialize(self) -> None:

        self._set_columns()

        if self.d == 0:
            raise ValueError("Nested sampling needs at least one sampled parameter.")

        self.priors = [self.model.parameters.specs[n].prior for n in self.names]
        self.results = None

    # ---------------------------------------------------------

    def _transform(self, unit):
        return np.array([p.ppf(u) for p, u in zip(self.priors, unit)])

    def _loglike(self, x):

        result = self.model.logposterior(dict(zip(self.names, x)), want_derived=self.want_derived)

        value = result.loglike if result.rejected is None else -math.inf

        return (
            value if math.isfinite(value) else _FLOOR,
            self._columns_of(x, result),
        )

    def run(self) -> None:

        try:
            from dynesty import NestedSampler
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                'Nested sampling needs dynesty: pip install "cosmofit[evidence]".'
            ) from error

        self._prepare_output()

        sampler = NestedSampler(
            self._loglike, self._transform, self.d,
            nlive=int(self.options["nlive"]),
            bound=self.options["bound"],
            sample=self.options["sample"],
            rstate=self.rng,
            blob=True,
        )

        sampler.run_nested(
            dlogz=float(self.options["dlogz"]),
            maxcall=self.options["max_calls"],
            print_progress=False,
        )

        self.results = sampler.results

        if self.output.enabled:

            products = self.products()

            self.output.write_updated(self.model, self.options)
            self.output.write_getdist_metadata(self.model, self.columns)

            chain = self.output.chain(1, self.columns)
            chain.create()
            chain.append(products["chains"][0])

            self.output.write_yaml(".evidence.yaml", {
                key: products[key] for key in (
                    "log_evidence", "log_evidence_error", "information",
                    "dimensionality", "nlive", "n_evaluations",
                )
            })

    # ---------------------------------------------------------

    def products(self) -> dict:
        """
        ``log_evidence`` and ``log_evidence_error``; the ``information``
        (prior-to-posterior KL divergence, nats) and the Bayesian model
        ``dimensionality`` ``2 Var[ln L]`` (Handley & Lemos 2019); the
        weighted ``samples``, their ``weights`` (largest 1) and the
        same as rows under ``chains`` with their ``columns``;
        ``nlive`` and ``n_evaluations``.
        """

        if self.results is None:
            raise RuntimeError("Call run() first.")

        r = self.results

        logwt = np.asarray(r.logwt, dtype=float)
        weights = np.exp(logwt - logwt.max())

        log_l = np.asarray(r.logl, dtype=float)

        # Points at the floor carry no weight; dropped before squaring,
        # where (1e300)^2 would overflow and inf * 0 give nan.
        keep = (weights > 0.0) & (log_l > _FLOOR / 10)

        w = weights[keep] / weights[keep].sum()
        mean = float(np.sum(w * log_l[keep]))
        dimensionality = float(2.0 * np.sum(w * (log_l[keep] - mean) ** 2))

        columns = np.array([blob for blob in r.blob])[keep]
        minuslogpost = -(log_l[keep] - columns[:, self.d + len(self.derived_names)])

        rows = np.column_stack([weights[keep], minuslogpost, columns])

        return {
            "log_evidence": float(r.logz[-1]),
            "log_evidence_error": float(r.logzerr[-1]),
            "information": float(r.information[-1]),
            "dimensionality": dimensionality,
            "samples": rows[:, 2: 2 + self.d],
            "weights": rows[:, 0],
            "chains": [rows],
            "columns": ["weight", "minuslogpost", *self.columns],
            "names": list(self.names),
            "nlive": int(self.options["nlive"]),
            "n_evaluations": int(np.sum(r.ncall)),
        }
