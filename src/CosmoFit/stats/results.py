"""
Consolidated fit results.

``Fitter.summary()``, ``.convergence()``, ``.best_fit_params``,
``.best_fit_chi2`` and friends each answer one question about a fit.
That's still the right shape for driving a specific plot or check,
so none of it is going away -- but there was no single object
answering "how did this fit turn out", printable in one line and
saveable without dragging the whole ``emcee`` sampler along with it.

``FitResult`` (built from ``Fitter.result``, any time after
``best_fit()`` and/or ``run_mcmc()`` have been called) is that
object.

Example
-------
>>> fit.run_mcmc(nwalkers=48, nsteps=6000, burnin=1000)
>>> fit.best_fit()
>>> print(fit.result)
>>> fit.result.save_json("cpl_fit.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from CosmoFit.typing import PathLike


def _json_default(value):
    """
    Last-resort JSON encoder for values ``json`` doesn't know:
    numpy scalars become their Python equivalents, numpy arrays
    become lists. Anything else still raises, so a genuinely
    unserializable object isn't silently stringified.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON "
        f"serializable"
    )


# ============================================================
# Best-fit result
# ============================================================

@dataclass
class BestFitResult:
    """
    Maximum-likelihood point, as found by ``Fitter.best_fit()``.
    """

    params: dict
    chi2: float
    ndim: int
    n_data: int
    success: bool
    message: str

    def aic(self) -> float:
        return self.chi2 + 2.0 * self.ndim

    def bic(self) -> float:
        return self.chi2 + self.ndim * np.log(self.n_data)

    # ------------------------------------------------------------

    def to_dict(self) -> dict:

        return {
            "params": dict(self.params),
            "chi2": self.chi2,
            "ndim": self.ndim,
            "n_data": self.n_data,
            "aic": self.aic(),
            "bic": self.bic(),
            "success": self.success,
            "message": self.message,
        }

    # ------------------------------------------------------------

    def __repr__(self):

        params = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())

        return (
            f"BestFitResult({params}, chi2={self.chi2:.4f}, "
            f"AIC={self.aic():.2f}, BIC={self.bic():.2f})"
        )


# ============================================================
# MCMC result
# ============================================================

@dataclass
class MCMCResult:
    """
    Posterior summary of an MCMC run, as found by
    ``Fitter.run_mcmc()``.
    """

    summary: dict
    convergence: dict
    nwalkers: int
    nsteps: int
    burnin: int
    ndim: int
    acceptance_fraction: float

    # ------------------------------------------------------------

    def to_dict(self) -> dict:

        return {
            "summary": self.summary,
            "convergence": self.convergence,
            "nwalkers": self.nwalkers,
            "nsteps": self.nsteps,
            "burnin": self.burnin,
            "ndim": self.ndim,
            "acceptance_fraction": self.acceptance_fraction,
        }

    # ------------------------------------------------------------

    def __repr__(self):

        lines = [
            f"{name} = {s['median']:.6g} "
            f"(+{s['plus']:.2g}/-{s['minus']:.2g})"
            for name, s in self.summary.items()
        ]

        status = "converged" if self.convergence["converged"] else "NOT converged"

        header = (
            f"MCMCResult({self.nwalkers} walkers x {self.nsteps} steps, "
            f"burnin={self.burnin}, {status}, "
            f"acceptance={self.acceptance_fraction:.2f})"
        )

        return "\n  ".join([header] + lines)


# ============================================================
# FitResult
# ============================================================

@dataclass
class FitResult:
    """
    Everything a finished (or in-progress) fit produced, in one
    object: which model/datasets/parameters it is, the best-fit
    point (if ``best_fit()`` was called), and the MCMC posterior
    (if ``run_mcmc()`` was called).

    Built on demand by ``Fitter.result`` -- it is a read-only
    snapshot, not a live view, so it reflects whatever state the
    fitter was in when it was accessed.
    """

    model: str
    datasets: list = field(default_factory=list)
    free_params: list = field(default_factory=list)
    best_fit: Optional[BestFitResult] = None
    mcmc: Optional[MCMCResult] = None

    # ------------------------------------------------------------

    def to_dict(self) -> dict:

        return {
            "model": self.model,
            "datasets": list(self.datasets),
            "free_params": list(self.free_params),
            "best_fit": self.best_fit.to_dict() if self.best_fit else None,
            "mcmc": self.mcmc.to_dict() if self.mcmc else None,
        }

    # ------------------------------------------------------------

    def save_json(self, path: PathLike) -> None:
        """
        Serialize to JSON: model/dataset/parameter names, the
        best-fit point and chi2/AIC/BIC, and the MCMC posterior
        summary and convergence diagnostics. Does *not* include
        the raw chain -- this is meant for keeping a fit's
        headline numbers around cheaply, not for re-plotting the
        posterior later (keep the ``Fitter`` itself, or
        ``fit.flat_samples()``, for that).
        """

        with open(path, "w") as f:
            # `default=`: numpy scalars reach here easily (a chain
            # read back from HDF5 reports `np.int64` counters, and
            # `np.float32` shows up from user-supplied arrays), and
            # `json` rejects anything that isn't a Python builtin --
            # `np.float64` only survives because it subclasses
            # `float`. Coerce rather than fail on a save.
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    # ------------------------------------------------------------

    @classmethod
    def load_json(cls, path) -> "FitResult":

        with open(path) as f:
            data = json.load(f)

        best_fit = None
        if data.get("best_fit") is not None:
            bf = data["best_fit"]
            best_fit = BestFitResult(
                params=bf["params"],
                chi2=bf["chi2"],
                ndim=bf["ndim"],
                n_data=bf["n_data"],
                success=bf["success"],
                message=bf["message"],
            )

        mcmc = None
        if data.get("mcmc") is not None:
            m = data["mcmc"]
            mcmc = MCMCResult(
                summary=m["summary"],
                convergence=m["convergence"],
                nwalkers=m["nwalkers"],
                nsteps=m["nsteps"],
                burnin=m["burnin"],
                ndim=m["ndim"],
                acceptance_fraction=m["acceptance_fraction"],
            )

        return cls(
            model=data["model"],
            datasets=data["datasets"],
            free_params=data["free_params"],
            best_fit=best_fit,
            mcmc=mcmc,
        )

    # ------------------------------------------------------------

    def __repr__(self):

        header = (
            f"FitResult(model={self.model}, datasets={self.datasets}, "
            f"free_params={self.free_params})"
        )

        parts = [header]

        if self.best_fit is not None:
            parts.append(f"  {self.best_fit!r}")

        if self.mcmc is not None:
            parts.append(f"  {self.mcmc!r}")

        if self.best_fit is None and self.mcmc is None:
            parts.append("  (call best_fit() and/or run_mcmc() first)")

        return "\n".join(parts)
