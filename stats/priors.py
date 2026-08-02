"""
Prior distributions.

CosmoFit currently implements independent uniform ("top-hat")
priors, which is what essentially every MCMC-based cosmology
analysis uses for the background parameters (H0, Omega_m, w0,
wa, rd, ...). More informative priors (Gaussian, etc.) can be
added later following the same interface.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Uniform prior
# ============================================================

class UniformPrior:
    """
    Independent uniform priors over a set of parameters.

    Parameters
    ----------
    names : list[str]
        Parameter names, in the order matching the ``theta``
        vector this prior will be evaluated on.

    bounds : dict[str, tuple[float, float]]
        Mapping of parameter name -> (lower, upper). Every name
        in ``names`` must have an entry.
    """

    def __init__(self, names, bounds):

        self.names = list(names)

        missing = [n for n in self.names if n not in bounds]

        if missing:
            raise ValueError(
                f"Missing bounds for parameter(s): {missing}"
            )

        self.lower = np.array(
            [bounds[n][0] for n in self.names],
            dtype=float,
        )

        self.upper = np.array(
            [bounds[n][1] for n in self.names],
            dtype=float,
        )

        if np.any(self.upper <= self.lower):
            raise ValueError(
                "Every parameter must have upper > lower bound."
            )

    # ---------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.names)

    # ---------------------------------------------------------

    def log_prior(self, theta) -> float:
        """
        Return 0.0 if every entry of ``theta`` lies within its
        bounds, otherwise -inf.
        """

        theta = np.asarray(theta, dtype=float)

        if theta.shape != self.lower.shape:
            raise ValueError(
                f"Expected a length-{self.ndim} vector, "
                f"got shape {theta.shape}."
            )

        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf

        return 0.0

    # Allow calling the prior directly: prior(theta)
    __call__ = log_prior

    # ---------------------------------------------------------

    def sample(self, n_samples: int, rng=None) -> np.ndarray:
        """
        Draw ``n_samples`` independent samples from the prior.

        Returns
        -------
        ndarray, shape (n_samples, ndim)
        """

        if rng is None:
            rng = np.random.default_rng()

        return rng.uniform(
            self.lower,
            self.upper,
            size=(n_samples, self.ndim),
        )

    # ---------------------------------------------------------

    def __repr__(self):

        bounds = ", ".join(
            f"{n}=({lo:g}, {hi:g})"
            for n, lo, hi in zip(self.names, self.lower, self.upper)
        )

        return f"UniformPrior({bounds})"
