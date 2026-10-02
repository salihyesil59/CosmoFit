"""
Prior distributions for sampled parameters.

Every prior here is a *normalized* one-dimensional density. The old
fitting path only knew a uniform box and returned 0 inside it, which
is enough for an MCMC (a constant cancels) and not enough for anything
that compares numbers across fits: an evidence, a MAP point under a
Gaussian prior, an importance weight. ``logpdf`` here is the real
log-density.

Each prior also carries what a sampler needs from it besides the
density:

``ppf``
    The inverse CDF, mapping the unit interval onto the parameter --
    the prior transform a nested sampler integrates through.

``sample``
    Draws, for starting points and prior-predictive checks.

``support``
    The interval outside which the density is zero (possibly infinite),
    for optimizer bounds.

``center``, ``scale``
    A typical value and a typical width, so that a sampler can start
    somewhere sensible and an optimizer can work in units where every
    parameter moves by order one.

Specified in an input as a dict -- ``{"min": 0.1, "max": 0.5}``,
``{"dist": "norm", "loc": 0.0224, "scale": 0.0005}``, ``{"dist":
"loguniform", "min": 1e-6, "max": 1e-2}`` -- or as a ``(min, max)``
pair; see :func:`make_prior`.
"""

from __future__ import annotations

import math

import numpy as np

from scipy import special


__all__ = [
    "Prior",
    "Uniform",
    "Gaussian",
    "LogUniform",
    "make_prior",
]


class Prior:
    """
    A normalized one-dimensional prior.
    """

    def logpdf(self, x: float) -> float:
        raise NotImplementedError

    def ppf(self, u: float) -> float:
        raise NotImplementedError

    @property
    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def center(self) -> float:
        raise NotImplementedError

    @property
    def scale(self) -> float:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> float:
        """One draw, by inverse-CDF sampling."""

        return float(self.ppf(rng.uniform()))

    def __contains__(self, x: float) -> bool:

        low, high = self.support

        return bool(low <= x <= high)


class Uniform(Prior):
    """
    Uniform on ``[min, max]``.
    """

    def __init__(self, min: float, max: float):

        self.min = float(min)
        self.max = float(max)

        if not (np.isfinite(self.min) and np.isfinite(self.max)):
            raise ValueError("A uniform prior needs finite bounds.")

        if not self.max > self.min:
            raise ValueError(
                f"A uniform prior needs max > min, got "
                f"[{self.min}, {self.max}]."
            )

        self._log_width = math.log(self.max - self.min)

    def logpdf(self, x: float) -> float:

        if self.min <= x <= self.max:
            return -self._log_width

        return -math.inf

    def ppf(self, u: float) -> float:

        return self.min + float(u) * (self.max - self.min)

    @property
    def support(self) -> tuple[float, float]:
        return self.min, self.max

    @property
    def center(self) -> float:
        return 0.5 * (self.min + self.max)

    @property
    def scale(self) -> float:
        # The standard deviation of the uniform distribution.
        return (self.max - self.min) / math.sqrt(12.0)

    def __repr__(self) -> str:
        return f"Uniform(min={self.min:g}, max={self.max:g})"


class Gaussian(Prior):
    """
    Normal with mean ``loc`` and standard deviation ``scale``,
    optionally truncated to ``[min, max]`` and renormalized there.
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        min: float = -math.inf,
        max: float = math.inf,
    ):

        self.loc = float(loc)
        self.sigma = float(scale)
        self.min = float(min)
        self.max = float(max)

        if not self.sigma > 0.0:
            raise ValueError(f"A Gaussian prior needs scale > 0, got {scale}.")

        if not self.max > self.min:
            raise ValueError(
                f"A truncated Gaussian prior needs max > min, got "
                f"[{self.min}, {self.max}]."
            )

        self._a = (self.min - self.loc) / self.sigma
        self._b = (self.max - self.loc) / self.sigma

        # Mass of the standard normal inside the truncation.
        self._cdf_a = special.ndtr(self._a)
        self._mass = special.ndtr(self._b) - self._cdf_a

        if not self._mass > 0.0:
            raise ValueError(
                "The truncation interval holds no probability: it lies "
                "too far into the Gaussian's tail."
            )

        self._log_norm = (
            -math.log(self.sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(self._mass)
        )

    def logpdf(self, x: float) -> float:

        if not self.min <= x <= self.max:
            return -math.inf

        t = (x - self.loc) / self.sigma

        return self._log_norm - 0.5 * t * t

    def ppf(self, u: float) -> float:

        p = self._cdf_a + float(u) * self._mass

        return float(self.loc + self.sigma * special.ndtri(p))

    @property
    def support(self) -> tuple[float, float]:
        return self.min, self.max

    @property
    def center(self) -> float:
        return float(np.clip(self.loc, self.min, self.max))

    @property
    def scale(self) -> float:
        return self.sigma

    def __repr__(self) -> str:

        text = f"Gaussian(loc={self.loc:g}, scale={self.sigma:g}"

        if math.isfinite(self.min) or math.isfinite(self.max):
            text += f", min={self.min:g}, max={self.max:g}"

        return text + ")"


class LogUniform(Prior):
    """
    Uniform in ``ln x`` on ``[min, max]``, ``0 < min < max`` -- the
    prior that says "no preferred order of magnitude".
    """

    def __init__(self, min: float, max: float):

        self.min = float(min)
        self.max = float(max)

        if not 0.0 < self.min < self.max:
            raise ValueError(
                f"A log-uniform prior needs 0 < min < max, got "
                f"[{self.min}, {self.max}]."
            )

        self._log_span = math.log(math.log(self.max / self.min))

    def logpdf(self, x: float) -> float:

        if self.min <= x <= self.max:
            return -math.log(x) - self._log_span

        return -math.inf

    def ppf(self, u: float) -> float:

        return float(self.min * (self.max / self.min) ** float(u))

    @property
    def support(self) -> tuple[float, float]:
        return self.min, self.max

    @property
    def center(self) -> float:
        return math.sqrt(self.min * self.max)

    @property
    def scale(self) -> float:
        # Half the interval's width in log space, mapped back at the
        # center: the step at which x moves by a "typical" factor.
        return self.center * 0.5 * math.log(self.max / self.min) / math.sqrt(3.0)

    def __repr__(self) -> str:
        return f"LogUniform(min={self.min:g}, max={self.max:g})"


#: Names accepted for ``dist``.
_DISTRIBUTIONS = {
    "uniform": "uniform",
    "norm": "norm",
    "normal": "norm",
    "gaussian": "norm",
    "loguniform": "loguniform",
    "log-uniform": "loguniform",
    "reciprocal": "loguniform",
}


def make_prior(spec) -> Prior:
    """
    Build a :class:`Prior` from how an input writes one.

    Accepts a :class:`Prior` (returned as is), a ``(min, max)`` pair
    (uniform), or a dict:

    * ``{"min": a, "max": b}`` -- uniform
    * ``{"dist": "norm", "loc": m, "scale": s}`` -- Gaussian, with
      optional ``min``/``max`` truncation
    * ``{"dist": "loguniform", "min": a, "max": b}``

    Raises
    ------
    ValueError
        For anything else, naming what was not understood.
    """

    if isinstance(spec, Prior):
        return spec

    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return Uniform(*spec)

    if not isinstance(spec, dict):
        raise ValueError(
            f"A prior is a dict, a (min, max) pair or a Prior; got {spec!r}."
        )

    spec = dict(spec)

    dist = _DISTRIBUTIONS.get(str(spec.pop("dist", "uniform")).lower())

    if dist is None:
        raise ValueError(
            f"Unknown prior distribution {spec!r}; known: "
            f"{sorted(set(_DISTRIBUTIONS))}."
        )

    allowed = {
        "uniform": {"min", "max"},
        "norm": {"loc", "scale", "min", "max"},
        "loguniform": {"min", "max"},
    }[dist]

    unknown = set(spec) - allowed

    if unknown:
        raise ValueError(
            f"Unexpected key(s) {sorted(unknown)} for a {dist} prior; "
            f"it takes {sorted(allowed)}."
        )

    try:

        if dist == "uniform":
            return Uniform(spec["min"], spec["max"])

        if dist == "loguniform":
            return LogUniform(spec["min"], spec["max"])

        return Gaussian(
            spec["loc"], spec["scale"],
            spec.get("min", -math.inf), spec.get("max", math.inf),
        )

    except KeyError as missing:

        raise ValueError(
            f"A {dist} prior needs {missing.args[0]!r}; got {spec}."
        ) from None
