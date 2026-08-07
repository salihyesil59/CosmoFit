from __future__ import annotations

from dataclasses import dataclass, MISSING, asdict, fields, field, make_dataclass
from collections.abc import Iterable, Iterator
from typing import Optional, Any
from copy import deepcopy

import numpy as np
import math
import numbers


@dataclass(frozen=True, slots=True)
class Parameter:
    """
    Metadata describing a cosmological parameter.

    Parameters
    ----------
    name : str
        Internal parameter name (e.g. ``"H0"``).

    label : str, optional
        Display label used in figures (usually LaTeX).

    unit : str, optional
        Physical unit.

    bounds : tuple(float, float), optional
        Allowed parameter range used by priors.

    description : str, optional
        Human-readable description.
    """

    name: str
    label: Optional[str] = None
    unit: Optional[str] = None
    bounds: Optional[tuple[float, float]] = None
    description: Optional[str] = None

    @property
    def latex(self) -> str:
        """
        Return the plotting label.

        Falls back to the parameter name if no LaTeX label
        is provided.
        """
        return self.label if self.label is not None else self.name

    @property
    def has_bounds(self) -> bool:
        """
        Whether the parameter has finite bounds.
        """
        return self.bounds is not None

    @property
    def lower(self) -> Optional[float]:
        """
        Lower bound.
        """
        if self.bounds is None:
            return None
        return self.bounds[0]

    @property
    def upper(self) -> Optional[float]:
        """
        Upper bound.
        """
        if self.bounds is None:
            return None
        return self.bounds[1]

    def __repr__(self) -> str:

        return (
            f"Parameter("
            f"name='{self.name}', "
            f"bounds={self.bounds})"
        )

# -----------------------------------------------------------------

class ParameterSet:
    """
    Immutable collection of Parameter objects.

    Examples
    --------
    >>> params = ParameterSet(
    ...     Parameter("H0"),
    ...     Parameter("Omega_m"),
    ...     Parameter("w0")
    ... )

    >>> params.names
    ['H0', 'Omega_m', 'w0']

    >>> params["H0"]
    Parameter(name='H0', bounds=None)

    >>> params[0]
    Parameter(name='H0', bounds=None)
    """

    def __init__(self, *parameters: Parameter):

        if len(parameters) == 1 and isinstance(parameters[0], Iterable):
            parameters = tuple(parameters[0])

        if len(parameters) == 0:
            raise ValueError(
                "ParameterSet cannot be empty."
            )

        names = [p.name for p in parameters]

        if len(names) != len(set(names)):
            raise ValueError(
                "Duplicate parameter names detected."
            )

        self._parameters = tuple(parameters)

        self._index = {
            p.name: i
            for i, p in enumerate(parameters)
        }

    # ------------------------------------------------------------------
    # Python API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __getitem__(self, item: int | str) -> Parameter:

        if isinstance(item, str):
            return self._parameters[self._index[item]]

        return self._parameters[item]

    def __repr__(self) -> str:

        names = ", ".join(self.names)

        return f"ParameterSet({names})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of parameters."""
        return len(self)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._parameters]

    @property
    def labels(self) -> list[str]:
        return [p.latex for p in self._parameters]

    @property
    def units(self) -> list[str | None]:
        return [p.unit for p in self._parameters]

    @property
    def bounds(self) -> list[tuple[float, float] | None]:
        return [p.bounds for p in self._parameters]

    @property
    def lower_bounds(self) -> list[Any]:
        return [p.lower for p in self._parameters]

    @property
    def upper_bounds(self) -> list[Any]:
        return [p.upper for p in self._parameters]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def index(self, name: str) -> int:
        """
        Return the index of a parameter.
        """
        return self._index[name]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self, theta) -> dict[str, float]:

        if len(theta) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} parameters, "
                f"got {len(theta)}."
            )

        return dict(zip(self.names, theta))

    def from_dict(self, values: dict[str, float]) -> list[float]:

        return [
            values[name]
            for name in self.names
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, theta) -> None:
        """
        Validate the structure of a parameter vector.

        This method checks only the structural validity of the parameter
        vector. Statistical constraints (priors, parameter bounds, etc.)
        are handled by the Prior classes.

        Checks
        ------
        1. Correct dimension.
        2. All values are real numbers.
        3. All values are finite.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Raises
        ------
        TypeError
            If a parameter is not a real number.

        ValueError
            If the dimension is incorrect or a parameter is not finite.
        """

        # --------------------------------------------------------------
        # Dimension
        # --------------------------------------------------------------

        if len(theta) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} parameters, "
                f"got {len(theta)}."
            )

        # --------------------------------------------------------------
        # Numerical validity
        # --------------------------------------------------------------

        for value, parameter in zip(theta, self._parameters):

            if not isinstance(value, numbers.Real):
                raise TypeError(
                    f"Parameter '{parameter.name}' "
                    "must be a real number."
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Parameter '{parameter.name}' "
                    "must be finite."
                )


    def is_valid(self, theta) -> bool:
        """
        Return True if the parameter vector is structurally valid.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Returns
        -------
        bool
            True if valid, otherwise False.
        """

        try:
            self.validate(theta)
            return True

        except (TypeError, ValueError):
            return False

# ==========================================================
# Base Parameters
# ==========================================================

@dataclass(slots=True)
class BaseParameters:
    """
    Base class for cosmological parameter containers.

    This class provides common functionality shared by all
    cosmological parameter classes (LCDMParameters,
    wCDMParameters, CPLParameters, ...).
    """

    # ==========================================================
    # Class utilities
    # ==========================================================

    @classmethod
    def names(cls) -> list[str]:
        """Return the parameter names."""
        return [field.name for field in fields(cls)]

    @classmethod
    def ndim(cls) -> int:
        """Return the number of parameters."""
        return len(fields(cls))

    @classmethod
    def defaults(cls) -> dict:
        """Return the default values."""

        defaults = {}

        for field in fields(cls):
            if field.default is not MISSING:
                defaults[field.name] = field.default

        return defaults

    # ==========================================================
    # Dictionary interface
    # ==========================================================

    def as_dict(self) -> dict:
        """Convert parameters to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict):
        """Create from a dictionary."""
        return cls(**values)

    # ==========================================================
    # NumPy interface
    # ==========================================================

    def to_numpy(self) -> np.ndarray:
        """Return parameters as a NumPy vector."""

        return np.asarray(
            [getattr(self, name) for name in self.names()],
            dtype=float,
        )

    @property
    def theta(self) -> np.ndarray:
        """
        Alias for to_numpy().

        This is the vector used by MCMC samplers.
        """
        return self.to_numpy()

    @classmethod
    def from_numpy(cls, theta):

        theta = np.asarray(theta, dtype=float)

        if theta.ndim != 1:
            raise ValueError(
                "Parameter vector must be one-dimensional."
            )

        if theta.size != cls.ndim():
            raise ValueError(
                f"Expected {cls.ndim()} parameters, "
                f"got {theta.size}."
            )

        return cls(
            **dict(zip(cls.names(), theta))
        )

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, theta=None, **kwargs):
        """
        Update parameter values.

        Examples
        --------

        Update from an MCMC vector

        >>> params.update(theta)

        Update individual parameters

        >>> params.update(H0=70.0)

        >>> params.update(
        ...     H0=70.0,
        ...     Omega_m=0.3,
        ... )
        """

        if theta is not None:

            theta = np.asarray(theta, dtype=float)

            if theta.ndim != 1:
                raise ValueError(
                    "Parameter vector must be one-dimensional."
                )

            if theta.size != self.ndim():
                raise ValueError(
                    f"Expected {self.ndim()} parameters, "
                    f"got {theta.size}."
                )

            for name, value in zip(self.names(), theta):
                setattr(self, name, float(value))

        valid = set(self.names())

        for key, value in kwargs.items():

            if key not in valid:
                raise KeyError(
                    f"Unknown parameter '{key}'."
                )

            setattr(self, key, value)

    # ==========================================================
    # Convenience
    # ==========================================================

    def copy(self):
        """Return a deep copy."""
        return deepcopy(self)

    def values(self):
        """Return parameter values as a tuple."""
        return tuple(self)

    def items(self):
        """Iterate over (name, value) pairs."""

        for name in self.names():
            yield name, getattr(self, name)

    # ==========================================================
    # Python API
    # ==========================================================

    def __iter__(self):
        for name in self.names():
            yield getattr(self, name)

    def __len__(self):
        return self.ndim()

    def __getitem__(self, key):

        if isinstance(key, str):
            return getattr(self, key)

        return list(self)[key]

    def __setitem__(self, key, value):

        if isinstance(key, str):
            setattr(self, key, value)
            return

        setattr(self, self.names()[key], value)

    def __repr__(self):

        values = ", ".join(
            f"{name}={getattr(self, name)}"
            for name in self.names()
        )

        return (
            f"{self.__class__.__name__}"
            f"({values})"
        )


# ==========================================================
# Concrete cosmology parameter container
# ==========================================================

#: Default (Planck-like) parameter bounds, used by the
#: statistics/priors module unless the user overrides them.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "H0": (50.0, 90.0),
    "Omega_m": (0.05, 0.50),
    "Omega_k": (-0.5, 0.5),
    "w0": (-2.5, 0.5),
    "wa": (-4.0, 4.0),
    "rd": (120.0, 170.0),
    "MB": (-21.0, -17.0),
    "Omega_b": (0.01, 0.10),
    "A_s": (0.01, 0.99),
    "alpha": (-0.9, 2.0),
    "sigma8": (0.6, 1.0),
}


@dataclass(slots=True)
class CosmologyParameters(BaseParameters):
    """
    Generic container for the parameters shared by every
    cosmological model implemented in CosmoFit (LCDM, CPL, ...).

    Models that do not use a given parameter simply ignore it
    (e.g. LCDM ignores ``w0``/``wa``).

    Parameters
    ----------
    H0 : float
        Hubble constant today [km/s/Mpc].
    Omega_m : float
        Present-day matter density parameter.
    Omega_k : float, optional
        Present-day curvature density parameter. Default 0
        (flat universe).
    w0 : float, optional
        Dark-energy equation-of-state parameter at z=0.
        Default -1 (cosmological constant).
    wa : float, optional
        CPL evolution parameter. Default 0.
    rd : float, optional
        Sound horizon at the drag epoch [Mpc], used as a free
        nuisance parameter for BAO analyses.
    MB : float, optional
        Supernova absolute magnitude nuisance parameter, used
        by likelihoods that do not analytically marginalize
        over it.
    Omega_b : float, optional
        Present-day baryon density parameter. Not used by the
        background expansion (E(z)) itself, but required by the
        recombination fitting formulas (redshift of decoupling
        z*, sound horizon r_s) used by :class:`~likelihoods.planck.PlanckLikelihood`.
        Default is the Planck 2018 best-fit value.
    A_s : float, optional
        Generalized Chaplygin Gas density parameter (0 < A_s < 1),
        used only by :class:`~cosmology.models.gcg.GCG`. Equal to
        -w(z=0) in that model (A_s=1 reduces GCG to flat/curved
        LCDM). Default 0.7.
    alpha : float, optional
        Generalized Chaplygin Gas exponent in p = -A/rho^alpha,
        used only by :class:`~cosmology.models.gcg.GCG`
        (alpha=1 is the original/"pure" Chaplygin gas). Default 0.
    sigma8 : float, optional
        Present-day linear matter power spectrum normalization
        (RMS density contrast in 8 Mpc/h spheres). Used only by
        the growth-of-structure machinery
        (:class:`~cosmology.calculators.growth.GrowthCalculator`,
        :class:`~likelihoods.fsigma8.FSigma8Likelihood`,
        :class:`~likelihoods.s8.S8Likelihood`) -- every model
        computes its own scale-independent growth factor D(z) from
        ``E(z)``/``dEdz``/``mu(a,k)``; ``sigma8`` only sets its
        overall normalization. Default is the Planck 2018 best-fit
        value.
    """

    H0: float
    Omega_m: float
    Omega_k: float = 0.0
    w0: float = -1.0
    wa: float = 0.0
    rd: float = 147.05
    MB: float = 0.0
    Omega_b: float = 0.0493
    A_s: float = 0.7
    alpha: float = 0.0
    sigma8: float = 0.811

    #: Extra bounds/labels contributed by :func:`build_params_class`
    #: for dynamically-created custom-model parameter classes. Empty
    #: for the standard container; not dataclass fields (no type
    #: annotation), so ``@dataclass`` ignores them.
    _EXTRA_BOUNDS = {}
    _EXTRA_LABELS = {}

    # ==========================================================
    # Parameter metadata (labels / bounds), used by ParameterSet
    # ==========================================================

    @classmethod
    def default_bounds(cls) -> dict:
        """
        Default prior bounds for every parameter in this
        container: the module-level :data:`DEFAULT_BOUNDS`, plus
        any extra bounds contributed by :func:`build_params_class`
        for a custom model's extra parameters.
        """

        merged = dict(DEFAULT_BOUNDS)
        merged.update(cls._EXTRA_BOUNDS)

        return merged

    # ------------------------------------------------------------

    @classmethod
    def parameter_set(cls, bounds: dict | None = None) -> "ParameterSet":
        """
        Build a :class:`ParameterSet` describing this container,
        with LaTeX labels and (by default) :meth:`default_bounds`.

        Parameters
        ----------
        bounds : dict, optional
            Mapping of parameter name -> (lower, upper) that
            overrides/extends :meth:`default_bounds`.
        """

        labels = {
            "H0": r"$H_0$",
            "Omega_m": r"$\Omega_m$",
            "Omega_k": r"$\Omega_k$",
            "w0": r"$w_0$",
            "wa": r"$w_a$",
            "rd": r"$r_d$",
            "MB": r"$M_B$",
            "Omega_b": r"$\Omega_b$",
            "A_s": r"$A_s$",
            "alpha": r"$\alpha$",
            "sigma8": r"$\sigma_8$",
        }
        labels.update(cls._EXTRA_LABELS)

        merged_bounds = cls.default_bounds()

        if bounds:
            merged_bounds.update(bounds)

        return ParameterSet(
            Parameter(
                name=name,
                label=labels.get(name, name),
                bounds=merged_bounds.get(name),
            )
            for name in cls.names()
        )


# ==========================================================
# Dynamic parameter classes for custom models
# ==========================================================

def build_params_class(
    name: str,
    extra_params: dict,
    base: type = CosmologyParameters,
) -> type:
    """
    Build a new :class:`BaseParameters` subclass that adds
    user-defined parameters to ``base`` (normally
    :class:`CosmologyParameters`).

    Used by :meth:`Cosmology.__init_subclass__` to back
    ``EXTRA_PARAMS`` on a custom model (see
    :mod:`cosmology.custom`) -- not usually called directly.

    Parameters
    ----------
    name : str
        Base name for the generated class (``f"{name}Parameters"``).

    extra_params : dict[str, dict]
        Mapping of new parameter name -> spec dict with keys
        ``"default"`` (float, defaults to 0.0), ``"bounds"``
        (``(lower, upper)``, optional), ``"label"`` (str,
        optional, for corner-plot axes).

    base : type
        The :class:`BaseParameters` subclass to extend. Must not
        already define any of the ``extra_params`` names.

    Returns
    -------
    type
        A new, slotted dataclass subclassing ``base`` with the
        extra fields added.
    """

    existing = set(base.names())
    collisions = [p for p in extra_params if p in existing]

    if collisions:
        raise ValueError(
            f"Extra parameter name(s) {collisions} collide with "
            f"{base.__name__}'s existing parameter(s). Choose a "
            f"different name, or omit `extra_params` and reuse the "
            f"existing parameter instead."
        )

    fields_spec = []
    extra_bounds = dict(getattr(base, "_EXTRA_BOUNDS", {}))
    extra_labels = dict(getattr(base, "_EXTRA_LABELS", {}))

    for pname, spec in extra_params.items():

        default = float(spec.get("default", 0.0))
        fields_spec.append((pname, float, field(default=default)))

        if "bounds" in spec:
            extra_bounds[pname] = tuple(spec["bounds"])

        if "label" in spec:
            extra_labels[pname] = spec["label"]

    def _default_bounds(cls) -> dict:

        merged = dict(DEFAULT_BOUNDS)
        merged.update(extra_bounds)

        return merged

    return make_dataclass(
        f"{name}Parameters",
        fields_spec,
        bases=(base,),
        slots=True,
        namespace={
            "default_bounds": classmethod(_default_bounds),
            "_EXTRA_BOUNDS": extra_bounds,
            "_EXTRA_LABELS": extra_labels,
        },
    )