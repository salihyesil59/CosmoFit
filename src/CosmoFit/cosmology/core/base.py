"""
Base cosmology class.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core.parameters import (
    CosmologyParameters,
    build_params_class,
)

from CosmoFit.cosmology.numerics import DistanceIntegrator

from CosmoFit.cosmology.calculators import (
    BackgroundCalculator,
    DistanceCalculator,
    SoundHorizon,
    RecombinationCalculator,
    GrowthCalculator,
)


class Cosmology:
    """
    Base class for all cosmological models.

    Subclasses that need a parameter beyond the standard
    :class:`CosmologyParameters` set (e.g. a genuinely new,
    not-in-the-literature model) can declare an ``EXTRA_PARAMS``
    class attribute -- a dict of ``{name: {"default": ..., "bounds":
    (lo, hi), "label": ...}}`` -- and ``__init_subclass__`` below
    builds a matching parameter dataclass (``PARAMS_CLASS``) and a
    ``self.<name>`` property automatically. See
    :func:`cosmology.custom.define_model` for the minimal-code way
    to do this without subclassing by hand.
    """

    #: Extra parameters this model adds beyond `CosmologyParameters`.
    #: Empty for every built-in model.
    EXTRA_PARAMS: dict = {}

    #: Parameter container class used to build `self.params`.
    #: Overridden automatically for subclasses that set
    #: `EXTRA_PARAMS`.
    PARAMS_CLASS = CosmologyParameters

    #: Plain-text name for this model, for tables, JSON and log
    #: lines. `None` falls back to the class name -- see
    #: :meth:`plain_name`.
    MODEL_NAME: str | None = None

    #: The same name written for a *figure*: a LaTeX math string
    #: where the plain name is really a set of symbols
    #: (``LCDM`` -> ``$\Lambda$CDM``), `None` where the plain name
    #: is already what a reader should see (an acronym like ``CPL``).
    #: See :meth:`plot_label`.
    MODEL_LABEL: str | None = None

    # ---------------------------------------------------------

    @classmethod
    def plain_name(cls) -> str:
        """
        This model's name as plain text: ``MODEL_NAME`` if it
        declares one, else the class name.
        """

        return cls.MODEL_NAME or cls.__name__

    @classmethod
    def plot_label(cls) -> str:
        """
        This model's name as it should appear in a figure legend
        or title -- ``MODEL_LABEL`` (LaTeX) if it declares one,
        else :meth:`plain_name`.

        Matplotlib renders ``$...$`` spans with mathtext, so a
        legend built from this shows the symbols a paper would
        (``ΛCDM``, ``wCDM``, ``f(R)``) rather than the ASCII
        spelling of a Python identifier.
        """

        return cls.MODEL_LABEL or cls.plain_name()

    # ---------------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:

        super().__init_subclass__(**kwargs)

        # Only trigger for a subclass's *own* `EXTRA_PARAMS`
        # declaration, not an inherited one -- so re-subclassing an
        # already-extended model doesn't rebuild `PARAMS_CLASS`
        # (and every built-in model, which never sets this, is
        # completely unaffected).
        extra = cls.__dict__.get("EXTRA_PARAMS")

        if not extra:
            return

        cls.PARAMS_CLASS = build_params_class(
            cls.__name__, extra, base=cls.PARAMS_CLASS,
        )

        for pname in extra:
            setattr(
                cls, pname,
                property(lambda self, _n=pname: getattr(self.params, _n)),
            )

    # ---------------------------------------------------------

    def __init__(
        self,
        params: CosmologyParameters,
    ):

        self.params = params

        # Numerical utilities
        self.integrator = DistanceIntegrator(self)

        # Cosmological calculators
        self.background = BackgroundCalculator(self)
        self.distance = DistanceCalculator(self)
        self.sound_horizon = SoundHorizon(self)
        self.recombination = RecombinationCalculator(self)
        self.growth = GrowthCalculator(self)

    # ---------------------------------------------------------

    @property
    def H0(self):
        return self.params.H0

    @property
    def Omega_m(self):
        return self.params.Omega_m

    @property
    def Omega_k(self):
        return self.params.Omega_k

    @property
    def w0(self):
        return self.params.w0

    @property
    def wa(self):
        return self.params.wa

    @property
    def rd(self):
        return self.params.rd

    @property
    def MB(self):
        return self.params.MB

    @property
    def Omega_b(self):
        return self.params.Omega_b

    @property
    def A_s(self):
        return self.params.A_s

    @property
    def alpha(self):
        return self.params.alpha

    @property
    def sigma8(self):
        return self.params.sigma8

    # ---------------------------------------------------------

    def refresh(self) -> None:
        """
        Rebuild internal numerical tables after the underlying
        ``params`` object has been mutated in place (e.g. by
        ``params.update(theta)`` during an MCMC step).

        ``Cosmology`` objects are intentionally mutated in place
        rather than reconstructed at every likelihood evaluation
        (reconstruction would mean re-reading every dataset from
        disk on every MCMC step). The one piece of cached state
        that depends on the parameters -- the χ(z) = ∫dz/E(z)
        interpolation table built by ``DistanceIntegrator`` --
        must therefore be explicitly refreshed. Call this right
        after updating ``self.params`` and before evaluating any
        likelihood.
        """

        self.integrator.rebuild()
        self.growth.rebuild()

    # ---------------------------------------------------------

    @property
    def Omega_de0(self):
        return 1.0 - self.Omega_m - self.Omega_k

    # ---------------------------------------------------------

    def E(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def dEdz(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def Omega_de(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def H(self, z):
        return self.H0 * self.E(z)

    # ---------------------------------------------------------

    def mu(self, a, k=None):
        """
        Effective-to-Newtonian gravitational coupling ratio,
        G_eff(a,k)/G_N, entering the linear growth equation solved
        by :class:`~cosmology.calculators.growth.GrowthCalculator`
        (see that module's docstring for the equation itself).

        Default is 1 everywhere -- standard GR growth, correct for
        every dark-energy-on-top-of-GR model in this library
        (LCDM, wCDM, CPL, JBP, BA, GCG all inherit this unchanged).
        A genuinely modified-gravity model (``FQExponential``,
        ``FRTLinear``, ``FRHuSawicki``) overrides this with its own
        derived ``mu(a,k)``.

        Parameters
        ----------
        a : float or ndarray
            Scale factor.

        k : float, optional
            Wavenumber [h/Mpc]. Ignored by every scale-independent
            ``mu`` (the default, and every override here except
            ``FRHuSawicki``'s).
        """

        return np.ones_like(np.asarray(a, dtype=float))