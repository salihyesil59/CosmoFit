"""
Base cosmology class.
"""

from __future__ import annotations

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