"""
The existing models and datasets, as components of the new core.

Temporary by design. :class:`LegacyCosmology` wraps a
:class:`~cosmology.core.base.Cosmology` subclass as a theory;
:class:`LegacyLikelihood` wraps any dataset in
:data:`~stats.fitter.DATASET_REGISTRY` as a likelihood. Together they
let the new core run every model and dataset the library has *today*,
through exactly the code ``Fitter`` runs -- so the two can be compared
point by point, and every later port (a model rewritten as a native
theory, a dataset as a native likelihood) has a reference it must
reproduce.

In an input, a dataset name is enough to reach these::

    theory:
      legacy_cosmology: {model: CPL, compute_rd: true}
    likelihood:
      desi: {version: desi2025}
      pantheon:

Every parameter of the model's parameter class -- cosmological and
nuisance alike (``MB``, ``A_planck``), since the old likelihoods read
those off the cosmology -- is an input of the theory; any not declared
in ``params`` keeps the class default, as with ``Fitter``.
"""

from __future__ import annotations

from .component import ComponentError, Likelihood, Theory


__all__ = ["LegacyCosmology", "LegacyLikelihood"]


def _resolve_model(model):
    """A ``Cosmology`` subclass from a class, a name, or an import path."""

    from CosmoFit.cosmology.core.base import Cosmology

    if isinstance(model, type) and issubclass(model, Cosmology):
        return model

    if not isinstance(model, str):
        raise ComponentError(f"Not a cosmological model: {model!r}.")

    if ":" in model or "." in model:

        from .registry import _import

        cls = _import(model)

    else:

        import CosmoFit

        cls = getattr(CosmoFit, model, None)

    if not (isinstance(cls, type) and issubclass(cls, Cosmology)):
        raise ComponentError(
            f"Unknown model {model!r}: give a built-in name (LCDM, CPL, "
            f"...) or an import path to a Cosmology subclass."
        )

    return cls


class LegacyCosmology(Theory):
    """
    A ``Cosmology`` subclass as a theory.

    Options
    -------
    model : str or type
        ``"LCDM"``, ``"CPL"``, ..., an import path, or the class.
    compute_rd : bool
        Compute ``rd`` from the densities, as ``Fitter(compute_rd=True)``.
        ``rd`` is then not an input.
    derive_sigma8 : bool
        Take ``sigma8`` from the Boltzmann code, as
        ``Fitter(derive_sigma8=True)``. ``sigma8`` is then not an input.

    Provides
    --------
    ``cosmology`` (the live object the wrapped likelihoods read),
    ``E``, ``H``, ``DM``, ``DH``, ``DV``, ``DA`` (each ``z=...``),
    ``rd``, ``fsigma8`` (``z=...``).

    Derived parameters: ``rd_computed`` and ``z_drag``.
    """

    #: One mutable cosmology object, so one state.
    cache_size = 1

    def initialize(self) -> None:

        self.model_cls = _resolve_model(self.info.get("model", "LCDM"))

        self.compute_rd = bool(self.info.get("compute_rd", False))
        self.derive_sigma8 = bool(self.info.get("derive_sigma8", False))

        unknown = set(self.info) - {"model", "compute_rd", "derive_sigma8"}

        if unknown:
            raise ComponentError(
                f"{self.name}: unknown option(s) {sorted(unknown)}."
            )

        params_cls = self.model_cls.PARAMS_CLASS

        values = dict(params_cls.defaults())

        # H0 and Omega_m have no defaults; any value builds the object,
        # and the first point overwrites them.
        values.setdefault("H0", 70.0)
        values.setdefault("Omega_m", 0.3)

        self.param_names = list(params_cls.names())

        self.cosmology = self.model_cls(
            params_cls(**{n: values[n] for n in self.param_names})
        )

        self.cosmology.compute_rd = self.compute_rd

    # ---------------------------------------------------------

    def accepts(self, name: str) -> bool:

        if self.compute_rd and name == "rd":
            return False

        if self.derive_sigma8 and name == "sigma8":
            return False

        return name in self.param_names

    def get_can_provide(self) -> list[str]:

        return ["cosmology", "E", "H", "DM", "DH", "DV", "DA", "rd", "fsigma8"]

    def get_derived_params(self) -> list[str]:

        return ["rd_computed", "z_drag"]

    def initialize_complete(self) -> None:

        if not self.derive_sigma8:
            return

        from CosmoFit.cosmology.boltzmann import CAMBBackend

        if CAMBBackend.attached(self.cosmology) is None:
            raise ComponentError(
                f"{self.name}: derive_sigma8 needs a likelihood that runs "
                f"CAMB (planck_lite, planck_lensing, planck_lowe, "
                f"act_lensing) -- there is nothing else to derive it from."
            )

        # Only now: the backend is created by the CMB likelihood.
        self.cosmology.derive_sigma8 = True

    # ---------------------------------------------------------

    def calculate(self, state: dict, want_derived: bool = True, **params):

        self.cosmology.params.update(**params)
        self.cosmology.refresh()

        if want_derived:

            horizon = self.cosmology.sound_horizon

            state["derived"] = {
                "rd_computed": float(horizon.rd_computed()),
                "z_drag": float(horizon.z_drag()),
            }

        return True

    # ---------------------------------------------------------

    def get_cosmology(self):
        return self.cosmology

    def get_E(self, z):
        return self.cosmology.E(z)

    def get_H(self, z):
        return self.cosmology.H(z)

    def get_DM(self, z):
        return self.cosmology.distance.DM(z)

    def get_DH(self, z):
        return self.cosmology.distance.DH(z)

    def get_DV(self, z):
        return self.cosmology.distance.DV(z)

    def get_DA(self, z):
        return self.cosmology.distance.DA(z)

    def get_rd(self):
        return float(self.cosmology.rd)

    def get_fsigma8(self, z):
        return self.cosmology.background.fsigma8(z)


class LegacyLikelihood(Likelihood):
    """
    A dataset of :data:`~stats.fitter.DATASET_REGISTRY` as a
    likelihood: ``ln L = -chi2 / 2`` of the wrapped class.

    Options
    -------
    dataset : str, optional
        Registry key; defaults to the name the likelihood is listed
        under, so ``desi:`` in an input is enough.

    Any other option is passed to the wrapped class's constructor
    (``version``, ``marginalize_MB``, ``include_cepheid``, ...).
    """

    def initialize(self) -> None:

        from CosmoFit.stats.fitter import DATASET_REGISTRY

        self.options = dict(self.info)
        self.dataset = self.options.pop("dataset", self.name)

        if self.dataset not in DATASET_REGISTRY:
            raise ComponentError(
                f"Unknown dataset {self.dataset!r}; known: "
                f"{sorted(DATASET_REGISTRY)}."
            )

        self.legacy_cls = DATASET_REGISTRY[self.dataset]
        self.legacy = None

    def get_requirements(self) -> dict:

        return {"cosmology": None}

    def initialize_with_provider(self, provider) -> None:

        super().initialize_with_provider(provider)

        self.legacy = self.legacy_cls(provider.get_cosmology(), **self.options)

    def logp(self, **params) -> float:

        return -0.5 * float(self.legacy.chi2())

    # ---------------------------------------------------------

    def check_model(self, model) -> None:
        """
        The dataset-combination warnings ``Fitter`` gives, once per
        model (from whichever wrapped likelihood is listed first).
        """

        wrapped = [
            lk for lk in model.likelihoods.values()
            if isinstance(lk, LegacyLikelihood)
        ]

        if wrapped[0] is not self:
            return

        from CosmoFit.stats import fitter as old

        names = [lk.dataset for lk in wrapped]
        sampled = model.sampled_params

        cosmologies = [
            t for t in model.theories.values()
            if isinstance(t, LegacyCosmology)
        ]

        compute_rd = any(t.compute_rd for t in cosmologies)

        old._warn_conflicting_datasets(names)
        old._warn_inconsistent_amplitude(names, sampled)
        old._warn_blind_neutrino_mass(names, sampled, compute_rd)
        old._warn_blind_neff(names, sampled)
        old._warn_free_rd_with_early_universe(names, sampled, compute_rd)
        old._warn_calibrated_twice([lk.legacy for lk in wrapped])
        old._warn_tau_counted_twice([lk.legacy for lk in wrapped])

        for theory in cosmologies:
            old._warn_derived_parameters(theory.model_cls, sampled)
            old._warn_ungrounded_coupling(theory.model_cls, names)
