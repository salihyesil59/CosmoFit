"""
The datasets as likelihoods of the 2.0 core, reading the native theories.

Each likelihood here is one of the library's datasets -- the same data,
the same covariance, the same chi2 -- with its predictions read from the
theories through the provider instead of from a model object it holds.
The data loading and the statistics stay where they were: each wraps
the dataset's class and hands it a read-only *view* of the provider,
:class:`ProviderCosmology`, in place of a cosmology. Nothing in a
likelihood can change a theory's state.

What used to be a parameter of the cosmology because a likelihood read
it there is now that likelihood's own:

* ``rd`` -- a BAO likelihood reads the sound horizon from the
  ``early_universe`` theory by default (``rd: computed``), or takes it
  as a parameter with ``rd: free``;
* ``MB`` -- a supernova likelihood that does not marginalize the
  absolute magnitude analytically takes it as a parameter;
* ``tau_reio`` -- read by the tau prior, shared with the CAMB theory
  when there is one;
* ``A_planck`` -- Planck's absolute calibration, for the bandpower and
  low-l EE likelihoods (default 1; ``planck_lite`` carries its Gaussian
  prior).

The CMB spectra likelihoods read the ``camb`` theory's ``Cl``, each
asking for the multipoles and lensing accuracy it needs.

Named in an input by family::

    likelihood:
      bao.desi: {version: desi2025}
      sn.pantheonplus:
      cmb.distance_priors:
      external.bbn:

The old dataset names (``desi``, ``pantheon``, ...) still reach the
old likelihoods through :class:`~core.legacy.LegacyLikelihood`, for
the old models.
"""

from __future__ import annotations

import math

import numpy as np

from CosmoFit.core.component import ComponentError, Likelihood
from CosmoFit.cosmology.core.constants import c


__all__ = [
    "ProviderCosmology",
    "DatasetLikelihood",
    "NATIVE_LIKELIHOODS",
]


# ============================================================
# The view
# ============================================================

class _Distances:

    def __init__(self, provider):
        self._provider = provider

    def DM(self, z):
        return self._provider.get_DM(z)

    def DH(self, z):
        return self._provider.get_DH(z)

    def DV(self, z):
        return self._provider.get_DV(z)

    def DA(self, z):
        return self._provider.get_DA(z)

    def DM_from_chi(self, chi):
        """For the distance priors: ``chi`` in units of ``c/H0``."""

        Omega_k = self._provider.get_expansion()["context"].Omega_k
        hubble = c / (100.0 * self._provider.get_background_densities()["h"])

        if Omega_k == 0.0:
            return hubble * chi

        root = math.sqrt(abs(Omega_k))

        if Omega_k > 0.0:
            return hubble / root * np.sinh(root * chi)

        return hubble / root * np.sin(root * chi)


class _CHW19Background:
    """
    The expansion the CMB distance priors were compressed with (Chen,
    Huang & Wang 2019), for the old recombination calculator: no
    radiation of its own (the calculator adds CHW19's), massive
    neutrinos inside ``Omega_m``, and the dark sector's own evolution
    ``f(z) = Omega_de(z) / Omega_de(0)`` from the native background.
    """

    def __init__(self, provider):

        d = provider.get_background_densities()

        self.H0 = 100.0 * d["h"]
        self.Omega_m = d["Omega_m"]
        self.Omega_b = d["omega_b"] / d["h"] ** 2
        self.Omega_k = provider.get_expansion()["context"].Omega_k
        self.Omega_de0 = 1.0 - self.Omega_m - self.Omega_k

        dark = provider.get_Omega_de_z
        dark_today = float(dark(0.0))

        def f(z):
            return np.asarray(dark(z), dtype=float) / dark_today

        self._f = f

    def Omega_de(self, z):
        return self.Omega_de0 * self._f(z)

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(
            self.Omega_m * (1.0 + z) ** 3 + self.Omega_k * (1.0 + z) ** 2
            + self.Omega_de(z)
        )


class _Background:

    def __init__(self, provider):
        self._provider = provider

    def H(self, z):
        return self._provider.get_H(z)

    def fsigma8(self, z):
        return self._provider.get_fsigma8(z)


class _Params:
    """``cosmology.params.<name>``: the likelihood's own parameters."""

    def __init__(self, likelihood):
        self._likelihood = likelihood

    def __getattr__(self, name):

        try:
            return self._likelihood.current_params[name]
        except KeyError:
            raise AttributeError(name) from None


class _Spectra:
    """
    What the CMB likelihoods' classes read from the old CAMB backend --
    ``cls``, ``lensing_spectra``, and the widening ``CAMBBackend.shared``
    applies -- served from the ``camb`` theory's ``Cl``. The largest
    ``lmax`` and lensing accuracy a class asks for become its likelihood's
    requirement.
    """

    #: Attributes ``CAMBBackend.shared`` resets on widening.
    _cache_key = None
    _cache_value = None

    def __init__(self, view):

        self._view = view
        self.lmax = 0
        self.lens_potential_accuracy = 1

    def _Cl(self, ell_factor):
        return self._view._provider.get_Cl(ell_factor=ell_factor)

    def cls(self, lmin: int = 2) -> dict:
        """Raw ``C_l`` [muK^2] from ``lmin`` to the highest multipole computed."""

        Cl = self._Cl(False)

        return {
            "ell": Cl["ell"][lmin:],
            "TT": Cl["tt"][lmin:], "TE": Cl["te"][lmin:], "EE": Cl["ee"][lmin:],
        }

    def lensing_spectra(self, lmax: int) -> dict:
        """``D_l`` [muK^2] and ``[L(L+1)]^2 C_L^phiphi / 2 pi``, from ``l = 0``."""

        Dl = self._Cl(True)

        if lmax + 1 > len(Dl["ell"]):
            raise ComponentError(
                f"Spectra asked for to l = {lmax}, computed to {len(Dl['ell']) - 1}."
            )

        stop = lmax + 1

        return {
            "TT": Dl["tt"][:stop], "EE": Dl["ee"][:stop], "TE": Dl["te"][:stop],
            "PP": Dl["pp"][:stop],
        }


class ProviderCosmology:
    """
    The part of an old-style cosmology the datasets' classes read, served
    from the provider and the likelihood's own parameters. Read-only:
    there is nothing on it to set.
    """

    def __init__(self, likelihood):

        self._likelihood = likelihood
        self.params = _Params(likelihood)

        # Where CAMBBackend.shared looks for an existing backend: the old
        # CMB classes find this one and never build their own.
        self._camb_backend = _Spectra(self)

    @property
    def _provider(self):
        return self._likelihood.provider

    @property
    def distance(self):
        return _Distances(self._provider)

    @property
    def background(self):
        return _Background(self._provider)

    def H(self, z):
        return self._provider.get_H(z)

    @property
    def _densities(self) -> dict:
        return self._provider.get_background_densities()

    @property
    def H0(self) -> float:
        return 100.0 * self._densities["h"]

    @property
    def Omega_m(self) -> float:
        return self._densities["Omega_m"]

    @property
    def Omega_b(self) -> float:
        d = self._densities
        return d["omega_b"] / d["h"] ** 2

    @property
    def rd(self) -> float:

        if self._likelihood.rd == "free":
            return self._likelihood.current_params["rd"]

        return self._provider.get_rdrag()

    @property
    def MB(self) -> float:
        return self._likelihood.current_params["MB"]

    @property
    def A_planck(self) -> float:
        return self._likelihood.current_params["A_planck"]

    @property
    def recombination(self):
        """The distance priors' own recombination, CHW19's conventions."""

        from CosmoFit.cosmology.calculators.recombination import RecombinationCalculator

        return RecombinationCalculator(_CHW19Background(self._provider))


# ============================================================
# The likelihoods
# ============================================================

class DatasetLikelihood(Likelihood):
    """
    One dataset of :data:`~stats.fitter.DATASET_REGISTRY`, reading the
    native theories. Options other than this class's own go to the
    dataset's class (``version``, ``marginalize_MB``, ...).
    """

    #: Key of the dataset in ``DATASET_REGISTRY``.
    dataset: str = ""

    #: What the predictions are computed from.
    needs: tuple = ()

    #: Whether the predictions involve the BAO sound horizon.
    uses_rd = False

    def initialize(self) -> None:

        from CosmoFit.stats.fitter import DATASET_REGISTRY

        self.options = dict(self.info)

        self.rd = self.options.pop("rd", "computed") if self.uses_rd else None

        if self.uses_rd and self.rd not in ("computed", "free"):
            raise ComponentError(
                f"{self.name}: rd must be 'computed' (from the early_universe "
                f"theory) or 'free' (a parameter), not {self.rd!r}."
            )

        self.current_params: dict = {}
        self.view = ProviderCosmology(self)

        try:
            self.legacy = DATASET_REGISTRY[self.dataset](self.view, **self.options)
        except TypeError as error:
            raise ComponentError(f"{self.name}: {error}") from None

    # ---------------------------------------------------------

    def get_params(self) -> list[str]:
        """The parameters this likelihood takes, given its options."""

        return ["rd"] if self.rd == "free" else []

    def accepts(self, name: str) -> bool:
        return name in self.get_params()

    def get_default_params(self) -> dict:
        return {}

    def get_required_params(self) -> list[str]:
        return self.get_params()

    def get_requirements(self) -> dict:

        requirements = {name: None for name in self.needs}

        if self.rd == "computed":
            requirements["rdrag"] = None

        return requirements

    def check_model(self, model) -> None:
        """
        How the datasets are combined, once per model (from whichever
        is listed first): pairs that share a sample
        (:func:`data.metadata.conflicts`), and the two double counts that
        depend on options -- Pantheon+'s Cepheids with a SH0ES ``H0``, the
        full-posterior ``tau`` with ``plik_lite``.
        """

        datasets = [
            lk for lk in model.likelihoods.values() if isinstance(lk, DatasetLikelihood)
        ]

        if datasets[0] is not self:
            return

        import warnings

        from CosmoFit.data.metadata import conflicts
        from CosmoFit.stats import fitter

        names = {lk.dataset: lk.name for lk in datasets}

        for (first, second), reason in conflicts(names).items():
            warnings.warn(
                f"Likelihoods '{names[first]}' and '{names[second]}' should not "
                f"be combined: {reason} Treating them as independent "
                f"double-counts that data, which understates the uncertainties "
                f"and can bias the result. Use one of the two.",
                UserWarning,
                stacklevel=2,
            )

        fitter._warn_calibrated_twice([lk.legacy for lk in datasets])
        fitter._warn_tau_counted_twice([lk.legacy for lk in datasets])

    # ---------------------------------------------------------

    def chi2(self) -> float:
        return float(self.legacy.chi2())

    def logp(self, **params) -> float:

        self.current_params = params

        return -0.5 * self.chi2()


# ----------------------------------------------------------- BAO

class _BAO(DatasetLikelihood):
    needs = ("DM", "DH", "DV")
    uses_rd = True


class DESI(_BAO):
    """DESI BAO (``version``: ``desi2025``, ``desi2024``)."""
    dataset = "desi"


class SDSSBAO(_BAO):
    """SDSS BOSS/eBOSS consensus BAO."""
    dataset = "sdss_bao"


class SDSSFullShape(_BAO):
    """SDSS full-shape BAO with ``f sigma8``."""
    dataset = "sdss_fsbao"
    needs = ("DM", "DH", "DV", "fsigma8")


class EBOSSELG(_BAO):
    """eBOSS DR16 ELG BAO, tabulated likelihood."""
    dataset = "eboss_elg"


class EBOSSELGFullShape(_BAO):
    """eBOSS DR16 ELG full shape, tabulated likelihood."""
    dataset = "eboss_elg_fs"
    needs = ("DM", "DH", "DV", "fsigma8")


class EBOSSLya(_BAO):
    """eBOSS DR16 Lyman-alpha BAO, tabulated likelihood."""
    dataset = "eboss_lya"


class BAOLowZ(_BAO):
    """6dFGS and SDSS MGS."""
    dataset = "bao_lowz"


# ----------------------------------------------------------- SN

class _Supernovae(DatasetLikelihood):
    needs = ("DM",)

    def get_params(self) -> list[str]:

        # Pantheon+ names the switch marginalize_MB, the others
        # marginalize_offset; either way, MB is a parameter when off.
        marginalized = getattr(
            self.legacy, "marginalize_MB", getattr(self.legacy, "marginalize_offset", True),
        )

        return [] if marginalized else ["MB"]


class PantheonPlus(_Supernovae):
    """Pantheon+ (with SH0ES Cepheid hosts: ``include_cepheid: true``)."""
    dataset = "pantheon"


class DESSN5YR(_Supernovae):
    """DES-SN5YR."""
    dataset = "des_sn5yr"


class Union3(_Supernovae):
    """Union3, binned."""
    dataset = "union3"


# ----------------------------------------------------------- the rest

class CosmicChronometers(DatasetLikelihood):
    """``H(z)`` from cosmic chronometers."""
    dataset = "cc"
    needs = ("H",)


class FSigma8(DatasetLikelihood):
    """RSD ``f sigma8(z)``, corrected to each survey's fiducial geometry."""
    dataset = "fsigma8"
    needs = ("fsigma8", "H", "DA")


class S8(DatasetLikelihood):
    """A weak-lensing ``S8``, of all matter."""
    dataset = "s8"
    needs = ("S8",)

    def chi2(self) -> float:

        residual = np.array([self.legacy.data.value - self.provider.get_S8()])

        return float(self.legacy.covariance.chi2(residual))


class DistancePriors(DatasetLikelihood):
    """
    The CMB distance priors ``(R, l_A, omega_b h^2)`` of Chen, Huang &
    Wang (2019), predicted in *their* conventions -- a compressed
    likelihood is a summary computed with particular definitions, and
    the prediction must share them. Their radiation is
    ``Omega_m / (1 + z_eq)`` with massive neutrinos left in ``Omega_m``,
    which puts ``l_A`` 0.1% (3 sigma of the prior) below the exact
    ``pi / theta_*`` the native background gives; at Planck's own best
    fit the exact value misses the prior by 2.7 sigma and theirs by 0.6.
    Only the dark sector's evolution is taken from the background.
    """
    dataset = "planck"
    needs = ("Omega_de_z", "expansion", "background_densities")


class _External(DatasetLikelihood):
    needs = ("background_densities",)


class H0(_External):
    """A local ``H0`` (``version``: ``sh0es2022``, ...)."""
    dataset = "h0"


class BBN(_External):
    """A BBN ``omega_b h^2``."""
    dataset = "omega_b"


class Tau(DatasetLikelihood):
    """A reionization optical depth; reads the parameter ``tau_reio``."""
    dataset = "tau"

    def get_params(self) -> list[str]:
        return ["tau_reio"]


# ----------------------------------------------------------- CMB spectra

class _CMBSpectra(DatasetLikelihood):

    def get_requirements(self) -> dict:

        spectra = self.view._camb_backend

        return {"Cl": {
            "lmax": spectra.lmax,
            "lens_potential_accuracy": spectra.lens_potential_accuracy,
        }}


class _Calibrated(_CMBSpectra):

    def get_params(self) -> list[str]:
        return ["A_planck"]

    def get_default_params(self) -> dict:
        return {"A_planck": 1.0}


class PlanckLite(_Calibrated):
    """Planck 2018 high-l ``plik_lite`` bandpowers (``spectra``, ``use_low_ell``)."""
    dataset = "planck_lite"


class PlanckLowE(_Calibrated):
    """Planck 2018 low-l EE, tabulated in ``tau``."""
    dataset = "planck_lowe"


class PlanckLensing(_CMBSpectra):
    """Planck 2018 lensing reconstruction."""
    dataset = "planck_lensing"


class ACTLensing(_CMBSpectra):
    """ACT DR6 lensing reconstruction."""
    dataset = "act_lensing"


#: Every native likelihood, by the name it is listed under.
NATIVE_LIKELIHOODS = {
    "bao.desi": DESI,
    "bao.sdss": SDSSBAO,
    "bao.sdss_fullshape": SDSSFullShape,
    "bao.eboss_elg": EBOSSELG,
    "bao.eboss_elg_fullshape": EBOSSELGFullShape,
    "bao.eboss_lya": EBOSSLya,
    "bao.lowz": BAOLowZ,
    "sn.pantheonplus": PantheonPlus,
    "sn.des_sn5yr": DESSN5YR,
    "sn.union3": Union3,
    "cc.chronometers": CosmicChronometers,
    "rsd.fsigma8": FSigma8,
    "lss.s8": S8,
    "cmb.distance_priors": DistancePriors,
    "cmb.planck_lite": PlanckLite,
    "cmb.planck_lowe": PlanckLowE,
    "cmb.planck_lensing": PlanckLensing,
    "cmb.act_lensing": ACTLensing,
    "external.h0": H0,
    "external.bbn": BBN,
    "external.tau": Tau,
}
