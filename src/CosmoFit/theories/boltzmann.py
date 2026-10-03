"""
CMB power spectra and ``sigma8`` from CAMB, on the native background.

The old backend (:class:`~cosmology.boltzmann.CAMBBackend`) translated
an old-style model into CAMB's parameters. This one reads them off the
background theory: the densities it was solved with -- ``omega_cb``
directly, so the neutrino density is never converted back from a mass
with a separate constant -- and the dark sector's own ``w(z)``.

What CAMB can be given is decided by what a sector *does*, as before,
now read off the sector's hooks:

* a dark-energy fluid on top of general relativity, with an equation
  of state ``w(z)``, is passed through CAMB's PPF module as a table of
  ``w(a)``;
* a sector that changes the gravitational coupling (``mu``), moves
  energy into matter (``matter_exchange``, ``clustering_matter``) or
  makes ``E(z)`` jump is refused. CAMB would run for each of them --
  on a background it was not given, or with perturbations it does
  not have -- and return a spectrum that looks reasonable and means
  nothing;
* a modified Friedmann equation or a holographic sector has no
  ``w(z)`` to give it, and is refused too.

CAMB is optional (``pip install "cosmofit[cmb]"``); only a model with
this theory in it imports it.
"""

from __future__ import annotations

import math

import numpy as np

from CosmoFit.core.component import ComponentError, Theory
from CosmoFit.cosmology.boltzmann import BoltzmannError, _import_camb
from CosmoFit.cosmology.core.constants import Tcmb

from .dark_energy import DarkEnergy
from .dark_sector import DarkSector


__all__ = ["CAMB", "cmb_support"]


#: Scale factors the dark energy's ``w(a)`` is tabulated on for PPF:
#: log-spaced, as every ``w(z)`` here varies fastest at late times.
_A_TABLE = np.logspace(-5.0, 0.0, 500)


def cmb_support(sector) -> tuple[bool, str]:
    """
    Whether CAMB can compute spectra for a dark sector, and why not when
    it cannot -- without importing CAMB.
    """

    cls = type(sector)
    name = sector.name

    def overrides(hook):
        return getattr(cls, hook) is not getattr(DarkSector, hook)

    if overrides("mu"):
        return False, (
            f"{name} changes the gravitational coupling of matter "
            f"perturbations (mu), and CAMB's perturbation equations are "
            f"general relativity's. It would return GR spectra, with "
            f"every parameter that enters only mu doing nothing."
        )

    if overrides("matter_exchange") or overrides("clustering_matter"):
        return False, (
            f"{name} moves energy into matter. CAMB's cold dark matter "
            f"obeys its own continuity equation, so the exchange would "
            f"drop out of the spectra."
        )

    if overrides("jumps"):
        return False, (
            f"{name}'s E(z) jumps, so its equation of state is singular "
            f"there; CAMB's dark-energy modules cannot be given it."
        )

    if not isinstance(sector, DarkEnergy):
        return False, (
            f"{name} has no dark-energy equation of state w(z) -- its "
            f"E(z) comes from a modified Friedmann equation or a "
            f"holographic density, and many perturbation histories share "
            f"one E(z)."
        )

    return True, ""


class CAMB(Theory):
    """
    Options
    -------
    lmax : int
        Highest multipole; likelihoods asking for more widen it. By
        default, the most any likelihood asks for -- 30 for low-l EE
        alone -- or 2508 if none says.
    lens_potential_accuracy : int
        CAMB's lensing accuracy; the largest any likelihood asks for is
        used. Default 1.

    The primordial parameters ``ln1e10As``, ``n_s`` and ``tau_reio``
    default to Planck 2018's best fit.

    Provides
    --------
    * ``Cl`` -- the lensed spectra from ``l = 0``, keyed ``ell``, ``tt``,
      ``te``, ``ee``, ``bb`` and ``pp``: ``C_l`` in muK^2, or
      ``D_l = l(l+1) C_l / 2 pi`` with ``ell_factor=True``. ``pp`` is
      always ``[L(L+1)]^2 C_L / 2 pi``, the convention Planck's lensing
      likelihood is defined on.
    * ``sigma8_0`` -- ``sigma8`` of the cold matter today, the amplitude
      the growth theory scales when its ``amplitude`` is ``boltzmann``.
    * ``S8`` -- of all matter, as the derived parameter.

    Derived parameters: ``sigma8`` (all matter, as usually quoted),
    ``S8 = sigma8 sqrt(Omega_m/0.3)`` and ``A_s``.
    """

    #: Planck 2018 TT,TE,EE+lowE+lensing, as the old parameter defaults.
    defaults = {"ln1e10As": 3.044, "n_s": 0.9649, "tau_reio": 0.0544}

    def initialize(self) -> None:

        unknown = set(self.info) - {"lmax", "lens_potential_accuracy"}

        if unknown:
            raise ComponentError(f"{self.name}: unknown option(s) {sorted(unknown)}.")

        self.lmax = int(self.info.get("lmax", 0))
        self.lens_potential_accuracy = int(self.info.get("lens_potential_accuracy", 1))

        try:
            self.camb = _import_camb()
        except BoltzmannError as error:
            raise ComponentError(str(error)) from None

    def accepts(self, name: str) -> bool:
        return name in self.defaults

    def get_default_params(self) -> dict:
        return dict(self.defaults)

    def get_requirements(self) -> dict:
        return {"expansion": None, "background_densities": None}

    def get_can_provide(self) -> list[str]:
        return ["Cl", "sigma8_0", "S8"]

    def get_derived_params(self) -> list[str]:
        return ["sigma8", "S8", "A_s"]

    def initialize_with_provider(self, provider) -> None:

        super().initialize_with_provider(provider)

        # The union of what is asked: more multipoles and more accuracy
        # can only help the likelihood that asked for less.
        for options in self.requested.get("Cl", []):

            options = options or {}

            self.lmax = max(self.lmax, int(options.get("lmax", 0)))
            self.lens_potential_accuracy = max(
                self.lens_potential_accuracy,
                int(options.get("lens_potential_accuracy", 0)),
            )

        if self.lmax == 0:
            self.lmax = 2508

    def check_model(self, model) -> None:

        background = self.provider.theory_for("expansion")

        if not getattr(background, "radiation", True):
            raise ComponentError(
                f"{self.name} needs a background with radiation, and "
                f"{background.name} has radiation: false."
            )

        supported, reason = cmb_support(background.sector)

        if not supported:
            raise ComponentError(
                f"{self.name} cannot compute spectra for this model: {reason} "
                f"The compressed CMB distance priors need only E(z), and "
                f"work for it."
            )

    # ---------------------------------------------------------

    def _parameters(self, ln1e10As, n_s, tau_reio):

        camb = self.camb

        expansion = self.provider.get_expansion()
        d = self.provider.get_background_densities()

        sector = expansion["sector"]
        ctx = expansion["context"]

        pars = camb.CAMBparams()

        pars.set_cosmology(
            H0=100.0 * d["h"], ombh2=d["omega_b"],
            omch2=d["omega_cb"] - d["omega_b"], omk=ctx.Omega_k,
            mnu=d["m_nu"], nnu=d["N_eff"], num_massive_neutrinos=1,
            TCMB=Tcmb, tau=tau_reio,
        )

        pars.InitPower.set_params(As=math.exp(ln1e10As) * 1.0e-10, ns=n_s)

        pars.set_matter_power(redshifts=[0.0], kmax=2.0)

        if type(sector) is not DarkEnergy:

            z = 1.0 / _A_TABLE - 1.0

            w = np.asarray(sector.w(z, **expansion["sector_params"]), dtype=float)

            if not np.all(np.isfinite(w)):
                raise BoltzmannError(f"{sector.name}: w(z) is not finite over a = 1e-5..1.")

            dark_energy = camb.dark_energy.DarkEnergyPPF()
            dark_energy.set_w_a_table(_A_TABLE, w)

            # Assigned only once the table is set: the assignment copies
            # into CAMB's Fortran state, and later changes to the Python
            # object would be lost -- LCDM spectra for a w0-wa model.
            pars.DarkEnergy = dark_energy

        pars.set_for_lmax(
            # CAMB's accuracy falls off near the lmax it is given.
            self.lmax + 500, lens_potential_accuracy=self.lens_potential_accuracy,
        )

        return pars

    def calculate(self, state: dict, want_derived: bool = True, **params):

        pars = self._parameters(**params)

        try:

            results = self.camb.get_results(pars)

            # Not passing `pars` again: that recomputes everything, and
            # doubled the cost of every call in the old backend.
            total = results.get_cmb_power_spectra(
                lmax=self.lmax, spectra=("total",), CMB_unit="muK", raw_cl=True,
            )["total"]

            potential = results.get_lens_potential_cls(lmax=self.lmax)[:, 0]

            sigma8 = float(results.get_sigma8_0())

            sigma8_cb = float(results.get_sigmaR(
                8.0, z_indices=[-1], var1="delta_nonu", var2="delta_nonu",
            )[0])

        except self.camb.CAMBError as error:
            raise BoltzmannError(f"CAMB refused the point: {error}") from error

        Cl = {
            "ell": np.arange(self.lmax + 1),
            "tt": total[:, 0], "ee": total[:, 1], "bb": total[:, 2], "te": total[:, 3],
            "pp": potential,
        }

        # CAMB can return NaN spectra without raising, for unexceptional
        # parameters (see the old backend). Not a point to sample.
        if not all(np.all(np.isfinite(v)) for v in Cl.values()) or not (
            math.isfinite(sigma8) and math.isfinite(sigma8_cb)
        ):
            return False

        Omega_m = self.provider.get_background_densities()["Omega_m"]
        S8 = sigma8 * math.sqrt(Omega_m / 0.3)

        state.update(Cl=Cl, sigma8_0=sigma8_cb, S8=S8)

        if want_derived:
            state["derived"] = {
                "sigma8": sigma8,
                "S8": S8,
                "A_s": math.exp(params["ln1e10As"]) * 1.0e-10,
            }

        return True

    # ---------------------------------------------------------

    def get_Cl(self, ell_factor: bool = False) -> dict:

        Cl = self.current_state["Cl"]

        if not ell_factor:
            return dict(Cl)

        ell = Cl["ell"]
        factor = ell * (ell + 1.0) / (2.0 * math.pi)

        return {
            name: (value if name in ("ell", "pp") else factor * value)
            for name, value in Cl.items()
        }

    def get_sigma8_0(self) -> float:
        return self.current_state["sigma8_0"]
