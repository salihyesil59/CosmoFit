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

from cosmofit.core.component import ComponentError, Theory
from cosmofit.cosmology.boltzmann import BoltzmannError, _import_camb
from cosmofit.cosmology.core.constants import Tcmb

from .dark_energy import DarkEnergy
from .dark_sector import DarkSector
from .power_spectrum import PowerSpectrumInterpolator


__all__ = ["CAMB", "cmb_support"]


#: Scale factors the dark energy's ``w(a)`` is tabulated on for PPF:
#: log-spaced, as every ``w(z)`` here varies fastest at late times.
_A_TABLE = np.logspace(-5.0, 0.0, 500)

#: What a request for a matter power spectrum or for ``sigma(R)`` may
#: say, and the pair of variables it means when it names none.
_PK_OPTIONS = {"z", "k_max", "nonlinear", "vars_pairs"}
_SIGMA_OPTIONS = {"z", "R", "vars_pairs"}
_TOTAL = ("delta_tot", "delta_tot")

#: The ``k_max`` [1/Mpc] CAMB's transfer functions go to when nothing
#: asks for more: enough for ``sigma8``.
_K_MAX = 2.0


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
    halofit_version : str
        CAMB's non-linear model, for ``P(k)`` and the lensing potential:
        ``mead2020`` (HMcode 2020, CAMB's own default) unless given; any
        of CAMB's ``halofit_version`` values.

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
    * ``Pk_interpolator`` -- ``P(z, k)``, a
      :class:`~cosmofit.theories.power_spectrum.PowerSpectrumInterpolator`;
      ``k`` in 1/Mpc, ``P`` in Mpc^3. Asked for with ``z`` (the
      redshifts it must cover), ``k_max`` [1/Mpc], ``nonlinear``
      (``True``, the default, ``False``, or both as a list) and
      ``vars_pairs`` (CAMB's variable names, ``[["delta_tot",
      "delta_tot"]]`` by default); read with
      ``get_Pk_interpolator(var_pair=("delta_tot", "delta_tot"),
      nonlinear=True)``.
    * ``Pk_grid`` -- the same spectra as the grid ``(k, z, P[z, k])``
      they are interpolated from; asked for and read the same way.
    * ``sigma_R`` -- the linear r.m.s. ``sigma(R, z)`` in spheres of
      radius ``R`` [Mpc], as ``(z, R, sigma[z, R])``. Asked for with
      ``z``, ``R`` and ``vars_pairs``; read with
      ``get_sigma_R(var_pair=("delta_tot", "delta_tot"))``.

    Every request widens what CAMB computes for all of them -- the
    union of the redshifts, the largest ``k_max`` -- so a spectrum may
    cover more than its own request asked for, never less.

    Derived parameters: ``sigma8`` (all matter, as usually quoted),
    ``S8 = sigma8 sqrt(Omega_m/0.3)`` and ``A_s``.
    """

    #: Planck 2018 TT,TE,EE+lowE+lensing, as the old parameter defaults.
    defaults = {"ln1e10As": 3.044, "n_s": 0.9649, "tau_reio": 0.0544}

    def initialize(self) -> None:

        unknown = set(self.info) - {"lmax", "lens_potential_accuracy", "halofit_version"}

        if unknown:
            raise ComponentError(f"{self.name}: unknown option(s) {sorted(unknown)}.")

        self.lmax = int(self.info.get("lmax", 0))
        self.lens_potential_accuracy = int(self.info.get("lens_potential_accuracy", 1))
        self.halofit_version = self.info.get("halofit_version", "mead2020")

        # What the matter power spectra and sigma(R) are wanted for,
        # gathered from the requests in `initialize_with_provider`.
        self.redshifts = {0.0}
        self.k_max = _K_MAX
        self.spectra: set[tuple[str, str, bool]] = set()
        self.sigma_pairs: set[tuple[str, str]] = set()
        self.radii: set[float] = set()

        try:
            self.camb = _import_camb()
        except BoltzmannError as error:
            raise ComponentError(str(error)) from None

        versions = self.camb.nonlinear.halofit_version_names

        if self.halofit_version not in versions:
            raise ComponentError(
                f"{self.name}: halofit_version {self.halofit_version!r} is not "
                f"one of CAMB's: {sorted(versions)}."
            )

    def accepts(self, name: str) -> bool:
        return name in self.defaults

    def get_default_params(self) -> dict:
        return dict(self.defaults)

    def get_requirements(self) -> dict:
        return {"expansion": None, "background_densities": None}

    def get_can_provide(self) -> list[str]:
        return ["Cl", "sigma8_0", "S8", "Pk_interpolator", "Pk_grid", "sigma_R"]

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

        for quantity in ("Pk_interpolator", "Pk_grid", "sigma_R"):
            for options in self.requested.get(quantity, []):
                self._add_request(quantity, options or {})

    def _add_request(self, quantity: str, options: dict) -> None:

        allowed = _SIGMA_OPTIONS if quantity == "sigma_R" else _PK_OPTIONS

        unknown = set(options) - allowed

        if unknown:
            raise ComponentError(
                f"{self.name}: {quantity} takes {sorted(allowed)}, "
                f"not {sorted(unknown)}."
            )

        z = np.atleast_1d(np.asarray(options.get("z", [0.0]), dtype=float))

        if z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ComponentError(f"{self.name}: {quantity} needs redshifts z >= 0.")

        self.redshifts.update(float(x) for x in z)

        known = set(self.camb.model.transfer_names) - {"k/h"}

        pairs = []

        for pair in options.get("vars_pairs", [_TOTAL]):

            if len(pair) != 2 or not set(pair) <= known:
                raise ComponentError(
                    f"{self.name}: {quantity}'s vars_pairs are pairs of "
                    f"{sorted(known)}; {pair!r} is not."
                )

            pairs.append((str(pair[0]), str(pair[1])))

        if quantity == "sigma_R":

            R = np.atleast_1d(np.asarray(options.get("R", []), dtype=float))

            if R.size == 0 or not np.all(R > 0):
                raise ComponentError(f"{self.name}: sigma_R needs radii R > 0 [Mpc].")

            self.radii.update(float(x) for x in R)
            self.sigma_pairs.update(pairs)

            return

        self.k_max = max(self.k_max, float(options.get("k_max", _K_MAX)))

        nonlinear = options.get("nonlinear", True)

        for flag in (nonlinear if isinstance(nonlinear, (list, tuple)) else [nonlinear]):
            for pair in pairs:
                self.spectra.add((*pair, bool(flag)))

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

        # Earliest first, as CAMB stores them -- so the last is z = 0.
        pars.set_matter_power(
            redshifts=sorted(self.redshifts, reverse=True), kmax=self.k_max, silent=True,
        )

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

        # After `set_for_lmax`, which decides whether the lensing is
        # non-linear; a non-linear P(k) must not undo that.
        if any(nonlinear for *_, nonlinear in self.spectra):

            model = self.camb.model

            lensing = pars.NonLinear in (model.NonLinear_lens, model.NonLinear_both)

            pars.NonLinear = model.NonLinear_both if lensing else model.NonLinear_pk

        pars.NonLinearModel.set_params(halofit_version=self.halofit_version)

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

            Pk_grid = {
                (var1, var2, nonlinear): results.get_linear_matter_power_spectrum(
                    var1=var1, var2=var2, hubble_units=False, k_hunit=False,
                    have_power_spectra=True, nonlinear=nonlinear,
                )
                for var1, var2, nonlinear in self.spectra
            }

            radii = np.array(sorted(self.radii))

            sigma_R = {
                # CAMB's rows are earliest first; turned to increasing z.
                (var1, var2): results.get_sigmaR(
                    radii, var1=var1, var2=var2, hubble_units=False,
                )[::-1]
                for var1, var2 in self.sigma_pairs
            }

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

        if not all(np.all(np.isfinite(P)) for _, _, P in Pk_grid.values()) or not all(
            np.all(np.isfinite(s)) for s in sigma_R.values()
        ):
            return False

        Omega_m = self.provider.get_background_densities()["Omega_m"]
        S8 = sigma8 * math.sqrt(Omega_m / 0.3)

        state.update(
            Cl=Cl, sigma8_0=sigma8_cb, S8=S8, Pk_grid=Pk_grid, sigma_R=sigma_R,
            sigma_R_axes=(np.array(sorted(self.redshifts)), radii),
            Pk_interpolators={},
        )

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

    def _spectrum(self, var_pair, nonlinear):

        key = (*var_pair, bool(nonlinear))

        grids = self.current_state["Pk_grid"]

        if key not in grids:
            raise ComponentError(
                f"{self.name} computed no {'non-linear' if nonlinear else 'linear'} "
                f"P(k) of {tuple(var_pair)}: ask for it, in the requirement's "
                f"'vars_pairs' and 'nonlinear'. Computed: {sorted(grids)}."
            )

        return key, grids[key]

    def get_Pk_grid(self, var_pair=_TOTAL, nonlinear: bool = True):
        """``(k, z, P[z, k])``: ``k`` in 1/Mpc, ``P`` in Mpc^3."""

        return self._spectrum(var_pair, nonlinear)[1]

    def get_Pk_interpolator(self, var_pair=_TOTAL, nonlinear: bool = True):
        """``P(z, k)`` as a :class:`PowerSpectrumInterpolator`."""

        key, (k, z, P) = self._spectrum(var_pair, nonlinear)

        built = self.current_state["Pk_interpolators"]

        if key not in built:
            built[key] = PowerSpectrumInterpolator(z, k, P)

        return built[key]

    def get_sigma_R(self, var_pair=_TOTAL):
        """``(z, R, sigma[z, R])``, linear, ``R`` in Mpc."""

        computed = self.current_state["sigma_R"]
        key = tuple(var_pair)

        if key not in computed:
            raise ComponentError(
                f"{self.name} computed no sigma(R) of {key}: ask for it, in "
                f"the requirement's 'vars_pairs'. Computed: {sorted(computed)}."
            )

        z, R = self.current_state["sigma_R_axes"]

        return z, R, computed[key]
