"""
Boltzmann-code backend: CMB angular power spectra from scratch.

Everything else in CosmoFit is a background calculation -- distances
and expansion rates from ``E(z)``, plus a one-equation linear growth
ODE. That is enough for CC, BAO, supernovae and the *compressed* CMB
distance priors, and it is why the library runs an MCMC in minutes
without a Fortran dependency.

It is not enough to predict C_l. The CMB anisotropy spectrum comes
out of the coupled Boltzmann hierarchy for photons, neutrinos,
baryons and cold dark matter, integrated through recombination with
a full ionization history -- thousands of coupled ODEs per
wavenumber, over hundreds of wavenumbers. Reimplementing that here
would be a Boltzmann code, not a feature of one, and a pure-Python
one would be far too slow to put inside an MCMC.

So this module does the honest thing and calls one. ``CAMBBackend``
translates a :class:`~cosmology.core.base.Cosmology` into CAMB's
parameter conventions, runs it, and hands back binned-ready C_l
arrays. CAMB is an *optional* dependency (``pip install
"cosmofit[cmb]"``); nothing else in the library imports this module,
and a fit that does not include the ``"planck_lite"`` dataset never
touches it.

What can and cannot be pushed through it
----------------------------------------
CAMB solves the perturbation equations for a specific set of
physical models. A CosmoFit model that is *only* an ``E(z)`` cannot
be handed to it -- the same ``E(z)`` is consistent with many
different perturbation histories, and picking one silently would be
inventing physics the model never specified. Three cases:

* **LCDM** maps exactly onto CAMB's default.

* **Any model exposing a dark-energy equation of state ``w(z)``**
  (wCDM, CPL, JBP, BA, GCG) is passed through CAMB's PPF dark-energy
  module as a tabulated ``w(a)``. This is exact at the background
  level and is the standard treatment of the perturbations for a
  smooth dark-energy fluid, including across ``w = -1``, where a
  quintessence-fluid treatment breaks down.

* **Modified-gravity models** (f(Q), f(R,T), f(R)) are refused.
  Their whole content is that the field equations differ from GR,
  which is exactly what CAMB's perturbation solver assumes. A model
  like ``FRHuSawicki`` would run -- its background is LCDM's by
  construction -- and would return LCDM's C_l while its ``f_R0``
  did nothing, which is worse than an error. So this raises instead.

Where a model is refused, the compressed distance priors
(``"planck"``) still work: they only need ``E(z)``.
"""

from __future__ import annotations

import numpy as np


#: Neutrino mass-to-density conversion, ``omega_nu h^2 = m_nu /
#: NEUTRINO_MASS_DENOM`` [eV]. The standard value for the
#: temperature and degeneracy CAMB assumes.
NEUTRINO_MASS_DENOM = 93.14


#: Models whose perturbations CAMB cannot represent, mapped to why.
#: Checked by class name so that importing the modified-gravity
#: modules is not required.
_UNSUPPORTED = {

    "FQExponential":
        "f(Q) gravity modifies the field equations themselves; "
        "CAMB's perturbation solver assumes GR.",

    "FRTLinear":
        "f(R,T) gravity modifies the field equations themselves; "
        "CAMB's perturbation solver assumes GR.",

    "FRHuSawicki":
        "f(R) Hu-Sawicki has an LCDM background by construction, so "
        "CAMB would run happily and return LCDM's C_l with f_R0 "
        "doing nothing at all -- a silently wrong answer rather "
        "than a missing one.",

}


def supports_cmb_spectra(model) -> tuple[bool, str]:
    """
    Whether CMB power spectra can be computed for a model, and why
    not when they cannot -- **without importing CAMB**.

    Answers the question :class:`CAMBBackend`'s constructor answers
    by raising, but as a value, for callers that need to decide
    *before* offering the choice: a GUI greying out a dataset, a
    script picking between the compressed priors and the full
    spectra, a table of what each model supports.

    Parameters
    ----------
    model : type or Cosmology
        A model class or an instance of one.

    Returns
    -------
    (bool, str)
        ``(True, "")`` if supported, else ``(False, reason)``.
    """

    cls = model if isinstance(model, type) else type(model)

    name = cls.__name__

    if name in _UNSUPPORTED:
        return False, _UNSUPPORTED[name]

    if name == "LCDM" or issubclass(cls, tuple()) if False else name == "LCDM":
        return True, ""

    if not hasattr(cls, "w"):

        return False, (

            f"{name} defines an expansion history E(z) but no "
            f"dark-energy equation of state w(z), and the two are "
            f"not interchangeable -- many different perturbation "
            f"histories share one E(z)."

        )

    return True, ""


class BoltzmannError(RuntimeError):
    """
    Raised when CMB spectra cannot be computed for this cosmology --
    because CAMB is not installed, because the model's perturbations
    are outside what CAMB can represent, or because CAMB itself
    rejected the parameters.
    """


def _import_camb():
    """
    Import CAMB, with an error message that says what to install.
    """

    try:

        import camb

    except ImportError as exc:

        raise BoltzmannError(

            "Computing CMB power spectra from scratch requires "

            "CAMB, which is an optional dependency:\n\n"

            "    pip install \"cosmofit[cmb]\"\n\n"

            "Every other dataset -- including the compressed "

            "Planck distance priors ('planck') -- works without "

            "it.",

        ) from exc

    return camb


class CAMBBackend:
    """
    Computes CMB angular power spectra for a CosmoFit cosmology by
    calling CAMB.

    Parameters
    ----------
    cosmology : Cosmology
        The model to compute spectra for. Read live on every call,
        so an in-place ``params.update(theta)`` during an MCMC step
        is picked up without rebuilding the backend.

    lmax : int, optional
        Maximum multipole to compute. CAMB is asked for somewhat
        more than this internally, since its high-l accuracy
        degrades near the requested limit.

    lens_potential_accuracy : int, optional
        CAMB's lensing accuracy setting. 1 (the default) is what
        Planck analyses use for parameter estimation; 0 disables
        lensing entirely and is wrong at the 10-sigma level for
        Planck's error bars, so it is not offered as a shortcut.

    Notes
    -----
    Parameter translation is where the errors hide, so it is
    spelled out:

    - ``H0``, ``Omega_b h^2``, ``Omega_k``, ``tau``, ``n_s`` map
      across directly.
    - ``A_s`` is ``exp(ln1e10As) * 1e-10``. The library's ``A_s``
      field is the *Chaplygin gas* parameter and is not this;
      see :class:`~cosmology.core.parameters.CosmologyParameters`.
    - ``omch2`` is ``Omega_m h^2 - Omega_b h^2 - Omega_nu h^2``.
      CosmoFit counts massive neutrinos inside ``Omega_m`` (they
      are non-relativistic across the whole redshift range its
      background calculations cover), while CAMB counts them
      separately -- so the neutrino density is *subtracted* here
      rather than added. Getting this backwards shifts
      ``Omega_c h^2`` by ~0.0006, which is ~0.5 sigma of Planck's
      constraint on it.
    """

    #: Attribute the shared instance is parked on. Private to this
    #: module -- nothing else should reach for it.
    _SHARED_ATTR = "_camb_backend"

    @classmethod
    def attached(cls, cosmology) -> "CAMBBackend | None":
        """
        The backend already attached to ``cosmology``, or ``None``.

        Unlike :meth:`shared` this never creates one. It exists for
        callers that want the Boltzmann code's answer *if* it is
        already being computed, and must not trigger a CAMB run of
        their own if it is not -- deriving ``sigma8`` for the growth
        machinery, in particular, which would otherwise make a
        growth-only fit pay for a CMB calculation nothing asked for.
        """

        return getattr(cosmology, cls._SHARED_ATTR, None)

    @classmethod
    def shared(
        cls,
        cosmology,
        lmax: int = 2508,
        lens_potential_accuracy: int = 1,
    ) -> "CAMBBackend":
        """
        The backend attached to ``cosmology``, creating it on first
        use and *widening* it if this caller needs more than the
        last one did.

        Two likelihoods can want CMB spectra from the same
        cosmology -- ``plik_lite`` and the lensing reconstruction
        do, in any fit that uses the full Planck data. Giving each
        its own backend runs CAMB twice per MCMC step for two views
        of one calculation, which doubles the cost of the single
        most expensive thing in the library. Measured on a
        615-bandpower + lensing fit: 1.93 s per evaluation with
        separate backends, 1.36 s sharing one. Not the clean halving
        it looks like it should be, because sharing also *raises*
        the accuracy the bandpower half is computed at -- see the
        widening rule below.

        Widening rather than asserting equality, because the two
        callers legitimately differ: the bandpower likelihood needs
        ``lmax = 2508`` at accuracy 1, the lensing one ``lmax =
        2500`` at accuracy 4. The union -- the larger of each -- is
        correct for both, since more multipoles and more accuracy
        can only help. Widening invalidates the cache, so a
        already-computed result is never served at the lower
        setting it was computed with.
        """

        backend = getattr(cosmology, cls._SHARED_ATTR, None)

        if backend is None:

            backend = cls(cosmology, lmax, lens_potential_accuracy)

            setattr(cosmology, cls._SHARED_ATTR, backend)

            return backend

        widened = False

        if lmax > backend.lmax:
            backend.lmax = int(lmax)
            widened = True

        if lens_potential_accuracy > backend.lens_potential_accuracy:
            backend.lens_potential_accuracy = int(lens_potential_accuracy)
            widened = True

        if widened:
            backend._cache_key = None
            backend._cache_value = None

        return backend

    # ---------------------------------------------------------

    def __init__(
        self,
        cosmology,
        lmax: int = 2508,
        lens_potential_accuracy: int = 1,
    ):

        self.cosmo = cosmology

        self.lmax = int(lmax)

        self.lens_potential_accuracy = int(lens_potential_accuracy)

        self._camb = _import_camb()

        self._check_supported()

        #: Cache keyed on the parameter vector the last call used,
        #: so repeated `cls()` calls at one MCMC step (TT, then TE,
        #: then EE) run CAMB once rather than three times.
        self._cache_key = None
        self._cache_value = None

    # ---------------------------------------------------------

    def _check_supported(self) -> None:
        """
        Refuse models whose perturbations CAMB cannot represent.

        The decision itself lives in :func:`supports_cmb_spectra`,
        so a caller that asks "can this model do it?" and a caller
        that simply tries cannot get different answers.
        """

        supported, reason = supports_cmb_spectra(self.cosmo)

        if supported:
            return

        raise BoltzmannError(

            f"Cannot compute CMB spectra for "

            f"{type(self.cosmo).__name__}: {reason}\n\n"

            f"Use the compressed distance priors "

            f"(datasets=['planck', ...]) for this model instead -- "

            f"they need only E(z).",

        )


    # ---------------------------------------------------------

    @property
    def omega_nu_h2(self) -> float:
        """
        Physical density of the massive neutrinos, from ``m_nu``.
        """

        return self.cosmo.m_nu / NEUTRINO_MASS_DENOM

    # ---------------------------------------------------------

    def _parameter_key(self) -> tuple:
        """
        Everything the CAMB call depends on, as a hashable tuple --
        the cache key.
        """

        p = self.cosmo.params

        base = (

            type(self.cosmo).__name__,

            self.lmax,

            self.lens_potential_accuracy,

            float(p.H0),

            float(p.Omega_m),

            float(p.Omega_b),

            float(p.Omega_k),

            float(p.n_s),

            float(p.ln1e10As),

            float(p.tau_reio),

            float(p.N_eff),

            float(p.m_nu),

        )

        if hasattr(self.cosmo, "w"):

            # A model's w(z) can depend on parameters beyond the
            # standard set (GCG's A_s/alpha, a custom model's
            # extras), so key on the whole container rather than on
            # a guessed subset.
            base = base + tuple(

                float(getattr(p, name))

                for name in p.names()

            )

        return base

    # ---------------------------------------------------------

    def _build_params(self):
        """
        Translate the cosmology into a ``camb.CAMBparams``.
        """

        camb = self._camb

        cosmo = self.cosmo

        omch2 = (

            cosmo.omega_m_h2

            - cosmo.omega_b_h2

            - self.omega_nu_h2

        )

        if omch2 <= 0.0:

            raise BoltzmannError(

                f"Non-physical cold dark matter density: "

                f"Omega_c h^2 = {omch2:.5f} <= 0 for "

                f"Omega_m = {cosmo.Omega_m:.4f}, "

                f"Omega_b = {cosmo.Omega_b:.4f}, "

                f"m_nu = {cosmo.m_nu:.3f} eV.",

            )

        pars = camb.CAMBparams()

        pars.set_cosmology(

            H0=cosmo.H0,

            ombh2=cosmo.omega_b_h2,

            omch2=omch2,

            omk=cosmo.Omega_k,

            mnu=cosmo.m_nu,

            nnu=cosmo.N_eff,

            tau=cosmo.tau_reio,

        )

        pars.InitPower.set_params(

            As=np.exp(cosmo.ln1e10As) * 1.0e-10,

            ns=cosmo.n_s,

        )

        # Enables `get_sigma8_0()`. Measured at 15% of the CAMB call
        # (0.475 s -> 0.546 s), which buys the amplitude in the form
        # everything outside the CMB talks about -- worth it as a
        # standing capability rather than an opt-in nobody sets.
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)

        self._set_dark_energy(pars)

        pars.set_for_lmax(

            # CAMB's accuracy falls off approaching the requested
            # lmax, so ask for headroom and slice back down.
            self.lmax + 500,

            lens_potential_accuracy=self.lens_potential_accuracy,

        )

        return pars

    # ---------------------------------------------------------

    def _set_dark_energy(self, pars) -> None:
        """
        Configure CAMB's dark-energy sector from the model's own
        ``w(z)``, where it has one.

        LCDM needs nothing (CAMB's default is a cosmological
        constant). Everything else is tabulated as ``w(a)`` on a
        log-spaced scale-factor grid and handed to the PPF module,
        which handles ``w`` crossing -1 -- CPL and JBP posteriors
        routinely do, and a quintessence-fluid treatment develops a
        gradient instability exactly there.
        """

        cosmo = self.cosmo

        if not hasattr(cosmo, "w"):
            return

        # Dense enough that linear interpolation between nodes is
        # far below CAMB's own accuracy, log-spaced because every
        # w(z) form here varies fastest at late times.
        a = np.logspace(-5.0, 0.0, 500)

        z = 1.0 / a - 1.0

        w = np.asarray(cosmo.w(z), dtype=float)

        if not np.all(np.isfinite(w)):

            raise BoltzmannError(

                f"{type(cosmo).__name__}.w(z) returned "

                f"non-finite values over a = 1e-5..1; CAMB cannot "

                f"be given this equation of state.",

            )

        dark_energy = self._camb.dark_energy.DarkEnergyPPF()

        dark_energy.set_w_a_table(a, w)

        # Assign only *after* the table is set: `pars.DarkEnergy =`
        # copies the object into CAMB's Fortran state, so mutating
        # the Python-side instance afterwards is silently lost --
        # which returns a perfectly valid LCDM spectrum for a
        # w0-wa model, with no error anywhere.
        pars.DarkEnergy = dark_energy

    # ---------------------------------------------------------

    def _spectra(self) -> dict[str, np.ndarray]:
        """
        Everything CAMB is asked for, from ``l = 0``, computed once
        per parameter point.

        Both consumers -- the bandpower likelihood (raw ``C_l``)
        and the lensing likelihood (``D_l`` plus the lensing
        potential) -- are served from this one cached result.
        Running CAMB twice per step for two views of the same
        calculation would double the cost of the single most
        expensive thing in the library.
        """

        key = self._parameter_key()

        if key != self._cache_key:

            self._cache_value = self._run()

            self._cache_key = key

        return self._cache_value

    # ---------------------------------------------------------

    def sigma8(self) -> float:
        r"""
        ``sigma_8`` as the Boltzmann code derives it, from the
        primordial amplitude and the transfer function.

        This is **not** ``cosmology.sigma8``. That one is a free
        parameter the growth machinery
        (:class:`~cosmology.calculators.growth.GrowthCalculator`,
        and the ``"fsigma8"``/``"s8"`` likelihoods) uses to
        normalize its own scale-independent growth factor. This one
        is a *derived* quantity, fixed by ``ln1e10As``, ``n_s``,
        ``tau_reio`` and the densities.

        A fit that varies the free ``sigma8`` while also using a
        CAMB-based CMB likelihood is therefore carrying two
        different amplitudes that nothing forces to agree.
        :class:`~stats.fitter.Fitter` warns about that combination;
        this method is how to check it.
        """

        return self._spectra()["sigma8"]

    # ---------------------------------------------------------

    def lensing_spectra(
        self,
        lmax: int,
    ) -> dict[str, np.ndarray]:
        r"""
        The inputs Planck's lensing likelihood is defined on,
        indexed from ``l = 0`` so array index and multipole
        coincide.

        Returns
        -------
        dict
            ``{"TT", "EE", "TE"}`` as ``D_l = l(l+1) C_l / 2 pi``
            in muK^2, and ``"PP"`` as
            ``[L(L+1)]^2 C_L^{phiphi} / 2 pi``.

        Notes
        -----
        These scalings are not stylistic -- they are what the
        bundled window functions were built against. The ``PP`` one
        is the easy one to get wrong: Planck's bandpowers are
        ~1.5e-7 at L ~ 30, where ``C_L^{phiphi}`` itself is ~1e-8
        and ``L(L+1)C_L/2pi`` is ~1.3e-6, so those two would land
        orders of magnitude out and be caught immediately -- but a
        stray ``2 pi`` would not, which is why the convention is
        written down rather than left to be inferred.
        """

        spectra = self._spectra()

        if lmax + 1 > len(spectra["ell"]):

            raise BoltzmannError(

                f"Lensing spectra requested to l = {lmax}, but CAMB "

                f"ran only to l = {len(spectra['ell']) - 1}. Raise "

                f"the backend's lmax.",

            )

        stop = lmax + 1

        return {

            "TT": spectra["D_TT"][:stop],

            "EE": spectra["D_EE"][:stop],

            "TE": spectra["D_TE"][:stop],

            "PP": spectra["PP"][:stop],

        }

    # ---------------------------------------------------------

    def cls(
        self,
        lmin: int = 2,
    ) -> dict[str, np.ndarray]:
        """
        Lensed CMB angular power spectra.

        Parameters
        ----------
        lmin : int, optional
            First multipole to return.

        Returns
        -------
        dict
            ``{"ell", "TT", "TE", "EE"}``, with the spectra given
            as ``C_l`` in muK^2 (not ``D_l``), which is what the
            Planck bandpower windows are defined on.
        """

        spectra = self._spectra()

        stop = self.lmax + 1

        keep = slice(lmin, stop)

        return {

            "ell": spectra["ell"][keep],

            "TT": spectra["TT"][keep],

            "TE": spectra["TE"][keep],

            "EE": spectra["EE"][keep],

        }

    # ---------------------------------------------------------

    def _run(self) -> dict[str, np.ndarray]:
        """
        Call CAMB once and keep every view the likelihoods need,
        indexed from l = 0.
        """

        camb = self._camb

        pars = self._build_params()

        try:

            results = camb.get_results(pars)

            powers = results.get_cmb_power_spectra(

                pars,

                CMB_unit="muK",

                raw_cl=True,

            )

            # [L(L+1)]^2 C_L^{phiphi} / 2pi in column 0 -- already
            # the convention Planck's lensing windows use.
            lens_potential = results.get_lens_potential_cls(

                lmax=self.lmax,

            )

            sigma8 = float(results.get_sigma8_0())

        except Exception as exc:

            raise BoltzmannError(

                f"CAMB failed for H0={self.cosmo.H0:.3f}, "

                f"Omega_m={self.cosmo.Omega_m:.4f}, "

                f"Omega_b={self.cosmo.Omega_b:.4f}, "

                f"tau={self.cosmo.tau_reio:.4f}: {exc}",

            ) from exc

        # (n_ell, 4) array indexed from l = 0, columns TT/EE/BB/TE.
        totals = powers["total"]

        n_ell = totals.shape[0]

        if self.lmax + 1 > n_ell:

            raise BoltzmannError(

                f"CAMB returned spectra only to l = {n_ell - 1}, "

                f"short of the requested lmax = {self.lmax}.",

            )

        ell = np.arange(n_ell)

        # D_l = l(l+1) C_l / 2pi. l = 0 and 1 are zero and unused;
        # the factor is written with the array's own `ell` so index
        # and multipole cannot drift apart.
        factor = ell * (ell + 1.0) / (2.0 * np.pi)

        potential = np.zeros(n_ell, dtype=float)
        potential[: lens_potential.shape[0]] = lens_potential[:, 0]

        return {

            "ell": ell,

            "TT": totals[:, 0],

            "EE": totals[:, 1],

            "TE": totals[:, 3],

            "D_TT": factor * totals[:, 0],

            "D_EE": factor * totals[:, 1],

            "D_TE": factor * totals[:, 3],

            "PP": potential,

            "sigma8": sigma8,

        }


    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (

            f"CAMBBackend(model="

            f"{type(self.cosmo).__name__}, "

            f"lmax={self.lmax})"

        )
