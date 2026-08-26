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

        key = self._parameter_key()

        if key != self._cache_key:

            self._cache_value = self._run(lmin)

            self._cache_key = key

        return self._cache_value

    # ---------------------------------------------------------

    def _run(self, lmin: int) -> dict[str, np.ndarray]:
        """
        Actually call CAMB and reshape the output.
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

        except Exception as exc:

            raise BoltzmannError(

                f"CAMB failed for H0={self.cosmo.H0:.3f}, "

                f"Omega_m={self.cosmo.Omega_m:.4f}, "

                f"Omega_b={self.cosmo.Omega_b:.4f}, "

                f"tau={self.cosmo.tau_reio:.4f}: {exc}",

            ) from exc

        # (n_ell, 4) array indexed from l = 0, columns TT/EE/BB/TE.
        totals = powers["total"]

        ell = np.arange(totals.shape[0])

        stop = self.lmax + 1

        if stop > totals.shape[0]:

            raise BoltzmannError(

                f"CAMB returned spectra only to l = "

                f"{totals.shape[0] - 1}, short of the requested "

                f"lmax = {self.lmax}.",

            )

        keep = slice(lmin, stop)

        return {

            "ell": ell[keep],

            "TT": totals[keep, 0],

            "EE": totals[keep, 1],

            "TE": totals[keep, 3],

        }

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (

            f"CAMBBackend(model="

            f"{type(self.cosmo).__name__}, "

            f"lmax={self.lmax})"

        )
