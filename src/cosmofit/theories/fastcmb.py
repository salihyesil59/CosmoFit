"""
CMB spectra from an emulator of CAMB that learns while the chain runs.

:class:`FastCMB` *is* the ``camb`` theory -- same options, parameters,
requests and results -- with `fastcmb <https://github.com/salihyesil59/fastcmb>`_'s
emulator in front of it. At each point the emulator predicts the
spectra and the error of its prediction in units of ``ln L``. Where
that error is below ``tolerance`` the prediction is the answer, in
milliseconds; elsewhere CAMB runs exactly as the ``camb`` theory runs
it, and the point is learned.

What is emulated is what a CMB likelihood reads: the lensed ``Cl`` and
``sigma8``. A model that asks for ``P(k)`` or ``sigma(R)`` gets CAMB at
every point, as with ``camb``.

The spectra are learned as functions of what CAMB is given, read off
the background as the ``camb`` theory reads it: ``h``, ``omega_b``,
``omega_cb``, ``m_nu``, ``N_eff``, ``Omega_k`` and the dark sector's
own parameters (EDE's ``z_c`` as ``log10_z_c``); then ``ln1e10As``,
``n_s`` and ``tau_reio``. With a dark sector of its own, also
``D_M_star``: the distance to last scattering, ``D_M(z = 1090)``.
The primary CMB is almost a function of
it alone among the late-time parameters, and the dark sector's
parameters trade off against ``h`` along it; given it, the spectra are
far smoother in them, and much less of CAMB is needed.

fastcmb is optional (``pip install "cosmofit[fast]"``); only a model
with this theory in it imports it.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from cosmofit.core.component import ComponentError

from .boltzmann import CAMB


__all__ = ["FastCMB"]

#: The emulator's options and their defaults; the rest are ``camb``'s.
EMULATOR_OPTIONS = {
    "tolerance": 0.05,
    "sigma8_tolerance": 1e-3,
    "min_train": 30,
    "retrain_fraction": 0.1,
    "max_train": 800,
    "training_file": None,
    "check_every": 200,
    "screen_tolerance": 1.0,
}

_DENSITIES = ("h", "omega_b", "omega_cb", "m_nu", "N_eff")
_PRIMORDIAL = ("ln1e10As", "n_s", "tau_reio")
#: Redshift of the distance to last scattering given to the emulator.
_Z_STAR = 1090.0
#: Dark-sector parameters that span decades, given to the emulator as
#: their log10: EDE's ``z_c``, whose spectra change far faster near
#: recombination than at 10^4, and are smooth in its logarithm.
_LOG_INPUTS = ("z_c",)


def _import_fastcmb():

    try:
        import fastcmb
    except ImportError:
        raise ComponentError(
            "The fastcmb theory needs the fastcmb package, an optional "
            "dependency:\n\n    pip install \"cosmofit[fast]\"\n\n"
            "or use the camb theory, which it emulates."
        ) from None

    return fastcmb


class FastCMB(CAMB):
    """
    Options
    -------
    Those of :class:`~cosmofit.theories.boltzmann.CAMB`, and:

    tolerance : float
        Largest predicted error of ``ln L`` at which the emulator
        answers, for an idealized Planck. Default 0.05.
    sigma8_tolerance : float
        Largest predicted relative error of ``sigma8``. Default 1e-3.
    min_train : int
        CAMB points before the emulator is first fitted. Default 30.
    retrain_fraction : float
        Refit once the new points are this fraction of the old.
        Default 0.1.
    max_train : int
        Most points the emulator is fitted on, the latest. Default 800.
    training_file : str
        Where the CAMB points are kept (no extension), so that parallel
        chains share them and a later run of the same model starts from
        them.
    check_every : int
        After this many predictions, CAMB runs as well and the error is
        measured; a prediction found worse than three times the
        tolerance is learned. While the checks pass the interval
        doubles, up to ten times this. Default 200; 0 never.

    screen_tolerance : float
        Largest predicted ``ln L`` error of an *approximate* answer,
        given only while :attr:`screening` is on: a sampler screening a
        proposal rejects it from that when it is clearly far worse than
        the current point, and asks again, with screening off, when it
        is not. Default 1.

    :meth:`stats` says how many points CAMB and the emulator computed.
    """

    #: Whether an approximate answer will do (see ``screen_tolerance``);
    #: set by the sampler around a screening evaluation.
    screening = False

    def initialize(self) -> None:

        self.settings = {
            name: self.info.pop(name, default) for name, default in EMULATOR_OPTIONS.items()
        }

        super().initialize()

        self.fastcmb = _import_fastcmb()
        self.emulator = None

    def get_requirements(self) -> dict:
        return {**super().get_requirements(), "DM": None}

    def _inputs(self, params) -> dict:

        d = self.provider.get_background_densities()
        expansion = self.provider.get_expansion()

        inputs = {name: float(d[name]) for name in _DENSITIES}
        inputs["Omega_k"] = float(expansion["context"].Omega_k)

        for name, value in expansion["sector_params"].items():
            if isinstance(value, numbers.Real):
                if name in _LOG_INPUTS:
                    inputs[f"log10_{name}"] = math.log10(value)
                else:
                    inputs[name] = float(value)
                inputs["D_M_star"] = None

        if "D_M_star" in inputs:
            inputs["D_M_star"] = float(self.provider.get_DM(_Z_STAR))

        inputs.update({name: float(params[name]) for name in _PRIMORDIAL})

        return inputs

    def _tag(self) -> str:

        sector = self.provider.get_expansion()["sector"]

        return (
            f"cosmofit|{type(sector).__module__}.{type(sector).__qualname__}"
            f"|lmax={self.lmax}|lens={self.lens_potential_accuracy}"
            f"|halofit={self.halofit_version}"
        )

    def calculate(self, state: dict, want_derived: bool = True, **params):

        # P(k) and sigma(R) are not emulated: CAMB, as before.
        if self.spectra or self.sigma_pairs:
            return super().calculate(state, want_derived, **params)

        if self.emulator is None:
            s = self.settings
            self.emulator = self.fastcmb.CMBEmulator(
                lmax=self.lmax, tag=self._tag(), tolerance=s["tolerance"],
                sigma8_tolerance=s["sigma8_tolerance"], min_train=s["min_train"],
                retrain_fraction=s["retrain_fraction"], max_train=s["max_train"],
                training_file=s["training_file"], check_every=s["check_every"],
            )

        Omega_m = self.provider.get_background_densities()["Omega_m"]

        def camb(_):
            # The camb theory's own computation, into this state.
            if not super(FastCMB, self).calculate(state, want_derived, **params):
                return None
            return {
                "Cl": state["Cl"],
                "sigma8": state["S8"] / math.sqrt(Omega_m / 0.3),
                "sigma8_cb": state["sigma8_0"],
            }

        loose = self.settings["screen_tolerance"] if self.screening else None
        result = self.emulator(self._inputs(params), oracle=camb, loose_tolerance=loose)

        if result is None:
            return False

        if result.approximate:
            state["approximate"] = result.error

        if not result.emulated:
            # CAMB ran, and filled the state.
            return True

        S8 = result.sigma8 * math.sqrt(Omega_m / 0.3)

        state.update(
            Cl=result.Cl, sigma8_0=result.sigma8_cb, S8=S8, Pk_grid={}, sigma_R={},
            sigma_R_axes=(np.array(sorted(self.redshifts)), np.array([])),
            Pk_interpolators={},
        )

        if want_derived:
            state["derived"] = {
                "sigma8": result.sigma8,
                "S8": S8,
                "A_s": math.exp(params["ln1e10As"]) * 1.0e-10,
            }

        return True

    def compute(self, params: dict, want_derived: bool = True, upstream: tuple = ()) -> bool:

        # An approximate state is never the answer without screening;
        # and it is told apart from the exact one in the key that those
        # reading this theory cache by.
        if not self.screening:
            self.drop_approximate()

        ok = super().compute(params, want_derived, upstream)

        if self.approximate_error is not None:
            self.current_key = (*self.current_key, "approximate")

        return ok

    @property
    def approximate_error(self) -> float | None:
        """The predicted ``ln L`` error if the current state is approximate."""

        state = self.current_state
        return None if state is None else state.get("approximate")

    def drop_approximate(self) -> None:
        """Forget the approximate states."""

        for key in [k for k, v in self._cache.items() if v is not None and "approximate" in v]:
            del self._cache[key]

    def stats(self) -> dict:
        """
        Points CAMB and the emulator computed (``camb``, ``emulated``),
        the training set's size, and the checks' errors.
        """

        if self.emulator is None:
            return {"camb": self.n_calculations, "emulated": 0, "approximate": 0, "training_points": 0,
                    "checks": 0, "check_error_median": None, "check_error_max": None}

        stats = self.emulator.stats()
        stats["camb"] = stats.pop("oracle")

        return stats
