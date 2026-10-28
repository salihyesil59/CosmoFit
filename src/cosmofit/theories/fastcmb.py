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
own parameters; then ``ln1e10As``, ``n_s`` and ``tau_reio``.

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
}

_DENSITIES = ("h", "omega_b", "omega_cb", "m_nu", "N_eff")
_PRIMORDIAL = ("ln1e10As", "n_s", "tau_reio")


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
        Every this many predictions, CAMB runs as well and the error is
        measured; a prediction found worse than three times the
        tolerance is learned. Default 200; 0 never.

    :meth:`stats` says how many points CAMB and the emulator computed.
    """

    def initialize(self) -> None:

        self.settings = {
            name: self.info.pop(name, default) for name, default in EMULATOR_OPTIONS.items()
        }

        super().initialize()

        self.fastcmb = _import_fastcmb()
        self.emulator = None

    def _inputs(self, params) -> dict:

        d = self.provider.get_background_densities()
        expansion = self.provider.get_expansion()

        inputs = {name: float(d[name]) for name in _DENSITIES}
        inputs["Omega_k"] = float(expansion["context"].Omega_k)

        for name, value in expansion["sector_params"].items():
            if isinstance(value, numbers.Real):
                inputs[name] = float(value)

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

        result = self.emulator(self._inputs(params), oracle=camb)

        if result is None:
            return False

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

    def stats(self) -> dict:
        """
        Points CAMB and the emulator computed (``camb``, ``emulated``),
        the training set's size, and the checks' errors.
        """

        if self.emulator is None:
            return {"camb": self.n_calculations, "emulated": 0, "training_points": 0,
                    "checks": 0, "check_error_median": None, "check_error_max": None}

        stats = self.emulator.stats()
        stats["camb"] = stats.pop("oracle")

        return stats
