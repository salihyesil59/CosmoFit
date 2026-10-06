"""
The early universe: the sound horizon at the drag epoch and at
recombination, computed from the background's own expansion.

The old sound-horizon calculator built its own ``H(a)`` from photons,
neutrinos and ``omega_cb`` and assumed nothing else mattered before
recombination; a model where something else did had its own ``E(z)``
patched in after the fact. Here the integral simply reads
:class:`~theories.background.Background`'s ``E(z)``, which already
contains everything -- an early dark energy, an extra matter-like
component, a coupling -- so there is nothing to patch.

Two quantities still come from fits to CAMB rather than from a
recombination calculation: the drag and decoupling redshifts
(:func:`~cosmology.calculators.sound_horizon.z_drag_fit`,
:func:`~cosmology.calculators.recombination.z_star_fit`). Both were
calibrated in LCDM, where the densities reach them through the
expansion rate they set; the drag fit is therefore evaluated at the
cold-matter density that gives the background's *actual* ``H(z_drag)``,
which for a standard early universe is ``omega_cb`` itself.

Requires a background with ``radiation: true``.
"""

from __future__ import annotations

import math

import numpy as np

from scipy.integrate import simpson

from cosmofit.core.component import ComponentError, Theory
from cosmofit.cosmology.core.constants import c
from cosmofit.cosmology.calculators.recombination import z_star_fit
from cosmofit.cosmology.calculators.sound_horizon import z_drag_fit

from .background import neutrino_density


__all__ = ["EarlyUniverse"]


class EarlyUniverse(Theory):
    """
    Provides
    --------
    ``rdrag`` and ``rstar`` [Mpc], ``zdrag``, ``zstar``, ``thetastar``
    (``100 r_s(z_*) / D_M(z_*)``), and ``rs`` (``z=...``, the comoving
    sound horizon at any redshift).

    Derived parameters
    ------------------
    The same five, under CAMB's names.
    """

    def initialize(self) -> None:

        if self.info:
            raise ComponentError(f"{self.name} takes no options, got {sorted(self.info)}.")

    def get_requirements(self) -> dict:
        return {"E": None, "DM": None, "background_densities": None}

    def check_model(self, model) -> None:

        background = self.provider.theory_for("background_densities")

        if not getattr(background, "radiation", True):
            raise ComponentError(
                f"{self.name} needs a background with radiation -- the sound "
                f"horizon is an integral through the radiation era -- but "
                f"{background.name} has radiation: false."
            )

    def get_can_provide(self) -> list[str]:
        return ["rdrag", "zdrag", "rstar", "zstar", "thetastar", "rs"]

    def get_derived_params(self) -> list[str]:
        return ["rdrag", "zdrag", "rstar", "zstar", "thetastar"]

    # ---------------------------------------------------------

    def calculate(self, state: dict, want_derived: bool = True, **params):

        d = self.provider.get_background_densities()

        E = self.provider.get_E
        h = d["h"]

        def rs(z_end: float, n: int = 1201, decades: float = 8.0) -> float:
            """
            ``r_s(z) = int_0^{a(z)} c_s / (a^2 H) da``, on a grid uniform in
            ``log10 a`` reaching ``decades`` below ``a(z)`` -- where the
            omitted piece is ~1e-8 of the total, the integrand being
            constant in radiation domination.
            """

            log_a = np.linspace(-math.log10(1.0 + z_end) - decades,
                                -math.log10(1.0 + z_end), n)
            a = 10.0 ** log_a

            R_b = 3.0 * d["omega_b"] / (4.0 * d["omega_gamma"]) * a
            sound_speed = c / np.sqrt(3.0 * (1.0 + R_b))

            H = 100.0 * h * E(1.0 / a - 1.0)

            return float(
                simpson(sound_speed / (a * H), x=log_a) * math.log(10.0)
            )

        omega_nu = d["omega_nu_massive"]

        z_standard = z_drag_fit(d["omega_b"], d["omega_cb"], d["N_eff"], omega_nu)

        # The cold-matter density that gives the background's actual
        # H at the drag epoch: omega_cb exactly for a standard early
        # universe, more or less for anything that changes it.
        a = 1.0 / (1.0 + z_standard)

        effective = (
            float(E(z_standard)) ** 2 * h * h
            - d["omega_gamma"] / a ** 4
            - float(neutrino_density(a, d["N_eff"], d["m_nu"], d["omega_gamma"]))
        ) * a ** 3

        zdrag = z_drag_fit(d["omega_b"], effective, d["N_eff"], omega_nu)

        zstar = z_star_fit(d["omega_b"], d["Omega_m"] * h * h)

        rdrag = rs(zdrag)
        rstar = rs(zstar)

        thetastar = 100.0 * rstar / float(self.provider.get_DM(z=zstar))

        state.update(
            rdrag=rdrag, zdrag=zdrag, rstar=rstar, zstar=zstar,
            thetastar=thetastar, rs_function=rs,
        )

        if want_derived:
            state["derived"] = {
                "rdrag": rdrag, "zdrag": zdrag, "rstar": rstar,
                "zstar": zstar, "thetastar": thetastar,
            }

        return True

    # ---------------------------------------------------------

    def get_rs(self, z):
        """Comoving sound horizon at redshift ``z`` [Mpc]."""

        rs = self.current_state["rs_function"]

        z = np.asarray(z, dtype=float)

        if z.ndim == 0:
            return rs(float(z))

        return np.array([rs(float(x)) for x in z])
