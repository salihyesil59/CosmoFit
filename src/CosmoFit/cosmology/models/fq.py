"""
f(Q) modified gravity (symmetric teleparallel, non-metricity scalar).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.powers import cube
from scipy.special import lambertw

from CosmoFit.cosmology.core import Cosmology


class FQExponential(Cosmology):
    r"""
    f(Q) gravity, exponential model.

    f(Q) modified gravity replaces the Ricci scalar R in the
    Einstein-Hilbert action with an arbitrary function f of the
    non-metricity scalar Q (symmetric teleparallel gravity; GR is
    recovered exactly at f(Q)=Q). This is a genuine modification of
    the gravitational field equations, not a dark-energy fluid
    bolted onto standard GR the way LCDM/wCDM/CPL/JBP/BA/GCG all
    are.

    This implements the specific model of Anagnostopoulos, Basilakos
    & Saridakis (2021):

        f(Q) = Q * exp(lambda * Q0 / Q),    Q = 6 H^2,  Q0 = 6 H0^2

    with lambda the model's only new quantity. Crucially, lambda is
    *not* an independent free parameter: their eq. (10) fixes it
    from Omega_m via the Lambert W function, so this model has
    exactly as many free parameters as flat LCDM (H0, Omega_m) --
    only ``Omega_m`` needs to be given/fit; lambda is derived
    automatically. Flat only -- ``Omega_k`` is ignored, since the
    Q=6H^2 identity this model is built on is itself a flat-FLRW
    result (see the reference below for the curved-space case, not
    implemented here).

    The Friedmann equation (their eq. 9, radiation dropped -- as
    with every other model in this library) is transcendental, not
    closed-form:

        (E^2 - 2*lambda) * exp(lambda / E^2) = Omega_m * (1+z)^3

    ``E(z)`` solves this via a fixed number of vectorized Newton
    iterations (the function is smooth and monotonic in E^2 over the
    physical range, so this converges to machine precision well
    before the iteration budget is used). ``dEdz`` comes from
    implicit differentiation of the same relation -- closed-form,
    no extra root-finding -- so ``background.q()`` /
    ``plots.deceleration()`` work exactly like every other model,
    not as some numerical-fallback special case.

    ``Omega_de(z)`` is presented, as is standard for this model
    class, as an *effective/geometric* dark-energy density
    ``E(z)^2 - Omega_m*(1+z)^3`` -- there is no actual second fluid
    here, the accelerated expansion comes entirely from the modified
    gravitational sector.

    **Growth of structure.** ``mu(a, k)`` implements the standard
    sub-horizon, quasi-static result for f(Q) gravity's effective
    gravitational coupling, G_eff/G_N = 1/f_Q where
    f_Q = df/dQ (Barros, Barreiro, Koivisto & Nunes 2020,
    arXiv:2004.07867, "Testing F(Q) gravity with redshift space
    distortions" -- the same growth-rate observable this feeds into
    here). For this model, f_Q = exp(lambda/E^2) (1 - lambda/E^2)
    (derived by differentiating f(Q) above and using
    Q0/Q = 1/E(z)^2), scale-independent (``k`` is accepted for
    interface consistency with :meth:`Cosmology.mu` but ignored).
    lambda=0 gives f_Q=1, i.e. mu=1 exactly -- by this model's own
    closure condition (see ``_lam`` above), that is the
    Omega_m -> 1 limit (matter-only, EdS), not Omega_m -> 0.

    References
    ----------
    Anagnostopoulos, Basilakos & Saridakis (2021), "First evidence
    that non-metricity f(Q) gravity could challenge LCDM", Phys.
    Lett. B 822, 136634, arXiv:2104.15123.

    Barros, Barreiro, Koivisto & Nunes (2020), "Testing F(Q) gravity
    with redshift space distortions", Phys. Dark Univ. 30, 100616,
    arXiv:2004.07867.
    """

    MODEL_NAME = "FQExponential"
    MODEL_LABEL = r"$f(Q)$ exponential"

    #: Newton iterations for solving E^2 -- the relation is smooth
    #: and monotonic over the physical range, so this converges to
    #: machine precision in well under half this budget; kept large
    #: since the cost is negligible (vectorized over the whole
    #: z-grid, once per MCMC step) and correctness matters more here
    #: than shaving a few iterations off.
    _NEWTON_ITERATIONS = 30

    # ---------------------------------------------------------

    @property
    def _lam(self) -> float:
        """
        lambda(Omega_m), from the E(z=0)=1 closure condition:
        lambda = 1/2 + W0(-Omega_m / (2 sqrt(e))).

        Re-derived directly rather than taken as-is from the
        reference's eq. 10, after that transcription failed a
        numerical closure check: solving (1-2*lambda)*exp(lambda) =
        Omega_m for lambda by substituting u=1-2*lambda, v=-u/2
        gives v*exp(v) = -Omega_m/(2 sqrt(e)), i.e.
        v = W0(-Omega_m/(2 sqrt(e))), lambda = (1+2v)/2 = 1/2 + v.
        Verified against a numerical root-find of the same equation
        across several Omega_m values.
        """

        w = lambertw(-self.Omega_m / (2.0 * np.sqrt(np.e)), k=0)

        return float(np.real(0.5 + w))

    # ---------------------------------------------------------

    def _solve_E2(self, z):
        """
        Vectorized Newton solve of
        (E^2 - 2*lambda) * exp(lambda/E^2) = Omega_m*(1+z)^3
        for E^2, at every z simultaneously.
        """

        z = np.asarray(z, dtype=float)
        lam = self._lam
        rhs = self.Omega_m * cube(1.0 + z)

        # LCDM-like starting point -- this model is close to LCDM
        # by construction, so this is already a good initial guess.
        x = rhs + (1.0 - self.Omega_m)

        for _ in range(self._NEWTON_ITERATIONS):
            expo = np.exp(lam / x)
            g = (x - 2.0 * lam) * expo
            dg = expo * (1.0 - lam * (x - 2.0 * lam) / x ** 2)
            x = x - (g - rhs) / dg

        return x

    # ---------------------------------------------------------

    def E(self, z):
        """
        Dimensionless Hubble parameter (solves the transcendental
        Friedmann equation -- see the class docstring).
        """

        return np.sqrt(self._solve_E2(z))

    # ---------------------------------------------------------

    def dEdz(self, z):
        """
        Derivative of E(z), by implicit differentiation of the
        Friedmann relation (closed-form given E(z), no extra
        root-finding).
        """

        z = np.asarray(z, dtype=float)
        lam = self._lam

        x = self._solve_E2(z)
        expo = np.exp(lam / x)
        dg_dx = expo * (1.0 - lam * (x - 2.0 * lam) / x ** 2)

        d_rhs_dz = 3.0 * self.Omega_m * (1.0 + z) ** 2

        dx_dz = d_rhs_dz / dg_dx

        return dx_dz / (2.0 * np.sqrt(x))

    # ---------------------------------------------------------

    def Omega_de(self, z):
        """
        Effective/geometric dark-energy density -- see the class
        docstring for why this isn't a real second fluid here.
        """

        z = np.asarray(z, dtype=float)

        return self.E(z) ** 2 - self.Omega_m * cube(1.0 + z)

    # ---------------------------------------------------------

    def mu(self, a, k=None):
        """
        Effective gravitational coupling G_eff/G_N = 1/f_Q -- see
        the class docstring for the derivation.
        """

        a = np.asarray(a, dtype=float)
        z = 1.0 / a - 1.0

        lam = self._lam
        x = 1.0 / self.E(z) ** 2

        f_Q = np.exp(lam * x) * (1.0 - lam * x)

        return 1.0 / f_Q
