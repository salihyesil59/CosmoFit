"""
Holographic dark energy (Li 2004), with the future event horizon
as the infrared cutoff.
"""

from __future__ import annotations

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from CosmoFit.cosmology.core import Cosmology


#: Scale factor the background solution is tabulated down to.
#: ``Omega_DE`` falls off proportional to ``a`` below matter-radiation
#: equality, so there is nothing to resolve further back; this is
#: simply well past anything the distance integrator asks for.
A_MIN = 1.0e-8

#: Scale factor the solution is carried *forward* to.
#:
#: Nothing in the library asks for z < 0, so this is not there for
#: distances. It is there because the model is *defined* by the
#: future event horizon: the only way to check the solution against
#: its own definition is to integrate that horizon, and that
#: integral runs over the future. A model whose physics depends on
#: the future should be able to be asked about it.
A_MAX = 1.0e5

#: Points in ``ln a``. The solution is smooth and monotonic, so this
#: is generous rather than tuned.
N_GRID = 6000


class HDE(Cosmology):
    r"""
    Holographic dark energy with a future-event-horizon cutoff.

    The holographic principle bounds the energy in a region by the
    area of its boundary, which for a cosmological infrared cutoff
    ``L`` gives ``rho_DE = 3 c^2 M_p^2 / L^2``. Li (2004) showed
    that the only choice of ``L`` that yields an accelerating
    universe *and* a sensible equation of state is the **future
    event horizon**,

        L(a) = a Int_a^inf da' / (a'^2 H(a')),

    which is what this class implements. The single free parameter
    is ``c_hde``, the dimensionless constant in that relation.

    What makes this model different from every other one here
    ---------------------------------------------------------
    ``E(z)`` has no closed form. ``Omega_DE`` obeys

        d Omega_DE / d ln a
            = Omega_DE (1 - Omega_DE) (1 + 2 sqrt(Omega_DE) / c),

    which is solved once per :meth:`refresh` and splined. The
    expansion rate then follows algebraically from flatness,

        E(z)^2 = Omega_m (1+z)^3 / (1 - Omega_DE(z)),

    so this is the library's first background model that integrates
    an ODE rather than evaluating a formula. Everything downstream
    -- distances, growth, ``r_d`` -- only ever asks for ``E`` and
    ``dEdz``, so nothing else changes.

    The equation of state is
    ``w = -1/3 - (2/3) sqrt(Omega_DE) / c``, and its behaviour is
    the model's signature: ``w -> -1/3`` at early times, when dark
    energy is negligible, and ``w -> -1/3 - 2/(3c)`` in the far
    future. So ``c = 1`` approaches a cosmological constant only
    asymptotically, ``c > 1`` stays quintessence-like, and
    ``c < 1`` crosses into phantom territory. Fits have generally
    preferred ``c`` somewhat below 1, which is the interesting part
    -- the crossing is not put in by hand.

    Flat only
    ---------
    The event-horizon integral, and hence the ODE above, assumes
    flatness. Curvature changes the causal structure the holographic
    bound is applied to rather than merely adding a term, so a
    curved version is a different model and is refused here instead
    of being silently approximated.

    Caveats
    -------
    The cutoff is the *future* event horizon, so the present energy
    density depends on the entire future expansion history. That is
    unusual as physics and has been argued about at length; it is
    also what the model is, and the alternative cutoffs (Hubble
    radius, particle horizon) do not accelerate.

    Its perturbations are outside what a standard Boltzmann code
    solves, so it is refused by
    :class:`~cosmology.boltzmann.CAMBBackend`; use the compressed
    distance priors for its CMB constraint.

    References
    ----------
    Li (2004), Phys. Lett. B 603, 1,
    `arXiv:hep-th/0403127 <https://arxiv.org/abs/hep-th/0403127>`_.

    Wang, Mortsell et al. (2017), Phys. Rept. 696, 1,
    `arXiv:1612.00345 <https://arxiv.org/abs/1612.00345>`_ (review).
    """

    MODEL_NAME = "HDE"
    MODEL_LABEL = "HDE"

    EXTRA_PARAMS = {

        "c_hde": {
            "default": 0.8,
            "bounds": (0.2, 2.0),
            "label": r"$c$",
        },

    }

    #: Cached background solution. Declared on the class, not in
    #: ``__init__``, because ``Cosmology.__init__`` builds the
    #: distance table -- which calls ``E(z)``, which reads this --
    #: before any subclass ``__init__`` body could run.
    _omega_de_spline = None

    # ---------------------------------------------------------

    def refresh(self) -> None:
        """
        Re-solve the background ODE, then rebuild everything that
        depends on it.

        Order matters: the distance table is built from ``E(z)``,
        which reads the spline, so the spline has to exist first.
        """

        self._solve_background()

        super().refresh()

    # ---------------------------------------------------------

    def _solve_background(self) -> None:
        """
        Integrate ``Omega_DE`` backwards from today and spline it
        against ``ln a``.
        """

        if abs(self.Omega_k) > 1.0e-12:

            raise ValueError(

                f"HDE is implemented for a flat universe only, but "
                f"Omega_k = {self.Omega_k:.4g}. The future event "
                f"horizon that defines the model is a statement "
                f"about the causal structure, which curvature "
                f"changes -- a curved holographic model is a "
                f"different model, not this one with an extra term.",

            )

        c = float(self.params.c_hde)

        if c <= 0.0:

            raise ValueError(
                f"HDE needs c > 0, got c_hde = {c:.4g}.",
            )

        def rhs(_x, y):

            omega = np.clip(y[0], 1.0e-16, 1.0 - 1.0e-16)

            return [

                omega

                * (1.0 - omega)

                * (1.0 + 2.0 * np.sqrt(omega) / c)

            ]

        x_min, x_max = np.log(A_MIN), np.log(A_MAX)

        pieces = []

        for x_end in (x_min, x_max):

            solution = solve_ivp(

                rhs,

                (0.0, x_end),

                [self.Omega_de0],

                rtol=1.0e-10,

                atol=1.0e-14,

                dense_output=True,

            )

            if not solution.success:

                raise RuntimeError(
                    f"HDE background ODE failed integrating to "
                    f"ln a = {x_end:.3g}: {solution.message}",
                )

            pieces.append(solution)

        past, future = pieces

        n_past = int(N_GRID * abs(x_min) / (abs(x_min) + x_max))

        x = np.concatenate([

            np.linspace(x_min, 0.0, n_past, endpoint=False),

            np.linspace(0.0, x_max, N_GRID - n_past),

        ])

        omega = np.where(

            x < 0.0,

            past.sol(np.minimum(x, 0.0))[0],

            future.sol(np.maximum(x, 0.0))[0],

        )

        self._omega_de_spline = CubicSpline(

            x,

            np.clip(omega, 1.0e-16, 1.0 - 1.0e-12),

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):
        """
        Dark-energy density parameter at ``z``, from the solved
        background.

        Below the tabulated range ``Omega_DE`` is continued with its
        analytic early-time limit, ``Omega_DE proportional to a``,
        rather than by extrapolating the spline -- a cubic run
        outside its knots would eventually go negative, and a
        negative density parameter would surface as a NaN in
        ``E(z)`` rather than as an error.
        """

        if self._omega_de_spline is None:

            self._solve_background()

        z = np.asarray(z, dtype=float)

        x = -np.log1p(z)

        x_min = self._omega_de_spline.x[0]

        x_max = self._omega_de_spline.x[-1]

        inside = (x >= x_min) & (x <= x_max)

        omega = np.empty_like(x)

        omega[inside] = self._omega_de_spline(x[inside])

        below = x < x_min

        if np.any(below):

            edge = float(self._omega_de_spline(x_min))

            omega[below] = edge * np.exp(x[below] - x_min)

        # Far future: Omega_DE -> 1. Held at the last tabulated
        # value rather than extrapolated, which for a cubic would
        # run above 1 and make ``E`` imaginary.
        above = x > x_max

        if np.any(above):

            omega[above] = float(self._omega_de_spline(x_max))

        return omega

    # ---------------------------------------------------------

    def w_de(self, z):
        r"""
        Equation of state,
        ``w = -1/3 - (2/3) sqrt(Omega_DE) / c``.
        """

        return (

            -1.0 / 3.0

            - (2.0 / 3.0)

            * np.sqrt(self.Omega_de(z))

            / float(self.params.c_hde)

        )

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * (1.0 + z) ** 3

            / (1.0 - self.Omega_de(z))

        )

    # ---------------------------------------------------------

    def dEdz(self, z):
        r"""
        Analytic, using the ODE rather than differencing ``E``.

        With ``f = 1 - Omega_DE`` and ``E^2 = Omega_m (1+z)^3 / f``,

            dE/dz = Omega_m (1+z)^2
                    [3 - Omega_DE (1 + 2 sqrt(Omega_DE)/c)]
                    / (2 E f)

        where the bracket is ``3 - (d Omega_DE / d ln a) / f``, and
        that derivative is exactly what the ODE gives -- so no
        numerical differentiation enters anywhere.
        """

        z = np.asarray(z, dtype=float)

        omega = self.Omega_de(z)

        f = 1.0 - omega

        c = float(self.params.c_hde)

        bracket = 3.0 - omega * (1.0 + 2.0 * np.sqrt(omega) / c)

        return (

            self.Omega_m

            * (1.0 + z) ** 2

            * bracket

            / (2.0 * self.E(z) * f)

        )
