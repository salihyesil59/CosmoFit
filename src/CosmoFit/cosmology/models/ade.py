"""
New agegraphic dark energy (Wei & Cai 2008).
"""

from __future__ import annotations

import numpy as np

from scipy.integrate import solve_ivp

from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.hermite import hermite_spline
from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


#: Scale factor the background is tabulated from and to. The
#: forward reach is not for distances -- nothing asks for z < 0 --
#: but so the solution can be checked against the conformal-time
#: integral that defines it, which runs over the future.
A_MIN = 1.0e-8
A_MAX = 1.0e3

#: Points in ``ln a``.
N_GRID = 4000


class ADE(Cosmology):
    r"""
    New agegraphic dark energy: the holographic cutoff is the age
    of the universe, in conformal time.

    The holographic family differs only in what length scale is put
    in ``rho_DE = 3 c^2 M_p^2 / L^2``.
    :class:`~cosmology.models.hde.HDE` uses the future event
    horizon, :class:`~cosmology.models.rde.RDE` the Ricci scalar,
    and this uses the **conformal age**,

        rho_DE = 3 n^2 M_p^2 / eta^2,
        eta = Int dt / a = Int da / (a^2 H),

    which is causal, local in time, and needs no reference to the
    future -- the objection HDE attracts.

    Like HDE there is no closed form. Writing
    ``Omega_DE = n^2 / (eta H)^2`` and differentiating the
    definition of ``eta`` gives

        d Omega_DE / d ln a
            = Omega_DE (1 - Omega_DE)
              [3 - (2/n) sqrt(Omega_DE) / a],

    solved and splined on every :meth:`refresh`, with the expansion
    rate following from flatness as
    ``E^2 = Omega_m (1+z)^3 / (1 - Omega_DE)``.

    The early-time behaviour is the model's signature and its own
    consistency check: for ``Omega_DE << 1`` the equation forces
    ``Omega_DE -> n^2 a^2 / 4`` exactly, so dark energy dilutes as
    ``a^2`` rather than the ``a`` of HDE, and the equation of state
    is

        w = -1 + (2 / 3n) sqrt(Omega_DE) / a,

    which tends to ``-2/3`` at early times and to ``-1`` in the far
    future for every ``n``. So unlike HDE, this model cannot be
    phantom at any ``n``: the crossing HDE predicts is simply not
    available here, which is the sharpest observational difference
    between them.

    Caveats
    -------
    Flat universes only, and refused by
    :class:`~cosmology.boltzmann.CAMBBackend`, for the same reasons
    as :class:`~cosmology.models.hde.HDE`.

    References
    ----------
    Wei & Cai (2008), Phys. Lett. B 660, 113,
    `arXiv:0708.0884 <https://arxiv.org/abs/0708.0884>`_.
    """

    MODEL_NAME = "ADE"
    MODEL_LABEL = "ADE"

    EXTRA_PARAMS = {

        "n_ade": {
            "default": 2.8,
            "bounds": (0.5, 8.0),
            "label": r"$n$",
        },

    }

    #: Parameters this model *derives* rather than accepts. ``ADE``
    #: fixes the matter density from ``n_ade`` -- see :meth:`Omega_m`
    #: -- so freeing it would sample a number nothing reads.
    DERIVED_PARAMS = frozenset({"Omega_m"})

    #: Cached background. On the class rather than in ``__init__``,
    #: because ``Cosmology.__init__`` builds the distance table --
    #: which calls ``E(z)``, which reads this -- before any subclass
    #: ``__init__`` body could run.
    _omega_de_spline = None

    # ---------------------------------------------------------

    def refresh(self) -> None:

        self._solve_background()

        super().refresh()

    # ---------------------------------------------------------

    def _solve_background(self) -> None:
        """
        Integrate the ODE **forward** from the early-time condition.

        This is what makes ADE the model it is rather than a
        one-parameter family. For ``Omega_DE << 1`` the equation of
        motion forces ``Omega_DE -> n^2 a^2 / 4`` exactly, so the
        solution is fixed by ``n`` alone -- and therefore so is
        today's ``Omega_DE``, and so is ``Omega_m``.

        Integrating backwards from today with ``Omega_m`` free, as
        :class:`~cosmology.models.hde.HDE` does, would silently
        produce a *different* model: one whose early-time limit is
        whatever the backward walk happens to reach, which is not
        ``n^2 a^2 / 4`` and does not satisfy the definition. That is
        how this was first written here, and the early-time check
        below is what caught it.
        """

        if abs(self.Omega_k) > 1.0e-12:

            raise ValueError(

                f"ADE is implemented for a flat universe only, but "
                f"Omega_k = {self.Omega_k:.4g}. The conformal age "
                f"that defines the model is a statement about the "
                f"causal structure, which curvature changes.",

            )

        n = float(self.params.n_ade)

        if n <= 0.0:

            raise ValueError(
                f"ADE needs n > 0, got n_ade = {n:.4g}.",
            )

        def rhs(x, y):

            omega = np.clip(y[0], 1.0e-16, 1.0 - 1.0e-16)

            return [

                omega

                * (1.0 - omega)

                * (3.0 - (2.0 / n) * np.sqrt(omega) / np.exp(x))

            ]

        x_min, x_max = np.log(A_MIN), np.log(A_MAX)

        # The initial condition *is* the model.
        start = n ** 2 * A_MIN ** 2 / 4.0

        solution = solve_ivp(

            rhs,

            (x_min, x_max),

            [start],

            rtol=1.0e-11,

            atol=1.0e-16,

            dense_output=True,

        )

        if not solution.success:

            raise RuntimeError(
                f"ADE background ODE failed: {solution.message}",
            )

        x = np.linspace(x_min, x_max, N_GRID)

        omega = np.clip(solution.sol(x)[0], 1.0e-16, 1.0 - 1.0e-12)

        derivative = omega * (1.0 - omega) * (

            3.0 - (2.0 / n) * np.sqrt(omega) / np.exp(x)

        )

        self._omega_de_spline = hermite_spline(x, omega, derivative)

        self._omega_de_today = float(solution.sol(0.0)[0])

    # ---------------------------------------------------------

    def omega_de_fraction(self, z: Redshift) -> Array:
        """
        The **density parameter** ``rho_DE(z) / rho_crit(z)``, which
        is what the ODE solves for -- not :meth:`Omega_de`, which is
        in units of *today's* critical density. They agree at
        ``z = 0`` and nowhere else.
        """

        if self._omega_de_spline is None:

            self._solve_background()

        z = np.asarray(z, dtype=float)

        x = -np.log1p(z)

        knots = self._omega_de_spline.x

        x_min, x_max = knots[0], knots[-1]

        omega = np.empty_like(x)

        inside = (x >= x_min) & (x <= x_max)

        omega[inside] = self._omega_de_spline(x[inside])

        # Below the table, the analytic early-time limit
        # ``Omega_DE -> n^2 a^2 / 4`` rather than an extrapolated
        # cubic, which would eventually go negative and surface as a
        # NaN in E(z).
        below = x < x_min

        if np.any(below):

            edge = float(self._omega_de_spline(x_min))

            omega[below] = edge * np.exp(2.0 * (x[below] - x_min))

        above = x > x_max

        if np.any(above):

            omega[above] = float(self._omega_de_spline(x_max))

        return omega

    # ---------------------------------------------------------

    @property
    def Omega_m(self) -> float:
        r"""
        Matter density, **derived** from ``n_ade`` rather than
        sampled.

        The early-time condition ``Omega_DE -> n^2 a^2 / 4`` fixes
        the whole solution from ``n``, so today's split between
        matter and dark energy is a prediction:

        =========  ===========
        ``n``      ``Omega_m``
        =========  ===========
        2.4          0.359
        2.8          0.280
        3.2          0.220
        =========  ===========

        This is the model's most striking feature -- it has *one
        fewer* free parameter than LCDM, and the published
        constraint ``n = 2.78-2.81`` therefore predicts
        ``Omega_m = 0.278-0.283`` without being told.

        ``params.Omega_m`` is ignored. Freeing it in a fit would
        sample a number nothing reads; see ``DERIVED_PARAMS``.
        """

        if self._omega_de_spline is None:
            self._solve_background()

        return 1.0 - self._omega_de_today

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:
        """
        Dark-energy density in units of today's critical density --
        the term inside ``E(z)^2``, as every other model here
        returns.
        """

        return self.omega_de_fraction(z) * self.E(z) ** 2

    # ---------------------------------------------------------

    def w_de(self, z: Redshift) -> Array:
        r"""
        ``w = -1 + (2 / 3n) sqrt(Omega_DE) / a``.

        Never below ``-1``: the model cannot be phantom at any
        ``n``, which is what most sharply separates it from
        :class:`~cosmology.models.hde.HDE`.
        """

        z = np.asarray(z, dtype=float)

        return (

            -1.0

            + (2.0 / (3.0 * float(self.params.n_ade)))

            * np.sqrt(self.omega_de_fraction(z))

            * (1.0 + z)

        )

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_m * cube(1.0 + z)

            / (1.0 - self.omega_de_fraction(z))

        )

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:
        """
        Analytic, from the ODE rather than by differencing ``E``.
        """

        z = np.asarray(z, dtype=float)

        omega = self.omega_de_fraction(z)

        n = float(self.params.n_ade)

        bracket = 3.0 - omega * (

            3.0 - (2.0 / n) * np.sqrt(omega) * (1.0 + z)

        )

        return (

            self.Omega_m

            * (1.0 + z) ** 2

            * bracket

            / (2.0 * self.E(z) * (1.0 - omega))

        )
