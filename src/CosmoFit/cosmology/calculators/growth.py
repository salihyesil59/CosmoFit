"""
Linear growth of structure.

Solves the standard sub-horizon, quasi-static linear growth
equation for the matter density contrast, generalized to modified
gravity through a single ``mu(a, k)`` hook (the ratio of the
effective to the Newtonian gravitational coupling, ``G_eff/G_N``;
``mu = 1`` recovers standard GR growth):

    d^2 D / dN^2 + (2 + dlnH/dN) dD/dN - (3/2) Omega_m(a) mu(a,k) D = 0

where ``N = ln a`` and ``D`` is the linear growth factor (normalized
``D(a=1) = 1``). This is the equation every textbook derivation of
LCDM structure growth starts from (e.g. Dodelson & Schmidt,
*Modern Cosmology*, 2nd ed., Ch. 7) with ``mu`` inserted in the
source term exactly as in the "designer"/effective-field-theory
modified-gravity growth literature (e.g. Pogosian & Silvestri 2008,
arXiv:0709.0296) -- the same generic mechanism every ``Cosmology``
subclass in this library already uses for ``E(z)``/``dEdz``: the
base class provides the default (``mu = 1``, GR), and a model
overrides it only if it actually modifies gravity.

``Omega_m(a)`` and ``dlnH/dN`` are built directly from the
cosmology's own ``E(z)``/``dEdz`` -- no new physics input beyond
``mu`` is needed, so this applies unchanged to every model in the
library (LCDM, wCDM, CPL, JBP, BA, GCG all inherit ``mu = 1``; the
three modified-gravity models override it).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.hermite import hermite_spline


#: Scale factor at which the growth ODE's initial conditions are
#: set (z ~ 9999) -- deep enough in matter domination that every
#: model implemented here is GR-like there (chameleon screening for
#: f(R); mu(a) -> 1 as a -> 0 for the f(Q)/f(R,T) forms implemented
#: -- both approach the Omega_m(a) -> 1, GR-growing-mode limit), so
#: the standard matter-domination growing-mode initial condition
#: (D proportional to a) is valid regardless of which model this is
#: attached to.
_A_INIT = 1.0e-4

#: Number of fixed RK4 steps from ``_A_INIT`` to today.
#:
#: The equation is linear, smooth and non-stiff, so an adaptive
#: solver spends its effort discovering a step size that never
#: needed to change. 300 steps agree with ``solve_ivp`` at
#: ``rtol=1e-8`` to 1.2e-8 in ``D(z)/D(0)`` and 7.0e-8 in ``f(z)``
#: across the redshifts growth data covers -- better than that
#: solver's own error, and roughly six orders of magnitude finer
#: than any RSD measurement.
_N_STEPS = 300


class GrowthCalculator:
    """
    Fast evaluator for the linear growth factor D(z), growth rate
    f(z) = dlnD/dlna, and fsigma8(z), given a :class:`Cosmology`.

    Built lazily: the ODE solve only happens the first time D/f/
    sigma8/fsigma8 is actually requested after construction or
    after :meth:`rebuild` -- most fits never touch growth-of-
    structure data, so this avoids paying for an ODE solve on every
    single MCMC step the way the (always-needed) ``DistanceIntegrator``
    does.

    Parameters
    ----------
    cosmology : Cosmology

    k : float, optional
        Wavenumber [h/Mpc], forwarded to ``cosmology.mu(a, k)``.
        Irrelevant for every scale-independent ``mu`` (the default,
        and every modified-gravity model implemented here except
        ``FRHuSawicki``), which ignores it. Default 0.1 h/Mpc, a
        representative galaxy-survey RSD scale.
    """

    def __init__(self, cosmology, k: float = 0.1):

        self.cosmo = cosmology
        self.k = float(k)

        self._dirty = True
        self._D_spline = None
        self._P_spline = None
        self._D0 = None

    # --------------------------------------------------------

    def rebuild(self) -> None:
        """
        Invalidate the cached growth solution -- call whenever the
        cosmology's parameters change. The actual ODE solve is
        deferred to the next D/f/sigma8/fsigma8 call (see the class
        docstring).
        """

        self._dirty = True

    # --------------------------------------------------------

    def _dlnH_dN(self, z):
        """
        dlnH/dN = dlnE/dlna = -(1+z) dE/dz / E(z).
        """

        E = self.cosmo.E(z)

        return -(1.0 + z) * self.cosmo.dEdz(z) / E

    # --------------------------------------------------------

    def _coefficients(self, N):
        """
        The ODE's two coefficients on a grid of ``N = ln a``,
        written as ``D'' = -friction D' + source D``.

        Evaluated for the whole grid in one pass. That is the
        point of solving on a fixed grid at all: an adaptive
        solver calls back into Python for every stage of every
        step, and each of those calls used to evaluate ``E(z)``
        three times -- once directly, once inside ``dEdz``, and
        once inside ``Omega_m``. Nineteen thousand scalar
        evaluations per growth solve became four array ones.
        """

        a = np.exp(N)

        z = 1.0 / a - 1.0

        friction = 2.0 + self._dlnH_dN(z)

        source = 1.5 * self.cosmo.background.Omega_m(z) * self.cosmo.mu(

            a,

            k=self.k,

        )

        return friction, source

    # --------------------------------------------------------

    def _solve(self) -> None:
        """
        Fixed-step RK4 from ``_A_INIT`` to today, with the
        coefficients precomputed.

        The growth equation is linear, smooth and non-stiff over
        the whole range, so adaptivity buys nothing: the step size
        an adaptive solver settles on is essentially constant, and
        discovering it costs several evaluations per step. See
        :data:`_N_STEPS` for what the fixed grid is checked
        against.

        Both ``D`` and ``dD/dN`` are known at every node, so the
        interpolant is a cubic Hermite spline rather than a plain
        cubic -- which matches the solver's own fourth-order
        accuracy instead of throwing most of it away between
        nodes.
        """

        N_init = np.log(_A_INIT)

        n = _N_STEPS

        h = -N_init / n

        # Nodes *and* midpoints: RK4 needs the coefficients at
        # both, and asking for them together is one vectorized
        # pass instead of two.
        fine = N_init + 0.5 * h * np.arange(2 * n + 1)

        friction, source = self._coefficients(fine)

        f0, s0 = friction[0:-1:2], source[0:-1:2]
        f1, s1 = friction[1::2], source[1::2]
        f2, s2 = friction[2::2], source[2::2]

        D = np.empty(n + 1)
        P = np.empty(n + 1)

        D[0] = P[0] = _A_INIT

        d, p = _A_INIT, _A_INIT

        for i in range(n):

            k1d = p
            k1p = -f0[i] * p + s0[i] * d

            k2d = p + 0.5 * h * k1p
            k2p = -f1[i] * k2d + s1[i] * (d + 0.5 * h * k1d)

            k3d = p + 0.5 * h * k2p
            k3p = -f1[i] * k3d + s1[i] * (d + 0.5 * h * k2d)

            k4d = p + h * k3p
            k4p = -f2[i] * k4d + s2[i] * (d + h * k3d)

            d += h / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
            p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

            D[i + 1] = d
            P[i + 1] = p

        nodes = fine[::2]

        # D'' from the equation itself, so the growth-rate spline is
        # Hermite too rather than falling back to a plain cubic.
        second = -friction[::2] * P + source[::2] * D

        self._D_spline = hermite_spline(nodes, D, P)
        self._P_spline = hermite_spline(nodes, P, second)

        self._D0 = float(D[-1])

        self._dirty = False

    def _ensure_built(self) -> None:

        if self._dirty or self._D_spline is None:
            self._solve()

    # --------------------------------------------------------

    def D(self, z):
        """
        Linear growth factor, normalized to D(z=0) = 1.
        """

        self._ensure_built()

        z = np.asarray(z, dtype=float)
        N = -np.log1p(z)

        return self._D_spline(N) / self._D0

    # --------------------------------------------------------

    def growth_rate(self, z):
        """
        Linear growth rate f(z) = dlnD/dlna.
        """

        self._ensure_built()

        z = np.asarray(z, dtype=float)
        N = -np.log1p(z)

        return self._P_spline(N) / self._D_spline(N)

    # --------------------------------------------------------

    def sigma8(self, z):
        """
        sigma8(z) = sigma8_0 * D(z).
        """

        return self.cosmo.sigma8 * self.D(z)

    # --------------------------------------------------------

    def fsigma8(self, z):
        """
        f(z) * sigma8(z), the RSD growth-rate observable.
        """

        return self.growth_rate(z) * self.sigma8(z)
