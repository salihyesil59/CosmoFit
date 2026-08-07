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

from scipy.integrate import solve_ivp


#: Scale factor at which the growth ODE's initial conditions are
#: set (z ~ 9999) -- deep enough in matter domination that every
#: model implemented here is GR-like there (chameleon screening for
#: f(R); mu(a) -> 1 as a -> 0 for the f(Q)/f(R,T) forms implemented
#: -- both approach the Omega_m(a) -> 1, GR-growing-mode limit), so
#: the standard matter-domination growing-mode initial condition
#: (D proportional to a) is valid regardless of which model this is
#: attached to.
_A_INIT = 1.0e-4


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
        self._sol = None
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

    def _rhs(self, N, y):

        a = np.exp(N)
        z = 1.0 / a - 1.0

        D, Dp = y

        Omega_m_a = self.cosmo.background.Omega_m(z)
        mu = self.cosmo.mu(a, k=self.k)

        Dpp = (
            -(2.0 + self._dlnH_dN(z)) * Dp
            + 1.5 * Omega_m_a * mu * D
        )

        return [Dp, Dpp]

    # --------------------------------------------------------

    def _solve(self) -> None:

        N_init = np.log(_A_INIT)

        sol = solve_ivp(
            self._rhs,
            (N_init, 0.0),
            y0=[_A_INIT, _A_INIT],
            dense_output=True,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(
                f"Growth ODE integration failed: {sol.message}"
            )

        self._sol = sol
        self._D0 = float(sol.sol(0.0)[0])
        self._dirty = False

    # --------------------------------------------------------

    def _ensure_built(self) -> None:

        if self._dirty or self._sol is None:
            self._solve()

    # --------------------------------------------------------

    def D(self, z):
        """
        Linear growth factor, normalized to D(z=0) = 1.
        """

        self._ensure_built()

        z = np.asarray(z, dtype=float)
        N = -np.log1p(z)

        y = self._sol.sol(N)

        return y[0] / self._D0

    # --------------------------------------------------------

    def growth_rate(self, z):
        """
        Linear growth rate f(z) = dlnD/dlna.
        """

        self._ensure_built()

        z = np.asarray(z, dtype=float)
        N = -np.log1p(z)

        y = self._sol.sol(N)

        return y[1] / y[0]

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
