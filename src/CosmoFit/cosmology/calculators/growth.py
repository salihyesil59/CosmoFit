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

from scipy.interpolate import PPoly

from CosmoFit.cosmology.numerics.hermite import hermite_spline
from CosmoFit.cosmology.numerics import kernels


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

        dlnH = self._dlnH_dN(z)

        friction = 2.0 + dlnH

        source = 1.5 * self.cosmo.background.Omega_m(z) * self.cosmo.mu(

            a,

            k=self.k,

        )

        exchange = getattr(self.cosmo, "matter_exchange", None)

        psi = None if exchange is None else exchange(z)

        if psi is not None:

            # Matter that gains energy from a non-clustering dark
            # sector. Perturbing rho_m' + 3 rho_m = Q/H with Q
            # homogeneous gives delta' + psi delta + theta/(aH) = 0;
            # with geodesic matter and Poisson, eliminating theta:
            #
            #   delta'' + (2 + dlnH/dN + psi) delta'
            #     - [3/2 Omega_m mu - psi (2 + dlnH/dN) - psi'] delta = 0
            #
            # (Gomez-Valent, Sola & Basilakos 2015, in N = ln a).
            # Omega_m(a) alone, which is all this solver used to
            # carry, leaves out the two psi terms.
            psi = np.asarray(psi, dtype=float)

            step = 1.0e-4

            dpsi = (
                np.asarray(exchange(np.exp(-(N + step)) - 1.0), dtype=float)
                - np.asarray(exchange(np.exp(-(N - step)) - 1.0), dtype=float)
            ) / (2.0 * step)

            friction = friction + psi

            source = source - psi * (2.0 + dlnH) - dpsi

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

        jumps = self._jumps(N_init)

        if jumps:
            self._solve_across(N_init, jumps)
            return

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

        if kernels.HAVE_NUMBA:

            # Sequential stepping, compiled. See
            # `cosmology.numerics.kernels` for why this is worth a
            # second implementation and what the two are checked
            # against each other on.
            D, P = kernels.rk4_growth(

                np.ascontiguousarray(f0),
                np.ascontiguousarray(s0),
                np.ascontiguousarray(f1),
                np.ascontiguousarray(s1),
                np.ascontiguousarray(f2),
                np.ascontiguousarray(s2),

                h,

                _A_INIT,

            )

        else:

            D, P = self._step_by_prefix_product(
                f0, s0, f1, s1, f2, s2, h, n,
            )

        nodes = fine[::2]

        # D'' from the equation itself, so the growth-rate spline is
        # Hermite too rather than falling back to a plain cubic.
        second = -friction[::2] * P + source[::2] * D

        self._D_spline = hermite_spline(nodes, D, P)
        self._P_spline = hermite_spline(nodes, P, second)

        self._D0 = float(D[-1])

        self._dirty = False

    # --------------------------------------------------------

    def _jumps(self, N_init) -> list[float]:
        """
        ``N = ln a`` of every jump in ``E(z)`` inside the integration
        range, in order of integration.
        """

        jumps = getattr(self.cosmo, "background_jumps", lambda: ())()

        nodes = sorted(
            -np.log1p(float(z)) for z in jumps
            if 0.0 < float(z) < 1.0 / _A_INIT - 1.0
        )

        return [N for N in nodes if N_init < N < 0.0]

    # --------------------------------------------------------

    def _solve_across(self, N_init, jumps) -> None:
        r"""
        The same RK4, in pieces, across jumps in ``E(z)``.

        Where ``E`` jumps, ``dlnH/dN`` carries a delta function that no
        grid can sample: a fixed grid simply steps over it, which keeps
        ``dD/dN`` continuous. That is the wrong matching condition.
        Writing the equation as

            (1/H) d(H D')/dN + 2 D' - (3/2) Omega_m mu D = 0

        shows what is continuous across a jump: ``H dD/dN`` -- i.e.
        ``dδ/dt``. A finite jump in ``H`` cannot change a velocity in
        zero time. So ``D`` carries straight across, and ``dD/dN``
        (and with it ``f``) is rescaled by ``H_before / H_after``.

        Each piece is integrated with the same fixed-step RK4 and
        Hermite-interpolated on its own; the pieces are joined into one
        piecewise polynomial whose breakpoint at the jump is a genuine
        discontinuity in ``f``, not a smoothed one.
        """

        edges = [N_init, *jumps, 0.0]

        total = -N_init

        D_coefficients, P_coefficients, breakpoints = [], [], []

        start = np.array([_A_INIT, _A_INIT])

        # Coefficients just inside each piece, so a piece ending at a
        # jump sees the value *before* it and the next piece the value
        # after. `E` itself picks one side exactly at the jump.
        inset = 1.0e-10

        for i, (N0, N1) in enumerate(zip(edges[:-1], edges[1:])):

            n = max(16, int(round(_N_STEPS * (N1 - N0) / total)))

            h = (N1 - N0) / n

            fine = N0 + 0.5 * h * np.arange(2 * n + 1)

            evaluate = fine.copy()

            if i > 0:
                evaluate[0] += inset

            if i < len(edges) - 2:
                evaluate[-1] -= inset

            friction, source = self._coefficients(evaluate)

            f0, s0 = friction[0:-1:2], source[0:-1:2]
            f1, s1 = friction[1::2], source[1::2]
            f2, s2 = friction[2::2], source[2::2]

            D, P = self._step_by_prefix_product(
                f0, s0, f1, s1, f2, s2, h, n, start=start,
            )

            nodes = fine[::2]

            second = -friction[::2] * P + source[::2] * D

            D_coefficients.append(hermite_spline(nodes, D, P).c)
            P_coefficients.append(hermite_spline(nodes, P, second).c)
            breakpoints.append(nodes if i == 0 else nodes[1:])

            if i < len(edges) - 2:

                # Just before (higher z) and at/after the jump.
                before = float(self.cosmo.E(np.expm1(-(N1 - inset))))
                after = float(self.cosmo.E(np.expm1(-(N1 + inset))))


                start = np.array([D[-1], P[-1] * before / after])

        x = np.concatenate(breakpoints)

        self._D_spline = PPoly.construct_fast(
            np.concatenate(D_coefficients, axis=1), x,
        )
        self._P_spline = PPoly.construct_fast(
            np.concatenate(P_coefficients, axis=1), x,
        )

        self._D0 = float(self._D_spline(0.0))
        self._dirty = False

    # --------------------------------------------------------

    @staticmethod
    def _step_by_prefix_product(f0, s0, f1, s1, f2, s2, h, n, start=None):
        """
        The same RK4, without a Python loop and without numba.

        The equation is *linear*, so one step is a fixed 2x2 matrix
        acting on ``(D, dD/dN)`` -- and that matrix depends only on
        the coefficients, not on the solution. Every step's matrix
        is therefore built at once, by pushing the 2x2 identity
        through the same RK4 formulas.

        Composing them is a prefix product, and a prefix product
        over an associative operator does not need a sequential
        loop: pairwise doubling gets there in ``log2(n)`` rounds of
        batched multiplies -- nine rounds of vectorized work
        instead of three hundred iterations of scalar Python, which
        is 493 microseconds down to 215.

        Matrix multiplication is associative but **not**
        commutative, so the operand order below is load-bearing:
        ``combined @ shifted`` keeps each product in step order.
        Checked against an adaptive solver at every node, not only
        at ``z = 0``.
        """

        eye = np.ones(n), np.zeros(n)

        basis_D = np.stack(eye, axis=-1)
        basis_P = np.stack(eye[::-1], axis=-1)

        def stage(d, p, friction_i, source_i):

            return p, (

                -friction_i[:, None] * p

                + source_i[:, None] * d

            )

        k1d, k1p = stage(basis_D, basis_P, f0, s0)
        k2d, k2p = stage(
            basis_D + 0.5 * h * k1d, basis_P + 0.5 * h * k1p, f1, s1,
        )
        k3d, k3p = stage(
            basis_D + 0.5 * h * k2d, basis_P + 0.5 * h * k2p, f1, s1,
        )
        k4d, k4p = stage(
            basis_D + h * k3d, basis_P + h * k3p, f2, s2,
        )

        step = np.empty((n, 2, 2))

        step[:, 0, :] = basis_D + h / 6.0 * (
            k1d + 2.0 * k2d + 2.0 * k3d + k4d
        )
        step[:, 1, :] = basis_P + h / 6.0 * (
            k1p + 2.0 * k2p + 2.0 * k3p + k4p
        )

        combined = step

        identity = np.eye(2)

        stride = 1

        while stride < n:

            shifted = np.empty_like(combined)

            shifted[:stride] = identity

            shifted[stride:] = combined[:-stride]

            combined = combined @ shifted

            stride *= 2

        if start is None:
            start = np.array([_A_INIT, _A_INIT])

        solution = np.empty((n + 1, 2))

        solution[0] = start
        solution[1:] = combined @ start

        return solution[:, 0], solution[:, 1]

    # --------------------------------------------------------

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
