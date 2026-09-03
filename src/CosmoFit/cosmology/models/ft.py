"""
f(T) modified gravity (metric teleparallel, torsion scalar).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core.utils import coupling_from_derivative
from CosmoFit.typing import Array, Redshift

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology
from CosmoFit.cosmology.core.errors import ModelConfigurationError


class FTPowerLaw(Cosmology):
    r"""
    f(T) gravity, Bengochea-Ferraro power-law model.

    f(T) modified gravity is the torsion counterpart of
    :class:`~cosmology.models.fq.FQExponential`. Where f(Q) builds
    gravity from non-metricity with a flat, torsion-free
    connection, f(T) builds it from torsion with a flat,
    metric-compatible one; both reduce to general relativity when
    f is the identity, and both differ from it otherwise. This is a
    genuine modification of the field equations, not a dark-energy
    fluid added on top of GR.

    The model implemented here is the power law of Bengochea &
    Ferraro (2009), the standard benchmark of the f(T) literature:

        f(T) = T + alpha * T^n,     T = 6 H^2

    **Conventions, and why they do not matter here.** Half the f(T)
    literature writes the flat-FLRW torsion scalar as ``T = +6H^2``
    and half as ``T = -6H^2`` (with the power law then written
    ``f = T + alpha (-T)^n``, so that the power is taken of a
    positive number). :mod:`CosmoFit.theory` uses the negative
    convention; this class was derived in the positive one. Both
    were worked through, and **they give the same E(z) and the same
    mu(a)** -- the sign difference is absorbed by the ``E(0) = 1``
    closure, and the two are checked against each other rather than
    assumed equal. Either way general relativity sits at
    ``f(T) = T``.

    **The Friedmann equation.** Varying the lapse in the
    minisuperspace tetrad action gives ``kappa*rho = (2 T f_T - f)/2``,
    the sign fixed by demanding that ``f(T) = T`` return
    ``3H^2 = kappa*rho``. For the power law, writing
    ``A = alpha (6 H0^2)^(n-1)`` for the dimensionless amplitude,
    this collapses to

        E^2 + (2n - 1) * A * E^(2n) = Omega_m * (1+z)^3

    which is transcendental in ``E^2`` for general ``n`` and is
    solved by vectorized Newton iteration, exactly as
    :class:`~cosmology.models.fq.FQExponential` does. ``dEdz`` comes
    from implicit differentiation of the same relation, closed-form
    given ``E(z)``.

    **``A`` is not a free parameter.** Requiring ``E(0) = 1`` closes
    the model:

        A = (Omega_m - 1) / (2n - 1)

    so ``n`` is the only quantity beyond flat LCDM's ``H0`` and
    ``Omega_m`` -- the same "one extra number" structure as this
    library's f(Q). Substituting the closure back collapses the
    Friedmann relation to a form with no free amplitude in it at
    all,

        E^2 + (Omega_m - 1) * E^(2n) = Omega_m * (1+z)^3

    which is what ``E`` and ``dEdz`` actually solve. Writing it this
    way matters numerically as well as tidily: ``A`` itself diverges
    at ``n = 1/2``, but the combination ``(2n-1) A`` that the
    background depends on is just ``Omega_m - 1``, finite
    everywhere. The background therefore has **no** pole at
    ``n = 1/2``; only ``mu`` does, and only ``mu`` guards against
    it.

    Two limits follow immediately and are both covered by the test
    suite:

    * ``n = 0`` gives ``E^2 = Omega_m (1+z)^3 + (1 - Omega_m)``,
      i.e. **flat LCDM exactly**, since ``f = T + alpha`` is TEGR
      plus a cosmological constant.
    * ``n = 1`` gives ``E^2 = (1+z)^3``, Einstein-de Sitter, since
      ``f = (1 + alpha) T`` is a rescaled TEGR with no constant
      term and the rescaling cancels out of the Friedmann equation.

    ``n = 1/2`` is a genuine singularity of the *growth* sector
    rather than a numerical one. Holding the background fixed there
    forces ``alpha`` to diverge, which sends ``f_T`` to infinity and
    ``mu`` to zero: gravity switches off. ``E(z)`` is perfectly well
    defined at that point and is returned; ``mu`` raises instead of
    handing back a number that came from cancelling infinities. The
    default bounds stay on the branch ``n < 1/2``, which contains
    the LCDM limit.

    On the branch ``n > 1/2`` the Friedmann relation can have **no
    real solution** past some redshift, because its left-hand side
    turns over: at ``n = 2`` and ``Omega_m = 0.315`` there is no
    ``E`` above ``z = 0.05``. ``E(z)`` returns NaN there rather than
    the value an unconverged Newton iteration happens to land on.
    This is not a corner worth reaching for -- the default bounds
    keep well away from it -- but it is worth stating, because the
    unguarded version returned finite, plausible, non-monotonic
    numbers instead.

    ``Omega_de(z)`` is reported, as for f(Q), as an
    *effective/geometric* dark-energy density
    ``E(z)^2 - Omega_m (1+z)^3``. There is no second fluid; the
    acceleration is entirely from the modified gravitational
    sector. Flat only -- ``Omega_k`` is ignored, since ``T = 6H^2``
    is itself a flat-FLRW identity. Radiation is dropped, as in
    every other model here.

    **Growth of structure.** ``mu(a, k)`` is the sub-horizon,
    quasi-static effective gravitational coupling
    ``G_eff/G_N = 1/f_T``, which for this model is

        mu = 1 / (1 + n * A * E^(2n-2))

    scale-independent, so ``k`` is accepted for interface
    consistency with :meth:`Cosmology.mu` and ignored -- the same
    situation as f(Q), and unlike f(R), whose chameleon screening
    makes its ``mu`` genuinely k-dependent. ``n = 0`` gives
    ``mu = 1`` identically, so the LCDM limit above is a limit of
    the growth history too, not only of the background.

    **Provenance.** The Friedmann equation, the closure condition,
    ``dEdz`` and ``mu`` above were derived symbolically in the
    wljs-gr-toolkit GR-06 notebook rather than transcribed, and
    every step was checked against a limit where the answer is
    already known: that varying the lapse returns
    ``a^3 (2 T f_T - f)``, that ``f(T) = T`` gives ``3H^2``, that
    ``n = 0`` reproduces flat LCDM exactly, that ``n = 1`` gives
    Einstein-de Sitter, that ``mu = 1`` at both ``n = 0`` and
    ``A = 0``, and that the ``dEdz`` expression agrees with implicit
    differentiation of the Friedmann relation. The tests in
    ``tests/test_ft.py`` re-check the LCDM and TEGR limits on the
    Python side.

    **One caveat worth carrying.** The same toolkit's GR-08
    notebook finds that f(T)'s extra Lorentz modes have kinetic
    terms that vanish identically around flat FLRW: they do not
    propagate at linear order there, which is strong coupling, and
    means linear perturbation theory is not a reliable guide to
    them. The ``mu`` above is a statement about the metric sector,
    which is well behaved; it is not a claim that the full theory
    is healthy around this background. Gravitational waves in f(T)
    are exactly luminal, ``c_GW^2 = 1``, so GW170817 does not
    constrain it.

    References
    ----------
    Bengochea & Ferraro (2009), "Dark torsion as the cosmic
    speed-up", Phys. Rev. D 79, 124019, arXiv:0812.1205.

    Linder (2010), "Einstein's Other Gravity and the Acceleration
    of the Universe", Phys. Rev. D 81, 127301, arXiv:1005.3039.

    Nesseris, Basilakos, Saridakis & Perivolaropoulos (2013),
    "Viable f(T) models are practically indistinguishable from
    LCDM", Phys. Rev. D 88, 103010, arXiv:1308.6142.
    """

    MODEL_NAME = "FTPowerLaw"
    MODEL_LABEL = r"$f(T)$ power law"

    #: Newton iterations for solving E^2. The relation is smooth
    #: and monotonic in E^2 over the physical range on the branch
    #: this model lives on, so this converges long before the
    #: budget is used; it is vectorized over the whole z-grid and
    #: run once per likelihood call, so the cost is negligible and
    #: correctness matters more than saving iterations.
    _NEWTON_ITERATIONS = 30

    #: How close ``n`` may come to the 1/2 pole before ``mu``
    #: refuses to answer. The background is unaffected -- see the
    #: class docstring.
    _POLE_MARGIN = 1e-3

    #: How well the Newton result must satisfy the Friedmann
    #: relation, relative to the right-hand side, before it is
    #: believed. Newton converges to machine precision where a root
    #: exists, so this only ever rejects the case where none does.
    _RESIDUAL_TOLERANCE = 1e-10

    EXTRA_PARAMS = {
        "n": {
            "default": 0.0, "bounds": (-3.0, 0.45), "label": r"$n$",
        },
    }

    # ---------------------------------------------------------

    @property
    def _c(self) -> float:
        """
        The Friedmann coefficient ``(2n-1) A``, which the ``E(0)=1``
        closure fixes to ``Omega_m - 1`` for every ``n``. Finite
        everywhere, including at the ``n = 1/2`` pole of ``A``
        itself.
        """

        return self.Omega_m - 1.0

    # ---------------------------------------------------------

    @property
    def _A(self) -> float:
        """
        Dimensionless amplitude ``A = alpha (6 H0^2)^(n-1)``, fixed
        by the ``E(0) = 1`` closure rather than free:
        ``A = (Omega_m - 1) / (2n - 1)``.

        Needed only by ``mu``; the background goes through ``_c``,
        which has no pole.
        """

        two_n_minus_1 = 2.0 * self.n - 1.0

        if abs(two_n_minus_1) < self._POLE_MARGIN:
            raise ModelConfigurationError(
                f"FTPowerLaw: n = {self.n!r} sits on the n = 1/2 pole "
                "of the effective gravitational coupling. Holding the "
                "background fixed there forces alpha to diverge, which "
                "sends f_T to infinity and mu to zero. E(z) is well "
                "defined at this n and is still returned; mu is not. "
                "Choose n away from 1/2 (the default bounds keep to "
                "the n < 1/2 branch, which contains the LCDM limit at "
                "n = 0)."
            )

        return (self.Omega_m - 1.0) / two_n_minus_1

    # ---------------------------------------------------------

    def _solve_E2(self, z):
        """
        Vectorized Newton solve of
        ``E^2 + (2n-1) A E^(2n) = Omega_m (1+z)^3`` for ``E^2``, at
        every z simultaneously.

        The root is **verified**, not assumed. For ``n > 1/2`` the
        left-hand side is not monotonic in ``E^2``: with
        ``Omega_m - 1 < 0`` it turns over at
        ``E^2 = [n(1-Omega_m)]^(1/(1-n))``, so beyond some redshift
        the equation has no real solution at all -- at ``n = 2`` and
        ``Omega_m = 0.315`` that happens above ``z = 0.05``. Newton
        does not report this. It wanders off and returns whatever it
        lands on, and those values look entirely plausible: ``E``
        came back finite and even non-monotonic in ``z`` before this
        check existed, which is a wrong answer of the worst kind,
        the kind nothing downstream can detect.

        So the iteration is kept on the physical branch, and the
        result is only returned if it actually satisfies the
        equation. Anything else is NaN, which is what the rest of
        this library uses for a model evaluated where it has no
        solution.
        """

        z = np.asarray(z, dtype=float)
        n = float(self.n)
        c = self._c

        rhs = self.Omega_m * cube(1.0 + z)

        # LCDM starting point: exact when n = 0, and a good guess
        # otherwise, since this model is LCDM-adjacent by
        # construction.
        x = rhs + (1.0 - self.Omega_m)

        with np.errstate(invalid="ignore", divide="ignore"):

            for _ in range(self._NEWTON_ITERATIONS):

                xn = x ** n
                g = x + c * xn - rhs
                dg = 1.0 + n * c * xn / x

                x = x - g / dg

                # E^2 <= 0 is off the branch the expansion history
                # lives on; once there, Newton has already lost the
                # root and anything it returns is noise.
                x = np.where(x > 0.0, x, np.nan)

            # Did it actually solve the equation? Scale the residual
            # by the right-hand side, which is the size of the terms
            # being cancelled.
            residual = np.abs(x + c * x ** n - rhs)

            x = np.where(
                residual <= self._RESIDUAL_TOLERANCE * np.maximum(rhs, 1.0),
                x,
                np.nan,
            )

        return x

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:
        """
        Dimensionless Hubble parameter (solves the transcendental
        Friedmann equation -- see the class docstring).
        """

        return np.sqrt(self._solve_E2(z))

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:
        """
        Derivative of E(z), by implicit differentiation of the
        Friedmann relation (closed-form given E(z), no extra
        root-finding).
        """

        z = np.asarray(z, dtype=float)
        n = float(self.n)
        c = self._c

        x = self._solve_E2(z)

        dg_dx = 1.0 + n * c * x ** n / x
        d_rhs_dz = 3.0 * self.Omega_m * (1.0 + z) ** 2

        dx_dz = d_rhs_dz / dg_dx

        return dx_dz / (2.0 * np.sqrt(x))

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:
        """
        Effective/geometric dark-energy density -- see the class
        docstring for why this isn't a real second fluid here.
        """

        z = np.asarray(z, dtype=float)

        return self.E(z) ** 2 - self.Omega_m * cube(1.0 + z)

    # ---------------------------------------------------------

    def mu(self, a: Redshift, k: float | None = None) -> Array:
        """
        Effective gravitational coupling
        ``G_eff/G_N = 1/f_T = 1/(1 + n A E^(2n-2))`` --
        scale-independent, so ``k`` is ignored.
        """

        a = np.asarray(a, dtype=float)
        z = 1.0 / a - 1.0

        n = float(self.n)
        A = self._A

        x = self._solve_E2(z)

        f_T = 1.0 + n * A * x ** (n - 1.0)

        return coupling_from_derivative(f_T, model=type(self).__name__)
