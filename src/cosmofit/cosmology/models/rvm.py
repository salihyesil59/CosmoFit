"""
Running Vacuum Model (RVM).
"""

from __future__ import annotations

import numpy as np

from cosmofit.typing import Array, Redshift

from cosmofit.cosmology.core import Cosmology


def _expm1_ratio(x):
    """
    ``expm1(x) / x``, equal to 1 at ``x = 0`` and accurate near it --
    where the ratio of two vanishing quantities would otherwise lose
    every digit.
    """

    x = np.asarray(x, dtype=float)

    small = np.abs(x) < 1.0e-8

    safe = np.where(small, 1.0, x)

    return np.where(small, 1.0 + 0.5 * x, np.expm1(safe) / safe)



class RunningVacuum(Cosmology):
    r"""
    Running Vacuum Model: a cosmological "constant" that runs with
    the expansion rate,

        Lambda(H) = c0 + 3 nu H^2

    Solving the Friedmann and continuity equations together with
    this gives a closed form with no numerical integration:

        E(z)^2 = 1 + Omega_k [(1+z)^2 - 1]
                 + (Omega_m / (1 - nu)) [(1+z)^{3(1-nu)} - 1]

    and ``nu = 0`` recovers curved LCDM exactly.

    The idea comes from quantum field theory in curved spacetime,
    where the vacuum energy density is a running quantity obeying a
    renormalization-group equation, and ``nu`` is the beta-function
    coefficient -- expected to be ``|nu| ~ 10^-3`` from a one-loop
    estimate, which is why the default bounds here are tight
    compared to the wide-open priors on ``w0``/``wa``. It is one of
    the few dark-energy models whose extra parameter has a
    *predicted magnitude* rather than an arbitrary one, so a fit
    that returns ``nu ~ 10^-3`` means something quite different
    from one that returns ``nu ~ 0.1``.

    Mechanically the model works by making matter dilute slightly
    differently from ``(1+z)^3`` -- the exponent is
    ``3(1 - nu)`` -- because vacuum and matter exchange energy.
    That is a different lever from any ``w(z)`` parametrization,
    which leaves the matter scaling alone, and it means ``nu``
    is constrained by anything sensitive to the matter density's
    redshift evolution, growth data included.

    Notes
    -----
    ``nu = 1`` is a coordinate singularity of the closed form
    above (the ``1/(1 - nu)`` prefactor), far outside any physical
    prior; :meth:`E` falls back to the ``nu -> 1`` limit there
    rather than dividing by zero.

    This is the simplest member of the RVM family. Fuller versions
    add an ``Hdot`` term (``Lambda = c0 + 3 nu H^2 + alpha Hdot``)
    or higher powers of ``H`` relevant to inflation; neither is
    implemented.

    References
    ----------
    Sola (2013), J. Phys. Conf. Ser. 453, 012015, arXiv:1306.1527
    (review).

    Sola, Gomez-Valent & de Cruz Perez (2017), ApJ 836, 43,
    arXiv:1602.02103 (cosmological constraints).
    """

    MODEL_NAME = "RunningVacuum"
    MODEL_LABEL = r"Running vacuum"

    EXTRA_PARAMS = {

        "nu": {
            "default": 0.0,
            "bounds": (-0.1, 0.1),
            "label": r"$\nu$",
        },

    }

    # ---------------------------------------------------------

    @property
    def _K(self) -> float:
        r"""
        The curvature coefficient, ``Omega_k / (1 - 3 nu)``.

        With ``Lambda = c0 + 3 nu H^2`` the Friedmann and continuity
        equations combine into ``d(H^2)/dN + 3(1-nu) H^2 = 3 c0 - k/a^2``,
        whose curvature solution is ``-k / ((1 - 3 nu) a^2)``: the
        running vacuum responds to the curvature term in ``H^2`` as it
        does to everything else in it. The closed form used to carry
        plain ``Omega_k (1+z)^2``, an ``O(nu Omega_k)`` error.
        """

        one_minus_3nu = 1.0 - 3.0 * self.nu

        if self.Omega_k != 0.0 and abs(one_minus_3nu) < 1.0e-8:

            raise ValueError(
                "RunningVacuum: nu = 1/3 is a resonance of the curved "
                "solution; it is far outside the model's physical "
                "range."
            )

        return self.Omega_k / one_minus_3nu if self.Omega_k else 0.0

    @property
    def _B(self) -> float:
        """
        Amplitude of the ``(1+z)^{3(1-nu)}`` mode, times ``1 - nu``:
        ``Omega_m - 2 nu K``, so that ``Omega_matter(0) = Omega_m``.
        """

        return self.Omega_m - 2.0 * self.nu * self._K

    # ---------------------------------------------------------

    def _matter_term(self, z):
        r"""
        ``(B / (1 - nu)) [(1+z)^{3(1-nu)} - 1]``, written through
        ``expm1`` so it is finite and accurate at ``nu -> 1``, where
        it tends to ``3 B ln(1+z)``.
        """

        z = np.asarray(z, dtype=float)

        L = np.log1p(z)

        return self._B * 3.0 * L * _expm1_ratio(3.0 * (1.0 - self.nu) * L)

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            1.0

            + self._K * ((1.0 + z) ** 2 - 1.0)

            + self._matter_term(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:

        z = np.asarray(z, dtype=float)

        # The 1/(1-nu) of the matter term and its 3(1-nu) exponent
        # cancel exactly: 3 B (1+z)^{2-3nu}.
        d_matter = 3.0 * self._B * (1.0 + z) ** (2.0 - 3.0 * self.nu)

        return (

            (

                2.0 * self._K * (1.0 + z)

                + d_matter

            )

            /

            (2.0 * self.E(z))

        )

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:
        r"""
        The running vacuum density,

            Omega_Lambda(z) = E(z)^2 - Omega_m(z) - Omega_k(z)

        where the matter term is ``Omega_m (1+z)^{3(1-nu)}``,
        *not* ``Omega_m (1+z)^3`` -- that modified scaling is the
        whole content of the model, and using the LCDM one here
        would silently report the wrong split between the two
        components while leaving ``E(z)`` correct.
        """

        z = np.asarray(z, dtype=float)

        matter = self.Omega_matter(z)

        return (

            self.E(z) ** 2

            - matter

            - self.Omega_k * (1.0 + z) ** 2

        )

    # ---------------------------------------------------------

    def matter_exchange(self, z: Redshift) -> Array:
        """
        ``Q / (H rho_m)``, from ``rho_m' + 3 rho_m = Q / H`` applied to
        :meth:`Omega_matter`: ``3 nu`` for a flat universe, and
        ``(3 nu M1 + M2) / (M1 + M2)`` with curvature, ``M1`` and
        ``M2`` the two terms of the matter density.
        """

        z = np.asarray(z, dtype=float)

        m1, m2 = self._matter_parts(z)

        return (3.0 * self.nu * m1 + m2) / (m1 + m2)

    # ---------------------------------------------------------

    def _matter_parts(self, z):
        """
        The two pieces of the matter density: the running mode
        ``B (1+z)^{3(1-nu)}`` and the curvature-fed
        ``2 nu K (1+z)^2``.
        """

        z = np.asarray(z, dtype=float)

        m1 = self._B * (1.0 + z) ** (3.0 * (1.0 - self.nu))

        m2 = 2.0 * self.nu * self._K * (1.0 + z) ** 2

        return m1, m2

    # ---------------------------------------------------------

    def Omega_matter(self, z: Redshift) -> Array:
        r"""
        ``B (1+z)^{3(1-nu)} + 2 nu K (1+z)^2`` -- matter dilutes more
        slowly than in LCDM because the running vacuum is feeding it,
        and with curvature the vacuum's response to the ``k/a^2`` term
        feeds it too. ``Omega_m (1+z)^{3(1-nu)}`` when flat.

        Overriding this is what makes ``nu`` visible to the linear
        growth equation; without it the growth source term would use
        LCDM's matter scaling while ``E(z)`` used the RVM one.
        """

        m1, m2 = self._matter_parts(z)

        return m1 + m2
