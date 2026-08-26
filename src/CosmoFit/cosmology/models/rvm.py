"""
Running Vacuum Model (RVM).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


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

    def _matter_term(self, z):
        r"""
        ``(Omega_m / (1 - nu)) [(1+z)^{3(1-nu)} - 1]``.
        """

        z = np.asarray(z, dtype=float)

        nu = self.nu

        if np.isclose(nu, 1.0):

            # lim_{nu->1} [(1+z)^{3(1-nu)} - 1] / (1 - nu)
            #   = 3 ln(1+z)
            return 3.0 * self.Omega_m * np.log1p(z)

        return (

            self.Omega_m / (1.0 - nu)

        ) * (

            (1.0 + z) ** (3.0 * (1.0 - nu)) - 1.0

        )

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            1.0

            + self.Omega_k * ((1.0 + z) ** 2 - 1.0)

            + self._matter_term(z)

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        nu = self.nu

        if np.isclose(nu, 1.0):

            d_matter = 3.0 * self.Omega_m / (1.0 + z)

        else:

            # The 1/(1-nu) prefactor and the 3(1-nu) exponent
            # cancel exactly, which is why this is simply
            # 3 Omega_m (1+z)^{2-3nu}.
            d_matter = (

                3.0 * self.Omega_m

                * (1.0 + z) ** (2.0 - 3.0 * nu)

            )

        return (

            (

                2.0 * self.Omega_k * (1.0 + z)

                + d_matter

            )

            /

            (2.0 * self.E(z))

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):
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

        matter = self.Omega_m * (1.0 + z) ** (

            3.0 * (1.0 - self.nu)

        )

        return (

            self.E(z) ** 2

            - matter

            - self.Omega_k * (1.0 + z) ** 2

        )

    # ---------------------------------------------------------

    def Omega_matter(self, z):
        r"""
        ``Omega_m (1+z)^{3(1-nu)}`` -- matter dilutes more slowly
        than in LCDM because the running vacuum is feeding it.

        Overriding this is what makes ``nu`` visible to the linear
        growth equation; without it the growth source term would
        use LCDM's matter scaling while ``E(z)`` used the RVM one,
        which is internally inconsistent rather than merely
        approximate.
        """

        z = np.asarray(z, dtype=float)

        return self.Omega_m * (1.0 + z) ** (

            3.0 * (1.0 - self.nu)

        )
