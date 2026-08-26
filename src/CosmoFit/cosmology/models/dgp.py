"""
DGP braneworld gravity (self-accelerating branch).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class DGP(Cosmology):
    r"""
    Dvali-Gabadadze-Porrati braneworld gravity, self-accelerating
    branch.

        E(z) = sqrt(Omega_rc + Omega_m (1+z)^3) + sqrt(Omega_rc)
        E(z)^2 -> that, plus Omega_k (1+z)^2 for a curved brane

    Gravity leaks into a fifth dimension above a crossover scale
    ``r_c``, weakening it at large distances. The universe then
    accelerates **with no dark energy at all** -- there is no
    ``Omega_de`` term in the Friedmann equation above, and none in
    the theory. That is what distinguishes DGP from every
    dark-energy parametrization in this library: those add a fluid,
    this changes gravity.

    ``Omega_rc = 1/(4 r_c^2 H0^2)`` is not free. The Friedmann
    constraint ``E(0) = 1`` fixes it:

        Omega_rc = (1 - Omega_k - Omega_m)^2 / (4 (1 - Omega_k))

    so DGP, like :class:`~cosmology.models.pede.PEDE`, has exactly
    LCDM's parameter count and a completely different expansion
    history -- a fair comparison with no free parameter absorbing
    the difference.

    Growth of structure
    -------------------
    This model overrides ``mu(a, k)`` with the standard
    sub-horizon DGP result (Lue, Scoccimarro & Starkman 2004;
    Koyama & Maartens 2006):

        mu = 1 + 1 / (3 beta),
        beta(a) = 1 - 2 H r_c [1 + Hdot / (3 H^2)]

    On the self-accelerating branch ``beta < 0``, so ``mu < 1`` --
    gravity is *weaker* than in GR and structure grows more slowly.
    Today, for ``Omega_m = 0.3``, ``mu ~ 0.72``. This is the
    model's real observational signature and the reason it is
    strongly disfavoured by growth data even where it can be tuned
    to fit distances: no dark-energy parametrization can suppress
    growth this way while keeping the same background.

    ``Hdot / H^2 = -(1+z) E'(z) / E(z)``, built from this model's
    own ``E``/``dEdz`` rather than from an LCDM approximation.

    Caveats
    -------
    The self-accelerating branch is known to carry a ghost
    instability in its scalar sector. It is implemented here as the
    historically and observationally important benchmark it is --
    the reference case against which "modified gravity can mimic
    dark energy" is tested -- not as a viable theory. Its
    perturbations are also outside what a standard Boltzmann code
    solves, so it is refused by
    :class:`~cosmology.boltzmann.CAMBBackend`; use the compressed
    distance priors for its CMB constraint.

    References
    ----------
    Dvali, Gabadadze & Porrati (2000), Phys. Lett. B 485, 208,
    arXiv:hep-th/0005016.

    Deffayet (2001), Phys. Lett. B 502, 199, arXiv:hep-th/0010186
    (the cosmological solution).

    Koyama & Maartens (2006), JCAP 01 (2006) 016,
    arXiv:astro-ph/0511634 (the growth of structure).
    """

    MODEL_NAME = "DGP"

    # ---------------------------------------------------------

    @property
    def Omega_rc(self) -> float:
        r"""
        The crossover-scale density parameter,
        ``Omega_rc = 1/(4 r_c^2 H0^2)``, fixed by ``E(0) = 1``.

        Solving ``sqrt(Omega_rc + Omega_m) + sqrt(Omega_rc) =
        sqrt(1 - Omega_k)`` gives

            Omega_rc = (1 - Omega_k - Omega_m)^2
                       / (4 (1 - Omega_k))

        which reduces to ``(1 - Omega_m)^2 / 4`` for a flat brane.
        """

        one_minus_k = 1.0 - self.Omega_k

        return (

            (one_minus_k - self.Omega_m) ** 2

            / (4.0 * one_minus_k)

        )

    # ---------------------------------------------------------

    def _root(self, z):
        """
        ``sqrt(Omega_rc + Omega_m (1+z)^3)``, the piece both
        ``E`` and ``dEdz`` are built from.
        """

        z = np.asarray(z, dtype=float)

        return np.sqrt(

            self.Omega_rc

            + self.Omega_m * (1.0 + z) ** 3

        )

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        brane = (

            self._root(z)

            + np.sqrt(self.Omega_rc)

        ) ** 2

        return np.sqrt(

            brane

            + self.Omega_k * (1.0 + z) ** 2

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        root = self._root(z)

        # d(E^2)/dz for E^2 = [root + sqrt(Omega_rc)]^2
        #                     + Omega_k (1+z)^2
        d_e2 = (

            2.0

            * (root + np.sqrt(self.Omega_rc))

            * (3.0 * self.Omega_m * (1.0 + z) ** 2)

            / (2.0 * root)

            +

            2.0 * self.Omega_k * (1.0 + z)

        )

        return d_e2 / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z):
        r"""
        The *effective* dark-energy density this model's expansion
        history would be attributed to, if one insisted on writing
        it as GR plus a fluid:

            Omega_de,eff(z) = E(z)^2
                              - Omega_m (1+z)^3
                              - Omega_k (1+z)^2

        There is no such fluid in DGP. This exists so the plotting
        and diagnostic machinery, which asks every model for an
        ``Omega_de(z)``, has a well-defined answer -- and because
        the effective density is itself the interesting thing to
        look at: it is what a dark-energy analysis would
        *mistakenly* reconstruct from DGP data.
        """

        z = np.asarray(z, dtype=float)

        return (

            self.E(z) ** 2

            - self.Omega_m * (1.0 + z) ** 3

            - self.Omega_k * (1.0 + z) ** 2

        )

    # ---------------------------------------------------------

    def mu(self, a, k=None):
        r"""
        Effective gravitational coupling ``G_eff/G_N`` on the
        self-accelerating branch,

            mu = 1 + 1/(3 beta),
            beta = 1 - 2 H r_c [1 + Hdot/(3 H^2)]

        with ``2 H r_c = E / sqrt(Omega_rc)`` (from
        ``r_c H0 = 1/(2 sqrt(Omega_rc))``) and
        ``Hdot/H^2 = -(1+z) E'/E``.

        Scale-independent, so ``k`` is accepted and ignored -- DGP
        modifies gravity above the crossover scale, not at a
        particular wavenumber.
        """

        a = np.asarray(a, dtype=float)

        z = 1.0 / a - 1.0

        e = self.E(z)

        # Hdot / H^2 in terms of the redshift derivative.
        hdot_over_h2 = -(1.0 + z) * self.dEdz(z) / e

        beta = 1.0 - (

            e / np.sqrt(self.Omega_rc)

        ) * (

            1.0 + hdot_over_h2 / 3.0

        )

        return 1.0 + 1.0 / (3.0 * beta)
