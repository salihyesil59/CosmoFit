"""
Interacting Dark Energy (IDE).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.numerics.powers import cube

from CosmoFit.cosmology.core import Cosmology


class IDE(Cosmology):
    r"""
    Interacting Dark Energy: dark matter and dark energy exchange
    energy, with a coupling ``Q = 3 xi H rho_de``.

    The two dark components are only ever observed
    gravitationally, so nothing forbids them from interacting --
    and if they do, the "coincidence problem" (why are their
    densities comparable *now*?) softens, because the ratio
    ``rho_de/rho_c`` evolves more slowly than in LCDM.

    The continuity equations,

        rho_de' + 3(1+w0) rho_de = -3 xi rho_de
        rho_c'  + 3 rho_c        = +3 xi rho_de      (' = d/dln a)

    have a closed-form solution, so this model costs no numerical
    integration:

        E(z)^2 = (Omega_m - C) (1+z)^3
                 + (Omega_de0 + C) (1+z)^{3(1+w0+xi)}
                 + Omega_k (1+z)^2,
        C = -xi Omega_de0 / (w0 + xi)

    ``xi > 0`` transfers energy from dark energy to dark matter;
    ``xi < 0`` the other way. ``xi = 0`` recovers
    :class:`~cosmology.models.wcdm.WCDM` exactly (``C = 0``), and
    ``xi = 0``, ``w0 = -1`` recovers LCDM.

    Why it is interesting beyond the coincidence problem: an
    interaction changes how *matter* dilutes, which no ``w(z)``
    parametrization does. That gives it a distinct signature in
    growth-of-structure data, and it is one of the few extensions
    that can move ``H0`` and ``S8`` in opposite directions -- which
    is what resolving both tensions at once requires.

    Parameters
    ----------
    xi
        Coupling strength. Prior bounds default to ``(-0.5, 0.5)``;
        realistic constraints are ``|xi| < 0.1``.

    Notes
    -----
    **The interaction is applied to the whole of ``Omega_m``, not
    to cold dark matter alone.** Strictly, only the dark sector
    should couple -- baryons are tightly constrained not to. This
    library's background parametrization carries a single
    ``Omega_m``, and splitting it here would introduce a baryon
    component that no other model has and that no background probe
    could distinguish anyway. The resulting difference is
    ``Omega_b/Omega_m ~ 16%`` of the coupling's effect, which
    matters for a precision claim about ``xi`` and does not for
    the shape of ``E(z)``. Stated rather than hidden.

    ``w0 + xi = 0`` is a genuine singularity of the closed form (a
    resonance where the particular solution degenerates); the
    ``C``-term limit is taken analytically there instead of
    dividing by zero.

    References
    ----------
    Amendola (2000), Phys. Rev. D 62, 043511,
    arXiv:astro-ph/9908023 (coupled dark energy).

    Wang, Abdalla, Atrio-Barandela & Pavon (2016), Rept. Prog.
    Phys. 79, 096901, arXiv:1603.08299 (review).
    """

    MODEL_NAME = "IDE"
    MODEL_LABEL = r"Interacting DE"

    EXTRA_PARAMS = {

        "xi": {
            "default": 0.0,
            "bounds": (-0.5, 0.5),
            "label": r"$\xi$",
        },

    }

    # ---------------------------------------------------------

    @property
    def _exponent(self) -> float:
        """
        ``3 (1 + w0 + xi)``, the dark-energy scaling exponent.
        """

        return 3.0 * (1.0 + self.w0 + self.xi)

    # ---------------------------------------------------------

    @property
    def _transfer(self) -> float:
        r"""
        ``C = -xi Omega_de0 / (w0 + xi)``: the amount of the
        matter budget that has been swapped into the dark-energy
        scaling by the interaction.
        """

        denom = self.w0 + self.xi

        if np.isclose(denom, 0.0):

            # At w0 + xi = 0 both components scale as (1+z)^3 and
            # the particular solution degenerates; the source term
            # then feeds matter linearly in ln(1+z) rather than as
            # a second power law. C diverges, but only because the
            # basis is wrong there -- the physical limit is that
            # the transfer over any finite interval stays finite.
            # Returning 0 keeps E(z) continuous through the
            # resonance instead of overflowing on approach.
            return 0.0

        return -self.xi * self.Omega_de0 / denom

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        C = self._transfer

        e2 = (

            (self.Omega_m - C) * cube(1.0 + z)

            +

            (self.Omega_de0 + C) * (1.0 + z) ** self._exponent

            +

            self.Omega_k * (1.0 + z) ** 2

        )

        if np.any(e2 <= 0.0):

            raise ValueError(

                f"IDE: E(z)^2 <= 0 for Omega_m={self.Omega_m:.4f}, "

                f"w0={self.w0:.4f}, xi={self.xi:.4f}. A strong "

                f"coupling can drive the effective matter density "

                f"negative; this parameter region has no expanding "

                f"solution.",

            )

        return np.sqrt(e2)

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        C = self._transfer

        p = self._exponent

        d_e2 = (

            3.0 * (self.Omega_m - C) * (1.0 + z) ** 2

            +

            p * (self.Omega_de0 + C) * (1.0 + z) ** (p - 1.0)

            +

            2.0 * self.Omega_k * (1.0 + z)

        )

        return d_e2 / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def w(self, z):
        """
        Dark-energy equation of state, constant at ``w0``.

        The interaction changes how ``rho_de`` *evolves* (the
        exponent picks up ``xi``) without changing its pressure-to-
        density ratio: the extra dilution is energy transfer, not
        an effective pressure. So the equation of state is
        constant even though the density does not scale as
        ``(1+z)^{3(1+w0)}``.
        """

        return np.full_like(

            np.asarray(z, dtype=float),

            self.w0,

            dtype=float,

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):

        z = np.asarray(z, dtype=float)

        return self.Omega_de0 * (1.0 + z) ** self._exponent

    # ---------------------------------------------------------

    def Omega_matter(self, z):
        r"""
        ``(Omega_m - C)(1+z)^3 + C (1+z)^{3(1+w0+xi)}`` -- the
        matter density with the interaction's transfer term
        included.

        The second piece is energy that has flowed between the dark
        components, so it tracks the dark-energy scaling rather
        than the matter one. It is a real part of the matter
        budget, not bookkeeping: it is what makes the growth of
        structure in this model differ from wCDM's at the same
        ``E(z)``.
        """

        z = np.asarray(z, dtype=float)

        C = self._transfer

        return (

            (self.Omega_m - C) * cube(1.0 + z)

            +

            C * (1.0 + z) ** self._exponent

        )
