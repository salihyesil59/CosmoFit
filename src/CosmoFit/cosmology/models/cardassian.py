"""
Modified Polytropic Cardassian expansion.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core import Cosmology


class Cardassian(Cosmology):
    r"""
    Modified Polytropic Cardassian expansion (Freese & Lewis 2002;
    Wang et al. 2003),

        E(z)^2 = Omega_k (1+z)^2
                 + Omega_m (1+z)^3
                   { 1 + [Omega_m*^{-q} - 1] (1+z)^{3q(n-1)} }^{1/q}

    with ``Omega_m* = Omega_m / (1 - Omega_k)``.

    The Cardassian idea is that the Friedmann equation itself gains
    an extra term, ``H^2 = A rho + B rho^n``, from the universe
    being a 3-brane embedded in higher dimensions -- so acceleration
    arises **from matter alone**, with no dark energy and no
    vacuum energy. The universe is matter-dominated at all times;
    it is the relation between ``H`` and ``rho`` that is modified.

    Two extra parameters, ``n`` and ``q``, and two reductions worth
    knowing:

    - ``n = 0``, ``q = 1`` is exactly LCDM.
    - ``q = 1`` (any ``n``) is the "original" Cardassian model,
      which is degenerate with wCDM at ``w = n - 1``: the extra
      term scales as a power law, and a power-law density is what
      constant-``w`` dark energy is. It is ``q != 1`` that makes
      this a genuinely distinct expansion history rather than a
      reparametrization -- which is precisely why the *modified
      polytropic* form is the one implemented here and the
      original one is not.

    The bracket is normalized so that ``E(0) = 1`` identically for
    any ``(n, q, Omega_k)``: at ``z = 0`` it collapses to
    ``Omega_m*^{-1}``, cancelling the ``Omega_m`` prefactor against
    the curvature term.

    Notes
    -----
    The original derivation assumes a flat universe. Curvature is
    admitted here through the ``Omega_m*`` rescaling above, which
    keeps the Friedmann constraint exact and reduces to the
    published flat form at ``Omega_k = 0``; it is a consistent
    extension of the parametrization, not a result taken from the
    literature. Fit ``Omega_k`` with this model deliberately.

    Numerically, ``Omega_m*^{-q}`` overflows for large ``q`` and
    small ``Omega_m``, and the bracket can go negative for
    ``n > 1``; both are outside the physically motivated region
    (``0 <= n < 1``, ``q > 0``) that the default bounds cover.

    References
    ----------
    Freese & Lewis (2002), "Cardassian Expansion: a Model in which
    the Universe is Flat, Matter Dominated, and Accelerating",
    Phys. Lett. B 540, 1, arXiv:astro-ph/0201229.

    Wang, Freese, Gondolo & Lewis (2003), ApJ 594, 25,
    arXiv:astro-ph/0302064 (the modified polytropic form).
    """

    MODEL_NAME = "Cardassian"

    EXTRA_PARAMS = {

        "n_card": {
            "default": 0.0,
            "bounds": (-1.0, 1.0),
            "label": r"$n$",
        },

        "q_card": {
            "default": 1.0,
            "bounds": (0.1, 10.0),
            "label": r"$q$",
        },

    }

    # ---------------------------------------------------------

    @property
    def _omega_m_star(self) -> float:
        """
        ``Omega_m / (1 - Omega_k)`` -- the matter fraction of the
        non-curvature part, which is what the flat derivation's
        ``Omega_m`` means.
        """

        return self.Omega_m / (1.0 - self.Omega_k)

    # ---------------------------------------------------------

    def _bracket(self, z):
        r"""
        ``1 + [Omega_m*^{-q} - 1] (1+z)^{3q(n-1)}``, the quantity
        raised to ``1/q``.
        """

        z = np.asarray(z, dtype=float)

        q = self.q_card
        n = self.n_card

        return 1.0 + (

            self._omega_m_star ** (-q) - 1.0

        ) * (1.0 + z) ** (3.0 * q * (n - 1.0))

    # ---------------------------------------------------------

    def E(self, z):

        z = np.asarray(z, dtype=float)

        bracket = self._bracket(z)

        if np.any(bracket <= 0.0):

            raise ValueError(

                f"Cardassian: the Friedmann bracket went "

                f"non-positive for Omega_m={self.Omega_m:.4f}, "

                f"n={self.n_card:.4f}, q={self.q_card:.4f}. The "

                f"modified polytropic form has no real expansion "

                f"history there.",

            )

        return np.sqrt(

            self.Omega_k * (1.0 + z) ** 2

            +

            self.Omega_m * (1.0 + z) ** 3

            * bracket ** (1.0 / self.q_card)

        )

    # ---------------------------------------------------------

    def dEdz(self, z):

        z = np.asarray(z, dtype=float)

        q = self.q_card
        n = self.n_card

        bracket = self._bracket(z)

        # d(bracket)/dz
        d_bracket = (

            self._omega_m_star ** (-q) - 1.0

        ) * (

            3.0 * q * (n - 1.0)

        ) * (1.0 + z) ** (3.0 * q * (n - 1.0) - 1.0)

        # Product rule on Omega_m (1+z)^3 bracket^{1/q}.
        d_cardassian = (

            3.0 * self.Omega_m * (1.0 + z) ** 2

            * bracket ** (1.0 / q)

            +

            self.Omega_m * (1.0 + z) ** 3

            * (1.0 / q) * bracket ** (1.0 / q - 1.0)

            * d_bracket

        )

        return (

            (

                2.0 * self.Omega_k * (1.0 + z)

                + d_cardassian

            )

            /

            (2.0 * self.E(z))

        )

    # ---------------------------------------------------------

    def Omega_de(self, z):
        r"""
        The effective dark-energy density an observer assuming GR
        plus a fluid would reconstruct from this expansion history:

            Omega_de,eff(z) = E(z)^2
                              - Omega_m (1+z)^3
                              - Omega_k (1+z)^2

        There is no dark energy in a Cardassian universe -- the
        whole point of the model is that there isn't one. This is
        the fictitious component its ``E(z)`` would be *mistaken*
        for, and it is what the plotting machinery shows.
        """

        z = np.asarray(z, dtype=float)

        return (

            self.E(z) ** 2

            - self.Omega_m * (1.0 + z) ** 3

            - self.Omega_k * (1.0 + z) ** 2

        )
