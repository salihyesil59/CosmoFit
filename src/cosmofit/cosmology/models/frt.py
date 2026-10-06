"""
f(R,T) modified gravity, linear matter-geometry coupling.
"""

from __future__ import annotations

import numpy as np

from cosmofit.typing import Array, Redshift

from cosmofit.cosmology.numerics.powers import cube

from cosmofit.cosmology.core import Cosmology


class FRTLinear(Cosmology):
    r"""
    f(R,T) gravity, linear model: f(R,T) = R + 2*lambda*T.

    f(R,T) gravity (Harko, Lobo, Nojiri & Odintsov 2011) makes the
    gravitational Lagrangian depend on the trace T of the matter
    stress-energy tensor as well as the Ricci scalar R, coupling
    gravity directly to the matter content rather than adding a
    separate dark-energy fluid on top of standard GR. This is the
    original, simplest member of that family: f(T) = lambda*T.

    Reference derivation (their eq. 26, for dust p=0, T=rho):
    3H^2 = (8*pi + 3*lambda)*rho. Extending their eq. 23 (general
    perfect fluid, not just dust) from a single dust fluid to a
    two-component universe -- pressureless matter (rho_m, p_m=0) and
    a Lambda-like component (rho_L, p_L=-rho_L) -- gives (cross-
    checked: this reduces to their eq. 26 exactly as rho_L -> 0):

        3H^2 = (8*pi + 3*lambda)*rho_m + (8*pi + 4*lambda)*rho_L

    Non-dimensionalizing with beta = lambda / (8*pi) (beta=0 is
    exactly GR):

        E(z)^2 = Omega_k*(1+z)^2
                 + (1 + 3*beta) * Omega_m * (1+z)^3
                 + (1 + 4*beta) * Omega_L

    ``Omega_L`` is *derived*, from the Friedmann equation at z = 0:

        Omega_L = (1 - Omega_k - (1 + 3*beta) * Omega_m) / (1 + 4*beta)

    which is what makes ``E(0) = 1`` -- i.e. what makes ``H0`` the
    Hubble rate today. It used to be a free parameter independent of
    ``Omega_m``. Then ``E(0)^2 = Omega_k + (1+3*beta) Omega_m +
    (1+4*beta) Omega_L`` was whatever the sampler made it, the
    parameter called ``H0`` was not H(z=0), and everything that reads
    ``H0`` as the Hubble constant -- an H0 prior, the SN absolute
    magnitude, the CMB shift parameter ``R`` -- compared it with the
    wrong number. f(R,T) fits that quote independent ``Omega_m`` and
    ``Omega_L`` posteriors are fitting a curvature they do not name;
    here ``Omega_k`` is that parameter, and it is explicit. Only the linear
    (f(T) proportional to T) case is implemented; the general
    f(R,T) = R + alpha*T^n form is not, since the units/normalization
    convention for alpha in the papers surveyed for this couldn't be
    pinned down with confidence -- see the project's dev notes.

    **Growth of structure.** ``mu(a, k)`` uses
    ``mu(a) = 1 + 3*beta`` -- the same rescaling already derived
    above for the matter term of this model's own ``E(z)^2``
    (internally consistent with the background by construction).
    This is a stated simplification, not a full derivation: unlike
    f(Q) (where G_eff = G_N/f_Q is a settled sub-horizon result --
    see ``FQExponential.mu``), f(R,T) does not separately conserve
    the matter stress-energy tensor, so a full covariant linear
    perturbation theory is genuinely more involved (see Asghari &
    Sheykhi 2024, arXiv:2405.11840, who derive it for a general
    f(R,T) form and find it suppresses structure growth relative to
    LCDM -- the same qualitative direction ``mu(a) = 1+3*beta`` with
    the fitted-negative ``beta`` typical of this model produces, but
    not the same derivation). Scale-independent (``k`` accepted for
    interface consistency, ignored). beta=0 (GR) gives mu=1 exactly.

    Notes
    -----
    Adds ``beta`` (the dimensionless matter-geometry coupling,
    default 0.0 = GR) via ``EXTRA_PARAMS``. ``Omega_L`` is a derived
    property, not a parameter.

    References
    ----------
    Harko, Lobo, Nojiri & Odintsov (2011), "f(R,T) gravity", Phys.
    Rev. D 84, 024020, arXiv:1104.2669.

    Asghari & Sheykhi (2025), "Growth of cosmic perturbations in the
    modified f(R,T) gravity", Phys. Dark Univ. 48, arXiv:2405.11840.
    """

    MODEL_NAME = "FRTLinear"
    MODEL_LABEL = r"$f(R,T)$ linear"

    #: ``beta``'s bounds are deliberately narrow: this is a weak-
    #: coupling perturbation around GR (beta=0), and literature fits
    #: of this model find it consistent with ``|beta|`` of a few percent
    #: -- values of order 1 aren't just "strongly coupled", they can
    #: make E(z)^2 negative (unphysical) at moderate-to-high z for
    #: otherwise-ordinary Omega_m/Omega_L, since (1+3*beta) and
    #: (1+4*beta) can go negative and dominate. `LogPosterior`
    #: treats that region as chi2=+inf / log-posterior=-inf rather
    #: than crashing, but bounding the prior to the physically
    #: sensible regime is the right fix, not just a safety net.
    EXTRA_PARAMS = {
        "beta": {
            "default": 0.0, "bounds": (-0.2, 0.2), "label": r"$\beta$",
        },
    }

    #: ``Omega_L`` is fixed by the Friedmann equation at z = 0 -- see
    #: :attr:`Omega_L` -- so freeing it would sample a number nothing
    #: reads.
    DERIVED_PARAMS = frozenset({"Omega_L"})

    # ---------------------------------------------------------

    @property
    def Omega_L(self) -> float:
        """
        The Lambda-like component's density parameter, from ``E(0) = 1``:

            Omega_L = (1 - Omega_k - (1 + 3*beta) Omega_m) / (1 + 4*beta)
        """

        return (
            (1.0 - self.Omega_k - (1.0 + 3.0 * self.beta) * self.Omega_m)
            / (1.0 + 4.0 * self.beta)
        )

    # ---------------------------------------------------------

    def E(self, z: Redshift) -> Array:
        """
        Dimensionless Hubble parameter.
        """

        z = np.asarray(z, dtype=float)

        return np.sqrt(
            self.Omega_k * (1.0 + z) ** 2
            + (1.0 + 3.0 * self.beta) * self.Omega_m * cube(1.0 + z)
            + (1.0 + 4.0 * self.beta) * self.Omega_L
        )

    # ---------------------------------------------------------

    def dEdz(self, z: Redshift) -> Array:
        """
        Derivative of E(z).
        """

        z = np.asarray(z, dtype=float)

        dE2_dz = (
            2.0 * self.Omega_k * (1.0 + z)
            + 3.0 * (1.0 + 3.0 * self.beta) * self.Omega_m * (1.0 + z) ** 2
        )

        return dE2_dz / (2.0 * self.E(z))

    # ---------------------------------------------------------

    def Omega_de(self, z: Redshift) -> Array:
        """
        Effective dark-energy density (the Lambda-like component,
        rescaled by the matter-geometry coupling -- constant in z,
        as in LCDM).
        """

        z = np.asarray(z, dtype=float)

        return np.full_like(z, (1.0 + 4.0 * self.beta) * self.Omega_L)

    # ---------------------------------------------------------

    def mu(self, a: Redshift, k: float | None = None) -> Array:
        """
        Effective gravitational coupling, mu(a) = 1 + 3*beta -- see
        the class docstring for the (stated-simplification) caveat.
        """

        return np.full_like(np.asarray(a, dtype=float), 1.0 + 3.0 * self.beta)
