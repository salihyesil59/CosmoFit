"""
f(R) modified gravity (Hu-Sawicki), background level.
"""

from __future__ import annotations

from .lcdm import LCDM


class FRHuSawicki(LCDM):
    r"""
    f(R) gravity, Hu & Sawicki (2007) model -- background level only.

    f(R) gravity replaces the Ricci scalar R in the Einstein-Hilbert
    action with an arbitrary function f(R). The Hu-Sawicki form,

        f(R) = -m^2 c1 (R/m^2)^n / (c2 (R/m^2)^n + 1),

    is the standard benchmark model in the literature, tuned (via
    ``n`` and the present-day scalaron value ``f_R0``) to satisfy
    Solar System tests through chameleon screening while still
    producing cosmic acceleration.

    **This class subclasses :class:`~cosmology.models.lcdm.LCDM`
    directly and does not override ``E``/``dEdz``/``Omega_de`` --
    its background expansion history *is* LCDM's, unchanged.** This
    is not a simplification or a placeholder bug: it is the actual
    physics of the standard "designer f(R)" construction, which
    builds f(R) to reproduce an assumed target background (usually
    LCDM's) essentially exactly. The Hu-Sawicki parameters ``n`` and
    ``f_R0`` are present (via ``EXTRA_PARAMS``) but are, at the
    background level implemented here, inert -- they don't enter
    ``E(z)`` at all.

    **What this means in practice:** fitting ``f_R0``/``n`` against
    CosmoFit's current datasets (CC, BAO, SNe, Planck distance
    priors -- all background/expansion-history probes) cannot
    meaningfully constrain them; the fit will simply show them
    unconstrained by the prior, since the likelihood genuinely
    doesn't depend on them. Hu-Sawicki f(R)'s actual observational
    signature is in the growth of structure (fsigma8, weak lensing)
    -- a perturbation-level calculation this library does not
    (yet) implement. This model is included for completeness and to
    make that distinction concrete rather than silently omitting
    f(R) altogether.

    Parameters
    ----------
    Adds ``f_R0`` (present-day scalaron value, default -1e-6,
    typically negative and small -- see the reference for viability
    bounds) and ``n`` (default 1) via ``EXTRA_PARAMS``. Neither
    affects ``E(z)`` here -- see above.

    References
    ----------
    Hu & Sawicki (2007), "Models of f(R) Cosmic Acceleration that
    Evade Solar-System Tests", Phys. Rev. D 76, 064004,
    arXiv:0705.1158.
    """

    MODEL_NAME = "FRHuSawicki"

    EXTRA_PARAMS = {
        "f_R0": {
            "default": -1e-6, "bounds": (-1e-4, -1e-8),
            "label": r"$f_{R0}$",
        },
        "n": {
            "default": 1.0, "bounds": (0.1, 4.0), "label": r"$n$",
        },
    }
