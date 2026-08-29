"""
Bayesian evidence by nested sampling.

Every model comparison in this library so far has been a
``delta chi2`` at the best fit, wrapped in AIC, BIC or a
likelihood-ratio test. Those are approximations, and the notebooks
say so repeatedly -- *"these are delta chi2 at the best fit, not
evidence ratios"*. This computes the thing itself:

    Z = Int L(theta) pi(theta) d theta

and ``ln Z`` is what a Bayes factor is a difference of.

Two reasons it is worth having rather than another information
criterion.

**It does not care about boundaries.** Wilks' theorem needs the
null to be interior to the parameter space, and three of this
library's most interesting comparisons violate that: ``LsCDM``
reduces to ``LCDM`` only as ``z_dagger -> infinity``, ``GEDE`` at
an edge of its parameter, ``DGP`` not at all. The notebooks
correctly refuse to quote a sigma in those cases. An evidence ratio
is defined regardless.

**It integrates rather than maximizes**, so a parameter that
improves the fit only in a sliver of its prior is penalized for it
-- the Occam factor that AIC and BIC only approximate, and that
counts the volume rather than the parameter.

Which brings the warning that has to come with it.

Prior sensitivity
-----------------
``Z`` depends on the prior, and this library's priors are uniform
over :data:`~cosmology.core.parameters.DEFAULT_BOUNDS`. Widening
the range of a parameter the data does not constrain divides its
evidence by roughly the widening factor, with the fit unchanged.
That is not a defect of the method -- it is the Occam factor doing
its job -- but it does mean **a Bayes factor here is a statement
about a model plus its priors**, and comparing two models means
having defended both. :func:`bayes_factor` reports the prior
volumes alongside the ratio so that the dependence is visible
rather than implicit.

How much it matters, measured on this library's own case. ``LsCDM``
against ``LCDM`` on CC + DESI DR2 + low-z BAO + DES-SN5YR + Planck
priors + BBN, changing nothing but the upper edge of the
``z_dagger`` prior:

=========================  ==========  ================
``z_dagger`` prior           ``ln B``    label
=========================  ==========  ================
``(0.5, 10)``                  +1.158    positive
``(0.5, 100)``                 +0.494    inconclusive
=========================  ==========  ================

The same data, the same model, the same fit -- and a different
verdict, because above the sign-switch cliff the parameter is
unconstrained and the model is charged for whatever room it was
given. (The shift is 0.66 rather than the ``ln 10 = 2.30`` a
completely flat direction would give, because the likelihood does
decline slowly towards the ``z_dagger -> infinity`` limit.)

For contrast, the frequentist number on the same fit is
``delta chi2 = 5.40``, which Wilks would read as ~2 sigma and which
does not apply here at all, ``LCDM`` being on the boundary. Neither
statistic is wrong; they answer different questions, and only one
of them has to be told what the prior is.

Optional dependency: ``pip install "cosmofit[evidence]"``.
"""

from __future__ import annotations

import numpy as np


#: Jeffreys' scale, as revised by Kass & Raftery (1995), on
#: ``ln B``. The labels are theirs; the thresholds are what most of
#: the cosmology literature quotes.
_JEFFREYS = (
    (1.0, "inconclusive"),
    (3.0, "positive"),
    (5.0, "strong"),
    (np.inf, "very strong"),
)


def interpret(ln_bayes_factor: float) -> str:
    """
    Kass & Raftery's label for a log Bayes factor, on its absolute
    value -- the sign says which model, the size says how much.
    """

    magnitude = abs(float(ln_bayes_factor))

    for threshold, label in _JEFFREYS:

        if magnitude < threshold:
            return label

    return _JEFFREYS[-1][1]


def bayes_factor(alt, null) -> dict:
    """
    Compare two completed nested-sampling runs.

    Parameters
    ----------
    alt, null : NestedResult
        The more general model and the simpler one. Order only
        fixes the sign.

    Returns
    -------
    dict
        ``ln_B`` (positive favours ``alt``), its uncertainty
        propagated from both runs, the Kass & Raftery label, and
        the two prior volumes -- because a Bayes factor computed
        against uniform priors is a statement about those priors
        as much as about the models.
    """

    ln_b = float(alt.log_evidence - null.log_evidence)

    error = float(
        np.hypot(alt.log_evidence_error, null.log_evidence_error),
    )

    return {
        "ln_B": ln_b,
        "ln_B_error": error,
        "interpretation": interpret(ln_b),
        "favours": "alt" if ln_b > 0 else "null",
        "prior_volume_alt": alt.prior_volume,
        "prior_volume_null": null.prior_volume,
    }
