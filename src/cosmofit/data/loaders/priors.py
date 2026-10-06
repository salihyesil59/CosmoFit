"""
Gaussian external priors: H0, BBN omega_b h^2, tau.
"""

from __future__ import annotations

import numpy as np

from ..dataset import GaussianPriorDataset
from ..covariance import make_covariance

from ._common import _get_dataset_path, _load_txt, _validate_version


#: External single-number Gaussian constraints (see
#: :class:`~data.dataset.GaussianPriorDataset`). Each entry names
#: the *quantity* it constrains, which
#: :class:`~likelihoods.priors.GaussianPriorLikelihood` maps to a
#: model prediction.
PRIOR_FILES = {

    "h0": {

        "quantity": "H0",

        "parent": "priors",

        "folder": "h0",

        "versions": {

            "sh0es2022": {
                "data": "h0_sh0es2022.txt",
                "reference": "Riess et al. (2022), ApJ 934, L7, arXiv:2112.04510",
            },

            "sh0es2024": {
                "data": "h0_sh0es2024.txt",
                "reference": "Breuval et al. (2024), ApJ 973, 30, arXiv:2404.08038",
            },

            "tdcosmo2025": {
                "data": "h0_tdcosmo2025.txt",
                "reference": "TDCOSMO Collaboration / Birrer et al. (2025), A&A 704, A63, arXiv:2506.03023",
            },

        },

    },

    "omega_b": {

        "quantity": "omega_b_h2",

        "parent": "priors",

        "folder": "omega_b",

        "versions": {

            "bbn2024": {
                "data": "omega_b_bbn2024.txt",
                "reference": "Schoeneberg (2024), JCAP 06 (2024) 006, arXiv:2401.15054",
            },

            "cooke2018": {
                "data": "omega_b_cooke2018.txt",
                "reference": "Cooke, Pettini & Steidel (2018), ApJ 855, 102, arXiv:1710.11129",
            },

        },

    },

    "tau": {

        "quantity": "tau_reio",

        "parent": "priors",

        "folder": "tau",

        # The low-l-only constraint first, so it is the default: it
        # is what the "tau" dataset exists for -- breaking plik_lite's
        # tau-A_s degeneracy -- and the only one of the two that can
        # sit next to plik_lite without counting its spectra twice.
        "versions": {

            "planck2018_lowe": {
                "data": "tau_planck2018_lowe_only.txt",
                "reference": "Planck Collaboration (2020), A&A 641, A6, arXiv:1807.06209 (lowE alone)",
            },

            "planck2018": {
                "data": "tau_planck2018_lowe.txt",
                "reference": "Planck Collaboration (2020), A&A 641, A6, arXiv:1807.06209 (TT,TE,EE+lowE)",
            },

        },

    },

}


#: Flattened ``PRIOR_FILES``, so each prior dataset ("h0",
#: "omega_b", "tau") looks like every other registry to
#: :func:`_validate_version` and :func:`available_versions`.
_PRIOR_REGISTRIES = {

    name: {

        version: {
            **spec,
            "parent": entry["parent"],
            "folder": entry["folder"],
            "quantity": entry["quantity"],
        }

        for version, spec in entry["versions"].items()

    }

    for name, entry in PRIOR_FILES.items()

}


# ============================================================
# External single-number Gaussian constraints
# ============================================================

def load_gaussian_prior(
    dataset: str,
    version: str | None = None,
) -> GaussianPriorDataset:
    """
    Load a single external Gaussian constraint -- a local
    distance-ladder ``H0``, a BBN ``omega_b h^2``, or a
    reionization ``tau``.

    These enter a fit as one-point datasets rather than as priors
    on the parameter; see
    :class:`~data.dataset.GaussianPriorDataset` for why.

    Parameters
    ----------
    dataset : str
        Which constraint: ``"h0"``, ``"omega_b"`` or ``"tau"``.

    version : str, optional
        Which measurement of it. Defaults to the first entry
        registered for that dataset (``"sh0es2022"``,
        ``"bbn2024"``, ``"planck2018"``).

    Returns
    -------
    GaussianPriorDataset
    """

    if dataset not in _PRIOR_REGISTRIES:

        raise ValueError(

            f"Unknown prior dataset '{dataset}'. "

            f"Available: {list(_PRIOR_REGISTRIES)}",

        )

    if version is None:

        version = next(iter(_PRIOR_REGISTRIES[dataset]))

    entry = _validate_version(dataset, version)
    dataset_path = _get_dataset_path(dataset, version)

    data = np.atleast_1d(

        _load_txt(dataset_path / entry["data"]),

    ).astype(float)

    if data.size != 2:

        raise ValueError(

            f"'{entry['data']}': expected one '<value> <sigma>' "

            f"row, but found {data.size} numbers.",

        )

    value = float(data[0])
    sigma = float(data[1])

    if sigma <= 0.0:

        raise ValueError(

            f"'{entry['data']}': sigma must be positive.",

        )

    return GaussianPriorDataset(

        quantity=entry["quantity"],

        value=value,

        sigma=sigma,

        covariance=make_covariance(sigma=np.array([sigma])),

        reference=entry["reference"],

    )
