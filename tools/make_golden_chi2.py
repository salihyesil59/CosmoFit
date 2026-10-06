"""
Write tests/data/golden_chi2.json: every built-in model's chi2 on every
dataset that needs no Boltzmann code, at one fixed point.

The reference the 2.0 rewrite has to reproduce. It is computed straight
from the original likelihood classes -- a model instance, a dataset's
class from ``DATASET_REGISTRY``, ``chi2()`` -- so it depends on neither
``Fitter`` nor the new core, and a later port of a model or a dataset
is checked against what the code computed before the port.

Regenerate only when a change to a model or a dataset is *meant* to
change its chi2, and say so in the CHANGELOG entry for that change:

    python tools/make_golden_chi2.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import cosmofit
from cosmofit.stats.fitter import DATASET_REGISTRY


#: Datasets that call CAMB: excluded, since their values move at the
#: 1e-6 level between CAMB versions and platforms.
CAMB_DATASETS = {"planck_lite", "planck_lensing", "planck_lowe", "act_lensing"}

MODELS = [
    "LCDM", "WCDM", "CPL", "JBP", "BA", "LogarithmicDE", "PEDE", "GEDE",
    "LsCDM", "GCG", "IDE", "RunningVacuum", "Cardassian", "DGP", "HDE",
    "ADE", "RDE", "FQExponential", "FTPowerLaw", "FRTLinear", "FRHuSawicki",
]

#: The point: each model's own defaults, with these overridden where
#: the model has them -- a little away from LCDM, so that extension
#: parameters matter.
POINT = {
    "H0": 68.0,
    "Omega_m": 0.31,
    "Omega_b": 0.049,
    "w0": -0.95,
    "wa": -0.2,
    "rd": 147.5,
    "MB": -19.3,
    "sigma8": 0.80,
}

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "golden_chi2.json"


def point_for(model_cls) -> dict:

    params_cls = model_cls.PARAMS_CLASS

    values = dict(params_cls.defaults())
    values.update({k: v for k, v in POINT.items() if k in params_cls.names()})

    return {name: values[name] for name in params_cls.names()}


def main() -> None:

    datasets = sorted(set(DATASET_REGISTRY) - CAMB_DATASETS)

    table = {}

    for name in MODELS:

        model_cls = getattr(cosmofit, name)

        values = point_for(model_cls)

        cosmology = model_cls(model_cls.PARAMS_CLASS(**values))

        row = {}

        for dataset in datasets:

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")

                try:
                    value = float(DATASET_REGISTRY[dataset](cosmology).chi2())
                except (ValueError, ArithmeticError, RuntimeError) as error:
                    row[dataset] = {"error": type(error).__name__}
                    continue

                # Outside a tabulated likelihood's grid, chi2 is +inf:
                # recorded as such rather than as JSON's non-standard
                # `Infinity`.
                row[dataset] = value if math.isfinite(value) else {"error": "non-finite"}

        table[name] = {"point": values, "chi2": row}

    OUTPUT.write_text(
        json.dumps(
            {
                "about": (
                    "chi2 per model and dataset at 'point', from the "
                    "original likelihood classes; see "
                    "tools/make_golden_chi2.py."
                ),
                "cosmofit_version": cosmofit.__version__,
                "models": table,
            },
            indent=1,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {OUTPUT} ({len(MODELS)} models x {len(datasets)} datasets)")


if __name__ == "__main__":
    main()
