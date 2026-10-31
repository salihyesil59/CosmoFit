"""
Best fit (maximum likelihood) of finished runs, each started from its
chain's best point. Writes <output>.bestfit.json next to the chain, where
the cpl_vs_lcdm_full_cmb notebook reads it. Run from the repository root:

    python examples/yaml/cpl_vs_lcdm/bestfit.py lcdm_pantheonplus cpl_pantheonplus ...

The names are the YAML files' in this folder.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from getdist import loadMCSamples

from cosmofit.core import load_info, run

HERE = Path(__file__).resolve().parent


def chain_best(prefix, names):
    s = loadMCSamples(str(prefix), settings={"ignore_rows": 0.3})
    chi2 = sum(s[p.name] for p in s.paramNames.names if p.name.startswith("chi2__"))
    i = int(np.argmin(chi2))
    return {n: float(s[n][i]) for n in names}, float(chi2[i])


def best_fit(name):
    info = load_info(str(HERE / f"{name}.yaml"))
    prefix = info.pop("output")
    sampled = [n for n, spec in info["params"].items() if isinstance(spec, dict) and "prior" in spec]
    start, chi2_chain = chain_best(prefix, sampled)
    for n in sampled:
        info["params"][n]["ref"] = {"dist": "norm", "loc": start[n], "scale": 1e-6}
    # Nelder-Mead: the emulated likelihood is smooth only to its tolerance,
    # too rough for finite-difference gradients.
    info["sampler"] = {"minimize": {"ignore_prior": True, "method": "Nelder-Mead", "max_evals": 15000}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, minimizer = run(info)
    out = minimizer.products()
    out["chi2_chain"] = chi2_chain
    out["k"] = len(sampled)
    Path(f"{prefix}.bestfit.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: chi2 chain {chi2_chain:.2f} -> minimum {out['chi2']['total']:.2f} "
          f"({out['n_evaluations']} evaluations)", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        best_fit(name)
