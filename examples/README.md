# CosmoFit Examples

Every notebook here is **Colab-ready**: click a badge and
*Runtime → Run all*, no local setup. They are also plain notebooks —
`pip install -e .` from the repository root and run them anywhere.

New here? Start with
[`01-getting-started/quickstart.ipynb`](01-getting-started/quickstart.ipynb).

---

## 01 · Getting started

| | |
|---|---|
| [`quickstart.ipynb`](01-getting-started/quickstart.ipynb) | The shortest path from nothing to a real MCMC posterior: flat ΛCDM on CC + DESI, a couple of minutes end to end. |
| [`dataset_zoo.ipynb`](01-getting-started/dataset_zoo.ipynb) | All **21** datasets against one fixed fiducial cosmology — including the three that ship a likelihood *surface* rather than a mean, and the sixteen pairs that must not be combined. |

## 02 · Models

| | |
|---|---|
| [`model_zoo_comparison.ipynb`](02-models/model_zoo_comparison.ipynb) | Six background-expansion models (ΛCDM, wCDM, CPL, JBP, BA, GCG) fit to the same data and compared with AIC/BIC. |
| [`modified_gravity_growth.ipynb`](02-models/modified_gravity_growth.ipynb) | f(Q) / f(R,T) / f(R) end to end: background `E(z)`, `mu(a,k)`, `fσ8`/`S8`, and `f_R0` going from unconstrained to constrained once growth data enter. |
| [`holographic_family.ipynb`](02-models/holographic_family.ipynb) | HDE, ADE and RDE — three models from one idea, differing only in what sets the horizon scale. Two have no closed-form `E(z)`. Includes ADE's *derived* `Omega_m` and a comparison against published constraints. |

## 03 · Building your own model

| | |
|---|---|
| [`custom_models.ipynb`](03-building-models/custom_models.ipynb) | Three routes to a model that isn't in the library: an `E(z)` **string**, an `E(z)` **function**, or a `Cosmology` subclass. Plus `mu(a,k)` for modified gravity. |
| [`models_from_an_action.ipynb`](03-building-models/models_from_an_action.ipynb) | Skip deriving `E(z)` by hand: give an **action** and let `CosmoFit.theory` do the variational calculus. Rederives ΛCDM and f(Q) exactly, fits a new f(T) model, and shows the three things it refuses to approximate. |
| [`scalar_field_models.ipynb`](03-building-models/scalar_field_models.ipynb) | Quintessence and k-essence from the action, where the history must be **integrated**. Validated against both Copeland–Liddle–Wands attractors, and explains why the field's initial conditions go early rather than today. |

## 04 · Inference

| | |
|---|---|
| [`evidence_and_model_selection.ipynb`](04-inference/evidence_and_model_selection.ipynb) | AIC/BIC, the likelihood-ratio test, and Bayesian evidence by nested sampling — what each assumes, where they disagree, and why a Bayes factor must be quoted with its prior. |
| [`profile_likelihood_and_fisher.ipynb`](04-inference/profile_likelihood_and_fisher.ipynb) | Three ways to get an error bar. Fisher against MCMC, profile against marginal, and why a parameter on a boundary breaks the usual Δχ² reading. |
| [`tension_statistics.ipynb`](04-inference/tension_statistics.ipynb) | "They disagree at 4σ" needs a definition. Four of them: Gaussian, sample-based, N-dimensional, and suspiciousness — which divides the prior dependence out. |

## 05 · Case studies

Longer analyses, each posing a real research question.

| | |
|---|---|
| [`cpl_mcmc_analysis.ipynb`](05-case-studies/cpl_mcmc_analysis.ipynb) | The deep dive: CPL on CC + DESI + Pantheon+, convergence diagnostics, every `fit.plots` figure, model comparison, an independent Planck cross-check. |
| [`cpl_mcmc_tfd42.ipynb`](05-case-studies/cpl_mcmc_tfd42.ipynb) | Publication-scale variant: all four datasets *jointly* (Planck included, making `rd`/`Omega_b` constrainable), a much longer chain, multi-core MCMC. |
| [`lscdm_mcmc.ipynb`](05-case-studies/lscdm_mcmc.ipynb) | Does Akarsu et al.'s **ΛsCDM** — a cosmological constant that switches sign at z† — still relieve the H₀ tension against 2024–2025 data? Includes the profile likelihood over z† that locates the answer. |
| [`s8_tension_cmb.ipynb`](05-case-studies/s8_tension_cmb.ipynb) | The **S₈ tension from the CMB side**: the full Planck 2018 spectrum computed from scratch, σ₈ *derived* rather than fitted, confronted with KiDS-1000 and DES Y3. |
| [`dark_energy_evidence_audit.ipynb`](05-case-studies/dark_energy_evidence_audit.ipynb) | **How much of the dark-energy evidence is a choice?** The same comparison across 20 combinations of BAO set, supernova compilation, and whether `r_d` is computed or fitted. |

[`cpl_mcmc_tfd42.py`](05-case-studies/cpl_mcmc_tfd42.py) is a plain-script
version of the CPL analysis — run it with
`python examples/05-case-studies/cpl_mcmc_tfd42.py` for the full-length run
(`n_processes` gets its proper speedup as a script). Figures go to
`05-case-studies/cpl_mcmc_tfd42_figures/`, chains to
`..._chains/`, and both are reused on a re-run.

---

## Optional extras

Most notebooks need only the base install. Three need an extra, and say so
in their setup cell:

| extra | needed by | for |
|---|---|---|
| `theory` | `models_from_an_action`, `scalar_field_models` | `sympy`, for deriving Friedmann equations |
| `evidence` | `evidence_and_model_selection`, `tension_statistics` | `dynesty`, for nested sampling |
| `cmb` | `s8_tension_cmb` | `camb`, for the from-scratch CMB spectrum |

```bash
pip install -e ".[theory,evidence,cmb]"
```
