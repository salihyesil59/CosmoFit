# CosmoFit

> **Modern Cosmological Parameter Estimation in Python**

**CosmoFit** is an open-source Python library for cosmological parameter estimation and Bayesian inference. It provides a modular framework for fitting cosmological models to observational data using Markov Chain Monte Carlo (MCMC) techniques.

The project is designed to make cosmological analyses simple, reproducible, and extensible while remaining flexible for research applications.

> **Current Version:** v0.9.0

---

## Features

* Modular cosmological model framework, with curvature (Omega_k) supported end-to-end (flat/open/closed E(z) and D_M(z))

  * **LCDM** -- flat/curved ΛCDM
  * **wCDM** -- constant dark-energy equation of state w0
  * **CPL** (Chevallier-Polarski-Linder) -- w(z) = w0 + wa z/(1+z)
  * **JBP** (Jassal-Bagla-Padmanabhan) -- w(z) = w0 + wa z/(1+z)^2
  * **BA** (Barboza-Alcaniz) -- w(z) = w0 + wa z(1+z)/(1+z^2)
  * **GCG** (Generalized Chaplygin Gas) -- unified dark matter/dark energy fluid,
    p = -A/rho^alpha

* Flexible parameter management
* Built-in observational datasets

  * Cosmic Chronometers (CC)
  * BAO (DESI 2024; SDSS BOSS DR12 + eBOSS DR16 LRG/QSO) -- don't combine the two, see the note below
  * Supernova (Pantheon+, DES-SN5YR) -- don't combine the two, see the note below
  * CMB distance priors (Planck 2018 R, l_A, omega_b_h2)

* Modular likelihood architecture
* Bayesian parameter estimation with MCMC, with autocorrelation-time convergence diagnostics
* Covariance matrix support, including precision (inverse-covariance) matrices as shipped
  directly by some data releases
* Dedicated plotting module (`fitter.plots`): MCMC chain/corner plots, Hubble diagram, H(z)
  diagram, BAO distance plot, Planck pull plot, w(z) evolution, deceleration parameter

---

## Installation

Clone the repository and install CosmoFit in editable mode.

```bash
git clone https://github.com/salihyesil59/CosmoFit.git

cd CosmoFit

pip install -e .
```

Everything is importable from the top-level package:

```python
from CosmoFit import CPL, Fitter
```

---

## Project Structure

CosmoFit uses a standard `src` layout. The importable package
(`CosmoFit`) is separate from the repository root, which keeps the
package import path unambiguous and matches how the library is meant
to be used -- `pip install`ed, then imported by name, not by adding
the repo root to `sys.path`.

```text
CosmoFit/
├── src/CosmoFit/
│   ├── __init__.py    # unified public API: from CosmoFit import ...
│   ├── cosmology/     # models (LCDM, wCDM, CPL), parameters, distances, background
│   ├── data/          # dataset loaders + bundled CC/DESI/Pantheon+/Planck data files
│   ├── likelihoods/   # per-dataset chi2/likelihood classes + joint likelihood
│   ├── stats/         # Fitter (MCMC/best-fit), priors, posterior, model comparison
│   └── plots/         # FitPlotter -- every figure, attached as fitter.plots
├── examples/          # example notebooks
└── pyproject.toml
```

The subpackages remain directly importable too
(`CosmoFit.stats.model_comparison`, `CosmoFit.data.loader`, ...) for
anything not re-exported at the top level -- `from CosmoFit import ...`
is a convenience layer over them, not a replacement.

---

## Example Notebooks

All four notebooks below are Colab-ready: click a badge to open it directly in Google Colab and
*Runtime → Run all* — no local setup needed.

| Notebook | What it covers |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/quickstart.ipynb) | The shortest path to a real MCMC fit: flat ΛCDM on CC+DESI, a couple of minutes end to end. Start here. |
| [`dataset_zoo.ipynb`](examples/dataset_zoo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/dataset_zoo.ipynb) | A tour of all six built-in datasets (CC, DESI, SDSS BAO, Pantheon+, DES-SN5YR, Planck), each plotted on its own, plus which combinations to avoid. |
| [`model_zoo_comparison.ipynb`](examples/model_zoo_comparison.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/model_zoo_comparison.ipynb) | All six cosmological models (LCDM, wCDM, CPL, JBP, BA, GCG) fit to the same data and compared with AIC/BIC, plus a full MCMC for GCG. |
| [`cpl_mcmc_analysis.ipynb`](examples/cpl_mcmc_analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/cpl_mcmc_analysis.ipynb) | The deep dive: CPL fit to CC+DESI+Pantheon+, convergence diagnostics, every `fit.plots` figure, model comparison, and an independent Planck cross-check. |

---

## Quick Example

```python
from CosmoFit import CPL, Fitter

fitter = Fitter(
    model=CPL,
    datasets=["cc", "desi", "pantheon", "planck"],
    free_params=[
        "H0",
        "Omega_m",
        "w0",
        "wa",
        "rd",
    ],
    initial={
        "H0": 67.4,
        "Omega_m": 0.315,
        "w0": -1.0,
        "wa": 0.0,
        "rd": 147.1,
        "Omega_b": 0.0493,
    },
)

fitter.run_mcmc(
    nwalkers=48,
    nsteps=650,
    burnin=100,
)

fitter.best_fit()
fitter.summary()
fitter.convergence()

fitter.plots.corner()
fitter.plots.hubble_diagram()
fitter.plots.w_of_z()
```

---

## Project Status

CosmoFit is currently under active development.

Version **v0.3.0** fixes a curvature bug in the LCDM/CPL Friedmann
equations and in the transverse comoving distance (Omega_k was
previously a fittable-but-inert parameter), and adds a Planck 2018
CMB distance-prior likelihood (shift parameter R, acoustic scale
l_A, omega_b_h2), backed by a radiation-aware sound-horizon
calculation and the Hu & Sugiyama (1996) z_star fitting formula.

Version **v0.4.0** fixes a Pantheon+ distance-modulus bug (the
zHD/zHEL redshift distinction from Brout et al. 2022 was not
applied) and a circular import between `data.loader` and
`likelihoods`, roughly halves the per-step cost of an MCMC run
that includes Pantheon+, adds the `wCDM` model, adds an
autocorrelation-time MCMC convergence check
(`Fitter.convergence()`), and moves all plotting into a dedicated
`plots.FitPlotter` (`fitter.plots`), with new Hubble diagram, H(z),
BAO distance, Planck pull, w(z) evolution and deceleration-parameter
figures alongside the existing chain/corner plots.

Version **v0.5.0** moves the package to a standard `src` layout
(`src/CosmoFit/...`) with a unified public API, so `from CosmoFit
import CPL, Fitter, ...` now works instead of importing the five
subpackages (`cosmology`, `data`, `likelihoods`, `stats`, `plots`)
separately from the repository root.

Version **v0.6.0** adds three dark-energy models: **JBP** and **BA**
(alternative w0-wa parametrizations to CPL, reusing the same `w0`/`wa`
parameters) and **GCG** (Generalized Chaplygin Gas, a genuinely
different unified dark-matter/dark-energy fluid, adding two new shared
parameters `A_s`/`alpha`). All three have closed-form `E(z)`/`dE/dz`
(no per-step numerical integration), verified against finite-difference
derivatives and independent numerical integration of the continuity
equation, so they're exactly as fast in an MCMC as LCDM/wCDM/CPL.

Version **v0.7.0** adds the **DES-SN5YR** (Dark Energy Survey 5-year)
supernova likelihood, downloaded directly from the official
[des-science/DES-SN5YR](https://github.com/des-science/DES-SN5YR)
data release and cross-validated against its own reference likelihood
implementation (the distance-modulus formula and analytic
marginalization both match it exactly). Its covariance is shipped as a
precision (inverse covariance) matrix rather than a covariance matrix,
so this version also adds `PrecisionCovariance`, used directly (no
inversion round-trip) instead of forcing it through the existing
Cholesky-based `DenseCovariance`. The Pantheon+ and DES-SN5YR
marginalization logic (previously duplicated) was factored into a
shared `AnalyticOffsetMixin`.

> **Note:** don't combine `"pantheon"` and `"des_sn5yr"` in the same
> fit -- DES-SN5YR's low-z anchor sample (~11% of it) is also compiled
> into Pantheon+, so fitting both double-counts those supernovae. See
> the `DESSN5YRLikelihood` docstring.

Version **v0.8.0** adds the **SDSS BAO** likelihood: BOSS DR12
(z=0.38, 0.51) + eBOSS DR16 LRG (z=0.698) + eBOSS DR16 QSO (z=1.48),
combined into one dataset with a block-diagonal covariance (each
component is an independent, non-overlapping-redshift measurement).
Data downloaded directly from
[CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
(the same repository that is, independently, also the exact source of
this project's existing DESI 2024 files -- confirmed byte-for-byte)
and cross-checked against the published eBOSS DR16 LRG/QSO
uncertainties. BOSS DR12's usual third bin (z=0.61) is deliberately
omitted, since it overlaps the eBOSS DR16 LRG redshift range. The
DESI/SDSS-BAO `model()`/`residuals()`/`chi2()` logic (previously
duplicated in `DESILikelihood`) was factored into a shared
`BAODistanceLikelihood` base class.

> **Note:** don't combine `"desi"` and `"sdss_bao"` in the same fit --
> DESI targets much of the same sky BOSS/eBOSS did, so treating them
> as independent double-counts structure. See the
> `SDSSBAOLikelihood` docstring.

Version **v0.9.0** is a performance pass, driven by profiling rather
than guesswork. The big one: evaluating `PlanckLikelihood` was
dominated by two integrals (the sound horizon and the comoving
distance to z\*) computed with `scipy.integrate.quad` -- an *adaptive
scalar* quadrature that calls the (already fully vectorized)
integrand at one z at a time, several hundred times per call. Both
are now evaluated on a fixed, vectorized grid (`scipy.integrate.simpson`)
instead -- one array-valued call instead of hundreds of scalar ones.
Getting this right took two tries: the first grid (linear in the
substituted variable) looked fine across randomized-parameter testing
but turned out to badly under-resolve the sound-horizon integral's
approach to its asymptote at realistic (Planck-fiducial-like)
parameter values specifically -- caught by comparing end-to-end
`PlanckLikelihood` output at a fixed point against the original
`quad`-based result, not just spot-checking the integral in
isolation. A log-spaced grid fixes it, verified to <1e-6 relative
error against `quad` across dozens of randomized and
literature-realistic parameter sets, across all 6 models. **Net
effect: ~8x faster evaluation of a joint CC+DESI+Pantheon+Planck
likelihood.** The distance-integrator's interpolation grid
(`cosmology.numerics.integrals`) was also shrunk several-fold
(verified: still >100x more accurate than any dataset's measurement
precision needs) for a smaller additional gain that applies to every
fit, Planck or not.

The package structure may continue to evolve before the first stable **v1.0.0** release.

---

## Roadmap

### v0.10.0

* Dedicated sampler module
* Improved result interface

### v1.0.0

* Stable public API
* Complete documentation
* Production-ready release

---

## References

CosmoFit bundles real observational data and implements published
cosmological models and statistical methods -- see
**[REFERENCES.md](REFERENCES.md)** for every dataset, model, and
methodology paper used, with links, and where each is used in the
code. If you use CosmoFit in a publication, cite the underlying
data/method papers relevant to what you used.

---

## License

This project is licensed under the MIT License.