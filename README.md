# CosmoFit

> **Modern Cosmological Parameter Estimation in Python**

**CosmoFit** is an open-source Python library for cosmological parameter estimation and Bayesian inference. It provides a modular framework for fitting cosmological models to observational data using Markov Chain Monte Carlo (MCMC) techniques.

The project is designed to make cosmological analyses simple, reproducible, and extensible while remaining flexible for research applications.

> **Current Version:** v0.22.0

---

## Features

* Modular cosmological model framework, with curvature (Omega_k) supported end-to-end (flat/open/closed E(z) and D_M(z))

  * **LCDM** -- flat/curved ΛCDM
  * **wCDM** -- constant dark-energy equation of state w0
  * **CPL** (Chevallier-Polarski-Linder) -- w(z) = w0 + wa z/(1+z)
  * **JBP** (Jassal-Bagla-Padmanabhan) -- w(z) = w0 + wa z/(1+z)^2
  * **BA** (Barboza-Alcaniz) -- w(z) = w0 + wa z(1+z)/(1+z^2)
  * **LogarithmicDE** (Efstathiou) -- w(z) = w0 + wa ln(1+z); the one w0-wa form here
    that does *not* saturate at high z, so it is the control case for asking whether a
    measured `wa` is the data or the assumed shape
  * **PEDE** (Phenomenologically Emergent Dark Energy) -- Omega_de(z) = Omega_de0
    [1 - tanh(log10(1+z))], with **no free dark-energy parameter at all**: the same
    parameter count as LCDM and a completely different expansion history, so an AIC/BIC
    comparison against LCDM is a pure comparison of fit
  * **GEDE** (Generalized EDE) -- contains both LCDM (Delta -> 0) and PEDE
    (Delta = 1, z_t = 0) as exact limits, so `Delta` measures the distance from a
    cosmological constant on a continuous scale
  * **LsCDM** (sign-switching Lambda, Akarsu et al.) -- Lambda flips sign at
    z_dagger ~ 2 (AdS below, dS above), shrinking r_d and so raising the BAO-inferred H0;
    a route to the H0 tension that late-time-only dark-energy models cannot take
  * **GCG** (Generalized Chaplygin Gas) -- unified dark matter/dark energy fluid,
    p = -A/rho^alpha
  * **IDE** (Interacting Dark Energy) -- Q = 3 xi H rho_de in closed form; changes how
    *matter* dilutes, which no w(z) parametrization does, so it has its own
    growth-of-structure signature
  * **RunningVacuum** -- Lambda(H) = c0 + 3 nu H^2, from renormalization-group running of
    the vacuum energy; one of the few extensions whose extra parameter has a *predicted*
    magnitude (|nu| ~ 10^-3) rather than an arbitrary one

* Models where acceleration comes from a modified Friedmann equation rather than any dark
  energy at all

  * **Cardassian** (modified polytropic) -- H^2 = A rho + B rho^n, a flat, matter-dominated,
    accelerating universe
  * **DGP** (Dvali-Gabadadze-Porrati braneworld, self-accelerating branch) -- gravity leaks
    into a fifth dimension; LCDM's parameter count, and a *suppressed* growth history
    (mu ~ 0.72 today) that is the real observational handle on it

* Modified-gravity models, at both background *and* growth-of-structure level (see Project
  Status for exactly what's real vs. simplified in each)

  * **FQExponential** -- f(Q) gravity (symmetric teleparallel), f(Q) = Q exp(lambda Q0/Q)
  * **FRTLinear** -- f(R,T) gravity (linear), f(R,T) = R + 2 lambda T
  * **FRHuSawicki** -- f(R) gravity (Hu-Sawicki); background is identical to LCDM's by
    construction (stated explicitly, see Project Status), but growth of structure -- where
    this model's `f_R0`/`n` parameters actually show up -- is now real: a scale- and
    time-dependent, chameleon-screened `mu(a,k)`

* Growth of structure: every model (not just the three above) gets a linear growth factor
  D(z), growth rate f(z), and fsigma8(z)/S8 via a generic `mu(a,k)` hook on top of its own
  E(z)/dE(z)dz (`cosmology.calculators.growth.GrowthCalculator`) -- `mu = 1` (standard GR
  growth) for LCDM/wCDM/CPL/JBP/BA/GCG, a real derived `mu(a,k)` for the three
  modified-gravity models above
* **The BAO sound horizon `r_d` computed from scratch** (`Fitter(..., compute_rd=True)`)
  rather than fitted as a free nuisance parameter -- the integral
  `r_d = int c_s/H dz` with photons, massless neutrinos and massive neutrinos carrying
  their exact Fermi-Dirac energy density, validated against CAMB's `rdrag` to 5e-5. This
  is what turns BAO from a *relative* distance measurement (which constrains only
  `H0 * r_d`) into an absolute one, and it is how "BAO + BBN gives H0" works
* Flexible parameter management
* Built-in observational datasets

  * Cosmic Chronometers (CC)
  * BAO (**DESI DR2 2025** or DESI DR1 2024; SDSS BOSS DR12 + eBOSS DR16 LRG/QSO;
    **low-z 6dFGS + SDSS DR7 MGS**) -- the low-z pair is independent of the other two and
    can join either; DESI and SDSS cannot be combined with each other, see the note below
  * Supernova (Pantheon+, DES-SN5YR, **Union3**) -- one per fit, see the note below
  * CMB, two ways: the compressed distance priors (Planck 2018 R, l_A, omega_b_h2 -- fast,
    works for every model) or the **full Planck 2018 TT/TE/EE spectra** (613 `plik_lite`
    bandpowers computed from scratch with CAMB -- slow, LCDM and w(z) models only)
  * Growth rate fsigma8(z) (Gold-2018 RSD compilation, 22 points)
  * S8 weak-lensing prior (KiDS-1000 or DES Y3, Gaussian) -- don't combine the two versions,
    see the note below
  * **External single-number measurements**, entering as datasets rather than as priors so
    they show up in the chi2 breakdown and the degrees-of-freedom count: local **H0**
    (SH0ES 2022/2024, or TDCOSMO 2025 time-delay lensing -- independent of the Cepheid
    ladder), a **BBN** constraint on omega_b h^2 (Schoeneberg 2024 or Cooke 2018), and the
    Planck lowE **tau** prior

* Modular likelihood architecture
* Bayesian parameter estimation with MCMC, with autocorrelation-time convergence diagnostics
* Dedicated sampling backend (`stats.sampler`), decoupled from `Fitter` and swappable (custom
  `emcee` moves today, room for other backends later)
* Multi-core MCMC (`fitter.run_mcmc(n_processes=...)`): evaluates walkers across multiple CPU
  cores, each worker building its own `Fitter` once rather than repeatedly shipping the whole
  (potentially large) likelihood across processes
* Consolidated result object (`fitter.result`): best-fit + MCMC posterior in one printable,
  JSON-serializable snapshot
* Saved MCMC chains (`fitter.run_mcmc(save="chains/fit.h5")`): the chain is written to HDF5 as
  it is sampled and reused instead of re-sampled next time, so reopening a notebook, adding a
  plot, or extending a run costs seconds rather than hours -- an interrupted run keeps every
  step it had already taken, and `Fitter.from_chain(...)` reopens a finished one in a later
  session with no configuration to retype
* Custom models (`define_model`): fit a brand-new, not-in-the-library `E(z)` -- with its own
  extra parameters -- against every built-in dataset/likelihood/MCMC, no library changes needed
* Graphical interface (`app/streamlit_app.py`, `pip install -e ".[gui]"`): tick datasets, configure
  one or more models (built-in or your own), edit free parameters, and run the MCMC fit(s) + plots
  with one click, no code -- compare models side by side statistically (AIC/BIC/a likelihood-ratio
  test) and on the same figures, all from the browser, with every figure downloadable as SVG, PNG,
  or PDF. It is also **self-explaining**: every dataset and model carries a note saying what it
  measures, over what redshift range, what it constrains and where it comes from; six presets
  configure a whole analysis in one click; the parameter table shows only the parameters *this*
  fit actually uses; and the app warns about combinations that will not work (or, worse, will
  quietly produce a posterior for something the data cannot constrain) before you run them
* Covariance matrix support, including precision (inverse-covariance) matrices as shipped
  directly by some data releases
* Dedicated plotting module (`fitter.plots`): MCMC chain/corner plots, Hubble diagram, H(z)
  diagram, BAO distance plot, Planck pull plot, w(z) evolution, deceleration parameter, and the
  w0-wa dark-energy plane (2D credible contours over the phantom/quintessence/quintom regions,
  with ΛCDM marked) -- the headline figure of the DESI evolving-dark-energy results
* Publication-ready figure text: every axis, title, legend and tick renders as LaTeX
  (`$\Omega_b$`, `$r_d$`, `$\Lambda$CDM`, `$f(R)$ Hu-Sawicki`) rather than as Python
  identifiers, corner-plot titles quote each parameter to its own precision, and each label
  states the quantity actually plotted (`$m_B$` with `$M_B$` marginalized, `$D_V/r_d$` with the
  `r_d` the code really divides by)
* Model comparison plots (`fitter.plots.compare_*`): any of the figures above, overlaying this
  fit's curve with one or more other models' curves on the same data/axes -- defaults to this
  model vs. a quick LCDM reference, or an arbitrary N-model comparison via `other_fits=[...]`
* Derived-quantity posteriors (`CosmoFit.stats.derived`): the acceleration transition
  redshift `z_t` (where q(z) changes sign) and today's deceleration parameter `q0`, with
  proper error bars -- every posterior sample is pushed back through the model's own
  E(z)/dE(z)dz rather than the quantity being evaluated once at the best fit, and it works
  for every model, not just CPL (whose w0/wa-specific diagnostics live in
  `stats.cpl_diagnostics`)

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
│   ├── stats/         # Fitter (MCMC/best-fit), priors, posterior, saved chains, model comparison
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

All seven notebooks below are Colab-ready: click a badge to open it directly in Google Colab and
*Runtime → Run all* — no local setup needed.

| Notebook | What it covers |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/quickstart.ipynb) | The shortest path to a real MCMC fit: flat ΛCDM on CC+DESI, a couple of minutes end to end. Start here. |
| [`dataset_zoo.ipynb`](examples/dataset_zoo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/dataset_zoo.ipynb) | A tour of all eight built-in datasets (CC, DESI, SDSS BAO, Pantheon+, DES-SN5YR, Planck, fsigma8, S8), each plotted on its own, plus which combinations to avoid. |
| [`model_zoo_comparison.ipynb`](examples/model_zoo_comparison.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/model_zoo_comparison.ipynb) | All six background-expansion models (LCDM, wCDM, CPL, JBP, BA, GCG) fit to the same data and compared with AIC/BIC, plus a full MCMC for GCG. |
| [`modified_gravity_growth.ipynb`](examples/modified_gravity_growth.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/modified_gravity_growth.ipynb) | f(Q)/f(R,T)/f(R) modified gravity end to end: background E(z), mu(a,k), fsigma8/S8, and a real MCMC showing FRHuSawicki's f_R0 going from completely unconstrained (background-only) to genuinely constrained (growth data). |
| [`cpl_mcmc_analysis.ipynb`](examples/cpl_mcmc_analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/cpl_mcmc_analysis.ipynb) | The deep dive: CPL fit to CC+DESI+Pantheon+, convergence diagnostics, every `fit.plots` figure, model comparison, and an independent Planck cross-check. |
| [`cpl_mcmc_tfd42.ipynb`](examples/cpl_mcmc_tfd42.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/cpl_mcmc_tfd42.ipynb) | Publication-scale variant of the above: CPL fit to all four datasets *jointly* (Planck included, making `rd`/`Omega_b` constrainable), a much longer chain, and multi-core MCMC (`n_processes`). |
| [`lscdm_mcmc.ipynb`](examples/lscdm_mcmc.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/dev/examples/lscdm_mcmc.ipynb) | A real research question, end to end: does Akarsu et al.'s **ΛsCDM** — a cosmological constant that switches sign at z† — still relieve the H0 tension against the 2024–2025 data (DESI DR2 BAO + DES-SN5YR + Planck priors + BBN, with `r_d` computed rather than fitted)? Includes a profile likelihood over z†, which locates the answer in a single measurement. |

[`cpl_mcmc_tfd42.py`](examples/cpl_mcmc_tfd42.py) is a plain-script version of the same analysis --
run it with `python examples/cpl_mcmc_tfd42.py` for the actual long run (`n_processes` gets its full
speedup as a script; inside a live Jupyter kernel it currently doesn't -- see Project Status below).
Results print to stdout as they run, every figure is saved as an SVG into
`examples/cpl_mcmc_tfd42_figures/`, and the numeric results to `examples/cpl_mcmc_tfd42_result.json`.
Both chains are saved to `examples/cpl_mcmc_tfd42_chains/` and reused on a re-run (see
[Saved Chains](#saved-chains)), so running it a second time -- or picking up after a Ctrl-C --
skips straight past the MCMC. The notebook version does the same.

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

print(fitter.result)

fitter.plots.corner()
fitter.plots.hubble_diagram()
fitter.plots.w_of_z()
```

---

## Saved Chains

The MCMC is the expensive part of a fit; everything built on top of it -- summaries,
convergence diagnostics, corner plots, derived quantities, model comparison -- is seconds.
Add `save=` and the chain goes to an HDF5 file as it is sampled, so the expensive part
happens once:

```python
fitter.run_mcmc(
    nwalkers=48,
    nsteps=6000,
    burnin=1000,
    save="chains/cpl.h5",
)
```

Run that same code again -- next week, in a new session, after adding three more plots below
it -- and nothing is re-sampled: the stored chain is read back and `summary()`,
`convergence()`, `best_fit()` and every `plots.*` figure work off it exactly as before.

`nsteps` is the *total* chain length to end up with, which is what makes re-running a script
or a notebook cell idempotent:

| what you do | what happens |
|---|---|
| run it again unchanged | nothing is sampled; the stored chain is loaded |
| raise `nsteps` to 10000 | only the missing 4000 steps are sampled, continuing the same chain |
| Ctrl-C halfway through | every step taken so far is already on disk; run again to carry on |
| change the model, datasets, free parameters, or priors | refused, loudly, naming what differs -- samples from two different posteriors must never be merged |

A resumed chain is bit-identical to the uninterrupted one it would have been: emcee's proposal
RNG state travels in the file with the walkers, so `nsteps=6000` in one go and `1500 -> 6000`
in four sittings give the same samples.

### Reopening a chain later

`Fitter.from_chain` rebuilds the whole fit from what the file records -- model, datasets, free
parameters, priors, fixed values -- with nothing to retype and nothing that can drift out of
sync with the samples:

```python
from CosmoFit import Fitter

fit = Fitter.from_chain("chains/cpl.h5")

fit.summary()
fit.plots.corner()
fit.best_fit()          # the datasets are live again, so this works too
```

For posterior summaries alone, skip the fitter entirely -- this reads no dataset and evaluates
no likelihood:

```python
from CosmoFit.stats.chains import open_chain, chain_info

chain = open_chain("chains/cpl.h5")
chain.summary()
chain.samples_dict()["w0"]

chain_info("chains/cpl.h5")   # model, datasets, parameters, steps, when it was run
```

### Details worth knowing

* **The file is self-describing.** Alongside emcee's own `mcmc` group (plain
  `h5py`/`emcee` can read it with or without CosmoFit) sits a `cosmofit` group recording the
  model, datasets, free parameters, prior bounds, fixed parameter values, burn-in, seed and
  versions. That record is what makes a resume safe, and what `from_chain` rebuilds from.
* **`resume`** defaults to `"auto"` (continue if there's a chain, start one if not). Pass
  `resume=True` to *require* an existing chain -- "analyze last night's run, don't quietly
  start a fresh 12-hour one if the file moved" -- or `resume=False` to discard what's stored
  and sample from scratch. That last one is the only destructive option and is never reached
  by default.
* **Naming files automatically.** `fitter.chain_id()` is a short, stable hash of the
  posterior (`"CPL_3f9a1c04"`), identical across sessions and machines for the same fit and
  different as soon as anything about it changes:

  ```python
  fit.run_mcmc(nsteps=6000, save=f"chains/{fit.chain_id(nwalkers=48)}.h5")
  ```

  Saving under it reuses a chain exactly when reuse is correct, and starts a separate file --
  rather than colliding with, or overwriting, the old one -- when it isn't. This is how the
  GUI keeps one chain per configuration.
* **Several chains in one file.** Pass a `ChainFile` instead of a path to put a model
  comparison's chains side by side:

  ```python
  from CosmoFit.stats.chains import ChainFile

  fit_cpl.run_mcmc(nsteps=6000, save=ChainFile("chains/comparison.h5", name="CPL"))
  fit_lcdm.run_mcmc(nsteps=6000, save=ChainFile("chains/comparison.h5", name="LCDM"))
  ```

---

## The w0-wa Plane

`fitter.plots.w0_wa_plane()` draws the figure the recent
evolving-dark-energy literature is built around: the 2D posterior of the CPL parameters over the
four dark-energy regions of the (w0, wa) plane, with ΛCDM marked at (-1, 0).

```python
from CosmoFit import CPL, Fitter

fit = Fitter(
    model=CPL,
    datasets=["cc", "desi", "pantheon"],
    free_params=["H0", "Omega_m", "w0", "wa"],
    initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "wa": 0.0, "rd": 147.1},
)
fit.run_mcmc(nwalkers=32, nsteps=15000, burnin=3000, save="chains/cpl.h5")

fit.plots.w0_wa_plane()
```

The plane is cut by two lines -- `w0 = -1` (the equation of state today) and `wa = -1 - w0`
(its high-z limit `w0 + wa`) -- into the four cases the classification is made of:

| region | w(z) | meaning |
|---|---|---|
| **Phantom** | below -1 at all times | ρ grows with expansion; a Big Rip future |
| **Quintessence** | above -1 at all times | reachable by an ordinary canonical scalar field |
| **Quintom-A** | quintessence in the past, phantom today | crosses w = -1 |
| **Quintom-B** | phantom in the past, quintessence today | crosses w = -1 the other way |

The crossing is the physically loaded part: no single canonical (or single phantom) scalar field
can cross `w = -1` at all, so a posterior sitting in either quintom region calls for something
more -- two fields, a non-canonical kinetic term, or modified gravity.

Useful options:

```python
fit.plots.w0_wa_plane(show_fractions=True)          # posterior probability of each region
fit.plots.w0_wa_plane(levels=(0.68, 0.95, 0.997))   # add a third contour
fit.plots.w0_wa_plane(w0_range=(-1.3, 0.0), wa_range=(-3, 1))
fit.plots.w0_wa_plane(bins=120, smooth=1.0)         # finer density estimate
```

`levels` are 2D credible **probabilities** (the smallest area holding that fraction of the
samples), not sigmas -- in two dimensions the familiar "1σ"/"2σ" contours enclose only 39% and
86.5%, and conflating the two is the usual way this figure gets over-read.

The same numbers without the picture:

```python
from CosmoFit.stats import cpl_diagnostics

cpl_diagnostics.region_fractions(*[fit.samples_dict()[k] for k in ("w0", "wa")])
# {'phantom': 0.004, 'quintessence': 0.221, 'quintom-a': 0.012, 'quintom-b': 0.763}

cpl_diagnostics.classify_region(-0.85, -0.6)   # 'quintom-b'
```

To overlay several dataset combinations -- the "what does adding this data do to w0-wa?"
figure -- every fit needs its own chain, and then:

```python
fit_desi.plots.compare_w0_wa_plane(other_fits=[fit_desi_sn, fit_desi_sn_cmb])
```

Each contour set labels itself with its own dataset combination (`CC + DESI + Pantheon+`);
pass `labels=[...]` to override.

Both methods need `w0` and `wa` to be free parameters of a completed MCMC, and a model whose
w(z) really does tend to `w0 + wa` at high z (CPL, BA). For JBP -- whose w(z) returns to `w0`
instead -- the diagonal boundary would not be that model's own limit, so they refuse rather than
mislabel the regions; use `plots.w_of_z()` there.

---

## Model Comparison Plots

Every `fitter.plots.*` figure above has a `compare_*` counterpart that overlays this fit's curve
with one or more other models' curves on the same data/axes -- the "model A vs model B" figures
cosmology papers use (`compare_hz`, `compare_deceleration`, `compare_w_of_z`,
`compare_hubble_diagram`, `compare_des_hubble_diagram`, `compare_bao_distances`,
`compare_sdss_bao_distances`):

```python
fitter.plots.compare_hz()             # vs. a quick LCDM reference, built automatically
fitter.plots.compare_deceleration()   # ditto, each curve's own transition redshift z_t marked
```

Called with no arguments, `other_fits` defaults to a quick best-fit-only LCDM reference (no MCMC
-- just enough for a comparison curve) built from this fit's own datasets, unless this fit already
*is* LCDM. Pass an already-fit `Fitter` (or a list of them, for more than two models at once) to
compare against instead:

```python
fitter.plots.compare_hz(other_fits=fit_lcdm, labels=["CPL", "LCDM"])
fitter.plots.compare_hz(other_fits=[fit_lcdm, fit_wcdm], labels=["CPL", "LCDM", "WCDM"])
```

This works for any model CosmoFit knows about, built-in or [custom](#custom-models).

---

## Custom Models

Testing a model that isn't in the literature -- and isn't one of
CosmoFit's six built-ins -- doesn't require touching the library's
internals. `define_model` builds a usable model from a single
`E(z)` function (that alone is enough to fit against every dataset
and produce every plot except `w_of_z()`/`deceleration()`); any new
parameter it needs beyond the standard set (`H0`, `Omega_m`,
`Omega_k`, `w0`, `wa`, ...) is declared right there, with a default
and prior bounds:

```python
from CosmoFit import define_model, Fitter
import numpy as np

MyModel = define_model(
    "MyModel",
    E=lambda p, z: np.sqrt(
        p["Omega_m"] * (1 + z) ** 3
        + (1 - p["Omega_m"]) * (1 + z) ** (3 * (1 + p["w0"])) * (1 + p["beta"] * z)
    ),
    extra_params={"beta": {"default": 0.0, "bounds": (-2.0, 2.0), "label": r"$\beta$"}},
)

fit = Fitter(
    model=MyModel,
    datasets=["cc", "desi", "pantheon"],
    free_params=["H0", "Omega_m", "w0", "beta"],
    initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "beta": 0.0},
)
fit.run_mcmc(nwalkers=48, nsteps=3000, burnin=500)
fit.best_fit()
print(fit.result)
fit.plots.corner()
```

For more control, the same mechanism is available by subclassing
`Cosmology` directly and declaring `EXTRA_PARAMS` on the class --
`define_model` is a thin convenience wrapper around exactly this:

```python
from CosmoFit import Cosmology

class MyModel(Cosmology):
    EXTRA_PARAMS = {"beta": {"default": 0.0, "bounds": (-2.0, 2.0)}}

    def E(self, z):
        z = np.asarray(z, dtype=float)
        return np.sqrt(
            self.Omega_m * (1 + z) ** 3
            + (1 - self.Omega_m) * (1 + z) ** (3 * (1 + self.w0)) * (1 + self.beta * z)
        )
```

---

## Graphical Interface

For datasets/models/parameters by clicking rather than coding, a
[Streamlit](https://streamlit.io) app (`app/streamlit_app.py`) sits
on top of the exact same public API as above -- it builds a
`Fitter` and calls `run_mcmc()`/`best_fit()`/`fit.plots.*` for you.

```bash
pip install -e ".[gui]"
streamlit run app/streamlit_app.py
```

Or, without touching a terminal: double-click `run_gui.sh` (Linux/macOS)
or `run_gui.bat` (Windows) in the repository root. Either installs
whatever's missing on first run and then opens the app in your
browser; safe to double-click again any time to relaunch it.

Chains are saved and reused (the "Saved chains" box in the sidebar,
on by default): add a second model to compare against and the first
one comes back instantly instead of being re-sampled, raise Steps and
only the extra steps are run, close the app and it's all still there.
Each distinct configuration gets its own file (see
[Saved Chains](#saved-chains)), so nothing is reused when it
shouldn't be.

It lets you: tick which built-in datasets to fit; pick one of the
seventeen built-in models, or write a custom one directly as an
`E(z)` expression (e.g. `sqrt(Omega_m*(1+z)**3 + ...)`, using the
same mechanism as `model_from_expression()` below) with your own
extra parameters; tick which parameters are free and edit their
initial values/bounds in a table; and, with one click, run the MCMC
fit and render whichever result plots apply to your model/dataset
choice. Results (best fit, posterior summary, convergence) can be
downloaded as JSON via `FitResult.save_json()`.

**It also explains itself.** Fourteen datasets and seventeen models
is a lot to face cold, and a wrong combination does not announce
itself -- it produces a perfectly ordinary-looking posterior. So:

* **Every dataset carries a note** -- what it measures, over what
  redshift range, how many points, what it actually constrains, and
  the paper it comes from. They are grouped by probe (expansion rate,
  BAO, supernovae, CMB, growth, external measurements), because a fit
  is normally built by taking one from each family rather than by
  ticking everything.
* **Every model carries a note** -- what it is, what its extra
  parameters mean, and *which parameter values reduce it to ΛCDM*,
  which is the number an AIC/BIC comparison is measuring the distance
  from. Capability badges say whether it has its own `w(z)`, whether
  it modifies growth, and whether the full CMB spectra can be
  computed for it.
* **Six presets** configure a whole analysis in one click --
  including "DESI DR2 + BBN → H₀ without the CMB", which sets the
  datasets, picks DR2, and turns on the computed sound horizon
  together, since none of those three is useful without the others.
* **The parameter table shows only what this fit uses.** The shared
  container carries every parameter any model needs; for an LCDM fit
  against CC and BAO all but four are inert. Relevance depends on the
  datasets too -- `rd` appears only with BAO, `sigma8` only with
  growth data, `n_s`/`tau` only with the CMB spectra.
* **It warns before you run, not after.** A model the CMB spectra
  cannot be computed for is an error you would otherwise meet as a
  stack trace minutes in. The quieter ones matter more: a
  modified-gravity model with no growth data ticked, a model whose
  own parameters are left fixed, a free `sigma8` that nothing in the
  fit constrains. Each of those runs, converges, and returns a
  posterior that looks like a measurement.
* **A χ² breakdown per dataset** on the results tab, which is what
  turns a total χ² into "the local H₀ measurement is contributing 24
  of it, on one data point" -- the entire content of a tension.
* **Derived quantities** (`q₀`, `z_t`, and `r_d` computed from the
  densities) with real error bars, pushed sample by sample back
  through the model's own `E(z)`.
* A **📖 Guide** panel listing every dataset and model in one
  browsable table, with the conflict rules and a short "how to use
  this" walkthrough.

The custom-model expression box evaluates with `eval()` but with
Python's builtins removed and only whitelisted `numpy` math plus the
model's own parameter names reachable -- appropriate for running
locally on your own machine, not for a publicly hosted, multi-tenant
deployment.

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

Version **v0.10.0** splits MCMC sampling out of `Fitter` and into a
dedicated `stats.sampler` module: `EnsembleSampler` (the existing
`emcee`-backed walker initialization/run logic, unchanged in
behavior) now implements a small `BaseSampler` interface, so the
sampling backend is a swappable component rather than logic
inlined in `Fitter.run_mcmc`, and `run_mcmc()` gains a `moves=`
argument to pass through custom `emcee` proposals (e.g.
`emcee.moves.DEMove()` for strongly correlated posteriors). It also
adds a consolidated result interface (`stats.results`): `fitter.result`
returns a `FitResult` bundling the best-fit point (`BestFitResult`)
and MCMC posterior summary/convergence (`MCMCResult`) that were
previously only available piecemeal via `best_fit_params`,
`best_fit_chi2`, `summary()`, `convergence()`, with a single
readable `repr` and `FitResult.save_json()` / `.load_json()` for
keeping a fit's headline numbers without pickling the `emcee`
sampler. Both are purely additive -- every existing `Fitter`
method/attribute (`run_mcmc()`, `best_fit()`, `summary()`,
`convergence()`, `samples_dict()`, `.sampler`, `.best_fit_result`,
...) is unchanged.

Version **v0.11.0** adds support for custom, not-in-the-literature
models: `define_model()` (`cosmology.custom`) builds a usable
`Cosmology` subclass from a single `E(z)` function -- which alone is
enough to fit against every built-in dataset/likelihood and produce
every plot except `w_of_z()`/`deceleration()` (an optional `dEdz=`
enables the latter, else a numerical finite-difference fallback is
used) -- plus any new parameters the model needs beyond the standard
set (`H0`, `Omega_m`, `Omega_k`, `w0`, `wa`, ...), each declared
inline with a default and prior bounds. The same mechanism is also
available by subclassing `Cosmology` directly and declaring an
`EXTRA_PARAMS` class attribute (`Cosmology.__init_subclass__` builds
a matching parameter dataclass and property automatically); this is
what `define_model` does under the hood. Required generalizing
`Fitter` to read the parameter container off the model
(`model.PARAMS_CLASS`) instead of hardcoding
`CosmologyParameters` -- every built-in model is unaffected (none
declare `EXTRA_PARAMS`), verified against all six.

Version **v0.12.0** adds a graphical interface: a
[Streamlit](https://streamlit.io) app (`app/streamlit_app.py`,
optional `pip install -e ".[gui]"`) for ticking datasets, picking a
built-in model or writing a custom `E(z)` as an expression, editing
free parameters/bounds in a table, and running the fit + rendering
plots with one click -- a pure UI layer over the existing
`Fitter`/`FitPlotter` API, with no changes to either. The one new
library addition is `model_from_expression()`
(`cosmology.custom`), a thin wrapper around `define_model()` that
takes `E`/`w`/`dEdz` as expression strings (e.g.
`"sqrt(Omega_m*(1+z)**3 + ...)"`) instead of Python callables --
evaluated with builtins stripped and only whitelisted `numpy` math
plus the model's own parameters reachable, appropriate for a
locally-run tool.

Version **v0.13.0** adds model comparison plots: every
`fitter.plots.*` figure (`hz`, `deceleration`, `w_of_z`,
`hubble_diagram`, `des_hubble_diagram`, `bao_distances`,
`sdss_bao_distances`) gets a `compare_*` counterpart that overlays
this fit's curve with one or more other models' curves on the same
data/axes -- the "model A vs model B" figures cosmology papers use,
rather than only being able to look at models one at a time or
compare them statistically (AIC/BIC/LRT) without a picture of what
the difference actually looks like. `other_fits=None` (the default)
auto-compares against a quick best-fit-only LCDM reference built
from the same datasets; passing an already-fit `Fitter`, or a list
of them, compares against exactly those instead, for an arbitrary
N-model figure. Every `compare_*` method reuses the corresponding
single-model method's existing evaluation logic (posterior-predictive
bands, per-fit analytic SN offsets, ...) rather than duplicating it,
and works for any model -- built-in or
[custom](#custom-models) -- since it only depends on
`Fitter`/`Cosmology`'s existing generic interface. The
`cpl_mcmc_analysis` notebook's CPL vs. LCDM section now includes
these alongside its existing AIC/BIC/likelihood-ratio comparison.

Version **v0.14.0** adds multi-core MCMC: `fitter.run_mcmc(n_processes=...)`
evaluates the ensemble's walkers across multiple CPU cores. This
isn't emcee's own naive recipe (pairing a raw `multiprocessing.Pool`
with `Fitter`'s already-built, data-carrying log-posterior) -- for a
dataset like Pantheon+ (a ~1600x1600 dense covariance matrix),
re-pickling and sending that to a worker on every single step
measured *slower* than one process (~19ms to pickle vs. ~2ms to
evaluate). Instead, each worker process builds its own `Fitter` once
(from a small, cheap-to-pickle recipe), via the pool's `initializer`,
and only a length-`ndim` float vector crosses the process boundary
per evaluation after that; workers also pin their own BLAS thread
pool to 1 (via the new `threadpoolctl` dependency) to avoid
oversubscribing the machine. Net effect, measured on an 8-core
machine for a 4-dataset CPL fit: ~2.4x, well under `n_processes`x
(per-step IPC has its own cost) but a real, significant speedup. Only
works for models picklable by reference (every built-in model; not a
dynamically-built `define_model()`/`model_from_expression()` model,
which raises a clear error rather than an obscure pickling failure
if `n_processes` is given), and is most reliable on Linux/macOS
(multiprocessing from a Windows notebook is fragile for reasons
outside this library's control). The new `examples/cpl_mcmc_tfd42.ipynb`
notebook is a publication-scale variant of `cpl_mcmc_analysis.ipynb`
using this: all four datasets (including Planck, which makes `rd`
and `Omega_b` constrainable and lets them join the free parameters
too) and a much longer chain (`nwalkers=64`, `nsteps=12000`), run in
parallel across every available core.

Version **v0.15.0** fixes `n_processes` on Python 3.14: it changed
`multiprocessing`'s default start method on Linux from `fork` to
`forkserver` (matching what macOS/Windows already used), which made
`run_mcmc(n_processes=...)` crash outright as a plain script
(`forkserver`/`spawn` require every process-spawning call to sit
behind an `if __name__ == "__main__":` guard) and, even inside a
Jupyter kernel where that particular crash doesn't surface, measured
no speedup at all. `Fitter._mcmc_pool` now explicitly requests the
`fork` context rather than relying on whatever the interpreter's
default happens to be -- still available on Linux/macOS even where
it's no longer automatic, and with neither of the above problems.
This version also brings the graphical interface up to parity with
`cpl_mcmc_tfd42.ipynb`: the GUI can now configure and fit multiple
models at once (built-in or custom) sharing one set of datasets, with
a statistical comparison tab (AIC/BIC, and a likelihood-ratio test
when exactly two models are properly nested), every `compare_*`
figure alongside the existing single-model ones, and the CPL-family
w(z)=-1-crossing/LCDM-distance posterior diagnostics -- so everything
that notebook does is now also reachable with zero code. Every figure
also gets a download button (SVG/PNG/PDF, picked once per session and
applied to all of them) -- the browser's own save dialog is what lets
you choose where it goes.

**Resolved (v0.18.0): the "no speedup inside Jupyter" limitation was
a misdiagnosis.** Multiprocessing was never the problem, and it was
never specific to notebooks. The real cause was the *per-evaluation*
cost: every likelihood evaluation solved the Pantheon+ covariance
with a Cholesky triangular solve (`cho_solve`), and a triangular
solve is an inherently sequential recurrence -- each element depends
on the one before it -- so BLAS cannot thread it and worker processes
contend on memory bandwidth instead of scaling. The whole MCMC was
therefore pinned near one core's throughput in *every* environment;
a plain script only looked better because that is where
`n_processes` was actually being passed.

The covariance is constant, so `DenseCovariance` now precomputes an
explicit inverse once (validated against the original matrix, and
falling back to the Cholesky path if it fails that check) and
`solve()` is a symmetric mat-vec, which BLAS *does* thread. Measured
on the bundled 1624x1624 Pantheon+ covariance:

| | per solve | 8-process scaling |
|---|---|---|
| `cho_solve` (before) | 1.70 ms | 4.8x |
| mat-vec, 1 BLAS thread | 0.80 ms | 7.5x |
| mat-vec, threaded BLAS | 0.18 ms | -- |

End result for a 3-dataset CPL chain on an 8-core machine: the
single-process path went from 1.1x to **7.9x** core utilization and
roughly halved in wall time, *without any multiprocessing at all* --
and it measures the same in a plain script, in Jupyter Lab, in VS
Code, and under `nbconvert`. chi2 is unchanged to ~1e-11 relative.

`n_processes` now also defaults to `"auto"`, so notebooks get
multi-core behaviour without passing anything: it uses every core the
process is *allowed* to run on (`os.sched_getaffinity`, not
`os.cpu_count()` -- the two differ inside a container, cgroup, or
SLURM allocation, and oversizing the pool there makes things slower),
but only when the run is long enough to earn back worker startup, and
it silently stays single-process for a `define_model()` model rather
than raising. The chain is unaffected: the proposal RNG lives in the
main process, so a given `seed` gives bit-identical results at any
`n_processes` (verified).

Version **v0.16.0** adds modified-gravity models -- **FQExponential**
(f(Q) gravity), **FRTLinear** (f(R,T) gravity), and **FRHuSawicki**
(f(R) gravity) -- a real category beyond dark-energy-on-top-of-GR
reparametrizations (LCDM/wCDM/CPL/JBP/BA) or a unified fluid (GCG):
these modify the gravitational field equations themselves.
**Background (E(z)) level only** -- CosmoFit's datasets (CC, BAO,
SNe, Planck distance priors) are all background/expansion-history
probes, and growth-of-structure data (fσ8/RSD) plus the perturbation
machinery to use it isn't implemented, so that's the honest ceiling
of what's testable here for now. Both FQExponential's and FRTLinear's
Friedmann equations were derived and verified directly against
primary sources (Anagnostopoulos, Basilakos & Saridakis 2021,
arXiv:2104.15123, for f(Q); Harko, Lobo, Nojiri & Odintsov 2011,
arXiv:1104.2669, for f(R,T)) rather than taken from a single
secondary source -- one transcription (a sign error in FQExponential's
Lambert-W closure formula) was caught this way, by a numerical
closure self-check (`E(z=0)` should equal 1 by construction; it
didn't, until the sign was fixed) that's now a permanent part of the
model's test coverage. **FRHuSawicki's background is identical to
LCDM's by construction** -- the standard "designer f(R)" approach
builds f(R) to reproduce an assumed target background, so `f_R0`/`n`
are present as parameters but don't affect `E(z)` at all; fitting
them against these datasets won't meaningfully constrain them. This
is stated explicitly in the class docstring and shown as a visible
warning next to the model picker in the GUI -- included for
completeness rather than silently omitted, but not silently
overstated either. All three plug into the existing `EXTRA_PARAMS`
mechanism (built for `define_model()`/custom models), so `Fitter`,
every plot including `compare_*`, and the GUI's multi-model
comparison all work with them with no further changes.

Version **v0.17.0** adds growth-of-structure: the "phase 2" v0.16.0
explicitly deferred, since a background-only treatment left
FRHuSawicki's `f_R0`/`n` genuinely untestable (its background *is*
LCDM's by construction). A new `cosmology.calculators.growth.GrowthCalculator`
solves the standard sub-horizon, quasi-static linear growth ODE
(`D'' + (2+dlnH/dN)D' - (3/2)Omega_m(a) mu(a,k) D = 0`) for every
model via a `Cosmology.mu(a,k)` hook (default 1, i.e. standard GR
growth -- LCDM/wCDM/CPL/JBP/BA/GCG all get this for free), exposing
`fitter.cosmology.background.{growth_rate,sigma8,fsigma8}(z)`, plus
a new `sigma8` cosmological parameter and two new datasets/likelihoods:
`"fsigma8"` (the "Gold-2018" RSD growth-rate compilation, Sagredo,
Nesseris & Sapone 2018, arXiv:1806.10822, 22 points, with the same
Alcock-Paczynski correction the reference likelihood applies) and
`"s8"` (a single Gaussian S8 = sigma8*sqrt(Omega_m/0.3) constraint,
KiDS-1000 or DES Y3). Each of the three modified-gravity models now
has a real, cited `mu(a,k)`: **FQExponential** uses the settled
sub-horizon result G_eff/G_N = 1/f_Q (Barros, Barreiro, Koivisto &
Nunes 2020, arXiv:2004.07867); **FRTLinear** uses
`mu = 1 + 3*beta` (the same coupling already in its own `E(z)^2`,
stated explicitly as a simplification rather than a full covariant
perturbation derivation -- see the class docstring); and
**FRHuSawicki** gets the standard chameleon-screened, scale- and
time-dependent `mu(a,k)` (Pogosian & Silvestri 2008, arXiv:0709.0296),
derived here directly from this model's own background rather than
transcribed, held at a fixed fiducial pivot `k=0.1` h/Mpc since
CosmoFit's fsigma8 data are single per-z points, not a P(k) shape --
`f_R0`/`n` are still inert for background-only fits, but now
genuinely shape `"fsigma8"`/`"s8"` predictions (verified end-to-end:
a short FRHuSawicki fit against `"fsigma8"` alone now gives `f_R0` a
real, visibly informative posterior, not a flat one spanning the
whole prior). `fitter.plots.growth()`/`compare_growth()` add the
fsigma8(z) diagram alongside the existing figures, and the GUI picks
up both new datasets/the new plot automatically through the same
`DATASET_REGISTRY`/`EXTRA_PARAMS` mechanism every other dataset/model
already goes through.

Version **v0.19.0** fixes a real scientific error in the Planck
distance-prior likelihood. Evaluated at *Planck's own best-fit
LCDM* -- where a correct implementation must return chi2 ~ 0 -- the
old code returned **chi2 ~ 100 for 3 data points**, with `l_A` off by
**-8.9 sigma** (`R` and `omega_b_h2` were fine at 0.13 and 0.07
sigma, which is what localized it to the sound horizon `r_s(z*)`).

The cause was not a bug in the physics but a **definitional
mismatch**, which is the classic trap with compressed likelihoods.
These priors are not a measurement of the sky; they are a summary of
Planck's own fit, computed by Chen, Huang & Wang (2019) under a
specific set of conventions. CosmoFit was computing a *more detailed*
prediction than the compression assumed -- radiation as photons plus
3.046 massless neutrinos (`omega_r = 4.18e-5`) where CHW19 define
`Omega_r = Omega_m/(1+z_eq)` (0.8% lower, massive neutrinos left in
`Omega_m`), and `z*` from the Hu & Sugiyama (1996) fitting formula
where CHW19 take it from the Planck chains, i.e. from CAMB. HS96 was
calibrated against 1990s recombination physics and runs 0.22% high
for Planck-like parameters (1091.9 vs CAMB's 1089.9); `l_A` is
sensitive enough to `z*` that this alone is a ~4 sigma shift. Being
*more* physical than the data's own definitions is still wrong when
the data is a compression.

`RecombinationCalculator` now follows CHW19 Eqs. (1)-(6) exactly, and
`z*` comes from a fit calibrated directly against **CAMB 2.0.1** over
a 12x14 grid in (`omega_b`, `omega_cb`) covering far more than the
Planck posterior, accurate to **0.0018%** in `z*` (~0.04 sigma of the
`l_A` prior). The radiation term is also renormalized properly:
adding `Omega_r(1+z)^4` on top of a model's `E(z)` left
`E(0)^2 = 1 + Omega_r`, over-closing the universe; the correction
reuses each model's own dark-energy evolution, so it stays exact for
curved and non-LCDM models alike. The same fiducial check now gives
`l_A` +0.54 sigma, `R` -0.03 sigma, **chi2 = 0.39**.

Cross-checked two independent ways: against CAMB (`z*` to 0.002%),
and against a separate `scipy.quad` implementation of the CHW19
recipe across flat/open/closed LCDM and CPL (`R` and `l_A` to
<0.01 sigma). Per-evaluation cost is unchanged.

**This changes results.** The bias did not show up as a bad fit --
the sampler absorbed it by shifting parameters, which is exactly what
makes this kind of error dangerous. Refitting CC+DESI+Pantheon+ +
Planck:

| | old | new |
|---|---|---|
| LCDM `H0` | 68.04 | **67.39** |
| LCDM `Omega_m` | 0.3118 | **0.3149** |
| CPL `w0` | -0.973 | **-0.881** |
| CPL `wa` | -0.000 | **-0.298** |

The CPL case is the headline: the old code put `wa` at essentially
zero -- perfectly consistent with a cosmological constant -- while
the corrected code prefers evolving dark energy, in the same
direction and of comparable size to DESI's own published w0waCDM
result. Any CPL/JBP/BA conclusion drawn from a Planck-including fit
made with v0.18.0 or earlier should be regenerated.

> **Note:** don't combine two different `"s8"` versions in the same
> fit (default is `"kids1000"`; pass
> `dataset_kwargs={"s8": {"version": "des_y3"}}` for the other) --
> they're independent survey constraints, not a joint one. See the
> `S8Likelihood` docstring.

Version **v0.20.0** makes MCMC chains persistent. Until now a chain
lived only in memory: closing the notebook, or adding one more plot
to the bottom of a script, meant sampling the whole posterior again
from step zero -- hours, to recompute samples that hadn't changed.
`run_mcmc(save="chains/fit.h5")` now writes the chain to HDF5 as it
is sampled (emcee's own `HDFBackend`, plus a `cosmofit` metadata
group), and picks it back up on the next call instead of re-running
it. `nsteps` counts the *total* length to reach, so re-running an
unchanged script samples nothing, raising `nsteps` samples only the
difference, and an interrupted run keeps every step it had already
taken. `Fitter.from_chain("chains/fit.h5")` reopens a finished run in
a later session -- model, datasets, free parameters, priors and fixed
values all come back out of the file -- and
`CosmoFit.stats.chains.open_chain()` reads the posterior with no
dataset loaded at all.

Resuming is exact, not approximate: emcee's proposal RNG state is
stored with the walkers, so a chain sampled in four sittings is
bit-identical to the same chain sampled in one (verified directly, as
is multi-core `n_processes` writing through the same file). The
metadata is also what keeps it *safe* -- a resume whose model,
datasets, free parameters, prior bounds or fixed parameter values
differ from the stored chain's is refused, naming exactly what
differs, rather than silently welding samples from two different
posteriors into one array. The GUI uses the same machinery
(`fitter.chain_id()`, a stable hash of the posterior, as the
filename), so adding a model to a comparison no longer re-runs the
models already fitted. `h5py` is now a dependency.

Version **v0.21.0** adds the w0-wa dark-energy plane
(`fitter.plots.w0_wa_plane()`, and `compare_w0_wa_plane()` for
several posteriors at once): 2D credible contours over the
phantom/quintessence/quintom-A/quintom-B regions with ΛCDM marked at
(-1, 0) -- the figure the DESI evolving-dark-energy results are
argued in. The classification behind it is available on its own as
`cpl_diagnostics.classify_region()` /
`cpl_diagnostics.region_fractions()`, which turns "the contours sit
in the quintom-B region" into a posterior probability. Contour levels
are stated as 2D credible probabilities (68%/95% of the samples), not
sigmas, since in two dimensions the familiar 1D contours enclose only
39%/86.5%.

The same release fixes a reporting bug in the GUI's w0-wa
diagnostics: it printed the Mahalanobis distance D of the ΛCDM point
with a σ suffix. In 2D, D is not a number of sigma (D² follows χ² with
2 degrees of freedom), so this overstated the tension -- D = 2.20
reads as "2.2σ" but is really 1.70σ. The library was corrected in
v0.19.0; the GUI now reports `sigma` (with the confidence level, and D
labelled as what it is) too.

Version **v0.22.0** is a figure-typography pass: everything a plot
says is now written the way a paper would write it, and a few labels
that were quietly *wrong* are fixed.

Parameter names reach every axis and corner-plot title as LaTeX
(`$\Omega_b$`, `$r_d$`, `$\sigma_8$`) instead of as Python
identifiers -- the labels were always declared on the parameter
container, the plots just weren't reading them. Model names do the
same: legends show `ΛCDM`, `wCDM`, `f(R)` Hu-Sawicki via a new
`Cosmology.MODEL_LABEL` / `plot_label()` (with `plain_name()` for
tables, dropdowns and JSON, where raw LaTeX would be shown
literally), and `define_model(..., label=...)` lets a custom model
supply its own. Corner-plot titles now size their precision per
parameter, so `Omega_b` reads `0.0491 +0.0011 -0.0011` instead of
corner's default `0.05 +0.00 -0.00`.

Three labels were misleading rather than merely plain:

- **The Pantheon+ Hubble diagram's y axis claimed to be
  `mu = m_B - M_B`.** It is neither: the plotted data is the
  corrected *apparent* magnitude `m_b_corr` (~11-27 mag, not a
  distance modulus's ~33-46), and `M_B` is analytically
  marginalized out by default, so it cannot appear in an axis
  label. Now `$m_B$ [mag]`, with the model curve's legend entry
  saying where its normalization came from
  (`Model ($M_B$ marginalized)`).
- **The BAO panels were titled `D_V/r_s` while their y axis said
  `r_d`.** Both denote the sound horizon at the drag epoch; the
  `rs` spelling comes from DESI's own data file, which
  `MODEL_MAP`'s keys keep, but the code divides by `rd` and now so
  do the titles.
- **The Planck pull plot's ticks were the raw dataset identifiers**
  (`lA`, `omega_b_h2`); they now render as `$\ell_A$`,
  `$\omega_b h^2$`.

Dataset combinations in legends also spell themselves out
(`CC + DESI + Pantheon+`, via `stats.dataset_label`) rather than
joining registry keys. No numerical result changes from any of this --
it is labels, titles and legends only.

The DES-SN5YR Hubble diagram is also legible for the first time. That
release ships 81 supernovae (of 1820) with `mu_err` between 5 and 468
mag -- entries the survey de-weights rather than removes -- and
matplotlib autoscales an errorbar plot to contain every whisker, so
the panel spanned +/-500 mag with the actual 35-45 mag Hubble diagram
compressed into a sliver of it. Panels are now scaled by the
measurements rather than by their largest error bars, and a bar may
only stretch the view if it is extreme both as a fraction of the
data's spread *and* as a multiple of that dataset's median error --
which caps nothing at all on CC, DESI, Pantheon+ or fsigma8, where
matplotlib's own limits are reproduced exactly. The 81 points stay on
the plot, drawn without the whiskers that no longer fit and counted
in the legend.

One functional bug surfaced while testing the above and is fixed here
too: **`FitResult.save_json()` (and the GUI's JSON download) failed
for any fit that used `run_mcmc(save=...)`**, with
`TypeError: Object of type int64 is not JSON serializable`. Reading a
chain's step count back through an HDF5 attribute yields `np.int64`
rather than `int`, and `json` rejects it (`np.float64` slips through
only because it subclasses `float`). The counters are now cast at the
source, and both JSON writers coerce numpy scalars rather than
failing on a save.

Version **v0.23.0** is the largest single addition since the library
began: six new datasets, eight new cosmological models, the first
test suite, and -- the headline -- **the CMB computed from scratch
rather than compressed**.

### Planck, uncompressed

Until now "Planck" meant three numbers: the distance priors
(R, l_A, omega_b h^2). That is fast, dependency-free and works for
every model, and v0.19.0 documents at length how badly it can go
wrong, because a compression carries the conventions of whoever
produced it and the theory prediction has to share them exactly.

`"planck_lite"` is the other end of that trade. It uses the actual
measured spectra: **613 binned TT/TE/EE bandpowers over
l = 30-2508 with their full 613x613 covariance**, compared against a
C_l spectrum computed by a Boltzmann code. No compression, no
summary statistic, no borrowed convention -- the theory prediction
is the same object the data is.

CosmoFit does not implement a Boltzmann hierarchy, and should not:
that is thousands of coupled ODEs per wavenumber through
recombination, and a pure-Python version would be far too slow for
an MCMC. A new `cosmology/boltzmann.py` translates a `Cosmology`
into **CAMB**'s parameter conventions and calls it, as an optional
dependency (`pip install "cosmofit[cmb]"`). Nothing else in the
library imports it.

Three details that decide whether this is right or subtly wrong:

- **Which models can go through it.** LCDM maps onto CAMB directly.
  Any model exposing a `w(z)` (wCDM, CPL, JBP, BA, GCG, and the new
  PEDE/GEDE/IDE) is passed through CAMB's PPF dark-energy module as
  a tabulated `w(a)` -- exact at background level, and stable across
  the `w = -1` crossing that CPL posteriors routinely visit. The
  **modified-gravity models are refused outright**. `FRHuSawicki`
  would have run happily -- its background *is* LCDM's by
  construction -- and returned LCDM's C_l with `f_R0` doing nothing,
  which is worse than an error.
- **Massive neutrinos.** CosmoFit counts them inside `Omega_m`;
  CAMB counts them separately. So `omch2` is
  `Omega_m h^2 - Omega_b h^2 - Omega_nu h^2` -- *subtracting*. Adding
  instead shifts `Omega_c h^2` by ~0.0006, about half a sigma of
  Planck's constraint on it, with no other symptom.
- **A CAMB API trap, found by testing.** `pars.DarkEnergy = obj`
  copies the object into CAMB's Fortran state, so setting the `w(a)`
  table on the Python instance *afterwards* is silently discarded --
  and CAMB then returns a perfectly valid cosmological-constant
  spectrum for a w0-wa model. Caught by asserting that CPL at
  `w0 = -1, wa = 0` reproduces LCDM *and* that CPL at
  `w0 = -0.9, wa = -0.4` does not.

**Validation.** Against the reference spectrum shipped with
`planck-lite-py`, this implementation reproduces its published
log-likelihood **exactly**: -291.33481235418026 for TTTEEE
(bit-identical) and -101.58123068722571 vs -101.58123068722583 for
TT (1e-13, matrix-inversion roundoff -- Cobaya's own plik_lite
differs from `planck-lite-py` by 2e-13 on the same number).
Independently, at Planck's best-fit LCDM it returns chi2 = 585 for
613 bandpowers.

**What it costs.** One CAMB call is ~0.7 s against ~1 ms for the
whole rest of a joint likelihood, so a chain including this is
roughly three orders of magnitude slower per step. That is not an
implementation flaw -- it is why full CMB chains run on clusters and
why compressed priors exist. Budget hours, use `n_processes`, and
save the chain. A CMB spectrum also needs `ln1e10As`, `n_s` and
`tau_reio`, which no background fit ever did; and because
`plik_lite` starts at l = 30, `tau` is degenerate with the amplitude
unless the new `"tau"` dataset is included.

> The primordial amplitude is `ln1e10As`, not `A_s` -- that name was
> already taken in this library by the Generalized Chaplygin Gas
> parameter, an unrelated quantity that shares the symbol in its own
> literature. Renaming the GCG one would invalidate every saved
> chain that names it.

### Six new datasets

- **DESI DR2 BAO (2025)**, `dataset_kwargs={"desi": {"version": "desi2025"}}` --
  three years of observations, >14 million galaxies and quasars,
  twice the DR1 sample, and the measurement the strengthened
  evolving-dark-energy claim rests on. Identical file format to DR1,
  so it drops into the same loader. **Not** to be stacked with DR1:
  DR2 contains every DR1 galaxy.
- **Union3** (`"union3"`) -- 2087 supernovae fit with the UNITY1.5
  hierarchical model and released as 22 binned distance moduli. The
  third of the three compilations the DESI dark-energy results are
  argued with, and they *disagree* about how far the data sits from
  a cosmological constant: DES-SN5YR pulls hardest, Pantheon+ least,
  Union3 in between. A library that can only fit one of them cannot
  reproduce that comparison, which is the actual state of the
  evidence.
- **Low-z BAO** (`"bao_lowz"`) -- 6dFGS (z = 0.106) and SDSS DR7 MGS
  (z = 0.15), the only BAO leverage below z = 0.2 (DESI starts at
  0.295, BOSS at 0.38). Independent of both, so unlike DESI-vs-SDSS
  these *can* be added to either. Two details are handled rather than
  papered over: 6dFGS reports `r_s/D_V`, kept as its own observable
  rather than inverted (inverting a Gaussian gives something that is
  neither Gaussian nor centred on 1/mean), and its measurement is
  calibrated against an Eisenstein-Hu fitting-formula sound horizon,
  so the theory `r_d` is rescaled by 153.9/149.8 -- 2.7% on a
  4.5%-precision point, the same class of definitional mismatch
  v0.19.0 documents.
- **`"h0"`, `"omega_b"`, `"tau"`** -- external single-number
  measurements: SH0ES 2022/2024 and TDCOSMO 2025 time-delay lensing
  for H0, BBN (Schoeneberg 2024 or Cooke 2018) for omega_b h^2, and
  Planck lowE for tau.

  These are **datasets, not priors**, and the distinction is
  deliberate. The posterior is identical either way, but a prior is a
  statement of belief before seeing data, while "SH0ES measured
  73.04 +- 1.04" is data from a telescope with a systematic error
  budget. As a dataset it shows up in the per-dataset chi2
  breakdown, in the degrees-of-freedom count AIC/BIC use, in figure
  legends, and in the chain metadata that decides whether a saved
  chain may be resumed. A fit that quietly assumed the local
  distance ladder should not look, from the outside, like a fit that
  did not. It also makes the H0 tension askable in the form it is
  argued: run the same model with and without `"h0"` and compare what
  each dataset contributes.

The BBN prior is more than an extra data point: BAO measures
`D/r_d`, and with a BBN constraint on `omega_b` a BAO-only fit gains
a CMB-independent route to H0 -- exactly how the DESI "BAO + BBN"
constraints are produced.

### Overlapping datasets now warn

Every "don't combine X and Y" rule in this README was, until now,
written down only in a docstring, where it protected nobody who did
not go looking. `Fitter` now checks and warns, naming the reason.
The failure it guards against is silent: overlapping data treated as
independent gives a perfectly ordinary-looking posterior with error
bars that are too small. It stays a warning, not an error --
quantifying how much an overlap matters is a legitimate thing to
want to do.

Two new rules join the existing ones: Union3 must not be combined
with Pantheon+ or DES-SN5YR, and `"planck"` must not be combined with
`"planck_lite"` (the distance priors are a compression of exactly
those bandpowers -- that is the entire CMB dataset twice).

### Eight new models

Grouped by what they actually change, since that decides what can be
done with them:

| Model | Extra parameters | Reduces to |
|---|---|---|
| **LogarithmicDE** | none (reuses `w0`, `wa`) | wCDM at `wa = 0` |
| **PEDE** | **none at all** | — |
| **GEDE** | `Delta`, `z_t` | LCDM at `Delta -> 0`; PEDE at `Delta = 1, z_t = 0` |
| **LsCDM** | `z_dagger` | LCDM below the transition |
| **IDE** | `xi` (with `w0`) | wCDM at `xi = 0` |
| **RunningVacuum** | `nu` | LCDM at `nu = 0` |
| **Cardassian** | `n_card`, `q_card` | LCDM at `n = 0, q = 1` |
| **DGP** | **none at all** | — |

Every reduction in that last column is a **test**
(`tests/test_models.py`), asserted to machine precision, not a
docstring claim. That matters more than it sounds: the Friedmann
closure `E(0) = 1` pins some normalizations but not all of them, and
a limit check pins the rest.

Two of these -- PEDE and DGP -- have **exactly LCDM's parameter
count** and a completely different expansion history. That makes the
model comparison unusually clean: identical AIC/BIC penalties, so a
chi2 difference is a difference in fit and nothing else.

DGP also overrides `mu(a, k)` with the standard Koyama-Maartens
result, so it joins the three modified-gravity models in predicting a
growth history that differs from GR's at fixed background. On the
self-accelerating branch gravity is *weaker* (`mu = 0.72` today,
matching the literature), and the bundled fsigma8 data feels it
directly.

### A matter-scaling bug this surfaced

RunningVacuum and IDE both change how *matter* dilutes -- that is
their whole content. But `GrowthCalculator` read `Omega_m(a)` off
`Omega_m (1+z)^3` for every model, so those two would have been given
LCDM's growth source term alongside their own `E(z)`: internally
inconsistent, and silent. `Cosmology` now has an overridable
`Omega_matter(z)` hook that both models implement and every other
model inherits unchanged. No existing result changes; the two new
models get a growth history that is actually theirs.

### The first test suite

The library had no tests. It now has **130**, running in ~10 s, and
they are aimed at the failure mode this project actually has:
physics that is *plausibly* wrong rather than broken.

- `test_models.py` -- Friedmann closure for every model (flat and
  curved), `dEdz` against a central finite difference (two
  independent hand transcriptions of the same algebra), every known
  limit, published signatures (PEDE's `w(0) = -1.145`, DGP's
  `Omega_rc` and growth suppression), and a bound on the error the
  distance integrator's grid makes on LsCDM's genuine discontinuity.
- `test_datasets.py` -- every dataset loads, every likelihood's
  covariance matches its data vector, and every chi2 per data point
  is O(1) at a concordance cosmology. The two deliberate exceptions
  are the tensions themselves: `"h0"` disagrees at ~5 sigma and
  `"s8"` at ~2.9 sigma, and the bounds are set just above those so a
  *further* error would still be caught rather than hidden.
- `test_planck_lite.py` -- the binning and covariance algebra against
  published log-likelihoods, and the CAMB translation separately
  against physics, so a failure says which half is wrong.

Run them with `pip install -e ".[dev]" && pytest`.

Version **v0.24.0** computes the BAO sound horizon `r_d` from the
physical densities instead of fitting it.

### What was wrong with fitting it

Nothing, exactly -- treating `r_d` as a free nuisance parameter is a
defensible and common choice, and it makes BAO immune to any
assumption about the early universe. It is also the reason this
library could not measure `H0` from BAO. `H0` and `r_d` enter every
BAO observable only through the product `H0 * r_d`, so with `r_d`
free they are perfectly degenerate and BAO constrains neither.

`SoundHorizon` was, until now, a nine-line stub that returned the
free parameter. It now does the integral:

    r_d = int_{z_d}^inf c_s(z)/H(z) dz,
    c_s = c / sqrt(3(1 + R_b)),  R_b = 3 omega_b/(4 omega_gamma) a

### What is actually computed, and what is not

**Computed from first principles:** photon density from `T_CMB`;
massless neutrinos; the baryon loading; the integral itself.

**Massive neutrinos, exactly.** They are relativistic at the drag
epoch (`y = m/kT ~ 0.34` for 0.06 eV) and matter-like today, and the
transition matters. The usual `[1 + (Ay)^p]^(1/p)` approximation
costs 0.05% in `r_d` at `Sum m_nu = 0.06 eV` and 0.15% at 0.6 eV --
the latter half of DESI's best BAO precision, spent on an
approximation with no reason to be there. Instead the Fermi-Dirac
energy density integral is tabulated once at import (a few ms) and
splined afterwards. The familiar `Sum m_nu / omega_nu h^2 = 93.14 eV`
shorthand is then not an input at all: the calculation **derives**
93.0378 eV, where CAMB gets 93.04.

**Fitted, and honestly labelled:** `z_drag` alone. The drag epoch is
defined by a Thomson-drag optical depth over a full recombination
history, which is a Boltzmann code's job. This takes the same route
the library already took for `z_star` in v0.19.0 -- a fit calibrated
directly against CAMB, over a 5850-point grid spanning `omega_b` in
[0.018, 0.026], `omega_cb` in [0.09, 0.20], `N_eff` in [2.0, 5.0] and
`Sum m_nu` in [0, 0.6] eV.

> Eisenstein & Hu (1998)'s `z_drag` formula is bundled for comparison
> and is **not** usable here: it gives 1020.7 where CAMB gives 1059.9,
> 3.7% low, which puts `r_d` 2.5% high -- ten times DESI DR2's best
> BAO error bar. That is not a flaw in EH98; their `z_d` was
> calibrated jointly with their own closed-form `r_s`, and the two
> halves are not separately meaningful. Splicing one of them onto a
> modern integral is exactly the convention error v0.19.0 documents
> at length, in a different place.

### Accuracy

Against CAMB's `rdrag` over that whole grid:

| | max error |
|---|---|
| the integral alone, given CAMB's `z_drag` | 2.2e-6 |
| the `z_drag` fit | 6.7e-5 |
| **end to end** | **5.0e-5** |
| end to end, within realistic priors | 1.4e-5 |

DESI DR2's single best BAO bin is a 0.24% measurement, so the worst
case is ~50 times smaller than the best data's error bar and the
typical case ~500 times. A trimmed copy of that grid ships as a test
fixture, so the comparison runs without CAMB installed.

### `r_d` depends on less than you might expect

The integral runs entirely through the radiation- and
matter-dominated eras, so **`r_d` does not depend on `H0`, on
curvature, or on the dark-energy model at all** -- only on
`omega_b`, `omega_cb`, `N_eff` and `m_nu`. Verified against CAMB,
which returns the same `rdrag` to 1e-7 across `H0` from 60 to 75.

Two consequences. It works identically for **every** model in the
library, including the modified-gravity ones a Boltzmann code will
not accept. And the cache keys on the densities alone, so an MCMC
step that moves only `w0` reuses it -- which matters, because the BAO
likelihoods ask for `r_d` once per data point.

### Using it

Off by default: switching a fitted nuisance parameter into a derived
quantity changes every BAO prediction, and that is a choice about the
analysis.

```python
fit = Fitter(
    model=LCDM,
    datasets=["desi", "omega_b"],          # BAO + the BBN prior
    dataset_kwargs={"desi": {"version": "desi2025"}},
    free_params=["H0", "Omega_m", "Omega_b"],
    initial={"H0": 68.0, "Omega_m": 0.30, "Omega_b": 0.0493},
    compute_rd=True,
)
```

`rd` must not be in `free_params` alongside it -- the likelihood
would ignore the sampled value and its "posterior" would be its
prior -- and the constructor says so rather than letting it happen.
`Omega_b` becomes the parameter BAO now needs and cannot pin down on
its own, which is what the `"omega_b"` BBN dataset is for. Running
that fit (DESI DR2 + BBN, flat ΛCDM) gives

    H0      = 68.55 +0.59 -0.60
    Omega_m = 0.2977 +0.0087 -0.0085
    Omega_b = 0.0472 +0.0008 -0.0008
    r_d     = 148.10 +1.59 -1.63 Mpc   (derived)

which lands where published DESI BAO+BBN constraints do -- an
end-to-end check of the whole chain that no single unit test
provides.

`r_d` is a *derived* quantity now, so it leaves `summary()` and joins
`z_t` and `q0` in `CosmoFit.stats.derived`:

```python
from CosmoFit.stats import derived
derived.summarize(derived.sound_horizon(fit))
```

That works whether or not the fit used it. With `compute_rd=False`
it returns what the early-universe physics *would* have predicted for
the same densities, and comparing it against the fitted `r_d` is a
real consistency test: a mismatch is the standard signature of new
physics before recombination, and one of the main ways the Hubble
tension is diagnosed.

`compute_rd` is part of the chain signature, so a chain sampled one
way cannot be resumed the other. The GUI gets a checkbox, which
un-ticks `rd` for you and points at the BBN dataset if it is missing.

### One bug this surfaced

Nothing in the growth machinery: `Fitter` now also refuses
`compute_rd=True` together with a free `rd`, which was previously
expressible and would have produced a `rd` "posterior" identical to
its prior with no indication anything was wrong.

Seventeen new tests (**147 total**) cover the neutrino
thermodynamics, the CAMB comparison, the independence claims above,
quadrature convergence, and the wiring.

Version **v0.25.0** is a pass over the graphical interface, on the
principle that a tool which cannot be used wrongly is worth more than
one with more knobs.

The app had grown to fourteen datasets and seventeen models presented
as a flat checkbox list and a flat dropdown, with no indication of
what any of them was. That is a problem specific to this domain: a
wrong combination does not error, it returns a perfectly
ordinary-looking posterior with error bars that are too small, or a
"measurement" of a parameter nothing in the fit constrains.

**Everything now explains itself.** Each dataset says what it
measures, over what redshift range, how many points, what it actually
constrains, and where it comes from -- grouped by probe, since a fit
is built by taking one from each family rather than by ticking
everything. Each model says what it is, what its extra parameters
mean, and which parameter values reduce it to ΛCDM, plus capability
badges for `w(z)`, growth, and whether the full CMB spectra can be
computed for it.

**Six presets** configure a whole analysis at once. "DESI DR2 + BBN →
H₀ without the CMB" sets the datasets, selects DR2, *and* turns on the
computed sound horizon -- none of the three is useful without the
other two, and expecting someone to discover that from three separate
widgets was optimistic.

**The parameter table now shows only the parameters this fit uses.**
The shared container carries every parameter any model needs; for an
LCDM fit against CC and BAO, all but four are inert. Relevance is a
property of the fit rather than the model, so it follows the datasets
too: `rd` appears only with BAO, `sigma8` only with growth data,
`n_s`/`τ` only with the CMB spectra. The hidden ones still reach the
`Fitter` -- they are inert, not absent.

**Warnings fire before the run, not after.** One is a hard error the
app can see coming (a modified-gravity model with the full CMB
spectra). The rest are quieter and matter more, because the fit will
run, converge, and produce something that looks like a result: a
modified-gravity model with no growth data ticked, a model whose own
parameters are left fixed, a free `sigma8` nothing constrains, a
computed `r_d` with `Ω_b` held fixed.

**Three things the GUI simply could not show before:**

* **χ² per dataset**, which turns a total χ² into a statement about
  *which* dataset is in tension. On the Hubble-tension preset it reads
  DESI 19.6/13, Planck 7.2/3, local H₀ **24.5/1** -- the whole
  argument, in one table.
* **Derived quantities** (`q₀`, `z_t`, `r_d` from the densities) with
  real error bars. These have existed in `CosmoFit.stats.derived`
  since v0.13.0 and were reachable only from Python.
* **Dataset versions.** DESI DR2, the SH0ES 2024 and TDCOSMO H₀
  measurements, DES Y3's `S₈` and the Cooke BBN prior all shipped in
  v0.23.0 and were **unreachable from the GUI**, which never passed
  `dataset_kwargs`. Each dataset with more than one version now has a
  picker.

A **📖 Guide** panel lists every dataset and model in one browsable
table with the conflict rules and a short walkthrough. Both tables are
generated from the same dictionaries the widgets use, so a dataset
added to the library without a note shows up as a blank row rather
than silently.

### Two bugs, and the first GUI tests

`st.checkbox` was being given both a `value=` and a session-state
entry under the same key -- the one thing Streamlit explicitly warns
about, since the two disagree about which is authoritative. And the
preset button called `st.rerun()`, which is the obvious idiom and was
never needed here (the button sits above the widgets it writes to, so
the values land on the same pass); it was also an infinite loop
anywhere a button's click state outlives the rerun.

That second one was found by the new `tests/test_gui.py`, which is the
point of it. Streamlit's `AppTest` runs the app in-process and exposes
the rendered element tree, so **12 tests** now drive the flows a user
actually takes: a preset writes the configuration it claims to, a fit
runs end to end and renders its tables, the parameter table follows
the datasets, and each warning appears when it should. **159 tests
total**, ~18 s.

Two small public additions came out of this, both because the GUI
needed them and reaching into private state to get them would have
been worse: `dataset_reference(dataset, version)` returns a dataset's
citation without loading its files, and
`cosmology.boltzmann.supports_cmb_spectra(model)` answers "can CAMB do
this model?" as a value rather than by raising -- the backend now
routes its own check through it, so the two cannot drift apart.

The package structure may continue to evolve before the first stable **v1.0.0** release.

---

## Roadmap

### Datasets

* **eBOSS DR16 ELG and Lyman-alpha BAO** -- released as tabulated
  (non-Gaussian) likelihood grids rather than mean+covariance, so
  they need a grid-interpolating likelihood the library does not
  have yet.
* **DESI DR1/DR2 full-shape** (`fsigma8` + BAO jointly, with their
  cross-covariance) -- more constraining than BAO alone, and the
  natural companion to the growth machinery already here.
* **Planck lensing** and **ACT DR6** -- more CMB, and an
  independent one.
* **Planck low-l (Commander/SimAll)** -- at l < 30 the C_l
  distribution is not Gaussian, so this needs a different
  likelihood form, not another bandpower vector. Until then the
  `"tau"` prior stands in for it.

### Physics

* **Massive neutrinos and `N_eff` in each model's low-redshift
  `E(z)`.** Both parameters now reach the Boltzmann backend and the
  sound horizon, and both are treated exactly there -- but the
  models' own `E(z)` still counts massive neutrinos inside
  `Omega_m` as pure matter, which is right below z ~ 100 and
  increasingly wrong above it.
* **`Sum m_nu` as a fitted parameter.** Everything needed is in
  place; what is missing is the growth-suppression signature that
  actually constrains it.
* **Holographic dark energy**, which needs a per-step ODE solve
  rather than a closed form.

### v1.0.0

* Stable public API
* Complete documentation
* Test coverage across the whole library, not only the newest parts
* Continuous integration
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