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

* Modified-gravity models, at both background *and* growth-of-structure level (see
  [CHANGELOG.md](CHANGELOG.md) at v0.16.0 and v0.17.0 for exactly what's real vs.
  simplified in each)

  * **FQExponential** -- f(Q) gravity (symmetric teleparallel), f(Q) = Q exp(lambda Q0/Q)
  * **FRTLinear** -- f(R,T) gravity (linear), f(R,T) = R + 2 lambda T
  * **FRHuSawicki** -- f(R) gravity (Hu-Sawicki); background is identical to LCDM's by
    construction (stated explicitly, see [CHANGELOG.md](CHANGELOG.md) at v0.16.0), but
    growth of structure -- where this model's `f_R0`/`n` parameters actually show up --
    is now real: a scale- and time-dependent, chameleon-screened `mu(a,k)`

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
* **σ₈ derived from the CMB rather than fitted** (`Fitter(..., derive_sigma8=True)`), when the fit
  contains a from-scratch CMB likelihood. Without it, `sigma8` is a free parameter *and* the CMB
  fixes an amplitude of its own through `ln1e10As` -- two unrelated numbers for one quantity, with
  the sampler reporting a posterior for each. It also changes what a fit can ask: with `sigma8`
  free, the S₈ measurement is absorbed (the parameter slides onto it and χ² goes to zero, which
  looks like agreement); derived, the CMB's own prediction meets the measurement and the S₈
  tension is visible
* Flexible parameter management
* Built-in observational datasets

  * Cosmic Chronometers (CC)
  * BAO (**DESI DR2 2025** or DESI DR1 2024; SDSS BOSS DR12 + eBOSS DR16 LRG/QSO;
    **low-z 6dFGS + SDSS DR7 MGS**) -- the low-z pair is independent of the other two and
    can join either; DESI and SDSS cannot be combined with each other, see the note below
  * The **SDSS BAO + full-shape consensus**: D_M/r_d, D_H/r_d *and* f*sigma8 at
    z = 0.38, 0.51, 0.698, 1.48 with the covariance between them. The same galaxies
    as the BAO-only entry above, so use one or the other -- and prefer this one over
    pairing `sdss_bao` with `fsigma8`, which covers the same galaxies while treating
    growth and geometry as uncorrelated when they are not
  * BAO as a **tabulated likelihood surface** rather than a mean and a covariance:
    **eBOSS DR16 ELG** (`D_V/r_d` at z=0.845, a 399-point curve) and **eBOSS DR16
    Lyman-alpha** (`(D_M/r_d, D_H/r_d)` at z=2.334, a 50x50 surface, the
    highest-redshift BAO here outside the CMB). eBOSS released both as grids because
    a Gaussian misrepresents them -- the ELG BAO is a 1.4-sigma detection whose
    likelihood is still rising at the low edge of the table
  * The **eBOSS DR16 ELG full-shape** analysis of the same galaxies: a 100x100x100
    grid in `(D_M/r_d, D_H/r_d, f*sigma8)`, so it constrains the **growth rate**
    alongside the geometry and carries the fsigma8/Alcock-Paczynski degeneracy
    exactly. Mutually exclusive with the BAO-only ELG entry
  * **Holographic dark energy** (Li 2004) -- the one background model here with
    no closed-form `E(z)`: `Omega_DE` comes from an ODE solved on every parameter
    change, with the future event horizon as the infrared cutoff. Its single
    parameter `c` decides whether `w` crosses below -1, as a prediction rather
    than a parametrization choice
  * **Agegraphic** and **Ricci** dark energy, the other two members of the holographic
    family -- same `rho_DE = 3c^2 M_p^2 / L^2`, different cutoff (conformal age, Ricci
    scalar). ADE has **one fewer free parameter than LCDM**: its early-time condition
    fixes `Omega_m` from `n`, so n = 2.8 *predicts* `Omega_m = 0.280`
  * Supernova (Pantheon+, DES-SN5YR, **Union3**) -- one per fit, see the note below
  * CMB, four ways: the compressed distance priors (Planck 2018 R, l_A, omega_b_h2 -- fast,
    works for every model); the **full Planck 2018 TT/TE/EE spectra** (615 `plik_lite`
    bandpowers including the two Commander low-l temperature bins, computed from scratch with
    CAMB -- slow, LCDM and w(z) models only); **Planck 2018 CMB lensing** (9 bandpowers of the
    reconstructed lensing potential, `8 <= L <= 400`), the CMB's own measurement of how much
    structure grew; and **Planck 2018 low-l EE** as the tabulated, *non-Gaussian* likelihood it
    actually is rather than the usual `tau = 0.0544 +- 0.0073` shorthand (which is still
    available as the `"tau"` dataset)
  * **ACT DR6 CMB lensing** -- a second, independent lensing reconstruction, tighter than
    Planck's (2.3% on the amplitude), built on the lensing convergence rather than the potential
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
* **Bayesian evidence by nested sampling** (`fitter.run_nested()`, needs `pip install
  "cosmofit[evidence]"`): `ln Z` and a proper Bayes factor, for the comparisons a
  likelihood-ratio test cannot make. `LsCDM` reduces to `LCDM` only as `z_dagger -> infinity`
  and `DGP` is not nested at all, so Wilks' theorem does not apply to either -- an evidence
  ratio is defined regardless, and integrates rather than maximizes, so it charges a model
  for prior volume it does not use. Validated against an analytically integrable Gaussian.
  Read `stats.evidence`'s note on prior sensitivity before quoting one
* **Profile likelihood** (`fitter.profile("z_dagger", values)`): `chi2` minimized over every
  other parameter at each fixed value. The honest tool where Wilks fails, and where a marginal
  posterior would smooth over structure -- it is how `lscdm_mcmc.ipynb` found a 28-unit cliff
* **Tension statistics** (`stats.tension`): the `np.hypot` this repo used to write by hand,
  named and with its assumptions stated -- plus the alternatives for when they fail.
  `sample_tension` works from posterior samples with no Gaussian assumption;
  `gaussian_tension_nd` because a tension in a plane is not the larger of its projections
  (two posteriors can sit 0.35 sigma apart in each parameter and 1.74 sigma apart jointly);
  and `suspiciousness`, which divides out the prior dependence a Bayes factor carries.
  Checked against the published Hubble (4.85 sigma) and S8 (2.67 sigma) tensions, and
  against the analytic parameter-difference chi2
* **Fisher matrix** (`fitter.fisher()`): parameter errors from the curvature at the best fit,
  in `~2n^2` evaluations rather than a chain. What `s8_tension_cmb.ipynb` needed when every
  likelihood call is a CAMB call and a converged chain is thirteen hours
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
* **Models from an action** (`CosmoFit.theory`, `pip install -e ".[theory]"`): give a
  gravitational action on an FLRW metric and the library does the variational calculus --
  reduces it to a point-like Lagrangian, varies the lapse for the Friedmann constraint,
  solves or integrates it, and hands back an ordinary model every dataset already works on

  * **Undeformed gravity in three sectors** -- `R`, `T` (torsion), `Q` (non-metricity);
    the sign conventions are fixed by requiring each to reproduce General Relativity
    exactly, and the tests assert it rather than trusting the convention
  * **`f(T)` / `f(Q)`** -- solved pointwise, transcendental constraints included: the
    exponential `f(Q)` model's Lambert-`W` Friedmann equation is rederived from the action
  * **Quintessence and k-essence**, any `L(X, phi)` and any number of fields -- integrated
    forwards from `z_init`, with `E(0) = 1` as a shooting condition. Validated against
    *both* Copeland-Liddle-Wands attractors to five decimals
  * **Scalar-tensor `F(phi) R`** -- the field sets the strength of gravity, bringing the
    `3 H dF/dt` term into the Friedmann equation and a moving `G_eff/G_N` into the growth
  * **A general `f(R)`** -- fourth-order, reduced by promoting `R` to an independent
    variable held to its geometric value by a Lagrange multiplier
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
│   ├── theory/        # build a model from an action: reduce, vary, solve (optional: sympy)
│   └── plots/         # FitPlotter -- every figure, attached as fitter.plots
├── examples/          # example notebooks, in five sections
├── tests/             # the test suite
├── app/               # the Streamlit GUI
├── CHANGELOG.md       # the release history
└── pyproject.toml
```

The subpackages remain directly importable too
(`CosmoFit.stats.model_comparison`, `CosmoFit.data.loader`, ...) for
anything not re-exported at the top level -- `from CosmoFit import ...`
is a convenience layer over them, not a replacement.

---

## Example Notebooks

Seventeen notebooks under [`examples/`](examples/), organised by what you are
trying to do — see [`examples/README.md`](examples/README.md) for the full
index. All are Colab-ready: click a badge and *Runtime → Run all*, no local
setup.

| | |
|---|---|
| **[01 · Getting started](examples/01-getting-started)** | [`quickstart`](examples/01-getting-started/quickstart.ipynb) — nothing to a real MCMC posterior in a couple of minutes; [`dataset_zoo`](examples/01-getting-started/dataset_zoo.ipynb) — all 21 datasets, the three non-Gaussian ones, and the sixteen pairs not to combine; [`cmb_from_scratch`](examples/01-getting-started/cmb_from_scratch.ipynb) — the CMB computed rather than compressed, and what that costs. |
| **[02 · Models](examples/02-models)** | [`model_zoo_comparison`](examples/02-models/model_zoo_comparison.ipynb) — six dark-energy parametrizations head to head; [`modified_gravity_growth`](examples/02-models/modified_gravity_growth.ipynb) — f(Q)/f(R,T)/f(R) with `mu(a,k)` and growth data; [`holographic_family`](examples/02-models/holographic_family.ipynb) — HDE/ADE/RDE against published constraints. |
| **[03 · Building models](examples/03-building-models)** | [`custom_models`](examples/03-building-models/custom_models.ipynb) — three routes to your own `E(z)`; [`models_from_an_action`](examples/03-building-models/models_from_an_action.ipynb) — give an *action* and let the library derive `E(z)`; [`scalar_field_models`](examples/03-building-models/scalar_field_models.ipynb) — quintessence, k-essence and scalar-tensor gravity, integrated. |
| **[04 · Inference](examples/04-inference)** | [`evidence_and_model_selection`](examples/04-inference/evidence_and_model_selection.ipynb) — AIC/BIC, likelihood-ratio, and Bayesian evidence; [`profile_likelihood_and_fisher`](examples/04-inference/profile_likelihood_and_fisher.ipynb) — three ways to an error bar; [`tension_statistics`](examples/04-inference/tension_statistics.ipynb) — four definitions of "they disagree at 4σ". |
| **[05 · Case studies](examples/05-case-studies)** | [`cpl_mcmc_analysis`](examples/05-case-studies/cpl_mcmc_analysis.ipynb) and [`cpl_mcmc_tfd42`](examples/05-case-studies/cpl_mcmc_tfd42.ipynb) — the CPL deep dive and its publication-scale variant; [`lscdm_mcmc`](examples/05-case-studies/lscdm_mcmc.ipynb) — does ΛsCDM still relieve the H₀ tension?; [`s8_tension_cmb`](examples/05-case-studies/s8_tension_cmb.ipynb) — the S₈ tension from a from-scratch Planck spectrum; [`dark_energy_evidence_audit`](examples/05-case-studies/dark_energy_evidence_audit.ipynb) — how much of the dark-energy evidence is a choice, across 20 combinations. |

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
CosmoFit's twenty built-ins -- doesn't require touching the library's
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

    def _dark(self, z):
        return (1 - self.Omega_m) * (1 + z) ** (3 * (1 + self.w0)) * (1 + self.beta * z)

    def E(self, z):
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.Omega_m * (1 + z) ** 3 + self._dark(z))

    def dEdz(self, z, h=1e-5):
        z = np.asarray(z, dtype=float)
        return (self.E(z + h) - self.E(z - h)) / (2 * h)      # or by hand
```

A direct subclass **must** define `dEdz`, and from the moment it is
constructed: the distance integrator interpolates `1/E(z)` with a Hermite
spline built from that derivative, which is what makes it exact to fourth
order instead of second. `define_model` installs a central-difference
fallback for you; subclassing does not, so that a model can supply the exact
derivative -- as every built-in one does -- rather than silently give up
about four orders of magnitude of accuracy in every distance.

---

## Models From an Action

`define_model` still asks for `E(z)`, which means somebody has
already done the variational calculus by hand. `CosmoFit.theory`
takes the other end: give it a gravitational action on an FLRW
metric and it derives `E(z)` itself.

The method is the standard minisuperspace one. FLRW is written
with an explicit lapse `N(t)`, the action is reduced to a
point-like Lagrangian in `a`, `adot` and `N`, and varying the
lapse produces the Friedmann *constraint* -- the lapse is a
non-dynamical gauge degree of freedom, which is exactly why its
variation gives a constraint rather than an evolution equation.
Setting `N = 1` afterwards recovers the familiar form. Writing
`N = 1` from the start would lose the equation altogether.

```python
from CosmoFit import Fitter
from CosmoFit.theory import Action

# Power-law f(T) gravity (Bengochea & Ferraro 2009).
model = Action(
    "T + A0*(-T)**b",
    geometry="teleparallel",
    params={
        "A0": {"default": -4.2, "bounds": (-30.0, 0.0)},
        "b": {"default": 0.0, "bounds": (-2.0, 0.9), "label": r"$b$"},
    },
    closure="A0",           # fixed by E(0) = 1, not fit
    growth="quasi_static",  # mu = 1/f', for fsigma8/s8
).build("PowerLawFT")

fit = Fitter(
    model=model,
    datasets=["cc", "desi"],
    free_params=["H0", "Omega_m", "b", "rd"],
    initial={"H0": 70.0, "Omega_m": 0.3, "b": 0.0, "rd": 147.0},
)
fit.best_fit()
```

What comes back is an ordinary `Cosmology` subclass. Every
dataset, likelihood, sampler and plot in the library works on it
unchanged.

`Action.constraint()` returns the derived Friedmann equation
symbolically, if the derivation is what you wanted rather than the
fit.

### What it checks, and what it refuses

The three geometry scalars carry different signs across the
literature, and picking one wrongly inverts every modification
built on it while leaving nothing downstream to complain. They are
fixed here by requiring that an undeformed `f` reproduce General
Relativity exactly, and the test suite asserts that in all three
sectors rather than trusting the convention.

* An action that does not satisfy `E(0) = 1` is **refused**. It
  would predict every distance wrong by a constant factor without
  looking broken. `closure=` names the parameter that condition
  fixes -- in Lambda-CDM, `"R - 2*Lam"` with `closure="Lam"` is
  what makes `Lam = 3 (1 - Omega_m - Omega_k)`.
* `growth="quasi_static"` is **opt-in**. `mu = 1/f'` is a
  statement about perturbations, which a background action does
  not by itself determine.

### A general `f(R)`

Fourth-order, and it gets its own reduction — applied
automatically, with nothing to ask for:

```python
model = Action(
    "R - 2*Lam + alpha_fr*R**2",
    params={"Lam":      {"default": 2.1, "bounds": (0.0, 6.0)},
            "alpha_fr": {"default": 1e-3, "bounds": (1e-6, 1.0)}},
).build("Starobinsky")
```

The ordinary reduction removes the `addot` in the Einstein–Hilbert
term by integrating by parts, which is legitimate only while the
Lagrangian is *linear* in it. For a general `f(R)` it is not — the
term that would be dropped is not a total derivative. Promoting `R`
to an **independent variable**, held to its geometric value by a
Lagrange multiplier, gives

```
L = (1/2) N a^3 [ f(R) - f'(R) R + f'(R) R_geom ]
```

which is linear in `addot` again, at the cost of one extra dynamical
variable. That variable is the theory's fourth order made visible,
and it appears as a parameter: **`R_0`**, the Ricci scalar today —
an initial condition General Relativity does not have. There is no
`closure` here, because `E(0) = 1` holds by construction.

**Checked three ways.** The derived Friedmann constraint equals the
textbook `3 f_R H² = (f_R R − f)/2 − 3H d(f_R)/dt + rho`
symbolically. `alpha -> 0` reproduces ΛCDM, and does so smoothly:
over `z <= 5` the departure falls 6.4e-01 → 1.8e-01 → 4.2e-03 as
`alpha` goes 1e-1 → 1e-3 → 1e-5. And the accuracy measure is the
**third** equation of
motion, from varying `a` — the one the integration never uses,
which holds by the Bianchi identity and comes out at 1e-15.

Unlike a scalar field, this integrates **backwards** from today, and
the difference is measured rather than assumed: a `1e-8` kick to
`R_0` moves `E(z)` by less than that out to `z = 1100`, where a
scalar field would have run away.

### Scalar fields

An action can carry dynamical fields, in which case the expansion
history is integrated rather than solved pointwise:

```python
model = Action(
    "R",
    fields={"phi": "X - V0*exp(-lam*phi)"},   # any L(X, phi)
    params={"V0":  {"default": 2.1, "bounds": (0.05, 50.0)},
            "lam": {"default": 0.5, "bounds": (0.0, 1.7)}},
    closure="V0",
).build("ExponentialQuintessence")
```

More than one field is fine, and a field's name is also in scope in
`gravity` -- which is how **scalar-tensor gravity** is written:

```python
Action(
    "(1 + xi*phi**2)*R",                       # F(phi) R, not R + phi
    fields={"phi": "X - V0"},
    params={"xi": {"default": 0.02, "bounds": (-0.5, 0.5)},
            "V0": {"default": 2.1, "bounds": (0.05, 20.0)}},
    closure="V0",
)
```

That is a different theory from a field sitting on top of General
Relativity: the field sets the strength of gravity, and the
Friedmann equation gains the `3 H dF/dt` term. The derivation
reproduces `3 F H^2 + 3 H dF/dt = rho` symbolically, and `xi = 0`
returns ΛCDM to 1e-9.

It also changes how structure grows, and that is easy to get
silently wrong. `growth="quasi_static"` gives the scalar-tensor
`mu = G_eff/G_N` of Boisseau, Esposito-Farèse, Polarski &
Starobinsky (2000), evaluated on the field's own solution so that
it moves as the field rolls. Left at the default `"gr"`, `mu` is 1
-- correct for a *minimally* coupled field, and wrong here -- so
`Fitter` warns if such a model meets growth data.

One more thing to know: a field's history starts at `z_init`
(default 3000), and the growth ODE starts at `z = 9999`. Fitting
`fsigma8` therefore needs `Action(z_init=20000)` or similar, and
saying so is now an error rather than an infinite chi-squared.

Each field adds two parameters, `phi_i` and `dphi_i` -- its value
and `dphi/dN` at `z_init` (default 3000, and the earliest
redshift the model can be evaluated at).

**They are set early, not today, and that is the whole design.**
Setting them at `a = 1` would make `E(0) = 1` algebraic and skip
the shooting entirely. It is also wrong: integrating backwards
from a field at rest today, the Hubble friction that damps the
field forwards becomes anti-friction, the generic past solution
is kinetic-dominated, and `rho_phi` grows as `a^-6`. For
exponential quintessence normalized to today's dark-energy
density that gives `E(2.5) = 9.3` where ΛCDM gives 3.7. Nothing
fails -- it is the correct past of those initial conditions, and
those initial conditions are not a universe.

So the state is given where a quintessence model actually gives
it, early and typically frozen, and `E(0) = 1` becomes a shooting
condition on `closure`. Forwards is also the numerically stable
direction.

Validated against Copeland, Liddle & Wands (1998), whose
late-time attractors are exact functions of the slope alone. On a
matter background there are two, selected by `lambda^2` against 3:

| | `w` | predicted | `Omega_phi` | predicted |
|---|---|---|---|---|
| λ = 0.5 | −0.916667 | −0.916667 | 1.00000 | 1 |
| λ = 1.0 | −0.666667 | −0.666667 | 1.00000 | 1 |
| λ = 1.9 | −0.000035 | 0 | 0.83097 | 0.83102 |
| λ = 2.0 | +0.000035 | 0 | 0.75006 | 0.75000 |

The first two are the field-dominated attractor
(`w = -1 + lambda^2/3`); the last two are the scaling attractor,
where the field tracks matter at a fixed fraction `3/lambda^2` --
which is why a steep potential cannot accelerate. Ask such a
model for more dark energy than its attractor allows and `H(0)`
saturates below 1 with no potential scale reaching it; that
arrives as a statement about the model rather than as a stalled
integrator.

A constant potential pins the other end: it reproduces `LCDM` to
1e-10 in `E(z)`, with `w = -1` and `Omega_de` constant to machine
precision at every redshift.

Cost is about 37 ms per parameter set -- a shooting solve plus an
integration, against ~150 µs for a closed-form model. `w` and
`Omega_de` are read off the field's own Lagrangian
(`p = L`, `rho = 2 X L_X - L`) rather than by subtracting the
fluids from `E(z)^2`, which at z = 2000 would be 2.4e9 minus
2.4e9 to get 0.7.

### Validation

Because this derives what the rest of the library has typed in by
hand, it can be checked against it:

| action | reproduces | agreement |
|---|---|---|
| `R - 2*Lam` | `LCDM`, curvature included | `E(z)`, `dE/dz` to 1e-16 |
| `Q*exp(lam*Q0/Q)` | `FQExponential` | constraint identical; `lam`, `E(z)`, `mu` to 1e-13 |
| `T + A0*(-T)**b`, `b = 0` | `LCDM`, via a different sector | `E(z)` to 1e-13 |

The `f(Q)` case is the demanding one. Its constraint is
transcendental -- the hand-written model inverts a Lambert `W` to
solve it -- so the derivation has to produce
`(E^2 - 2 lam) exp(lam/E^2) = Omega_m (1+z)^3` from the action
alone, and the solver has to land on the same branch `W_0` picks
rather than on any root of that equation.

Installed as an extra, since nothing else in the library needs
sympy:

```bash
pip install "cosmofit[theory]"
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

CosmoFit is under active development. The full release history --
thirty-three versions, what each one changed, and for several of them
how the bug was found rather than only that it was fixed -- lives in
**[CHANGELOG.md](CHANGELOG.md)**.

Where it stands today:

| | |
|---|---|
| datasets | **21**, from cosmic chronometers to the from-scratch CMB |
| models | **20** written out by hand, plus three routes to one that is not here |
| tests | **697** at 93% coverage, on Python 3.11-3.13, with and without every optional extra |
| notebooks | **17**, in five sections under [`examples/`](examples/) |

`main` is deliberately held at **v0.22.0**; everything since has been
published as a `-dev` pre-release from the `dev` branch. The package
structure may still evolve before the first stable **v1.0.0** -- see
the [Roadmap](#roadmap) for what that release is waiting on.

---

## Roadmap

Most of what this section used to list has been built. What is left:

### Datasets

* **Strong-lensing time delays** (TDCOSMO / H0LiCOW) -- the one genuinely
  independent probe of `H0` still missing, and the third leg of the H0
  tension alongside the CMB and the distance ladder. Its per-lens
  likelihoods are skewed log-normals, which the tabulated machinery
  already here would handle.
* **DESI DR1/DR2 full-shape** -- deliberately *not* planned. DESI publish
  MCMC chains and the inputs to their modelling pipeline, but no
  compressed Gaussian summary, so using it means implementing an EFT
  model with its own nuisance parameters. That is outside a
  background-plus-linear-growth library, and it is not cheaply
  validatable. See the `2d59fe7` commit message.

### Physics

* **Massive neutrinos and `N_eff` in each model's own `E(z)`.** They
  reach the Boltzmann backend and the sound horizon and are treated
  exactly there, but the models' `E(z)` counts massive neutrinos inside
  `Omega_m` as pure matter -- right below z ~ 100, and increasingly wrong
  above it.

  Worth saying what this is *not*, because the three places that consume
  `E(z)` above z ~ 100 are each already handled. The sound horizon
  integrates the exact Fermi-Dirac density. The compressed Planck priors
  deliberately keep massive neutrinos inside `Omega_m` at every redshift,
  because a prediction has to share the compression's definitions rather
  than improve on them -- see `likelihoods/planck.py` on what mixing
  conventions costs. And the growth ODE solves a matter-plus-dark-energy
  equation, so starting it deep in *that* matter era is self-consistent.
  So this is narrower than it looks, and doing it carelessly would break
  agreements the library has validated.
* **`Sum m_nu` as a fitted parameter.** Everything needed is in place;
  what is missing is enough growth-suppression signal to constrain it.
  The compressed Planck priors cannot supply it -- they are blind to
  `m_nu` by construction, and `Fitter` says so.
* **`f(R)` growth.** The background is solved from the action, but `mu`
  there is scale-dependent -- a Compton wavelength enters -- so it is
  refused rather than given a scale-free answer. `FRHuSawicki` carries
  the standard form if that is what you need.

### Performance

* **Batched log-posterior** (`emcee`'s `vectorize=True`): evaluating every
  walker in one call rather than one at a time. The one structural lever
  left -- worth perhaps 2-4x, and the only one that helps inside a
  Jupyter kernel, where `n_processes` does not. It needs a batch axis
  through every calculator, so it is a real refactor rather than a flag.

### v1.0.0

This section used to be three bullet points. Here is where each of
them stands.

**Stable public API.** Done. Nothing was holding it: every other test
imports what it happens to need, so a name could be renamed, moved
between subpackages or dropped from `__all__` and the suite would go
on passing as long as *some* path to the object still existed.
`tests/test_public_api.py` now types the surface out by hand -- 64
names -- and asserts the set in both directions. Writing it down
found two names that had already drifted out of `CosmoFit.cosmology`
while every one of their siblings was re-exported:
`ModelConfigurationError`, which is the one exception a user is asked
to tell apart from an ordinary failure, and `GrowthCalculator`.

**Complete documentation.** Done, in two halves. The release history
is in **[CHANGELOG.md](CHANGELOG.md)** -- it had stopped at v0.25.0,
so fourteen releases, including the whole of `CosmoFit.theory`,
existed only as GitHub release notes. And there is an API reference
under [`docs/`](docs/), built by CI with warnings as errors. Getting
it there fixed ten docstrings that rendered wrong on the page:
equations parsed as bullet lists, `Parameters` sections holding
prose, and `|beta|` read as a substitution reference.

**Test coverage across the whole library, not only the newest parts.**
Done: **93%**, from 79%, across **697 tests**. The phrase turned out
to be exactly right -- `theory`, the newest subpackage, was at
86-94% while the oldest code was not:

| | was | now |
|---|---|---|
| `plots/plotter.py` | 16% | 94% |
| `stats/cpl_diagnostics.py` | 37% | 100% |
| `stats/results.py` | 58% | 97% |
| `stats/chains.py` | 64% | 92% |
| `cosmology/core/parameters.py` | 67% | 96% |
| `likelihoods/joint.py` | 70% | 100% |
| `stats/priors.py` | 73% | 100% |
| `data/covariance.py` | 78% | 93% |
| `stats/posterior.py` | 85% | 100% |

**And the package is typed.** `py.typed` ships, which tells every
downstream type checker to trust these annotations -- so it had to be
true first. It was 58 of 201 public callables fully annotated; it is
**201 of 201** now, and two tests hold it there: one that every
annotation resolves (including the ones deferred to `TYPE_CHECKING`,
which is what a checker sees and `get_type_hints` cannot), and one
that nothing on the public surface is left unannotated. Adding a
public method without annotating it fails the suite.

`CosmoFit.typing` spells out the two aliases the whole surface is
written in. `E(0.5)` returns `np.float64` and `E([0.1, 0.2])` returns
an `ndarray`, so annotating the return as `np.ndarray` alone would
have been wrong for the commonest call in the library.

Getting there also found four annotations that were simply false --
`load_chain` declared `MCMCResult` and returns a `StoredSampler`,
`theory.fields.build_system` declared one object and returns two --
which is exactly the class of thing `py.typed` would have started
telling other people's type checkers.

What is left before the version number changes is not physics:

Everything that could be prepared has been, and verified rather than
assumed:

* the name `cosmofit` is free on PyPI and TestPyPI;
* both artifacts build and pass `twine check --strict`, and the
  wheel is 22 MB against PyPI's 100 MB per-file limit -- the
  Pantheon+ covariance is 10 MB of that on its own;
* the sdist carries the bundled data *and* `CITATION.cff`,
  `REFERENCES.md` and the changelog, which it did not before there
  was a `MANIFEST.in`;
* the wheel was installed into a clean virtual environment, outside
  this source tree, and fits ΛCDM to CC+DESI+Pantheon++Planck
  (1671 points, H₀ = 67.5, Ω_m = 0.313) with `py.typed` present and
  the `theory` extra correctly absent;
* GitHub Pages is enabled, with Actions as the source.

**Run `pages` before `publish`**, so the `Documentation` link in
PyPI's sidebar is live rather than a 404.

What is left is three things nobody but the maintainer can do:

* **A Trusted Publisher on PyPI.** The `publish` workflow uploads
  through OIDC rather than an API token, so there is no secret to
  add here -- but PyPI has to be told to trust this repository, this
  workflow filename, and the environment name. Until then the
  workflow will run and be rejected at the last step.
* **Running the two workflows.** Both are `workflow_dispatch` only,
  deliberately: a version number, once used, can never be reused,
  and a site once deployed is public under this repository's name.
  `publish` offers TestPyPI first and defaults to it.
* **`main` catching up.** It is deliberately at v0.22.0, and moving
  it *is* the v1.0.0 moment rather than a chore that follows one.
  When it does, the `Changelog` and `Examples` entries in
  `[project.urls]` move from `dev` to `main` with it -- they point
  at `dev` today because `main` does not have those files.

## Contributing

**[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to get set up, why there
is no code formatter, and what a finished change looks like here.
The short version of the last one: a test that fails before the fix,
and a validation against something that does not share the machinery
being validated.

---

## References

CosmoFit bundles real observational data and implements published
cosmological models and statistical methods -- see
**[REFERENCES.md](REFERENCES.md)** for every dataset, model, and
methodology paper used, with links, and where each is used in the
code. If you use CosmoFit in a publication, cite the underlying
data/method papers relevant to what you used.

[CITATION.cff](CITATION.cff) describes the software itself, if you
also want to cite the library -- but the papers behind whatever you
actually used matter more.

---

## License

This project is licensed under the MIT License.
