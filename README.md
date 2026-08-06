# CosmoFit

> **Modern Cosmological Parameter Estimation in Python**

**CosmoFit** is an open-source Python library for cosmological parameter estimation and Bayesian inference. It provides a modular framework for fitting cosmological models to observational data using Markov Chain Monte Carlo (MCMC) techniques.

The project is designed to make cosmological analyses simple, reproducible, and extensible while remaining flexible for research applications.

> **Current Version:** v0.14.0

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
* Dedicated sampling backend (`stats.sampler`), decoupled from `Fitter` and swappable (custom
  `emcee` moves today, room for other backends later)
* Multi-core MCMC (`fitter.run_mcmc(n_processes=...)`): evaluates walkers across multiple CPU
  cores, each worker building its own `Fitter` once rather than repeatedly shipping the whole
  (potentially large) likelihood across processes
* Consolidated result object (`fitter.result`): best-fit + MCMC posterior in one printable,
  JSON-serializable snapshot
* Custom models (`define_model`): fit a brand-new, not-in-the-library `E(z)` -- with its own
  extra parameters -- against every built-in dataset/likelihood/MCMC, no library changes needed
* Graphical interface (`app/streamlit_app.py`, `pip install -e ".[gui]"`): tick datasets, pick or
  write a model, edit free parameters, and run the MCMC fit + plots with one click, no code
* Covariance matrix support, including precision (inverse-covariance) matrices as shipped
  directly by some data releases
* Dedicated plotting module (`fitter.plots`): MCMC chain/corner plots, Hubble diagram, H(z)
  diagram, BAO distance plot, Planck pull plot, w(z) evolution, deceleration parameter
* Model comparison plots (`fitter.plots.compare_*`): any of the figures above, overlaying this
  fit's curve with one or more other models' curves on the same data/axes -- defaults to this
  model vs. a quick LCDM reference, or an arbitrary N-model comparison via `other_fits=[...]`

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

All five notebooks below are Colab-ready: click a badge to open it directly in Google Colab and
*Runtime → Run all* — no local setup needed.

| Notebook | What it covers |
|---|---|
| [`quickstart.ipynb`](examples/quickstart.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/quickstart.ipynb) | The shortest path to a real MCMC fit: flat ΛCDM on CC+DESI, a couple of minutes end to end. Start here. |
| [`dataset_zoo.ipynb`](examples/dataset_zoo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/dataset_zoo.ipynb) | A tour of all six built-in datasets (CC, DESI, SDSS BAO, Pantheon+, DES-SN5YR, Planck), each plotted on its own, plus which combinations to avoid. |
| [`model_zoo_comparison.ipynb`](examples/model_zoo_comparison.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/model_zoo_comparison.ipynb) | All six cosmological models (LCDM, wCDM, CPL, JBP, BA, GCG) fit to the same data and compared with AIC/BIC, plus a full MCMC for GCG. |
| [`cpl_mcmc_analysis.ipynb`](examples/cpl_mcmc_analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/cpl_mcmc_analysis.ipynb) | The deep dive: CPL fit to CC+DESI+Pantheon+, convergence diagnostics, every `fit.plots` figure, model comparison, and an independent Planck cross-check. |
| [`cpl_4data.ipynb`](examples/cpl_4data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salihyesil59/CosmoFit/blob/main/examples/cpl_4data.ipynb) | Publication-scale variant of the above: CPL fit to all four datasets *jointly* (Planck included, making `rd`/`Omega_b` constrainable), a much longer chain, and multi-core MCMC (`n_processes`). |

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

It lets you: tick which built-in datasets to fit; pick one of the
six built-in models, or write a custom one directly as an `E(z)`
expression (e.g. `sqrt(Omega_m*(1+z)**3 + ...)`, using the same
mechanism as `model_from_expression()` below) with your own extra
parameters; tick which parameters are free and edit their initial
values/bounds in a table; and, with one click, run the MCMC fit and
render whichever result plots apply to your model/dataset choice.
Results (best fit, posterior summary, convergence) can be downloaded
as JSON via `FitResult.save_json()`.

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
outside this library's control). The new `examples/cpl_4data.ipynb`
notebook is a publication-scale variant of `cpl_mcmc_analysis.ipynb`
using this: all four datasets (including Planck, which makes `rd`
and `Omega_b` constrainable and lets them join the free parameters
too) and a much longer chain (`nwalkers=64`, `nsteps=12000`), run in
parallel across every available core.

The package structure may continue to evolve before the first stable **v1.0.0** release.

---

## Roadmap

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