# CosmoFit

> **Modern Cosmological Parameter Estimation in Python**

**CosmoFit** is an open-source Python library for cosmological parameter estimation and Bayesian inference. It provides a modular framework for fitting cosmological models to observational data using Markov Chain Monte Carlo (MCMC) techniques.

The project is designed to make cosmological analyses simple, reproducible, and extensible while remaining flexible for research applications.

> **Current Version:** v0.4.0

---

## Features

* Modular cosmological model framework, with curvature (Omega_k) supported end-to-end (flat/open/closed E(z) and D_M(z))
* Flexible parameter management
* Built-in observational datasets

  * Cosmic Chronometers (CC)
  * BAO (DESI)
  * Supernova (Pantheon+)
  * CMB distance priors (Planck 2018 R, l_A, omega_b_h2)

* Modular likelihood architecture
* Bayesian parameter estimation with MCMC, with autocorrelation-time convergence diagnostics
* Covariance matrix support
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

---

## Quick Example

```python
from cosmology import CPL
from stats.fitter import Fitter

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

The package structure may continue to evolve before the first stable **v1.0.0** release.

---

## Roadmap

### v0.5.0

* Refactor package structure (`src` layout)
* Unified import interface (`from CosmoFit import ...`)
* Dedicated sampler module
* Improved result interface

### v0.6.0

* Additional cosmological models
* Additional observational datasets
* Performance improvements

### v1.0.0

* Stable public API
* Complete documentation
* Production-ready release

---

## License

This project is licensed under the MIT License.