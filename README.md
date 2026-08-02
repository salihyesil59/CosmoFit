# CosmoFit

> **Modern Cosmological Parameter Estimation in Python**

**CosmoFit** is an open-source Python library for cosmological parameter estimation and Bayesian inference. It provides a modular framework for fitting cosmological models to observational data using Markov Chain Monte Carlo (MCMC) techniques.

The project is designed to make cosmological analyses simple, reproducible, and extensible while remaining flexible for research applications.

> **Current Version:** v0.2.0

---

## Features

* Modular cosmological model framework
* Flexible parameter management
* Built-in observational datasets

  * Cosmic Chronometers (CC)
  * BAO (DESI)
  * Supernova (Pantheon+)

* Modular likelihood architecture
* Bayesian parameter estimation with MCMC
* Covariance matrix support
* Initial plotting and analysis utilities

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
    datasets=["cc", "desi", "pantheon"],
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
    },
)

fitter.run_mcmc(
    nwalkers=48,
    nsteps=650,
    burnin=100,
)

fitter.best_fit()
fitter.summary()
```

---

## Project Status

CosmoFit is currently under active development.

Version **v0.2.0** introduces the first working Bayesian inference framework together with a modular architecture for cosmological parameter estimation.

The package structure may continue to evolve before the first stable **v1.0.0** release.

---

## Roadmap

### v0.3.0

* Refactor package structure (`src` layout)
* Unified import interface (`from CosmoFit import ...`)
* Dedicated sampler module
* Improved result interface

### v0.4.0

* Modular plotting package
* Advanced diagnostics
* Improved visualization tools

### v0.5.0

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