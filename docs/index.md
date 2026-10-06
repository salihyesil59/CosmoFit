# CosmoFit

A modular Python library for cosmological parameter estimation:
twenty-one bundled datasets, twenty models written out by hand, three
routes to one that is not here, and the sampling, evidence and tension
machinery to judge between them.

```python
from cosmofit import CPL, Fitter

fit = Fitter(
    model=CPL,
    datasets=["cc", "desi", "pantheon"],
    free_params=["H0", "Omega_m", "w0", "wa"],
    initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "wa": 0.0, "rd": 147.1},
)
fit.best_fit()

from cosmofit.compat import sample_on_core   # the 2.0 core samples it
sample_on_core(fit, "mcmc", {"Rminus1_stop": 0.01}, output="chains/cpl")

fit.summary()
fit.plots.corner()
```

Or with no Python: `cosmofit run examples/yaml/cpl_bao_sn_cmb.yaml`.

## Where to read what

This site is the **API reference** -- one page per subpackage, with
every public class and function, its parameters and its defaults.

The narrative documentation is elsewhere, and is better at being
narrative:

- the [README](readme.md) for what the library is, what is in it, and
  the physics behind each piece;
- the [notebooks](https://github.com/salihyesil59/CosmoFit/tree/main/examples)
  -- seventeen of them, in five sections, every one executed end to
  end against real data and Colab-ready;
- the [changelog](changelog.md) for how it got here, including how
  several of the bugs were found rather than only that they were
  fixed;
- [REFERENCES.md](https://github.com/salihyesil59/CosmoFit/blob/main/REFERENCES.md)
  for every dataset, model and method paper, with links and where each
  is used in the code.

## Installing

```bash
pip install cosmofit
```

Everything in the core works with no extras at all. The four
optional ones are `cmb` (CAMB, for the from-scratch CMB spectra --
the compressed Planck distance priors need nothing), `theory`
(sympy, for deriving a model from an action), `evidence` (dynesty,
for nested sampling) and `speed` (numba, worth about 1.7x on
growth-heavy fits and nothing elsewhere):

```bash
pip install "cosmofit[cmb,theory,evidence,speed]"
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: The rest

readme
changelog
```
