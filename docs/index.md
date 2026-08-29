# CosmoFit

A modular Python library for cosmological parameter estimation:
twenty-one bundled datasets, twenty models written out by hand, three
routes to one that is not here, and the sampling, evidence and tension
machinery to judge between them.

```python
from CosmoFit import CPL, Fitter

fit = Fitter(
    model=CPL,
    datasets=["cc", "desi", "pantheon"],
    free_params=["H0", "Omega_m", "w0", "wa"],
    initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "wa": 0.0, "rd": 147.1},
)
fit.run_mcmc(nwalkers=48, nsteps=6000, burnin=1000)
fit.summary()
fit.plots.corner()
```

## Where to read what

This site is the **API reference** -- one page per subpackage, with
every public class and function, its parameters and its defaults.

The narrative documentation is elsewhere, and is better at being
narrative:

- the [README](readme.md) for what the library is, what is in it, and
  the physics behind each piece;
- the [notebooks](https://github.com/salihyesil59/CosmoFit/tree/dev/examples)
  -- seventeen of them, in five sections, every one executed end to
  end against real data and Colab-ready;
- the [changelog](changelog.md) for how it got here, including how
  several of the bugs were found rather than only that they were
  fixed;
- [REFERENCES.md](https://github.com/salihyesil59/CosmoFit/blob/dev/REFERENCES.md)
  for every dataset, model and method paper, with links and where each
  is used in the code.

## Installing

```bash
pip install -e ".[cmb,theory,evidence,speed]"
```

Everything in the core works with no extras at all. The four above
are: `cmb` (CAMB, for the from-scratch CMB spectra -- the compressed
Planck distance priors need nothing), `theory` (sympy, for deriving a
model from an action), `evidence` (dynesty, for nested sampling) and
`speed` (numba, worth about 1.7x on growth-heavy fits and nothing
elsewhere).

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
