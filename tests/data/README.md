# Test fixtures

**`Dl_planck2015fit.dat`** — a CMB power spectrum (`ell`, `D_l^TT`,
`D_l^TE`, `D_l^EE`, `l = 2..2508`) computed by CLASS at the Planck
2015 best-fit ΛCDM, redistributed from
[heatherprince/planck-lite-py](https://github.com/heatherprince/planck-lite-py).

It exists so that `test_planck_lite.py` can check CosmoFit's
`plik_lite` implementation against a *published* log-likelihood value
for a *fixed* input spectrum, with no Boltzmann code in the loop. That
separates the two things that can go wrong — the binning/covariance
algebra, and the CAMB parameter translation — instead of testing them
as one blob and having a failure mean either.

Reference values (from `planck_lite_py.py`'s own `test()`):

| selection | log-likelihood |
|---|---|
| 2018 TTTEEE, high-ℓ | −291.33481235418026 |
| 2018 TT, high-ℓ | −101.58123068722583 |

**`HzTable_MM_BC03.dat`** — Moresco's 15 cosmic chronometer H(z)
measurements (BC03 models), copied unmodified from
[gitlab.com/mmoresco/CCcovariance](https://gitlab.com/mmoresco/CCcovariance)
at commit `881413330a7f1e1e5203607d6964db49b4c6c461`.

It exists so that `test_datasets.py` can check the CC covariance
recipe against Moresco's own flat-ΛCDM fit to these points, quoted in
that repository's `CC_fit.ipynb` (an emcee run, so a few hundredths of
sampling noise):

| covariance | H0 |
|---|---|
| with systematics | 65.995 +5.545 −5.591 |
| statistical only | 66.171 +3.770 −3.956 |
