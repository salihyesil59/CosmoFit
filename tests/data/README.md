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
