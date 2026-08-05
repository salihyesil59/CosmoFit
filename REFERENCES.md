# References

CosmoFit bundles real observational data and implements textbook (and
some less-textbook) cosmological models and statistical techniques.
This page collects, in one place, every paper, dataset, and
methodology reference used in the code -- what for, and where in the
codebase it's cited from. Every arXiv identifier below was verified
directly against arXiv's own API before being added here.

If you use CosmoFit in a publication, please cite the underlying
data/method papers relevant to what you used, not just this
repository.

---

## Datasets

### Cosmic Chronometers (CC)

- **Favale, Gómez-Valent & Migliaccio (2023)**, *Cosmic chronometers
  to calibrate the ladders and measure the curvature of the Universe.
  A model-independent study*, MNRAS 523, 3406.
  [arXiv:2301.09591](https://arxiv.org/abs/2301.09591)
- Used by: [`data/cc/favale2023/`](src/CosmoFit/data/cc/favale2023/),
  [`likelihoods/cc.py`](src/CosmoFit/likelihoods/cc.py)

### DESI 2024 BAO

- **DESI Collaboration et al. (2024)**, *DESI 2024 VI: Cosmological
  Constraints from the Measurements of Baryon Acoustic Oscillations*,
  JCAP 02 (2025) 021.
  [arXiv:2404.03002](https://arxiv.org/abs/2404.03002)
- Data source: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
  (also linked from DESI's own
  [official likelihood repository](https://github.com/cosmodesi/desi-kp-cosmological-likelihoods))
- Used by: [`data/bao/desi2024/`](src/CosmoFit/data/bao/desi2024/),
  [`likelihoods/desi.py`](src/CosmoFit/likelihoods/desi.py)

### SDSS BAO (BOSS DR12 + eBOSS DR16 LRG/QSO)

- **Alam et al. (2017)**, *The clustering of galaxies in the completed
  SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological
  analysis of the DR12 galaxy sample*, MNRAS 470, 2617 (BOSS DR12,
  z=0.38 and z=0.51 bins).
  [arXiv:1607.03155](https://arxiv.org/abs/1607.03155)
- **eBOSS Collaboration / Alam et al. (2021)**, *The Completed SDSS-IV
  extended Baryon Oscillation Spectroscopic Survey: Cosmological
  Implications from two Decades of Spectroscopic Surveys at the Apache
  Point Observatory*, Phys. Rev. D 103, 083533 (eBOSS DR16 LRG
  z=0.698, QSO z=1.48).
  [arXiv:2007.08991](https://arxiv.org/abs/2007.08991)
- Data source: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
  (data as originally distributed with
  [CosmoMC](https://github.com/cmbant/CosmoMC))
- Used by: [`data/bao/sdss/`](src/CosmoFit/data/bao/sdss/),
  [`likelihoods/sdss_bao.py`](src/CosmoFit/likelihoods/sdss_bao.py)
- **Note:** BOSS DR12's usual third bin (z=0.61) is deliberately
  omitted from this combination -- see the module docstring in
  `data/loader.py` for why.

### Pantheon+SH0ES (Type Ia Supernovae)

- **Brout et al. (2022)**, *The Pantheon+ Analysis: Cosmological
  Constraints*, ApJ 938, 110.
  [arXiv:2202.04077](https://arxiv.org/abs/2202.04077)
- Data source: [PantheonPlusSH0ES/DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)
  (official data release)
- Used by: [`data/sn/pantheon-plus-sh0es/`](src/CosmoFit/data/sn/pantheon-plus-sh0es/),
  [`likelihoods/pantheon.py`](src/CosmoFit/likelihoods/pantheon.py)

### DES-SN5YR (Type Ia Supernovae)

- **Sanchez et al. (2024)**, *The Dark Energy Survey Supernova Program:
  Data Release*, ApJ 975, 5.
  [arXiv:2406.05046](https://arxiv.org/abs/2406.05046)
- **DES Collaboration et al. (2024)**, *The Dark Energy Survey:
  Cosmology Results with ~1500 New High-redshift Type Ia Supernovae
  Using the Full 5-year Dataset*, ApJL 973, L14.
  [arXiv:2401.02929](https://arxiv.org/abs/2401.02929)
- Data source: [des-science/DES-SN5YR](https://github.com/des-science/DES-SN5YR)
  (official data release)
- Used by: [`data/sn/des-sn5yr/`](src/CosmoFit/data/sn/des-sn5yr/),
  [`likelihoods/des_sn5yr.py`](src/CosmoFit/likelihoods/des_sn5yr.py)
- **Note:** don't combine with Pantheon+ in the same fit -- see the
  module docstring for the sample overlap this would double-count.

### Planck 2018 CMB distance priors

- **Chen, Huang & Wang (2019)**, *Distance Priors from Planck Final
  Release*, JCAP 02 (2019) 028.
  [arXiv:1808.05724](https://arxiv.org/abs/1808.05724)
- Used by: [`data/cmb/planck2018/`](src/CosmoFit/data/cmb/planck2018/),
  [`likelihoods/planck.py`](src/CosmoFit/likelihoods/planck.py)

---

## Cosmological models

### LCDM, wCDM

Standard Friedmann-Lemaître-Robertson-Walker background expansion; no
single-paper citation applies. wCDM (constant dark-energy equation of
state w0) is the textbook one-parameter generalization of the
cosmological constant.

### CPL (Chevallier-Polarski-Linder)

w(z) = w0 + wa z/(1+z)

- **Chevallier & Polarski (2001)**, *Accelerating Universes with
  Scaling Dark Matter*, Int. J. Mod. Phys. D 10, 213.
  [arXiv:gr-qc/0009008](https://arxiv.org/abs/gr-qc/0009008)
- **Linder (2003)**, *Exploring the Expansion History of the
  Universe*, Phys. Rev. Lett. 90, 091301.
  [arXiv:astro-ph/0208512](https://arxiv.org/abs/astro-ph/0208512)
- Implemented in: [`cosmology/models/cpl.py`](src/CosmoFit/cosmology/models/cpl.py)

### JBP (Jassal-Bagla-Padmanabhan)

w(z) = w0 + wa z/(1+z)^2

- **Jassal, Bagla & Padmanabhan (2005)**, *WMAP constraints on low
  redshift evolution of dark energy*, MNRAS 356, L11.
  [arXiv:astro-ph/0404378](https://arxiv.org/abs/astro-ph/0404378)
- Implemented in: [`cosmology/models/jbp.py`](src/CosmoFit/cosmology/models/jbp.py)

### BA (Barboza-Alcaniz)

w(z) = w0 + wa z(1+z)/(1+z^2)

- **Barboza & Alcaniz (2008)**, *A parametric model for dark energy*,
  Phys. Lett. B 666, 415.
  [arXiv:0805.1713](https://arxiv.org/abs/0805.1713)
- Implemented in: [`cosmology/models/ba.py`](src/CosmoFit/cosmology/models/ba.py)

### GCG (Generalized Chaplygin Gas)

p = -A/rho^alpha, a unified dark matter/dark energy fluid.

- **Kamenshchik, Moschella & Pasquier (2001)**, *An Alternative to
  Quintessence*, Phys. Lett. B 511, 265.
  [arXiv:gr-qc/0103004](https://arxiv.org/abs/gr-qc/0103004)
- **Bento, Bertolami & Sen (2002)**, *Generalized Chaplygin Gas,
  Accelerated Expansion and Dark Energy-Matter Unification*, Phys.
  Rev. D 66, 043507.
  [arXiv:gr-qc/0202064](https://arxiv.org/abs/gr-qc/0202064)
- Implemented in: [`cosmology/models/gcg.py`](src/CosmoFit/cosmology/models/gcg.py)

---

## Methodology

### Photon-decoupling redshift z\* fitting formula

- **Hu & Sugiyama (1996)**, *Small-Scale Cosmological Perturbations:
  An Analytic Approach*, ApJ 471, 542, Eq. (E-1).
  [arXiv:astro-ph/9510117](https://arxiv.org/abs/astro-ph/9510117)
- Implemented in: [`cosmology/calculators/recombination.py`](src/CosmoFit/cosmology/calculators/recombination.py)
  (`z_star()`)

### Sound-horizon fitting formula (comparison only)

- **Eisenstein & Hu (1998)**, *Baryonic Features in the Matter
  Transfer Function*, ApJ 496, 605, Eq. (26).
  [arXiv:astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112)
- Implemented in: [`cosmology/calculators/recombination.py`](src/CosmoFit/cosmology/calculators/recombination.py)
  (`sound_horizon_eh98()` -- CosmoFit's primary sound-horizon
  calculation is a direct radiation-aware integral, not this fit; see
  that module's docstring)

### Analytic marginalization over a supernova absolute-magnitude offset

- **Conley et al. (2011)**, *Supernova Constraints and Systematic
  Uncertainties from the First 3 Years of the Supernova Legacy
  Survey*, ApJS 192, 1, Appendix (Eq. A9-A12).
  [arXiv:1104.1443](https://arxiv.org/abs/1104.1443)
- Implemented in: [`likelihoods/base.py`](src/CosmoFit/likelihoods/base.py)
  (`AnalyticOffsetMixin`), used by
  [`likelihoods/pantheon.py`](src/CosmoFit/likelihoods/pantheon.py) and
  [`likelihoods/des_sn5yr.py`](src/CosmoFit/likelihoods/des_sn5yr.py)

### MCMC

- **Foreman-Mackey, Hogg, Lang & Goodman (2013)**, *emcee: The MCMC
  Hammer*, PASP 125, 306.
  [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- CosmoFit's `Fitter.run_mcmc()` is a thin wrapper around the
  [`emcee`](https://emcee.readthedocs.io/) package.
