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

### DESI DR2 BAO (2025)

- **DESI Collaboration et al. (2025)**, *DESI DR2 Results II:
  Measurements of Baryon Acoustic Oscillations and Cosmological
  Constraints*.
  [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- Three years of observations and >14 million galaxies and quasars --
  twice the DR1 sample, and the measurement the strengthened
  evolving-dark-energy claim rests on. Same file format and same
  source as DR1, so it loads through the identical code path.
- **Do not combine with DESI DR1** (`"desi2024"`): DR2 contains every
  DR1 galaxy.
- Data source: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
  (`desi_bao_dr2/`)
- Used by: [`data/bao/desi_dr2/`](src/CosmoFit/data/bao/desi_dr2/),
  [`likelihoods/desi.py`](src/CosmoFit/likelihoods/desi.py)

### Low-redshift BAO (6dFGS + SDSS DR7 MGS)

- **Beutler et al. (2011)**, *The 6dF Galaxy Survey: Baryon Acoustic
  Oscillations and the Local Hubble Constant*, MNRAS 416, 3017
  (z=0.106).
  [arXiv:1106.3366](https://arxiv.org/abs/1106.3366)
- **Ross et al. (2015)**, *The Clustering of the SDSS DR7 Main Galaxy
  Sample I: A 4 per cent Distance Measure at z = 0.15*, MNRAS 449,
  835.
  [arXiv:1409.3242](https://arxiv.org/abs/1409.3242)
- The only BAO measurements below z = 0.2 in the library (DESI starts
  at z = 0.295, BOSS at z = 0.38), and independent of both, so they
  can be combined with either.
- Measurement values and 6dFGS's Eisenstein-Hu sound-horizon rescale
  (153.9/149.8) as tabulated by
  [CobayaSampler/cobaya](https://github.com/CobayaSampler/cobaya)
  (`bao.sixdf_2011_bao`, `bao.sdss_dr7_mgs`). The MGS release is a
  tabulated non-Gaussian likelihood; the Gaussian compression used
  here is the standard one.
- Used by: [`data/bao/lowz/`](src/CosmoFit/data/bao/lowz/),
  [`likelihoods/bao_lowz.py`](src/CosmoFit/likelihoods/bao_lowz.py)

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

### eBOSS DR16 tabulated BAO (ELG, Lyman-alpha)

The two DR16 tracers released as likelihood *surfaces* rather than a
mean and a covariance, because a Gaussian would misrepresent them.

- **de Mattia et al. (2020)**, *The Completed SDSS-IV extended Baryon
  Oscillation Spectroscopic Survey: measurement of the BAO and growth
  rate of structure of the emission line galaxy sample from the
  anisotropic power spectrum between redshift 0.6 and 1.1*, MNRAS 501,
  5616. D_V/r_d = 18.33 (+0.57/-0.62) at z_eff = 0.845, from a
  1.4-sigma BAO detection.
  [arXiv:2007.09008](https://arxiv.org/abs/2007.09008)
- **du Mas des Bourboux et al. (2020)**, *The Completed SDSS-IV
  Extended Baryon Oscillation Spectroscopic Survey: Baryon Acoustic
  Oscillations with Lyman-alpha Forests*, ApJ 901, 153.
  D_M/r_d = 37.5 +- 1.1 and D_H/r_d = 8.99 +- 0.19 at z_eff = 2.334,
  combining the forest auto-correlation with its cross-correlation
  with quasars.
  [arXiv:2007.08995](https://arxiv.org/abs/2007.08995)
- Data source: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
- **de Mattia et al. (2020)** again, for the ELG **full-shape**
  (RSD + BAO) grid: D_M/r_d = 19.5 +- 1.0, D_H/r_d = 19.6
  (-2.1/+2.2), f*sigma8 = 0.315 +- 0.095 at z_eff = 0.85, from the
  consensus of the Fourier- and configuration-space analyses.
  [arXiv:2007.09008](https://arxiv.org/abs/2007.09008)
- Used by: [`data/bao/sdss/`](src/CosmoFit/data/bao/sdss/),
  [`likelihoods/eboss_dr16.py`](src/CosmoFit/likelihoods/eboss_dr16.py)
- **Note on the full-shape grid:** it is the one dataset in this
  package that is *not* shipped as released. The original is 60 MB
  of ASCII with 10.3% of its probabilities underflowed to exact
  zero, which have no logarithm.
  [`tools/convert_eboss_elg_fs_grid.py`](tools/convert_eboss_elg_fs_grid.py)
  converts it to a 1.95 MB compressed archive, is committed so the
  shipped file can be regenerated, and documents both lossy steps
  along with the check that the marginals survive them unchanged.
- **Note:** eBOSS quote the Lyman-alpha constraint above from a joint
  fit but ship only the auto and cross surfaces separately, so this
  library multiplies them. That treats them as independent;
  `tests/test_eboss_tables.py` checks the assumption rather than
  making it, by recovering the published errors from the product.

### Holographic dark energy (HDE)

- **Li (2004)**, *A Model of Holographic Dark Energy*, Phys. Lett. B
  603, 1. The paper that identifies the future event horizon as the
  only infrared cutoff giving an accelerating universe.
  [arXiv:hep-th/0403127](https://arxiv.org/abs/hep-th/0403127)
- **Wang, Mortsell, et al. (2017)**, *Holographic Dark Energy*,
  Phys. Rept. 696, 1 (review).
  [arXiv:1612.00345](https://arxiv.org/abs/1612.00345)
- Used by: [`cosmology/models/hde.py`](src/CosmoFit/cosmology/models/hde.py)
- **Note:** the only background model here without a closed-form
  `E(z)`; `Omega_DE` is obtained by solving
  `dOmega/dln a = Omega(1-Omega)(1 + 2 sqrt(Omega)/c)` on each
  parameter change. Validated against the model's *definition*
  rather than the ODE -- `tests/test_hde.py` computes the future
  event horizon by quadrature from the solved `E(z)` and checks
  `H L = c / sqrt(Omega_DE)`.

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

### Union3 (Type Ia Supernovae, binned)

- **Rubin et al. (2023)**, *Union Through UNITY: Cosmology with 2,000
  SNe Using a Unified Bayesian Framework* (ApJ, accepted).
  [arXiv:2311.12098](https://arxiv.org/abs/2311.12098)
- 2087 supernovae from 24 datasets, fit with the UNITY1.5 Bayesian
  hierarchical model (light-curve standardization, host-mass
  dependence, selection effects and outliers marginalized internally)
  and released as 22 binned distance moduli with a 22x22 covariance.
- The third of the three SN compilations the DESI dark-energy results
  are argued with, and the one that sits between Pantheon+ and
  DES-SN5YR in how far it pulls from a cosmological constant.
- **Do not combine with Pantheon+ or DES-SN5YR** -- substantial
  supernova overlap with both.
- Data source: [CobayaSampler/sn_data](https://github.com/CobayaSampler/sn_data)
  (`Union3/`)
- Used by: [`data/sn/union3/`](src/CosmoFit/data/sn/union3/),
  [`likelihoods/union3.py`](src/CosmoFit/likelihoods/union3.py)

### Planck 2018 CMB distance priors

- **Chen, Huang & Wang (2019)**, *Distance Priors from Planck Final
  Release*, JCAP 02 (2019) 028.
  [arXiv:1808.05724](https://arxiv.org/abs/1808.05724)
- Used by: [`data/cmb/planck2018/`](src/CosmoFit/data/cmb/planck2018/),
  [`likelihoods/planck.py`](src/CosmoFit/likelihoods/planck.py)

### Planck 2018 plik_lite TT/TE/EE bandpowers

- **Planck Collaboration (2020)**, *Planck 2018 results. V. CMB power
  spectra and likelihoods*, A&A 641, A5.
  [arXiv:1907.12875](https://arxiv.org/abs/1907.12875)
- **Planck Collaboration (2020)**, *Planck 2018 results. VI.
  Cosmological parameters*, A&A 641, A6.
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- The measured CMB power spectra themselves -- 613 binned TT/TE/EE
  bandpowers over l = 30-2508 with their full covariance -- rather
  than the three-number compression above. `plik_lite` is the
  foreground-marginalized variant, so only the calibration parameter
  `A_planck` is left for a downstream fit.
- Requires a Boltzmann code to predict C_l; CosmoFit calls **CAMB**
  (Lewis, Challinor & Lasenby 2000, ApJ 538, 473,
  [astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)) as an
  optional dependency.
- **Do not combine with the distance priors** -- they are a
  compression of exactly these bandpowers.
- The two **Commander low-multipole temperature bins** (l = 2-29)
  can be prepended with
  `dataset_kwargs={"planck_lite": {"use_low_ell": True}}`, taking
  the data vector to 615 bandpowers. They come from a different
  Planck likelihood with its own windows and an uncorrelated
  covariance, so the TT block gains a diagonal 2x2 corner and its
  own binning while TE and EE are untouched.
- Data source: [heatherprince/planck-lite-py](https://github.com/heatherprince/planck-lite-py),
  redistributing the [Planck Legacy Archive](https://pla.esac.esa.int/)
  PR3 release. This library's implementation reproduces
  `planck-lite-py`'s published log-likelihood values exactly for all
  four selections -- TT and TTTEEE, each with and without the low-l
  bins ([`tests/test_planck_lite.py`](tests/test_planck_lite.py)).
- Used by: [`data/cmb/plik_lite/`](src/CosmoFit/data/cmb/plik_lite/),
  [`likelihoods/planck_lite.py`](src/CosmoFit/likelihoods/planck_lite.py),
  [`cosmology/boltzmann.py`](src/CosmoFit/cosmology/boltzmann.py)

### Planck 2018 low-multipole EE (SimAll)

- **Planck Collaboration (2020)**, *Planck 2018 results. V. CMB power
  spectra and likelihoods*, A&A 641, A5.
  [arXiv:1907.12875](https://arxiv.org/abs/1907.12875)
- A *tabulated, non-Gaussian* likelihood: for each multipole
  `l = 2..29` and each value of `D_l^EE` on a `1e-4 muK^2` grid, the
  log-probability. Below `l = 30` there are only `2l+1` modes on the
  sky, so the C_l distribution is strongly skewed — and that regime
  carries essentially all of the CMB's information about the
  reionization optical depth `tau`. A mean and an error bar cannot
  represent it.
- CosmoFit's Gaussian `"tau"` dataset (`0.0544 +- 0.0073`) is a
  compression of this; the two must not be combined.
- **Validated by reconstruction:** profiling `tau` against this table
  plus the high-l bandpowers, with the primordial amplitude
  re-optimized, returns `tau = 0.0541 +- 0.0072` against Planck's
  published `0.0544 +- 0.0073` — with that number used nowhere as an
  input. See
  [`tests/test_planck_lowe.py`](tests/test_planck_lowe.py).
- Data source: [CobayaSampler/planck_native_data](https://github.com/CobayaSampler/planck_native_data)
  (release v1, `planck_2018_lowE.zip`), a Python translation of the
  public Planck `clik` likelihood
  `simall_100x143_offlike5_EE_Aplanck_B.clik`.
- Used by: [`data/cmb/lowE2018/`](src/CosmoFit/data/cmb/lowE2018/),
  [`likelihoods/planck_lowe.py`](src/CosmoFit/likelihoods/planck_lowe.py)

### Planck 2018 CMB lensing

- **Planck Collaboration (2020)**, *Planck 2018 results. VIII.
  Gravitational lensing*, A&A 641, A8.
  [arXiv:1807.06210](https://arxiv.org/abs/1807.06210)
- Nine bandpowers of `[L(L+1)]^2 C_L^phiphi / 2pi` over the
  conservative range `8 <= L <= 400`, from the SMICA
  minimum-variance TEB reconstruction. The CMB's *own* growth
  measurement: every other CMB dataset here constrains
  recombination and reaches the present only through a distance,
  while this constrains `sigma8 * Omega_m^0.25` directly.
- The likelihood includes the linear correction for the
  reconstruction's dependence on the fiducial CMB spectra used to
  normalize it. That correction vanishes at the fiducial cosmology
  by construction, which is why the tests deliberately check it away
  from there.
- Requires CAMB (`pip install "cosmofit[cmb]"`). Reproduces
  `chi2 = 8.8` for 9 bandpowers at Planck's own best-fit LCDM,
  against the ~9 Planck reports.
- Data source: [CobayaSampler/planck_supp_data_and_covmats](https://github.com/CobayaSampler/planck_supp_data_and_covmats)
  (`lensing/2018/`), redistributing the Planck Legacy Archive PR3
  release.
- Used by: [`data/cmb/lensing2018/`](src/CosmoFit/data/cmb/lensing2018/),
  [`likelihoods/planck_lensing.py`](src/CosmoFit/likelihoods/planck_lensing.py)

### ACT DR6 CMB lensing

- **Madhavacheril et al. (ACT Collaboration, 2024)**, *The Atacama
  Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological
  Parameters*, ApJ 962, 113.
  [arXiv:2304.05203](https://arxiv.org/abs/2304.05203)
- **Qu et al. (ACT Collaboration, 2024)**, *The Atacama Cosmology
  Telescope: A Measurement of the DR6 CMB Lensing Power Spectrum and
  its Implications for Structure Growth*, ApJ 962, 112.
  [arXiv:2304.05202](https://arxiv.org/abs/2304.05202)
- A second, **independent** lensing reconstruction — different
  telescope, different sky, different pipeline — and tighter than
  Planck's: 2.3% on the lensing amplitude, over `40 <= L <= 763`
  (baseline, 10 bandpowers) or `L < 1250` (extended, 13).
- Built on the lensing **convergence** `C_L^kappakappa`, where
  Planck's products use the **potential**
  `[L(L+1)]^2 C_L^phiphi / 2pi`. The two differ by `2pi/4`; the
  dataset carries which one its windows act on rather than leaving
  it to be remembered.
- Uses ACT's **CMB-marginalized** covariance, which already accounts
  for the reconstruction's dependence on the primary CMB. Combining
  with primary CMB data anyway is conservative, not wrong — see the
  module docstring. A Hartlap correction for the 796 simulations
  behind the covariance is applied.
- **Validated against the published amplitude:** fitting a single
  scaling of the theory returns `A_lens = 1.017 +- 0.026` against
  ACT's `1.013 +- 0.023`, with that number used nowhere as an input.
  A wrong convergence/potential conversion would put it at 0.65 or
  1.6. See [`tests/test_act_lensing.py`](tests/test_act_lensing.py).
- Data source: [NASA LAMBDA](https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html),
  `ACT_dr6_likelihood_v1.2.tgz`. The reference implementation is
  [ACTCollaboration/act_dr6_lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike).
- Used by: [`data/cmb/act_dr6_lensing/`](src/CosmoFit/data/cmb/act_dr6_lensing/),
  [`likelihoods/act_lensing.py`](src/CosmoFit/likelihoods/act_lensing.py)

### External single-number priors (H0, BBN, tau)

- **Riess et al. (2022)**, *A Comprehensive Measurement of the Local
  Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the
  Hubble Space Telescope and the SH0ES Team*, ApJ 934, L7
  (H0 = 73.04 +- 1.04).
  [arXiv:2112.04510](https://arxiv.org/abs/2112.04510)
- **Breuval et al. (2024)**, *Small Magellanic Cloud Cepheids Observed
  with the Hubble Space Telescope Provide a New Anchor for the SH0ES
  Distance Ladder*, ApJ 973, 30 (H0 = 73.17 +- 0.86).
  [arXiv:2404.08038](https://arxiv.org/abs/2404.08038)
- **TDCOSMO Collaboration / Birrer et al. (2025)**, *TDCOSMO 2025:
  Cosmological constraints from strong lensing time delays*, A&A 704,
  A63 (H0 = 71.6 +3.9/-3.3; symmetrized to +-3.6 here).
  [arXiv:2506.03023](https://arxiv.org/abs/2506.03023)
- **Schoeneberg (2024)**, *The 2024 BBN baryon abundance update*
  (omega_b h^2 = 0.02218 +- 0.00055, the prior DESI DR1/DR2 adopt).
  [arXiv:2401.15054](https://arxiv.org/abs/2401.15054)
- **Cooke, Pettini & Steidel (2018)**, *One Percent Determination of
  the Primordial Deuterium Abundance*, ApJ 855, 102
  (omega_b h^2 = 0.02166 +- 0.00019).
  [arXiv:1710.11129](https://arxiv.org/abs/1710.11129)
- **Planck Collaboration (2020)**, A&A 641, A6, Table 1
  (tau = 0.0544 +- 0.0073, the lowE constraint).
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- Used by: [`data/priors/`](src/CosmoFit/data/priors/),
  [`likelihoods/priors.py`](src/CosmoFit/likelihoods/priors.py)

### fsigma8 growth-rate compilation ("Gold-2018")

- **Sagredo, Nesseris & Sapone (2018)**, *Internal Robustness of
  Growth Rate data*, Phys. Rev. D 98, 083543.
  [arXiv:1806.10822](https://arxiv.org/abs/1806.10822)
- Data source: bundled with the public MontePython likelihood
  [snesseris/RSD-growth](https://github.com/snesseris/RSD-growth)
  (Arjona, García-Bellido & Nesseris 2020,
  [arXiv:2006.01762](https://arxiv.org/abs/2006.01762)), fetched
  byte-for-byte from `data/growth_SN/data_growth_2018_main.txt`,
  `Cij_WiggleZ.txt`, `Cij_SDSS.txt` there.
- 22 points, each from its own survey: Huterer, Shafer, Scolnic &
  Schmidt (2017, arXiv:1611.09862); Turnbull et al. (2012,
  arXiv:1111.0631) / Hudson & Turnbull (2013, arXiv:1203.4814);
  Davis et al. (2011, arXiv:1011.3114) / Hudson & Turnbull (2013);
  Feix, Nusser & Branchini (2015, arXiv:1503.05945); Howlett, Ross,
  Samushia, Percival & Manera (2015, arXiv:1409.3238); Song &
  Percival (2009, arXiv:0807.0810); Blake et al. (2013,
  arXiv:1309.5556, GAMA); Samushia, Percival & Raccanelli (2012,
  arXiv:1102.1014, SDSS LRG); Sánchez et al. (2014, arXiv:1312.4854,
  BOSS DR10/11); Chuang et al. (2016, arXiv:1312.4889, BOSS DR12
  CMASS); Blake et al. (2012, arXiv:1204.3674, WiggleZ, correlated
  z=0.44/0.60/0.73 bins); Pezzotta et al. (2017, arXiv:1612.05645,
  VIPERS); Okumura et al. (2016, arXiv:1511.08083, FastSound);
  Zhao et al. (2018, arXiv:1801.03043, eBOSS DR14 quasars,
  correlated tomographic bins).
- Used by: [`data/growth/gold2018/`](src/CosmoFit/data/growth/gold2018/),
  [`likelihoods/fsigma8.py`](src/CosmoFit/likelihoods/fsigma8.py)

### S8 weak-lensing prior

- **KiDS-1000** (default): Asgari et al. (2021), *KiDS-1000
  Cosmology: Cosmic shear constraints and comparison between two
  point statistics*, A&A 645, A104. S8 = 0.759 (+0.024, -0.021).
  [arXiv:2007.15633](https://arxiv.org/abs/2007.15633)
- **DES Y3**: DES Collaboration / Abbott et al. (2022), *Dark Energy
  Survey Year 3 Results: Cosmological Constraints from Galaxy
  Clustering and Weak Lensing*, Phys. Rev. D 105, 023520.
  S8 = 0.776 +/- 0.017.
  [arXiv:2105.13549](https://arxiv.org/abs/2105.13549)
- Used by: [`data/s8/`](src/CosmoFit/data/s8/),
  [`likelihoods/s8.py`](src/CosmoFit/likelihoods/s8.py)

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

### LogarithmicDE (logarithmic w(z))

- **Efstathiou (1999)**, *Constraining the equation of state of the
  Universe from Distant Type Ia Supernovae and Cosmic Microwave
  Background Anisotropies*, MNRAS 310, 842.
  [arXiv:astro-ph/9904356](https://arxiv.org/abs/astro-ph/9904356)
- w(z) = w0 + wa ln(1+z). The one w0-wa form in the library that does
  *not* saturate at high z, which makes it the control case for
  asking whether a measured `wa` reflects the data or the assumed
  shape.
- Used by: [`cosmology/models/logarithmic.py`](src/CosmoFit/cosmology/models/logarithmic.py)

### PEDE (Phenomenologically Emergent Dark Energy)

- **Li & Shafieloo (2019)**, *A Simple Phenomenological Emergent Dark
  Energy Model can Resolve the Hubble Tension*, ApJ 883, L3.
  [arXiv:1906.08275](https://arxiv.org/abs/1906.08275)
- Omega_de(z) = Omega_de0 [1 - tanh(log10(1+z))], with **no free
  dark-energy parameter** -- the same parameter count as LCDM.
- Used by: [`cosmology/models/pede.py`](src/CosmoFit/cosmology/models/pede.py)

### GEDE (Generalized Emergent Dark Energy)

- **Li & Shafieloo (2020)**, *Evidence for Emergent Dark Energy*,
  ApJ 902, 58.
  [arXiv:2001.05103](https://arxiv.org/abs/2001.05103)
- Contains both LCDM (`Delta -> 0`) and PEDE (`Delta = 1, z_t = 0`)
  as limits, so `Delta` is a continuous measure of the distance from
  a cosmological constant.
- Used by: [`cosmology/models/gede.py`](src/CosmoFit/cosmology/models/gede.py)

### LsCDM (sign-switching cosmological constant)

- **Akarsu, Kumar, Ozulker & Vazquez (2021)**, *Relaxing cosmological
  tensions with a sign switching cosmological constant*, Phys. Rev. D
  104, 123512.
  [arXiv:2108.09239](https://arxiv.org/abs/2108.09239)
- **Akarsu, Kumar, Ozulker, Vazquez & Yadav (2023)**, *Relaxing
  cosmological tensions with a sign switching cosmological constant:
  Improved results with Planck, BAO and Pantheon data*, Phys. Rev. D
  108, 023513.
  [arXiv:2211.05742](https://arxiv.org/abs/2211.05742)
- Lambda flips sign at `z_dagger ~ 2` (AdS below, dS above), which
  shrinks the sound horizon `r_d` and so raises the BAO-inferred H0 --
  a route to the H0 tension that late-time-only dark-energy models
  cannot take.
- Used by: [`cosmology/models/lscdm.py`](src/CosmoFit/cosmology/models/lscdm.py)

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

### IDE (Interacting Dark Energy)

- **Amendola (2000)**, *Coupled Quintessence*, Phys. Rev. D 62,
  043511.
  [arXiv:astro-ph/9908023](https://arxiv.org/abs/astro-ph/9908023)
- **Wang, Abdalla, Atrio-Barandela & Pavon (2016)**, *Dark Matter and
  Dark Energy Interactions: Theoretical Challenges, Cosmological
  Implications and Observational Signatures*, Rept. Prog. Phys. 79,
  096901.
  [arXiv:1603.08299](https://arxiv.org/abs/1603.08299)
- Q = 3 xi H rho_de, solved in closed form. Changes how *matter*
  dilutes, which no w(z) parametrization does -- so it has its own
  growth-of-structure signature.
- Used by: [`cosmology/models/ide.py`](src/CosmoFit/cosmology/models/ide.py)

### RunningVacuum (Lambda(H) = c0 + 3 nu H^2)

- **Sola (2013)**, *Cosmological constant and vacuum energy: old and
  new ideas*, J. Phys. Conf. Ser. 453, 012015.
  [arXiv:1306.1527](https://arxiv.org/abs/1306.1527)
- **Sola, Gomez-Valent & de Cruz Perez (2017)**, *First evidence of
  running cosmic vacuum: challenging the concordance model*, ApJ 836,
  43.
  [arXiv:1602.02103](https://arxiv.org/abs/1602.02103)
- One of the few dark-energy extensions whose extra parameter has a
  *predicted* magnitude (`|nu| ~ 10^-3`, from a one-loop
  renormalization-group estimate) rather than an arbitrary one.
- Used by: [`cosmology/models/rvm.py`](src/CosmoFit/cosmology/models/rvm.py)

### Cardassian (modified polytropic)

- **Freese & Lewis (2002)**, *Cardassian Expansion: a Model in which
  the Universe is Flat, Matter Dominated, and Accelerating*, Phys.
  Lett. B 540, 1.
  [arXiv:astro-ph/0201229](https://arxiv.org/abs/astro-ph/0201229)
- **Wang, Freese, Gondolo & Lewis (2003)**, *Future Type Ia Supernova
  Data as Tests of Dark Energy from Modified Friedmann Equations*,
  ApJ 594, 25 (the modified polytropic form implemented here).
  [arXiv:astro-ph/0302064](https://arxiv.org/abs/astro-ph/0302064)
- Acceleration from matter alone, via an extra term in the Friedmann
  equation. The modified polytropic form is used rather than the
  original, which is degenerate with wCDM at w = n - 1.
- Used by: [`cosmology/models/cardassian.py`](src/CosmoFit/cosmology/models/cardassian.py)

### DGP (braneworld gravity, self-accelerating branch)

- **Dvali, Gabadadze & Porrati (2000)**, *4D Gravity on a Brane in 5D
  Minkowski Space*, Phys. Lett. B 485, 208.
  [arXiv:hep-th/0005016](https://arxiv.org/abs/hep-th/0005016)
- **Deffayet (2001)**, *Cosmology on a Brane in Minkowski Bulk*, Phys.
  Lett. B 502, 199 (the cosmological solution).
  [arXiv:hep-th/0010186](https://arxiv.org/abs/hep-th/0010186)
- **Koyama & Maartens (2006)**, *Structure formation in the DGP
  cosmological model*, JCAP 01 (2006) 016 (the `mu(a)` used for
  growth).
  [arXiv:astro-ph/0511634](https://arxiv.org/abs/astro-ph/0511634)
- Acceleration with no dark energy at all, and LCDM's parameter count.
  Its growth is *suppressed* (`mu ~ 0.72` today), which is the real
  observational handle on it. Implemented as the historically
  important benchmark it is -- the self-accelerating branch is known
  to carry a ghost instability.
- Used by: [`cosmology/models/dgp.py`](src/CosmoFit/cosmology/models/dgp.py)

### FQExponential (f(Q) gravity, exponential model)

f(Q) = Q exp(λQ₀/Q), Q = 6H² -- a genuine modification of the
gravitational field equations (symmetric teleparallel gravity), not
a dark-energy fluid on top of standard GR. λ is not independently
free: it's fixed by Ωm via a Lambert-W closure condition, so this
has exactly as many free parameters as flat LCDM. Background level
only (the Friedmann equation is transcendental, solved numerically);
implemented and cross-checked against the source paper's own eq. 26
dust limit and a numerical closure check (a transcription error in
the Lambert-W closure formula was caught this way during
development -- see the commit history).

Growth of structure: `mu(a) = 1/f_Q` (G_eff/G_N = 1/f_Q, the
standard sub-horizon quasi-static result for f(Q) gravity),
scale-independent, closing exactly to mu=1 at this model's own GR
limit (lambda=0).

- **Anagnostopoulos, Basilakos & Saridakis (2021)**, *First evidence
  that non-metricity f(Q) gravity could challenge ΛCDM*, Phys. Lett.
  B 822, 136634.
  [arXiv:2104.15123](https://arxiv.org/abs/2104.15123)
- **Barros, Barreiro, Koivisto & Nunes (2020)**, *Testing F(Q)
  gravity with redshift space distortions*, Phys. Dark Univ. 30,
  100616.
  [arXiv:2004.07867](https://arxiv.org/abs/2004.07867)
- Implemented in: [`cosmology/models/fq.py`](src/CosmoFit/cosmology/models/fq.py)

### FRTLinear (f(R,T) gravity, linear model)

f(R,T) = R + 2λT -- gravity coupled directly to the trace of the
matter stress-energy tensor. Only the linear (in T) case is
implemented; the field equations for a two-fluid (matter + Λ-like)
universe were derived from the source paper's own general-perfect-
fluid equation and cross-checked against its dust-only special case
(their eq. 26).

Growth of structure: `mu(a) = 1 + 3*beta`, the same coupling already
in this model's own `E(z)^2` -- a stated simplification (f(R,T) does
not separately conserve the matter stress-energy tensor, so a full
covariant perturbation theory is more involved), not a full
derivation; see the class docstring.

- **Harko, Lobo, Nojiri & Odintsov (2011)**, *f(R,T) gravity*, Phys.
  Rev. D 84, 024020.
  [arXiv:1104.2669](https://arxiv.org/abs/1104.2669)
- **Asghari & Sheykhi (2025)**, *Growth of cosmic perturbations in
  the modified f(R,T) gravity*, Phys. Dark Univ. 48.
  [arXiv:2405.11840](https://arxiv.org/abs/2405.11840)
- Implemented in: [`cosmology/models/frt.py`](src/CosmoFit/cosmology/models/frt.py)

### FRHuSawicki (f(R) gravity, Hu-Sawicki model)

f(R) = -m²c₁(R/m²)ⁿ / (c₂(R/m²)ⁿ+1), the standard f(R) benchmark
model. **Background expansion is identical to LCDM's by
construction** -- the standard "designer f(R)" approach builds f(R)
to reproduce an assumed target background; `f_R0`/`n` are invisible
to any background-only probe (CC/BAO/SNe/Planck). **Growth of
structure is where this model actually differs from LCDM**: a
scale- and time-dependent, chameleon-screened `mu(a,k)`, derived
directly from this model's own background (not transcribed) and
numerically self-consistent (k->0 gives mu->1, k->infinity gives the
well-known mu->4/3, f_R0->0 gives mu->1 at any k), held at a fixed
fiducial pivot k=0.1 h/Mpc since this library's fsigma8 data are
single per-z points, not a P(k) shape. Stated explicitly in the
class docstring and surfaced in the GUI -- not silently omitted, but
not silently overstated either.

- **Hu & Sawicki (2007)**, *Models of f(R) Cosmic Acceleration that
  Evade Solar-System Tests*, Phys. Rev. D 76, 064004.
  [arXiv:0705.1158](https://arxiv.org/abs/0705.1158)
- **Pogosian & Silvestri (2008)**, *The pattern of growth in viable
  f(R) cosmologies*, Phys. Rev. D 77, 023503.
  [arXiv:0709.0296](https://arxiv.org/abs/0709.0296)
- Implemented in: [`cosmology/models/fr.py`](src/CosmoFit/cosmology/models/fr.py)

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

### Linear growth of structure

D'' + (2 + dlnH/dN) D' - (3/2) Omega_m(a) mu(a,k) D = 0, the standard
sub-horizon, quasi-static growth equation for the linear matter
density contrast, generalized to modified gravity through a single
`mu(a,k)` (G_eff/G_N) hook -- no single-paper citation applies to
the equation itself (a standard textbook result, e.g. Dodelson &
Schmidt, *Modern Cosmology*, 2nd ed., Ch. 7); see each
modified-gravity model's own section above for its `mu(a,k)`'s
citation.

- Implemented in: [`cosmology/calculators/growth.py`](src/CosmoFit/cosmology/calculators/growth.py)
  (`GrowthCalculator`)

### Sound horizon at the drag epoch (`r_d`)

- **Eisenstein & Hu (1998)**, *Baryonic Features in the Matter
  Transfer Function*, ApJ 496, 605 (the `z_drag` fitting formula, kept
  for comparison only -- it runs ~4% low relative to CAMB for
  Planck-like parameters).
  [arXiv:astro-ph/9709112](https://arxiv.org/abs/astro-ph/9709112)
- **Lewis, Challinor & Lasenby (2000)**, ApJ 538, 473 (CAMB -- the
  reference `z_drag` is calibrated against, and the reference the
  whole calculation is validated against).
  [arXiv:astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)
- **Aubourg et al. (2015)**, *Cosmological implications of baryon
  acoustic oscillation measurements*, Phys. Rev. D 92, 123516 (the
  "BAO + BBN" programme that computing `r_d` makes possible).
  [arXiv:1411.1074](https://arxiv.org/abs/1411.1074)
- CosmoFit integrates `r_d = int c_s/H dz` directly, with photons
  from `T_CMB`, massless neutrinos, and massive neutrinos carrying
  their *exact* Fermi-Dirac energy density (tabulated once at import
  rather than approximated). The mass-to-density relation
  `Sum m_nu / omega_nu h^2 = 93.04 eV` is derived from that integral
  rather than assumed. Only `z_drag` is a fit, for the same reason
  `z_star` is: it needs a full recombination history. Validated
  end-to-end against CAMB's `rdrag` to 5e-5 over a 5850-point grid --
  see [`tests/test_sound_horizon.py`](tests/test_sound_horizon.py).
- Used by: [`cosmology/calculators/sound_horizon.py`](src/CosmoFit/cosmology/calculators/sound_horizon.py)

### CMB power spectra from a Boltzmann code

- **Lewis, Challinor & Lasenby (2000)**, *Efficient Computation of CMB
  anisotropies in closed FRW models*, ApJ 538, 473 (CAMB).
  [arXiv:astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)
- CosmoFit does not implement a Boltzmann hierarchy; it translates a
  `Cosmology` into CAMB's parameter conventions and calls it. Models
  with a `w(z)` are passed through CAMB's PPF dark-energy module as a
  tabulated `w(a)`, which handles the `w = -1` crossing that CPL and
  JBP posteriors routinely visit and where a quintessence-fluid
  treatment develops a gradient instability. Modified-gravity models
  are refused rather than silently given GR perturbations.
- Used by: [`cosmology/boltzmann.py`](src/CosmoFit/cosmology/boltzmann.py)

### MCMC

- **Foreman-Mackey, Hogg, Lang & Goodman (2013)**, *emcee: The MCMC
  Hammer*, PASP 125, 306.
  [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- CosmoFit's `Fitter.run_mcmc()` is a thin wrapper around the
  [`emcee`](https://emcee.readthedocs.io/) package.
