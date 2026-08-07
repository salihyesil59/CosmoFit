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

### MCMC

- **Foreman-Mackey, Hogg, Lang & Goodman (2013)**, *emcee: The MCMC
  Hammer*, PASP 125, 306.
  [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- CosmoFit's `Fitter.run_mcmc()` is a thin wrapper around the
  [`emcee`](https://emcee.readthedocs.io/) package.
