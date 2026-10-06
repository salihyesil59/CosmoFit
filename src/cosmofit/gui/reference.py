"""
What the GUI knows without computing anything: the models and
their descriptions, the datasets, their groups and presets, the
parameters each model and dataset brings, and the labels of every
plot.
"""

from __future__ import annotations

from cosmofit import (
    LCDM, WCDM, CPL, JBP, BA, GCG,
    LogarithmicDE, PEDE, GEDE, LsCDM,
    IDE, RunningVacuum, Cardassian, DGP, HDE, ADE, RDE,
    FQExponential, FTPowerLaw, FRTLinear, FRHuSawicki,
)
from cosmofit.stats.fitter import CONFLICTING_DATASETS
from cosmofit.data.metadata import DATASETS, FAMILIES


try:
    from cosmofit.theory import Action, GEOMETRIES, STANDARD_FLUIDS

    HAVE_THEORY = True

except ModuleNotFoundError:

    Action = None
    GEOMETRIES = ()
    STANDARD_FLUIDS = {}
    HAVE_THEORY = False


# ============================================================
# Static reference data
# ============================================================

BUILTIN_MODELS = {
    "LCDM": LCDM,
    "WCDM": WCDM,
    "CPL": CPL,
    "JBP": JBP,
    "BA": BA,
    "LogarithmicDE": LogarithmicDE,
    "PEDE": PEDE,
    "GEDE": GEDE,
    "LsCDM": LsCDM,
    "GCG": GCG,
    "IDE": IDE,
    "RunningVacuum": RunningVacuum,
    "Cardassian": Cardassian,
    "HDE": HDE,
    "ADE": ADE,
    "RDE": RDE,
    "DGP": DGP,
    "FQExponential": FQExponential,
    "FTPowerLaw": FTPowerLaw,
    "FRTLinear": FRTLinear,
    "FRHuSawicki": FRHuSawicki,
}


#: Models grouped by *what they change*, which is also what decides
#: what can be done with them: only models with a w(z) can be given
#: to a Boltzmann code, and only models that modify gravity predict a
#: growth history differing from GR's at fixed background.
#:
#: A seventeen-entry flat dropdown gives no hint that ΛCDM and f(Q)
#: are different kinds of object; this does.
MODEL_GROUPS = [
    ("Dark energy on top of GR",
     ["LCDM", "WCDM", "CPL", "JBP", "BA", "LogarithmicDE",
      "PEDE", "GEDE", "LsCDM"]),
    ("Unified or interacting dark sector",
     ["GCG", "IDE", "RunningVacuum"]),
    ("Modified Friedmann equation (no dark energy)",
     ["Cardassian", "DGP"]),
    ("Holographic",
     ["HDE", "ADE", "RDE"]),
    ("Modified gravity",
     ["FQExponential", "FTPowerLaw", "FRTLinear", "FRHuSawicki"]),
]


#: The two routes to a model the library does not ship. `Custom`
#: takes an ``E(z)`` -- the *result* of a derivation somebody did by
#: hand. `From an action` takes the input instead: a gravitational
#: Lagrangian, from which `cosmofit.theory` derives the Friedmann
#: equation itself.
CUSTOM_CHOICE = "Custom"


ACTION_CHOICE = "From an action"


#: Worked actions offered as starting points, so the box is never
#: blank. Each is (label, gravity, geometry, params, closure, growth,
#: fields) -- the same arguments `cosmofit.theory.Action` takes.
ACTION_PRESETS = {
    "— start blank —": None,
    "General Relativity + Λ (rederives ΛCDM)": dict(
        gravity="R - 2*Lam",
        geometry="metric",
        params="Lam = 2.1, 0.0, 6.0, $\\Lambda$",
        closure="Lam",
        growth="gr",
        fields="",
    ),
    "Power-law f(T)  ·  Bengochea & Ferraro (2009)": dict(
        gravity="T + A0*(-T)**b",
        geometry="teleparallel",
        params=(
            "A0 = -4.2, -30.0, 0.0, $A_0$\n"
            "b = 0.0, -2.0, 0.9, $b$"
        ),
        closure="A0",
        growth="quasi_static",
        fields="",
    ),
    "Power-law f(T)  ·  reproduces FTPowerLaw": dict(
        gravity="T + A0*(-T)**b",
        geometry="teleparallel",
        params=(
            "A0 = -4.2, -30.0, 0.0, $A_0$\n"
            "b = 0.0, -2.0, 0.45, $n$"
        ),
        closure="A0",
        growth="quasi_static",
        fields="",
    ),
    "Exponential f(Q)  ·  reproduces FQExponential": dict(
        gravity="Q*exp(lam*Q0/Q)",
        geometry="symmetric",
        params="lam = 0.1, 0.0, 0.9, $\\lambda$",
        closure="lam",
        growth="quasi_static",
        fields="",
    ),
    "Starobinsky f(R) = R - 2Λ + αR²": dict(
        gravity="R - 2*Lam + alpha_fr*R**2",
        geometry="metric",
        params=(
            "Lam = 2.1, 0.0, 6.0, $\\Lambda$\n"
            "alpha_fr = 0.001, 0.000001, 1.0, $\\alpha$"
        ),
        closure="",
        growth="gr",
        fields="",
    ),
    "Exponential quintessence  ·  a rolling scalar field": dict(
        gravity="R",
        geometry="metric",
        params=(
            "V0 = 2.1, 0.05, 50.0, $V_0$\n"
            "lam = 0.5, 0.0, 1.7, $\\lambda$"
        ),
        closure="V0",
        growth="gr",
        fields="phi = X - V0*exp(-lam*phi)",
    ),
    "Scalar-tensor  ·  F(φ)R, gravity's strength rolls": dict(
        gravity="(1 + xi*phi**2)*R",
        geometry="metric",
        params=(
            "xi = 0.02, -0.5, 0.5, $\\xi$\n"
            "V0 = 2.1, 0.05, 20.0, $V_0$"
        ),
        closure="V0",
        growth="quasi_static",
        fields="phi = X - V0",
    ),
}


#: One paragraph per model: what it is, what its extra parameters
#: mean, and which parameter values collapse it back to ΛCDM.
#:
#: ``reduces`` is the most useful line for someone deciding what to
#: fit -- it says exactly which point in parameter space the null
#: hypothesis sits at, which is what an AIC/BIC or likelihood-ratio
#: comparison is measuring the distance from.
MODEL_INFO = {

    "LCDM": dict(
        family="The concordance model",
        what="A cosmological constant plus cold dark matter. Two free "
             "parameters (H₀, Ω_m) and no dark-energy freedom at all. "
             "Everything else here is measured against it.",
        params="—",
        reduces=None,
        ref="Standard.",
    ),

    "WCDM": dict(
        family="Constant equation of state",
        what="Dark energy with a constant w₀ instead of exactly -1. "
             "The simplest possible test of whether dark energy is a "
             "cosmological constant.",
        params="**w₀** — the equation of state. w₀ < -1 is 'phantom'.",
        reduces="ΛCDM at w₀ = -1",
        ref="Standard.",
    ),

    "CPL": dict(
        family="Evolving equation of state",
        what="The standard two-parameter dark-energy parametrization, "
             "and the one the DESI evolving-dark-energy results are "
             "stated in. w(z) is linear in the scale factor, so it "
             "stays finite at high z.",
        params="**w₀** today's equation of state · **w_a** its rate of "
               "change",
        reduces="ΛCDM at (w₀, w_a) = (-1, 0)",
        ref="Chevallier & Polarski (2001); Linder (2003).",
    ),

    "JBP": dict(
        family="Evolving equation of state",
        what="Like CPL, but w(z) peaks at intermediate redshift and "
             "returns to w₀ at both ends. Fitting it alongside CPL "
             "tests how much of a detected w_a is the data and how "
             "much is the assumed shape.",
        params="**w₀**, **w_a** (shared with CPL)",
        reduces="ΛCDM at (w₀, w_a) = (-1, 0)",
        ref="Jassal, Bagla & Padmanabhan (2005).",
    ),

    "BA": dict(
        family="Evolving equation of state",
        what="A w₀–w_a form that stays well-behaved at high redshift "
             "and into the future, where CPL diverges.",
        params="**w₀**, **w_a** (shared with CPL)",
        reduces="ΛCDM at (w₀, w_a) = (-1, 0)",
        ref="Barboza & Alcaniz (2008).",
    ),

    "LogarithmicDE": dict(
        family="Evolving equation of state",
        what="w(z) = w₀ + w_a ln(1+z). The one w₀–w_a form here that "
             "does **not** saturate at high z -- CPL, JBP and BA all "
             "approach a finite limit, so this is the control case for "
             "asking whether a measured w_a reflects the data or the "
             "shape you assumed.",
        params="**w₀**, **w_a** (shared with CPL)",
        reduces="ΛCDM at (w₀, w_a) = (-1, 0)",
        ref="Efstathiou (1999).",
    ),

    "PEDE": dict(
        family="Emergent dark energy",
        what="Dark energy that is absent at high redshift and "
             "'emerges' toward the present. **It has no free "
             "dark-energy parameter at all** -- the same parameter "
             "count as ΛCDM and a completely different expansion "
             "history, so an AIC/BIC comparison against ΛCDM is a pure "
             "comparison of fit with identical penalties.",
        params="— (none beyond H₀, Ω_m)",
        reduces=None,
        ref="Li & Shafieloo (2019).",
    ),

    "GEDE": dict(
        family="Emergent dark energy",
        what="The family containing both ΛCDM and PEDE, so Δ measures "
             "the distance from a cosmological constant on a "
             "continuous scale rather than at a model boundary.",
        params="**Δ** how sharply dark energy emerges · **z_t** when",
        reduces="ΛCDM at Δ → 0; PEDE at Δ = 1, z_t = 0",
        ref="Li & Shafieloo (2020).",
    ),

    "LsCDM": dict(
        family="Sign-switching Λ",
        what="ΛCDM, except Λ **changes sign** at z_† ≈ 2 (anti-de "
             "Sitter before, de Sitter after). A lower expansion rate "
             "before the transition shrinks r_d, which raises the "
             "BAO-inferred H₀ -- a route to the Hubble tension that "
             "late-time-only dark-energy models cannot take. E(z) is "
             "genuinely discontinuous there; that is the model, not a "
             "bug.",
        params="**z_†** the transition redshift",
        reduces="ΛCDM for z_† above every data point",
        ref="Akarsu, Kumar, Özülker & Vázquez (2021).",
    ),

    "GCG": dict(
        family="Unified dark sector",
        what="A single fluid that behaves as dark matter early and "
             "dark energy late, with p = -A/ρ^α. One component doing "
             "both jobs rather than two.",
        params="**A_s** the density parameter (= -w today) · "
               "**α** the exponent",
        reduces="ΛCDM at A_s = 1",
        ref="Bento, Bertolami & Sen (2002).",
    ),

    "IDE": dict(
        family="Interacting dark sector",
        what="Dark matter and dark energy exchange energy, Q = 3ξHρ_DE. "
             "This changes how **matter** dilutes, which no w(z) "
             "parametrization does -- so it leaves its own signature "
             "in growth-of-structure data.",
        params="**ξ** the coupling (ξ > 0 feeds dark matter) · "
               "**w₀**",
        reduces="wCDM at ξ = 0; ΛCDM at ξ = 0, w₀ = -1",
        ref="Amendola (2000); Wang et al. (2016).",
    ),

    "RunningVacuum": dict(
        family="Unified dark sector",
        what="A cosmological 'constant' that runs with the expansion "
             "rate, Λ(H) = c₀ + 3νH². One of the few extensions whose "
             "extra parameter has a **predicted magnitude** "
             "(|ν| ~ 10⁻³, from a one-loop estimate) rather than an "
             "arbitrary one -- so ν ~ 10⁻³ means something quite "
             "different from ν ~ 0.1.",
        params="**ν** the renormalization-group running coefficient",
        reduces="ΛCDM at ν = 0",
        ref="Solà (2013); Solà, Gómez-Valent & de Cruz Pérez (2017).",
    ),

    "Cardassian": dict(
        family="Modified Friedmann equation",
        what="An extra term in the Friedmann equation itself, "
             "H² = Aρ + Bρⁿ, from the universe being a brane in higher "
             "dimensions. Acceleration **from matter alone** -- there "
             "is no dark energy in this model.",
        params="**n**, **q** — the modified-polytropic exponents",
        reduces="ΛCDM at n = 0, q = 1",
        ref="Freese & Lewis (2002); Wang et al. (2003).",
    ),

    "HDE": dict(
        family="Holographic",
        what="The holographic principle bounds the energy in a region "
             "by its **boundary area**, giving ρ_DE = 3c²M_p²/L². Li "
             "(2004) showed the infrared cutoff L has to be the "
             "**future event horizon** for the universe to accelerate "
             "at all. This is the only model here whose E(z) has **no "
             "closed form** — Ω_DE obeys an ODE, solved and splined "
             "whenever the parameters change. Flat universes only: "
             "curvature changes the causal structure the holographic "
             "bound is applied to.",
        params="**c** — the holographic constant, which fixes w: "
               "w → −1/3 early and −1/3 − 2/(3c) in the far future, "
               "so **c < 1 crosses into phantom** and c > 1 stays "
               "quintessence-like. The crossing is a prediction, not "
               "a parametrization choice.",
        reduces="never exactly — w evolves for every c",
        ref="Li (2004), arXiv:hep-th/0403127; "
            "Wang, Mörtsell et al. (2017), arXiv:1612.00345 (review).",
    ),

    "ADE": dict(
        family="Holographic",
        what="The same holographic idea as HDE with a different "
             "infrared cutoff: the **conformal age** of the universe, "
             "which is causal and needs no reference to the future — "
             "the usual objection to HDE. Its most striking feature "
             "is that it has **one fewer free parameter than ΛCDM**: "
             "the early-time condition Ω_DE → n²a²/4 fixes the whole "
             "background from n, so Ω_m is *derived* rather than "
             "fitted (n = 2.8 predicts Ω_m = 0.280). Flat only.",
        params="**n** — the agegraphic constant. It also sets Ω_m, "
               "so freeing Ω_m alongside it does nothing.",
        reduces="never — w → −2/3 early and −1 in the far future, and "
                "**never below −1**: unlike HDE this model cannot be "
                "phantom at any n",
        ref="Wei & Cai (2008), arXiv:0708.0884.",
    ),

    "RDE": dict(
        family="Holographic",
        what="Holographic dark energy with the **Ricci scalar** as "
             "the cutoff — a local curvature scale rather than a "
             "horizon, so again no reference to the future. Unlike "
             "the other two this has a closed-form E(z): the dark "
             "sector is a power law (1+z)^(4−2/γ). Fits want γ "
             "slightly above 1/2 and, more awkwardly, a low matter "
             "density around 0.22, which is its main observational "
             "problem. Flat only.",
        params="**γ** — sets the power law. Note that part of the "
               "Ricci density scales like matter, so the coefficient "
               "of (1+z)³ is (4/3)γ-corrected and larger than Ω_m.",
        reduces="a constant dark-energy density at γ = 1/2 (ΛCDM, "
                "with an effective matter density (4/3)Ω_m)",
        ref="Gao, Chen, Shen & Saridakis (2009), arXiv:0712.1394.",
    ),

    "DGP": dict(
        family="Braneworld gravity",
        what="Gravity leaks into a fifth dimension above a crossover "
             "scale, and the universe accelerates with **no dark "
             "energy at all**. Like PEDE it has exactly ΛCDM's "
             "parameter count. Its real signature is growth: gravity "
             "is *weaker* (μ ≈ 0.72 today), so structure grows more "
             "slowly than in any dark-energy model with the same "
             "E(z).",
        params="— (Ω_rc is fixed by E(0) = 1)",
        reduces=None,
        ref="Dvali, Gabadadze & Porrati (2000); Deffayet (2001).",
    ),

    "FQExponential": dict(
        family="Modified gravity",
        what="f(Q) symmetric teleparallel gravity. The field equations "
             "themselves differ from Einstein's, so both the expansion "
             "history and the growth of structure change.",
        params="**λ** the exponential coupling",
        reduces="ΛCDM-like at λ → 0",
        ref="Anagnostopoulos, Basilakos & Saridakis (2021).",
    ),

    "FTPowerLaw": dict(
        family="Modified gravity",
        what="f(T) metric teleparallel gravity, the torsion "
             "counterpart of f(Q). The field equations themselves "
             "differ from Einstein's, so both the expansion history "
             "and the growth of structure change.",
        params="**n** the power; the amplitude is fixed by Ω_m, not free",
        reduces="ΛCDM exactly at n = 0",
        ref="Bengochea & Ferraro (2009); Linder (2010).",
    ),

    "FRTLinear": dict(
        family="Modified gravity",
        what="f(R,T) gravity: gravity couples to the trace of the "
             "matter stress-energy tensor as well as to curvature. "
             "Ω_L follows from Ω_m, Ω_k and β through E(0) = 1, so "
             "H₀ is the Hubble rate today.",
        params="**β** the matter-geometry coupling (Ω_L is derived)",
        reduces="GR at β = 0 (with Ω_L = 1 - Ω_m)",
        ref="Harko, Lobo, Nojiri & Odintsov (2011).",
    ),

    "FRHuSawicki": dict(
        family="Modified gravity",
        what="The benchmark f(R) model, built to pass Solar-System "
             "tests through chameleon screening. Its **background is "
             "ΛCDM's by construction** -- f_R0 and n do nothing to "
             "E(z) -- so it can only be constrained by growth data.",
        params="**f_R0** today's scalaron value · **n** the shape "
               "exponent",
        reduces="ΛCDM at f_R0 → 0 (and always, at background level)",
        ref="Hu & Sawicki (2007); Pogosian & Silvestri (2008).",
    ),

}


#: Each dataset's descriptive title, from the library's metadata.
DATASET_LABELS = {key: info.title for key, info in DATASETS.items()}


#: Which probe family each dataset belongs to, for grouping the
#: sidebar. A fit is usually built by picking *one* from each family
#: rather than by ticking everything, and the flat checkbox list made
#: that impossible to see. The families and their members are the
#: library's (:mod:`data.metadata`); only the icons are the GUI's.
_FAMILY_ICONS = {
    "expansion": "📏", "bao": "🌀", "sn": "💥",
    "cmb": "🔥", "growth": "🕸️", "external": "📌",
}


DATASET_GROUPS = [
    (
        f"{_FAMILY_ICONS[family]} {title}",
        [key for key, info in DATASETS.items() if info.family == family],
    )
    for family, title in FAMILIES.items()
]


#: A short, honest note per dataset: what it measures, over what
#: redshift range, how many points, and -- the part a bare label
#: cannot carry -- *what it is for*, i.e. which parameter it is the
#: thing that actually constrains.
#:
#: ``n`` and ``z`` are stated rather than loaded: reading Pantheon+'s
#: 1600x1600 covariance off disk to print "1590 points" in a tooltip
#: would make the sidebar slow for no reason.
DATASET_INFO = {

    "cc": dict(
        observable="H(z), directly",
        n="32 points", z="0.07 – 1.97",
        what=(
            "Differential ages of passively-evolving galaxies give "
            "dz/dt and hence H(z) **without assuming a cosmology** -- "
            "the only truly model-independent expansion-rate probe "
            "here."
        ),
        constrains="H₀ directly (no r_d or M_B degeneracy)",
    ),

    "desi": dict(
        observable="D_M/r_d, D_H/r_d, D_V/r_d",
        n="13 points (DR2) / 12 (DR1)", z="0.30 – 2.33",
        what=(
            "The BAO standard ruler across seven tracers. DR2 is three "
            "years of data and >14 million galaxies and quasars -- the "
            "measurement the evolving-dark-energy claim rests on. "
            "Choose DR1 or DR2 in **Versions** below; they must not be "
            "combined (DR2 contains every DR1 galaxy)."
        ),
        constrains="Ω_m tightly; H₀ only via r_d",
    ),

    "sdss_bao": dict(
        observable="D_M/r_d, D_H/r_d",
        n="6 points", z="0.38 – 1.48",
        what=(
            "BOSS DR12 plus eBOSS DR16 LRG/QSO -- the pre-DESI BAO "
            "standard. Useful as an independent cross-check of DESI, "
            "not as an addition to it."
        ),
        constrains="Ω_m, H₀·r_d",
    ),

    "bao_lowz": dict(
        observable="r_d/D_V, D_V/r_d",
        n="2 points", z="0.106, 0.15",
        what=(
            "6dFGS and the SDSS DR7 Main Galaxy Sample: the only BAO "
            "leverage below z = 0.2, where DESI starts at 0.295 and "
            "BOSS at 0.38. Independent of both, so unlike DESI-vs-SDSS "
            "this **can** be added to either."
        ),
        constrains="extends the BAO lever arm to low z",
    ),

    "sdss_fsbao": dict(
        observable="D_M/r_d, D_H/r_d and fσ₈ per bin",
        n="12 points", z="0.38, 0.51, 0.698, 1.48",
        what=(
            "The **same BOSS/eBOSS galaxies as `sdss_bao`**, "
            "analysed for their full anisotropic clustering rather "
            "than the BAO peak alone. So it measures the growth "
            "rate too — and, more importantly, the **correlation "
            "between growth and geometry** (0.19 to 0.64 within a "
            "bin). Using `sdss_bao` together with the separate "
            "`fsigma8` compilation covers the same galaxies while "
            "pretending those are independent; this is the product "
            "that does not."
        ),
        constrains="σ₈ and the distance scale jointly",
    ),

    "eboss_elg": dict(
        observable="D_V/r_d, as a tabulated likelihood",
        n="1 quantity", z="0.845",
        what=(
            "eBOSS DR16 emission-line galaxies. Released as a "
            "**likelihood curve**, not a mean and an error bar, "
            "because the BAO feature is only a 1.4σ detection: the "
            "curve is asymmetric and still rising at the low edge of "
            "the released table. About a tenth of its probability "
            "sits below D_V/r_d = 16.5, where a Gaussian summary "
            "would put a thousandth."
        ),
        constrains="D_V at z ≈ 0.85, weakly but non-Gaussianly",
    ),

    "eboss_elg_fs": dict(
        observable="(D_M/r_d, D_H/r_d, fσ₈), a 3-D tabulated likelihood",
        n="3 quantities", z="0.845",
        what=(
            "The **same eBOSS ELG galaxies** as above, analysed for "
            "their full anisotropic shape rather than one isotropic "
            "BAO scale — so it measures the **growth rate** as well "
            "as the geometry. A 100×100×100 grid, which is the only "
            "way to carry the degeneracy between fσ₈ and the "
            "Alcock–Paczynski distortion honestly. Use this **or** "
            "`eboss_elg`, never both."
        ),
        constrains="σ₈ and the geometry jointly at z ≈ 0.85",
    ),

    "eboss_lya": dict(
        observable="(D_M/r_d, D_H/r_d), as a 2-D tabulated likelihood",
        n="2 quantities", z="2.334",
        what=(
            "The Lyman-α forest: the **highest-redshift BAO here "
            "outside the CMB**, and the only lever arm on expansion "
            "between the supernovae and recombination. Released as a "
            "50×50 likelihood surface. Its main value over two error "
            "bars is the −0.46 correlation between the two ratios. "
            "Sits about 2σ from a Planck-like ΛCDM, which is a real "
            "and much-discussed feature of the measurement."
        ),
        constrains="H(z) at z ≈ 2.3 -- where late-time DE models differ",
    ),

    "pantheon": dict(
        observable="corrected apparent magnitude m_B",
        n="1590 SNe", z="0.01 – 2.26",
        what=(
            "The largest SN Ia compilation here. The absolute "
            "magnitude M_B is analytically marginalized, so this "
            "measures the *shape* of the distance-redshift relation, "
            "not its normalization."
        ),
        constrains="Ω_m, w₀/w_a; not H₀",
    ),

    "des_sn5yr": dict(
        observable="distance modulus μ",
        n="1820 SNe", z="0.025 – 1.14",
        what=(
            "Dark Energy Survey 5-year sample, in its DES-Dovekie "
            "recalibration (Popovic et al. 2026), which supersedes "
            "the original 2024 release. Of the three SN "
            "compilations this one pulls hardest away from a "
            "cosmological constant -- which is exactly why it is worth "
            "running all three separately."
        ),
        constrains="Ω_m, w₀/w_a; not H₀",
    ),

    "union3": dict(
        observable="binned distance modulus μ",
        n="22 bins (2087 SNe)", z="0.05 – 2.26",
        what=(
            "Fit with the UNITY1.5 hierarchical model, which "
            "marginalizes light-curve standardization and selection "
            "effects internally -- so it ships as 22 bins rather than "
            "a catalogue, and sits between Pantheon+ and DES-SN5YR in "
            "how far it moves from ΛCDM."
        ),
        constrains="Ω_m, w₀/w_a; not H₀",
    ),

    "planck": dict(
        observable="(R, ℓ_A, ω_b h²)",
        n="3 numbers", z="z* ≈ 1090",
        what=(
            "The CMB compressed to three numbers. Fast, needs no extra "
            "dependency, and works for **every** model -- but it "
            "inherits the conventions the compression was built with, "
            "and throws away nearly all the information in the "
            "spectra."
        ),
        constrains="Ω_m·h², the distance to last scattering",
    ),

    "planck_lite": dict(
        observable="C_ℓ^TT, C_ℓ^TE, C_ℓ^EE",
        n="613 bandpowers", z="ℓ = 30 – 2508",
        what=(
            "The measured CMB spectra themselves, against C_ℓ computed "
            "from scratch by CAMB. No compression and no borrowed "
            "convention -- at the cost of ~0.7 s per likelihood "
            "evaluation and only working for ΛCDM and models with a "
            "w(z)."
        ),
        constrains="everything, tightly -- needs n_s, ln10¹⁰A_s, τ free",
    ),

    "planck_lensing": dict(
        observable="C_L^φφ (lensing potential)",
        n="9 bandpowers", z="L = 8 – 400",
        what=(
            "The CMB lensed by everything it passed through, "
            "inverted to map the matter back to z ~ 2. **A growth "
            "measurement made by the CMB itself** — every other CMB "
            "dataset here constrains recombination and reaches the "
            "present only through a distance."
        ),
        constrains="σ₈·Ω_m^0.25 — the CMB's own side of the S₈ question",
    ),

    "planck_lowe": dict(
        observable="D_ℓ^EE at ℓ = 2–29",
        n="28 multipoles", z="z ≈ 8 (reionization)",
        what=(
            "Planck's low-ℓ polarization, as the **tabulated, "
            "non-Gaussian** likelihood it actually is rather than a "
            "mean and an error bar. Below ℓ = 30 there are only "
            "2ℓ+1 modes on the sky, so the C_ℓ distribution is "
            "strongly skewed — and that is exactly the regime "
            "carrying the CMB's information about τ."
        ),
        constrains="τ — the real thing, not the Gaussian shorthand",
    ),

    "act_lensing": dict(
        observable="C_L^κκ (lensing convergence)",
        n="10 bandpowers", z="L = 40 – 763",
        what=(
            "A **second, independent** lensing reconstruction — "
            "different telescope, different sky, different pipeline "
            "— and tighter than Planck's: 2.3% on the lensing "
            "amplitude. CMB lensing is the cleanest handle anyone "
            "has on σ₈Ω_m^0.25, which is what the S₈ tension is "
            "about."
        ),
        constrains="σ₈·Ω_m^0.25, more tightly than Planck lensing",
    ),

    "fsigma8": dict(
        observable="fσ₈(z)",
        n="22 points", z="0.02 – 1.94",
        what=(
            "Redshift-space distortions: how fast structure grows, not "
            "how fast the universe expands. This is the **only** kind "
            "of data that can tell a modified-gravity model from a "
            "dark-energy one with the same E(z)."
        ),
        constrains="σ₈, and μ(a,k) for modified gravity",
    ),

    "s8": dict(
        observable="S₈ = σ₈√(Ω_m/0.3)",
        n="1 number", z="lensing kernel, z ≲ 1",
        what=(
            "A single Gaussian weak-lensing constraint (KiDS-1000 by "
            "default, DES Y3 available). It sits ~2-3σ below what "
            "Planck ΛCDM predicts -- the S₈ tension -- so expect a "
            "χ² of a few here even for a good fit."
        ),
        constrains="σ₈ and Ω_m jointly",
    ),

    "h0": dict(
        observable="H₀",
        n="1 number", z="z ≈ 0",
        what=(
            "The local distance ladder (SH0ES) or time-delay lensing "
            "(TDCOSMO). Enters as a **dataset**, not a prior, so it "
            "shows up in the χ² breakdown and the degrees-of-freedom "
            "count -- a fit that assumed the local ladder should not "
            "look like one that did not."
        ),
        constrains="H₀ directly (and disagrees with CMB at ~5σ)",
    ),

    "omega_b": dict(
        observable="ω_b = Ω_b h²",
        n="1 number", z="z ≈ 10⁸ (BBN)",
        what=(
            "Big Bang Nucleosynthesis, completely independent of the "
            "CMB. On its own it does little; paired with **Compute "
            "r_d** below it is what turns BAO into an absolute "
            "distance measurement and lets BAO measure H₀."
        ),
        constrains="Ω_b -- and through r_d, H₀",
    ),

    "tau": dict(
        observable="τ (reionization optical depth)",
        n="1 number", z="z ≈ 8",
        what=(
            "Planck's large-scale polarization constraint, "
            "τ = 0.0506 ± 0.0086 from low-ℓ EE alone. Only "
            "meaningful alongside the full CMB spectra, which cover "
            "ℓ ≥ 30 where τ is degenerate with the primordial "
            "amplitude. Without it, ln10¹⁰A_s is unconstrained."
        ),
        constrains="τ, breaking the τ–A_s degeneracy",
    ),

}


#: Ready-made dataset combinations, each one an analysis someone
#: actually runs. Picking from a flat list of fourteen checkboxes
#: without knowing which ones conflict is the single hardest part of
#: using this app cold.
DATASET_PRESETS = {

    "Late-time background (default)": dict(
        datasets=["cc", "desi"],
        note="Expansion rate plus the BAO ruler. Fast, and enough to "
             "constrain Ω_m and H₀·r_d.",
    ),

    "DESI DR2 + BBN → H₀ without the CMB": dict(
        datasets=["desi", "omega_b"],
        compute_rd=True,
        versions={"desi": "desi2025"},
        note="The 'BAO + BBN' measurement: with r_d computed rather "
             "than fitted, BAO becomes an absolute distance and H₀ is "
             "measurable with no CMB and no distance ladder. Free "
             "Ω_b as well as H₀ and Ω_m.",
    ),

    "Dark-energy workhorse (BAO + SNe + CMB priors)": dict(
        datasets=["desi", "pantheon", "planck"],
        versions={"desi": "desi2025"},
        note="The combination the w₀–w_a results are argued with. Try "
             "it with CPL and two free parameters w₀, w_a.",
    ),

    "The Hubble tension, both sides": dict(
        datasets=["desi", "planck", "h0"],
        versions={"desi": "desi2025"},
        note="CMB-anchored data plus the local H₀ measurement. The χ² "
             "breakdown in the results shows how much each side is "
             "being stretched.",
    ),

    "Growth of structure (tests modified gravity)": dict(
        datasets=["cc", "desi", "fsigma8", "s8"],
        note="The only combination that can distinguish modified "
             "gravity from dark energy. Pair it with f(R) Hu-Sawicki, "
             "f(Q), f(R,T) or DGP and free σ₈.",
    ),

    "Full CMB from scratch (slow)": dict(
        datasets=["planck_lite", "tau", "desi"],
        versions={"desi": "desi2025"},
        note="The measured CMB spectra rather than a compression. "
             "Hours, not minutes -- free n_s, ln10¹⁰A_s and τ too, and "
             "leave the chain saving on.",
    ),

}


#: Dataset pairs that double-count data if combined -- see README.
#: Dataset pairs that must not be combined, with why. Derived from the
#: library's own list rather than kept as a copy: a copy here held 7 of
#: the library's 17 pairs by the time anyone compared them.
INCOMPATIBLE_PAIRS = [
    (
        {first, second},
        f"{reason} Treating them as independent double-counts that data.",
    )
    for (first, second), reason in CONFLICTING_DATASETS.items()
]


#: Datasets that are slow enough to be worth warning about before
#: someone ticks them and waits.
SLOW_DATASETS = {
    "act_lensing": (
        "ACT DR6 lensing runs CAMB on every likelihood evaluation, "
        "like the other from-scratch CMB datasets. Cheap next to "
        "them (one CAMB call serves all), expensive alone."
    ),
    "planck_lowe": (
        "Planck low-ℓ EE runs CAMB on every likelihood evaluation, "
        "like the other from-scratch CMB datasets. Cheap to add "
        "*next to* them (one CAMB call serves all), expensive alone."
    ),
    "planck_lensing": (
        "Planck lensing runs CAMB on every likelihood evaluation, "
        "same as the full spectra below. It is cheap to add *next "
        "to* them (one CAMB call serves both) and expensive on its "
        "own. LCDM and w(z) models only."
    ),
    "planck_lite": (
        "Planck TT/TE/EE computes the CMB power spectrum from scratch "
        "with CAMB on every likelihood evaluation (~0.7 s per step, "
        "against ~1 ms for every other dataset combined). A full chain "
        "takes hours, not minutes -- use a saved chain, run the "
        "chains in several Processes, and consider the compressed "
        "distance priors "
        "('Planck 2018 CMB distance priors') unless you specifically "
        "need the spectra. It also only works for LCDM and models with "
        "a w(z), not the modified-gravity ones."
    ),
}


#: LaTeX preview shown next to each model picker -- background
#: expansion for the LCDM/WCDM family, dark-energy equation of state
#: for the w0-wa family, and the GCG fluid equation for GCG (its E(z)
#: doesn't have as illuminating a one-line form).
MODEL_EQUATIONS = {
    "LCDM": r"E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_k (1+z)^2 + \Omega_{DE}}",
    "WCDM": r"E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_k (1+z)^2 + \Omega_{DE}(1+z)^{3(1+w_0)}}",
    "CPL": r"w(z) = w_0 + w_a \dfrac{z}{1+z}",
    "JBP": r"w(z) = w_0 + w_a \dfrac{z}{(1+z)^2}",
    "BA": r"w(z) = w_0 + w_a \dfrac{z(1+z)}{1+z^2}",
    "LogarithmicDE": r"w(z) = w_0 + w_a \ln(1+z)",
    "PEDE": r"\Omega_{DE}(z) = \Omega_{DE,0}\left[1 - \tanh\left(\log_{10}(1+z)\right)\right]",
    "GEDE": r"\Omega_{DE}(z) \propto 1 - \tanh\left[\Delta \log_{10}\dfrac{1+z}{1+z_t}\right]",
    "LsCDM": r"E(z)^2 = \Omega_m (1+z)^3 + \Omega_k (1+z)^2 + \Omega_{\Lambda_s} \,\mathrm{sgn}(z_\dagger - z)",
    "GCG": r"p = -\dfrac{A}{\rho^{\alpha}}",
    "IDE": r"Q = 3\xi H \rho_{DE}, \quad w = w_0",
    "RunningVacuum": r"\Lambda(H) = c_0 + 3\nu H^2",
    "Cardassian": r"H^2 = A\rho + B\rho^{n} \ \ (\text{modified polytropic})",
    "HDE": r"\rho_{\rm DE} = 3c^2 M_p^2 / L^2, \ \ L = \text{future event horizon}",
    "ADE": r"\rho_{\rm DE} = 3n^2 M_p^2 / \eta^2, \ \ \eta = \text{conformal age}",
    "RDE": r"\rho_{\rm DE} = 3\gamma M_p^2 (\dot H + 2H^2) \ \ (\text{Ricci scalar})",
    "DGP": r"E(z) = \sqrt{\Omega_{rc} + \Omega_m (1+z)^3} + \sqrt{\Omega_{rc}}",
    "FQExponential": r"f(Q) = Q\, e^{\lambda Q_0/Q},\quad Q=6H^2",
    "FTPowerLaw": r"f(T) = T + \alpha T^{n},\quad T=6H^2",
    "FRTLinear": r"f(R,T) = R + 2\lambda T",
    "FRHuSawicki": r"f(R) = -m^2\dfrac{c_1(R/m^2)^n}{c_2(R/m^2)^n+1}",
}


#: Modified-gravity models whose background (E(z)) is, by
#: construction, indistinguishable from LCDM here -- see
#: cosmology/models/fr.py's docstring. Surfaced as a visible caveat
#: next to the model picker, not just in a docstring nobody reads
#: from the GUI.
BACKGROUND_DEGENERATE_MODELS = {
    "FRHuSawicki": (
        "This model's background expansion is, by construction, "
        "identical to LCDM's -- f_R0/n don't affect E(z) here, so "
        "fitting them against background-only datasets (CC/BAO/SNe/"
        "Planck) won't meaningfully constrain them. Hu-Sawicki f(R)'s "
        "actual signature is in the growth of structure -- tick "
        "'fsigma8' and/or 's8' above to actually constrain f_R0/n."
    ),
}


#: Single-model figures (Fitter.plots.<name>()).
PLOT_LABELS = {
    "chain": "MCMC chain (trace plot)",
    "corner": "Corner plot",
    "hubble_diagram": "Hubble diagram (Pantheon+)",
    "des_hubble_diagram": "Hubble diagram (DES-SN5YR)",
    "union3_hubble_diagram": "Hubble diagram (Union3)",
    "hz": "H(z) diagram (CC)",
    "bao_distances": "BAO distances (DESI)",
    "sdss_bao_distances": "BAO distances (SDSS)",
    "lowz_bao_distances": "BAO distances (6dFGS + MGS)",
    "planck_residuals": "Planck residuals (pull plot)",
    "cmb_spectra": "CMB power spectra (TT/TE/EE)",
    "cmb_lensing": "CMB lensing bandpowers",
    "w_of_z": "w(z) evolution",
    "w0_wa_plane": "w0-wa dark-energy plane",
    "deceleration": "Deceleration parameter q(z)",
    "growth": "Growth rate fsigma8(z)",
    "eboss_surface": "eBOSS likelihood surface (released grid)",
}


#: Model-comparison figures (Fitter.plots.compare_<name>(other_fits=...)).
COMPARE_PLOT_LABELS = {
    "compare_hz": "H(z) diagram (CC)",
    "compare_hubble_diagram": "Hubble diagram (Pantheon+)",
    "compare_des_hubble_diagram": "Hubble diagram (DES-SN5YR)",
    "compare_bao_distances": "BAO distances (DESI)",
    "compare_sdss_bao_distances": "BAO distances (SDSS)",
    "compare_w_of_z": "w(z) evolution",
    "compare_w0_wa_plane": "w0-wa dark-energy plane",
    "compare_deceleration": "Deceleration parameter q(z)",
    "compare_growth": "Growth rate fsigma8(z)",
}


MAX_MODELS = 5

#: The samplers the page offers -- the 2.0 core's -- by label.
SAMPLERS = {
    "Adaptive Metropolis-Hastings": "mcmc",
    "emcee ensemble": "emcee",
}


#: Export formats offered for every figure -- (file extension, MIME
#: type). SVG/PDF are vector (best for papers/further editing); PNG
#: is raster (best for slides/quick sharing).
PLOT_EXPORT_FORMATS = {
    "SVG": ("svg", "image/svg+xml"),
    "PNG": ("png", "image/png"),
    "PDF": ("pdf", "application/pdf"),
}


#: Which *standard* parameters each model's E(z) actually uses.
#: Extra parameters are read off ``EXTRA_PARAMS`` automatically and
#: need no entry here.
#:
#: This exists because the parameter table lists every field of the
#: shared container -- twenty-odd of them now -- and for LCDM all but
#: three are inert. Showing w_a, A_s, ξ, ν, n_s and τ in an LCDM fit
#: does not offer flexibility, it just hides which three numbers
#: matter.
MODEL_STANDARD_PARAMS = {
    "LCDM": set(),
    "WCDM": {"w0"},
    "CPL": {"w0", "wa"},
    "JBP": {"w0", "wa"},
    "BA": {"w0", "wa"},
    "LogarithmicDE": {"w0", "wa"},
    "PEDE": set(),
    "GEDE": set(),
    "LsCDM": set(),
    "GCG": {"A_s", "alpha"},
    "IDE": {"w0"},
    "RunningVacuum": set(),
    "Cardassian": set(),
    "HDE": set(),
    "ADE": set(),
    "RDE": set(),
    "DGP": set(),
    "FQExponential": set(),
    "FTPowerLaw": set(),
    "FRTLinear": set(),
    "FRHuSawicki": set(),
}


#: Parameters that only matter because a *dataset* needs them, keyed
#: by dataset. Relevance is a property of the fit, not of the model
#: alone: ``rd`` means nothing without BAO, ``sigma8`` nothing
#: without growth data, ``tau_reio`` nothing without the CMB spectra.
DATASET_PARAMS = {
    "desi": {"rd"},
    "sdss_bao": {"rd"},
    "sdss_fsbao": {"rd", "sigma8"},
    "bao_lowz": {"rd"},
    "eboss_elg": {"rd"},
    "eboss_elg_fs": {"rd", "sigma8"},
    "eboss_lya": {"rd"},
    "planck": {"Omega_b"},
    "planck_lite": {"Omega_b", "n_s", "ln1e10As", "tau_reio",
                    "N_eff", "m_nu", "A_planck"},
    "planck_lensing": {"Omega_b", "n_s", "ln1e10As", "tau_reio",
                       "N_eff", "m_nu"},
    "act_lensing": {"Omega_b", "n_s", "ln1e10As", "tau_reio",
                    "N_eff", "m_nu"},
    "planck_lowe": {"Omega_b", "n_s", "ln1e10As", "tau_reio",
                    "N_eff", "m_nu", "A_planck"},
    "fsigma8": {"sigma8"},
    "s8": {"sigma8"},
    "omega_b": {"Omega_b"},
    "tau": {"tau_reio"},
}


#: Parameters the sound-horizon calculation needs when ``r_d`` is
#: computed rather than fitted.
COMPUTE_RD_PARAMS = {"Omega_b", "N_eff", "m_nu"}


#: Always shown: the two every model has, plus curvature.
ALWAYS_RELEVANT = {"H0", "Omega_m", "Omega_k"}
