"""
CosmoFit -- graphical interface.

A thin Streamlit layer over the public ``CosmoFit`` API
(``Fitter``, ``FitPlotter``, ``define_model`` / ``model_from_expression``):
tick which datasets to fit, configure one or more models (built-in or
your own ``E(z)`` expression), choose which parameters are free, and
run an MCMC fit + look at the resulting plots and model-comparison
statistics -- no code required. Chains are saved to disk and reused,
so adding a model (or reopening the app) doesn't re-sample the fits
that haven't changed. Everything here is a consumer of
``CosmoFit``'s existing public API; no fitting/plotting logic lives
in this file.

Run with:

    pip install -e ".[gui]"
    streamlit run app/streamlit_app.py

Local use only: the custom-model expression box below is evaluated
with ``eval()`` (builtins stripped, only whitelisted numpy math and
the model's own parameter names reach it -- see
``CosmoFit.cosmology.custom._compile_expression``). That is a
reasonable trust boundary for a tool you run on your own machine,
not for a public, multi-tenant deployment.
"""

from __future__ import annotations

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from CosmoFit import (
    __version__,
    LCDM, WCDM, CPL, JBP, BA, GCG,
    LogarithmicDE, PEDE, GEDE, LsCDM,
    IDE, RunningVacuum, Cardassian, DGP, HDE, ADE, RDE,
    FQExponential, FTPowerLaw, FRTLinear, FRHuSawicki,
    Fitter,
    model_from_expression,
    CCLikelihood,
    DESILikelihood,
    SDSSBAOLikelihood,
    BAOLowZLikelihood,
    PantheonLikelihood,
    DESSN5YRLikelihood,
    Union3Likelihood,
    PlanckLikelihood,
    PlanckLiteLikelihood,
    PlanckLensingLikelihood,
    ACTDR6LensingLikelihood,
    FSigma8Likelihood,
    EBOSSELGLikelihood,
    EBOSSLyaLikelihood,
)
from CosmoFit import available_versions, dataset_reference

# `CosmoFit.theory` needs sympy, which is an optional extra. The GUI
# offers the "From an action" model route only when it is installed,
# and says how to get it when it is not -- rather than presenting a
# dropdown entry that fails the moment it is chosen.
try:
    from CosmoFit.theory import Action, GEOMETRIES, STANDARD_FLUIDS

    HAVE_THEORY = True

except ModuleNotFoundError:

    Action = None
    GEOMETRIES = ()
    STANDARD_FLUIDS = {}
    HAVE_THEORY = False
from CosmoFit.stats import DATASET_REGISTRY, model_comparison, cpl_diagnostics
from CosmoFit.stats.chains import ChainFile, StoredSampler
from CosmoFit.stats.results import _json_default
from CosmoFit.stats.fitter import usable_cpu_count


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
#: Lagrangian, from which `CosmoFit.theory` derives the Friedmann
#: equation itself.
CUSTOM_CHOICE = "Custom"
ACTION_CHOICE = "From an action"

#: Worked actions offered as starting points, so the box is never
#: blank. Each is (label, gravity, geometry, params, closure, growth,
#: fields) -- the same arguments `CosmoFit.theory.Action` takes.
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
             "Note that Ω_m and Ω_L are **independent** here, not tied "
             "by flatness -- so E(0) = 1 does not hold automatically, "
             "which is how this model is actually fitted in the "
             "literature.",
        params="**β** the matter-geometry coupling · **Ω_L** the "
               "Λ-like component, independent of Ω_m",
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

DATASET_LABELS = {
    "cc": "Cosmic Chronometers (CC)",
    "desi": "DESI BAO",
    "sdss_bao": "SDSS BAO (BOSS DR12 + eBOSS DR16)",
    "bao_lowz": "Low-z BAO (6dFGS + SDSS MGS)",
    "pantheon": "Pantheon+ (SNe Ia)",
    "des_sn5yr": "DES-SN5YR (SNe Ia)",
    "union3": "Union3 (SNe Ia, binned)",
    "planck": "Planck 2018 CMB (distance priors)",
    "planck_lite": "Planck 2018 CMB (full TT/TE/EE spectra)",
    "planck_lensing": "Planck 2018 CMB lensing",
    "planck_lowe": "Planck 2018 low-ℓ EE (τ, tabulated)",
    "act_lensing": "ACT DR6 CMB lensing",
    "fsigma8": "Growth rate fσ₈(z) (RSD)",
    "s8": "S₈ weak-lensing prior",
    "h0": "Local H₀ (distance ladder)",
    "omega_b": "BBN prior on ω_b",
    "tau": "Reionization τ prior",
}

#: Which probe family each dataset belongs to, for grouping the
#: sidebar. A fit is usually built by picking *one* from each family
#: rather than by ticking everything, and the flat checkbox list made
#: that impossible to see.
DATASET_GROUPS = [
    ("📏 Expansion rate", ["cc"]),
    ("🌀 BAO (standard ruler)", ["desi", "sdss_bao", "sdss_fsbao", "bao_lowz",
                                "eboss_elg", "eboss_elg_fs",
                                "eboss_lya"]),
    ("💥 Supernovae (standard candle)", ["pantheon", "des_sn5yr", "union3"]),
    ("🔥 CMB", ["planck", "planck_lite", "planck_lowe",
                "planck_lensing", "act_lensing"]),
    ("🕸️ Growth of structure", ["fsigma8", "s8"]),
    ("📌 External measurements", ["h0", "omega_b", "tau"]),
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
        n="1590 SNe", z="0.001 – 2.26",
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
        n="1829 SNe", z="0.025 – 1.13",
        what=(
            "Dark Energy Survey 5-year sample. Of the three SN "
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
            "Planck's large-scale polarization constraint. Only "
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
INCOMPATIBLE_PAIRS = [
    ({"desi", "sdss_bao"}, "DESI and SDSS BAO target much of the same sky; combining them double-counts structure."),
    ({"pantheon", "des_sn5yr"}, "DES-SN5YR's low-z sample overlaps Pantheon+; combining them double-counts those supernovae."),
    ({"pantheon", "union3"}, "Union3 and Pantheon+ compile substantially the same supernovae."),
    ({"des_sn5yr", "union3"}, "Union3's high-z half overlaps the DES sample."),
    ({"planck", "planck_lite"}, "The distance priors are a compression of exactly these bandpowers -- this is the whole Planck dataset twice."),
    ({"planck_lensing", "act_lensing"}, "ACT's lensing map overlaps Planck's on the sky, so the two reconstructions are correlated; combining the separate likelihoods overstates the joint constraint."),
    ({"planck_lowe", "tau"}, "The τ prior is a Gaussian compression of exactly this low-ℓ EE likelihood -- the same measurement twice."),
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
        "takes hours, not minutes -- use a saved chain, raise "
        "n_processes, and consider the compressed distance priors "
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


# ============================================================
# Helpers
# ============================================================

def _model_capabilities(model_cls) -> dict:
    """
    What a model can be asked to do, read off the class itself
    rather than hardcoded -- so a custom model gets honest badges
    too.

    ``mu`` is checked for an *override*: every model inherits
    ``Cosmology.mu`` returning 1 (standard GR growth), and only a
    model that replaces it predicts a growth history differing from
    GR's at the same background.
    """

    from CosmoFit.cosmology.core.base import Cosmology
    from CosmoFit.cosmology.boltzmann import supports_cmb_spectra

    cmb_ok, cmb_reason = supports_cmb_spectra(model_cls)

    return {
        "w": hasattr(model_cls, "w"),
        "mu": getattr(model_cls, "mu", None) is not Cosmology.mu,
        "cmb": cmb_ok,
        "cmb_reason": cmb_reason,
        "extra": list(getattr(model_cls, "EXTRA_PARAMS", {}) or {}),
    }


# ------------------------------------------------------------

def _relevant_parameters(model_choice, model_cls, datasets, compute_rd,
                         derive_sigma8=False) -> set:
    """
    Which parameters actually do something in *this* fit.

    The union of what the model's E(z) uses, what its own
    ``EXTRA_PARAMS`` add, and what the ticked datasets require. A
    model the user built -- from an expression or from an action --
    is opaque, since it could reference anything, so everything is
    reported relevant rather than guessing and hiding something it
    needs.
    """

    if model_choice in (CUSTOM_CHOICE, ACTION_CHOICE):
        params_cls = getattr(model_cls, "PARAMS_CLASS", None)
        return set(params_cls.names()) if params_cls else set()

    relevant = set(ALWAYS_RELEVANT)

    relevant |= MODEL_STANDARD_PARAMS.get(model_choice, set())

    relevant |= set(getattr(model_cls, "EXTRA_PARAMS", {}) or {})

    for name in datasets:
        relevant |= DATASET_PARAMS.get(name, set())

    if compute_rd:
        # rd stops being a parameter and the densities behind it
        # start being ones.
        relevant.discard("rd")
        relevant |= COMPUTE_RD_PARAMS

    if derive_sigma8:
        # Same trade: sigma8 stops being sampled, ln1e10As carries
        # the amplitude instead.
        relevant.discard("sigma8")
        relevant |= {"ln1e10As"}

    return relevant


# ------------------------------------------------------------

def _fit_warnings(model_choice, model_cls, datasets, free_params,
                  compute_rd) -> list[tuple[str, str]]:
    """
    Model-and-dataset combinations worth flagging *before* the run,
    as ``(icon, message)`` pairs.

    Two kinds. Some are outright errors that would only surface as a
    stack trace minutes in (the CMB spectra with a modified-gravity
    model). The rest are quieter: fits that will run, finish, and
    produce a posterior for a parameter the data cannot constrain --
    which looks exactly like a real result.
    """

    warnings = []

    caps = _model_capabilities(model_cls)

    selected = set(datasets)

    if "planck_lite" in selected and not caps["cmb"]:
        warnings.append((
            "🚫",
            f"**{model_choice}** cannot be used with the full CMB "
            f"spectra: {caps['cmb_reason']} Use the compressed "
            f"distance priors instead, or change the model.",
        ))

    # Extra parameters left fixed: the model reduces to something
    # simpler and the fit is not testing what it looks like it is.
    idle_extra = [
        name for name in caps["extra"] if name not in free_params
    ]
    if idle_extra:
        warnings.append((
            "ℹ️",
            f"**{', '.join(idle_extra)}** left fixed. "
            f"{model_choice}'s own parameter(s) are not being fit, so "
            f"this is a fit of whatever it reduces to at those "
            f"values -- tick them under **Parameters** to actually "
            f"test the model.",
        ))

    # Growth-only models against background-only data.
    background_only = not (selected & {"fsigma8", "s8"})
    if caps["mu"] and background_only:
        warnings.append((
            "⚠️",
            f"**{model_choice}** modifies gravity, and its signature "
            f"is in how structure *grows*. With no growth dataset "
            f"ticked, only its expansion history is being tested -- "
            f"add **fσ₈** and/or **S₈**.",
        ))

    # sigma8 free but nothing measures it.
    if "sigma8" in free_params and background_only:
        warnings.append((
            "⚠️",
            "**σ₈** is free but no dataset constrains it -- its "
            "posterior will be its prior. Tick fσ₈ or S₈, or untick "
            "σ₈.",
        ))

    # BAO with rd free and nothing else to anchor H0.
    if compute_rd and not (selected & {"desi", "sdss_bao", "bao_lowz"}):
        warnings.append((
            "ℹ️",
            "**Compute r_d** is on but no BAO dataset is ticked. "
            "r_d only enters through BAO, so this changes nothing.",
        ))

    if compute_rd and "Omega_b" not in free_params:
        warnings.append((
            "ℹ️",
            "With r_d computed, **Ω_b** is what carries H₀ -- leaving "
            "it fixed pins r_d to one value and throws away the "
            "reason to compute it. Free Ω_b and tick the BBN prior.",
        ))

    if (selected & {"planck_lite", "planck_lensing", "act_lensing",
                    "planck_lowe"}) and "sigma8" in free_params:
        warnings.append((
            "⚠️",
            "**σ₈ is defined twice here.** The CMB datasets compute "
            "it from `ln10¹⁰A_s` through the transfer function, "
            "while fσ₈/S₈ are compared against the *free* σ₈ you "
            "are sampling — and nothing makes the two agree. Tick "
            "**Derive σ₈ from the CMB** in the sidebar, or fix σ₈.",
        ))

    if "planck_lite" in selected:
        missing = [
            name for name in ("ln1e10As", "n_s", "tau_reio")
            if name not in free_params
        ]
        if missing:
            warnings.append((
                "⚠️",
                f"The full CMB spectra depend on "
                f"**{', '.join(missing)}**, which are fixed. The fit "
                f"will run but is conditioning on those values rather "
                f"than measuring them.",
            ))
        if "tau" not in selected and "tau_reio" in free_params:
            warnings.append((
                "⚠️",
                "**τ** is free but the τ prior dataset is not ticked. "
                "plik_lite covers ℓ ≥ 30 only, where τ is degenerate "
                "with the primordial amplitude -- both posteriors will "
                "be unconstrained.",
            ))

    return warnings


# ------------------------------------------------------------

def _parse_extra_params(text: str) -> dict:
    """
    Parse the "one parameter per line" extra-parameter box:

        beta = 0.0, -2.0, 2.0, $\\beta$

    into the ``extra_params`` dict ``model_from_expression()``
    expects. The label is optional.
    """

    extra = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):

        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"Line {line_no}: expected "
                f"'name = default, lower, upper[, label]', got {raw!r}"
            )

        name, rest = line.split("=", 1)
        name = name.strip()
        parts = [p.strip() for p in rest.split(",")]

        if len(parts) < 3:
            raise ValueError(
                f"Line {line_no}: need at least "
                f"'default, lower, upper', got {raw!r}"
            )

        default, lower, upper = (float(parts[0]), float(parts[1]), float(parts[2]))

        spec = {"default": default, "bounds": (lower, upper)}

        if len(parts) > 3 and parts[3]:
            spec["label"] = parts[3]

        extra[name] = spec

    return extra


# ------------------------------------------------------------

def _parse_fields(text: str) -> dict:
    """
    Parse the scalar-field box:

        phi = X - V0*exp(-lam*phi)

    into the ``fields`` dict :class:`CosmoFit.theory.Action` takes:
    field name -> Lagrangian density ``L(X, phi)``, with ``X`` the
    kinetic scalar. More than one line is more than one field.
    """

    fields = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):

        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"Field line {line_no}: expected "
                f"'name = L(X, name)', got {raw!r}"
            )

        name, lagrangian = line.split("=", 1)

        name = name.strip()
        lagrangian = lagrangian.strip()

        if not name or not lagrangian:
            raise ValueError(
                f"Field line {line_no}: both a name and a Lagrangian "
                f"are needed, got {raw!r}"
            )

        fields[name] = lagrangian

    return fields


# ------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=32)
def _action_and_model(
    name: str,
    gravity: str,
    geometry: str | None,
    fluids: tuple,
    params_text: str,
    closure: str,
    growth: str,
    fields_text: str,
    z_init: float,
    background: str,
):
    """
    Build an :class:`~CosmoFit.theory.Action` and the model class it
    derives, returning both.

    Cached on the definition itself, because Streamlit re-runs the
    whole script on every widget interaction and this is the one
    genuinely expensive thing in it. Reducing the action, varying the
    lapse and solving the constraint is symbolic work -- and a field
    action additionally compiles an integrator -- where everything
    else on this page is a dictionary lookup.

    Both objects come back from one call so the page can show the
    Friedmann equation that was derived without deriving it twice.
    """

    action = Action(
        gravity,
        geometry=geometry,
        fluids=tuple(fluids),
        params=_parse_extra_params(params_text),
        closure=closure or None,
        growth=growth,
        fields=_parse_fields(fields_text) or None,
        z_init=float(z_init),
        background=background,
    )

    return action, action.build(name)


# ------------------------------------------------------------

def _action_widgets(slot: int) -> tuple:
    """
    This slot's action widgets, as the argument tuple
    :func:`_action_and_model` is keyed on.

    Reading them in one place is what keeps the cache key and the
    thing built from it in step -- a second reader that forgot one
    widget would return a stale model for a changed definition,
    which is exactly the failure a cache is good at hiding.
    """

    gravity = st.session_state.get(f"action_gravity_{slot}", "").strip()

    if not gravity:
        raise ValueError(
            "Enter a gravitational Lagrangian -- `R - 2*Lam` is "
            "General Relativity with a cosmological constant."
        )

    geometry = st.session_state.get(f"action_geometry_{slot}", "auto")

    return (
        (
            st.session_state.get(f"action_name_{slot}", "").strip()
            or f"Action{slot + 1}"
        ),
        gravity,
        None if geometry == "auto" else geometry,
        tuple(
            st.session_state.get(f"action_fluids_{slot}", None) or ("matter",)
        ),
        st.session_state.get(f"action_params_{slot}", ""),
        st.session_state.get(f"action_closure_{slot}", "").strip(),
        st.session_state.get(f"action_growth_{slot}", "gr"),
        st.session_state.get(f"action_fields_{slot}", ""),
        float(st.session_state.get(f"action_zinit_{slot}", 3000.0)),
        st.session_state.get(f"action_background_{slot}", "backward"),
    )


# ------------------------------------------------------------

def _build_model_class(slot: int, model_choice: str):
    """
    Resolve one model slot's widgets into a ``Cosmology`` subclass,
    raising a clear error for an incomplete/invalid custom-model
    definition.
    """

    if model_choice == ACTION_CHOICE:

        if not HAVE_THEORY:
            raise ValueError(
                "Deriving a model from an action needs sympy: "
                "pip install 'cosmofit[theory]'."
            )

        return _action_and_model(*_action_widgets(slot))[1]

    if model_choice != CUSTOM_CHOICE:
        return BUILTIN_MODELS[model_choice]

    name = st.session_state.get(f"custom_name_{slot}", "").strip() or f"Custom{slot + 1}"
    E_expr = st.session_state.get(f"custom_E_{slot}", "").strip()

    if not E_expr:
        raise ValueError("Enter an E(z) expression for the custom model.")

    w_expr = st.session_state.get(f"custom_w_{slot}", "").strip() or None
    dEdz_expr = st.session_state.get(f"custom_dEdz_{slot}", "").strip() or None
    mu_expr = st.session_state.get(f"custom_mu_{slot}", "").strip() or None
    extra_text = st.session_state.get(f"custom_extra_params_{slot}", "")

    extra_params = _parse_extra_params(extra_text)

    return model_from_expression(
        name,
        E=E_expr,
        extra_params=extra_params,
        w=w_expr,
        dEdz=dEdz_expr,
        mu=mu_expr,
    )


# ------------------------------------------------------------

def _fit_labels(fits: list[Fitter]) -> tuple[list[str], list[str]]:
    """
    Two index-aligned label lists for a set of fits: plain text for
    the UI (dropdowns, tables, JSON keys) and LaTeX for figure
    legends.

    Both are needed because they go to renderers with different
    abilities. Streamlit shows a string literally, so a selectbox
    option must read ``wCDM``, not ``$w$CDM``; matplotlib renders
    ``$...$`` as mathtext, so a legend should read ``ΛCDM`` rather
    than the ASCII spelling of the class name. Disambiguating
    suffixes (two fits of the same model) are applied to the plain
    names and then copied onto the LaTeX ones, so the two lists
    never disagree about which fit is "(2)".
    """

    plain = _dedupe_labels([f.model_cls.plain_name() for f in fits])

    latex = [
        f.model_cls.plot_label() + label[len(f.model_cls.plain_name()):]
        for f, label in zip(fits, plain)
    ]

    return plain, latex


# ------------------------------------------------------------

def _dedupe_labels(labels: list[str]) -> list[str]:
    """``["CPL", "CPL"]`` -> ``["CPL (1)", "CPL (2)"]``; leaves
    already-unique labels untouched."""

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    seen = {}
    out = []
    for label in labels:
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} ({seen[label]})")

    return out


# ------------------------------------------------------------

def _available_plots(fit: Fitter) -> list[str]:
    """Which single-model ``fit.plots.<name>()`` methods apply."""

    methods = []

    if fit.sampler is not None:
        methods += ["chain", "corner"]

    likelihood_plots = [
        (PantheonLikelihood, "hubble_diagram"),
        (DESSN5YRLikelihood, "des_hubble_diagram"),
        (Union3Likelihood, "union3_hubble_diagram"),
        (CCLikelihood, "hz"),
        (DESILikelihood, "bao_distances"),
        (SDSSBAOLikelihood, "sdss_bao_distances"),
        (BAOLowZLikelihood, "lowz_bao_distances"),
        (PlanckLikelihood, "planck_residuals"),
        (PlanckLiteLikelihood, "cmb_spectra"),
        (PlanckLensingLikelihood, "cmb_lensing"),
        (ACTDR6LensingLikelihood, "cmb_lensing"),
        (FSigma8Likelihood, "growth"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in fit.likelihoods):
            methods.append(method)

    # The two released likelihood *surfaces*, which go down different
    # branches of the same method -- a 1-D curve and a 2-D contour
    # set. Worth offering because they are the figure that shows why
    # a Gaussian summary of these two would be wrong.
    for cls in (EBOSSELGLikelihood, EBOSSLyaLikelihood):
        if any(isinstance(lk, cls) for lk in fit.likelihoods):
            methods.append("eboss_surface")
            break

    if hasattr(fit.cosmology, "w"):
        methods.append("w_of_z")

    # The w0-wa plane is a 2D posterior, so it needs both parameters
    # sampled -- not just present in the model.
    if fit.sampler is not None and {"w0", "wa"} <= set(fit.free_params):
        methods.append("w0_wa_plane")

    methods.append("deceleration")

    return methods


# ------------------------------------------------------------

def _available_compare_plots(fits: list[Fitter]) -> list[str]:
    """
    Which ``compare_*`` methods apply, based on the anchor (first)
    fit's datasets -- same logic as `_available_plots`, mapped to
    the comparison-plot names. `compare_w_of_z`/`compare_deceleration`
    are always valid (every model has an E(z), and models without
    their own w(z) fall back to the w=-1 line -- see
    `FitPlotter.compare_w_of_z`).

    `compare_w0_wa_plane` is the exception that needs *every* fit,
    not just the anchor: it overlays one posterior per model, so a
    single model without w0/wa free has nothing to contribute to it.
    """

    anchor_fit = fits[0]

    methods = []

    likelihood_plots = [
        (PantheonLikelihood, "compare_hubble_diagram"),
        (DESSN5YRLikelihood, "compare_des_hubble_diagram"),
        (CCLikelihood, "compare_hz"),
        (DESILikelihood, "compare_bao_distances"),
        (SDSSBAOLikelihood, "compare_sdss_bao_distances"),
        (FSigma8Likelihood, "compare_growth"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in anchor_fit.likelihoods):
            methods.append(method)

    methods.append("compare_w_of_z")

    if all(
        fit.sampler is not None and {"w0", "wa"} <= set(fit.free_params)
        for fit in fits
    ):
        methods.append("compare_w0_wa_plane")

    methods.append("compare_deceleration")

    return methods


# ------------------------------------------------------------

def _render_best_fit(fit: Fitter) -> None:

    result = fit.result

    if result.best_fit is None:
        st.caption("No best-fit result.")
        return

    chi2 = result.best_fit.chi2
    dof = fit.n_data - result.best_fit.ndim

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("χ²", f"{chi2:.2f}")
    m2.metric(
        "χ²/dof", f"{chi2 / dof:.3f}" if dof > 0 else "—",
        help=f"{fit.n_data} data points − {result.best_fit.ndim} free "
             f"parameters = {dof} degrees of freedom. Around 1 is a "
             f"good fit; well above 1 means the model cannot describe "
             f"the data, well below usually means the error bars are "
             f"conservative.",
    )
    m3.metric(
        "AIC", f"{result.best_fit.aic():.2f}",
        help="χ² + 2k. Lower is better; a difference below ~2 is "
             "not evidence either way.",
    )
    m4.metric(
        "BIC", f"{result.best_fit.bic():.2f}",
        help="χ² + k·ln(n). Penalizes extra parameters harder than "
             "AIC does, so it favours simpler models more strongly.",
    )

    st.dataframe(
        pd.DataFrame(
            {"parameter": list(result.best_fit.params),
             "value": list(result.best_fit.params.values())}
        ),
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # Which dataset is the fit actually struggling with?
    # --------------------------------------------------------
    #
    # A single total chi2 says a fit is bad without saying where.
    # The per-dataset breakdown is what turns "chi2 = 640" into
    # "the local H0 measurement is contributing 23 of it, on one
    # data point" -- which is the whole content of a tension.

    st.markdown("**χ² by dataset**")

    rows = []
    for likelihood in fit.likelihoods:
        summary = likelihood.summary()
        n = summary["n_data"]
        rows.append({
            "dataset": summary["name"],
            "N": n,
            "χ²": summary["chi2"],
            "χ²/N": summary["chi2"] / n if n else None,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "χ²": st.column_config.NumberColumn(format="%.2f"),
            "χ²/N": st.column_config.ProgressColumn(
                "χ²/N", format="%.2f", min_value=0.0, max_value=4.0,
                help="Per-point χ². A dataset sitting far above the "
                     "others is the one in tension with the rest.",
            ),
        },
    )

    st.caption(
        "Evaluated at the best-fit point. These sum to the total χ² "
        "above; a single dataset carrying a disproportionate share is "
        "where a tension lives."
    )

    _render_gate(fit)


# ------------------------------------------------------------

def _render_gate(fit: Fitter) -> None:
    """
    Whether the fitted model is a theory worth having fitted.

    Two questions, kept apart because they have different answers.
    ``viability()`` asks whether the theory is *consistent* -- no
    ghost graviton, no tachyonic scalaron, a positive effective
    coupling. ``screening()`` asks whether it is *allowed*, which
    is a statement about local gravity tests and not about the
    theory's health. A model can pass one and fail the other, and
    the arctan f(R) does exactly that: it fits as well as LCDM and
    is excluded by the Solar System by four orders of magnitude.

    Only shown for models that answer -- most of the library is
    dark energy on top of General Relativity, where neither
    question arises.
    """

    cosmology = getattr(fit, "cosmology", None)

    has_viability = hasattr(cosmology, "viability")
    has_screening = hasattr(cosmology, "screening")

    if not (has_viability or has_screening):
        return

    st.markdown("#### Is this a theory worth fitting?")

    columns = st.columns(2)

    with columns[0]:

        if has_viability:
            try:
                verdict = cosmology.viability()

                if verdict["ok"]:
                    st.success("**Consistent** — no ghost, no tachyon.")
                else:
                    st.error(
                        "**Not consistent.** "
                        + " ".join(verdict["reasons"])
                    )

            except Exception as exc:
                st.warning(f"Could not check consistency: {exc}")

    with columns[1]:

        if has_screening:
            try:
                verdict = cosmology.screening()

                over = verdict["deviation"] / verdict["bound"]

                if verdict["ok"]:
                    st.success(
                        f"**Allowed by local tests** — "
                        f"|f_R0| = {verdict['deviation']:.3g}, "
                        f"within the {verdict['bound']:.0e} bound."
                    )
                else:
                    st.error(
                        f"**Excluded by local tests** — "
                        f"|f_R0| = {verdict['deviation']:.3g}, "
                        f"{over:.3g}× the Solar System bound of "
                        f"{verdict['bound']:.0e}."
                    )

            except Exception as exc:
                st.warning(f"Could not check screening: {exc}")

    st.caption(
        "Consistency and exclusion are different questions and are "
        "asked separately, because a model can pass one and fail the "
        "other. The screening number is the *linear* estimate: "
        "chameleon screening is non-linear, so failing it means "
        "\"excluded unless screening rescues it\" rather than a proof."
    )


# ------------------------------------------------------------

def _render_posterior(fit: Fitter) -> None:

    result = fit.result

    if result.mcmc is None:
        st.caption("No MCMC run.")
        return

    st.dataframe(
        pd.DataFrame([
            {"parameter": name, "median": s["median"],
             "+": s["plus"], "-": s["minus"]}
            for name, s in result.mcmc.summary.items()
        ]),
        hide_index=True,
        width="stretch",
    )

    if result.mcmc.convergence["converged"]:
        st.success(
            "Converged -- chain length exceeds 50x the "
            "autocorrelation time.", icon="✅",
        )
    elif fit.chain is not None:
        st.warning(
            f"Not converged yet -- raise Steps above "
            f"{result.mcmc.nsteps} and run again before trusting "
            f"this posterior. The steps already sampled are saved, "
            f"so only the new ones cost anything.", icon="⚠️",
        )
    else:
        st.warning(
            "Not converged yet -- consider more steps before "
            "trusting this posterior.", icon="⚠️",
        )

    if fit.chain is not None:
        st.caption(
            f"Chain saved in `{fit.chain.path}` "
            f"({result.mcmc.nsteps} steps x {result.mcmc.nwalkers} walkers)."
        )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------
    #
    # `stats.derived` pushes every posterior sample back through the
    # model's own E(z), so these carry real error bars rather than
    # being evaluated once at the best fit. Nothing in the GUI
    # surfaced them before, which meant the acceleration transition
    # redshift -- a headline number in most dark-energy papers --
    # was reachable only from Python.

    with st.expander("Derived quantities (z_t, q₀, r_d)", expanded=False):

        st.caption(
            "Each posterior sample is pushed back through this "
            "model's own E(z) and dE/dz, so these are proper "
            "posteriors, not the best-fit value with no uncertainty."
        )

        try:
            from CosmoFit.stats import derived

            rows = []

            q0 = derived.summarize(derived.deceleration_today(fit))
            rows.append({
                "quantity": "q₀ (deceleration today)",
                "median": q0["median"],
                "+": q0["plus"], "−": q0["minus"],
            })

            z_t = derived.summarize(derived.transition_redshift(fit))
            rows.append({
                "quantity": "z_t (acceleration begins)",
                "median": z_t["median"],
                "+": z_t["plus"], "−": z_t["minus"],
            })

            r_d = derived.summarize(derived.sound_horizon(fit))
            rows.append({
                "quantity": "r_d [Mpc], from the densities",
                "median": r_d["median"],
                "+": r_d["plus"], "−": r_d["minus"],
            })

            st.dataframe(
                pd.DataFrame(rows), hide_index=True,
                width="stretch",
                column_config={
                    "median": st.column_config.NumberColumn(format="%.4g"),
                    "+": st.column_config.NumberColumn(format="%.3g"),
                    "−": st.column_config.NumberColumn(format="%.3g"),
                },
            )

            if q0["median"] < 0:
                st.caption(
                    f"q₀ < 0: the expansion is accelerating today, and "
                    f"began doing so at z ≈ {z_t['median']:.2f}."
                )
            else:
                st.caption(
                    "q₀ > 0: this fit does **not** have an "
                    "accelerating universe today."
                )

            if z_t.get("n_undefined"):
                st.caption(
                    f"{z_t['n_undefined']} sample(s) never cross "
                    f"q = 0 in the search range and are excluded."
                )

            if not fit.compute_rd:
                st.caption(
                    f"r_d here is what the early-universe physics "
                    f"*predicts* for these densities; the fit used "
                    f"the free parameter "
                    f"({fit.result.mcmc.summary['rd']['median']:.2f} Mpc) "
                    f"instead. A disagreement between the two is the "
                    f"standard signature of new physics before "
                    f"recombination."
                    if "rd" in fit.free_params else
                    "r_d here is what the early-universe physics "
                    "predicts for these densities; the fit used the "
                    "fixed `rd` value instead."
                )

        except Exception as exc:
            st.caption(f"Could not compute derived quantities: {exc}")

    # CPL-family diagnostics (w(z)=-1 crossing redshift, direction,
    # distance from the LCDM point) -- only meaningful when both w0
    # and wa were actually fit.
    if "w0" in fit.free_params and "wa" in fit.free_params:

        with st.expander("w0-wa posterior diagnostics"):

            samples = fit.samples_dict()

            z_cross, frac_cross = cpl_diagnostics.crossing_redshift(
                samples["w0"], samples["wa"],
            )

            st.write(
                f"**w(z) = -1 crossing:** {frac_cross:.1%} of samples "
                f"cross in z in [0, 2.5]"
                + (
                    f", at z = {z_cross.mean():.2f} (mean) if they do"
                    if len(z_cross) else ""
                )
            )

            if len(z_cross) > 0:
                direction = cpl_diagnostics.crossing_direction(
                    samples["w0"], samples["wa"],
                )
                st.write(
                    f"Quintessence → phantom: "
                    f"{direction['quintessence_to_phantom']:.1%}  ·  "
                    f"Phantom → quintessence: "
                    f"{direction['phantom_to_quintessence']:.1%}"
                )

            regions = cpl_diagnostics.region_fractions(
                samples["w0"], samples["wa"],
            )
            st.write(
                "**Dark-energy region** (see the w0-wa plane plot): "
                + "  ·  ".join(
                    f"{name}: {fraction:.1%}"
                    for name, fraction in regions.items()
                )
            )

            lcdm_distance = cpl_diagnostics.mahalanobis_from_lcdm(
                samples["w0"], samples["wa"],
            )
            # Report `sigma`, not `distance`: the Mahalanobis
            # distance D is not a number of sigma in 2D (D^2 follows
            # chi2 with 2 d.o.f.), and quoting it as one overstates
            # the tension -- see `mahalanobis_from_lcdm`.
            st.write(
                f"**LCDM point** (w0, wa) = (-1, 0) is excluded at "
                f"**{lcdm_distance['sigma']:.2f}σ** "
                f"({lcdm_distance['confidence_level']:.1%} confidence; "
                f"Mahalanobis distance D = "
                f"{lcdm_distance['distance']:.2f}, which is *not* a "
                f"number of sigma in 2D)."
            )


# ------------------------------------------------------------

def _render_profile(fit: Fitter, label: str) -> None:
    """
    Profile likelihood: chi2 minimized over every *other* free
    parameter, at each fixed value of one.

    The honest tool where Wilks' theorem does not apply -- a
    parameter against a prior edge, or a surface with a plateau,
    where the marginal posterior and the profile say different
    things and the marginal is the one that smooths a real feature
    away.
    """

    import numpy as np

    free = list(fit.free_params)

    if not free:
        st.info("This fit has no free parameters to profile.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        name = st.selectbox(
            "Parameter", options=free, key=f"profile_param_{label}",
        )

    index = free.index(name)

    centre = float(fit.result.best_fit.params[name])

    sigma = None

    if fit.result.mcmc is not None:
        entry = fit.result.mcmc.summary.get(name)
        if entry:
            sigma = 0.5 * (entry["plus"] + entry["minus"])

    if not sigma or not np.isfinite(sigma) or sigma <= 0.0:
        lo, hi = fit.prior.lower[index], fit.prior.upper[index]
        sigma = 0.05 * (hi - lo)

    with col2:
        width = st.number_input(
            "Half-width (σ)", min_value=1.0, max_value=8.0, value=3.0,
            step=0.5, key=f"profile_width_{label}",
            help="How far either side of the best fit to scan, in "
                 "units of this parameter's own posterior width.",
        )

    with col3:
        n_points = st.number_input(
            "Points", min_value=5, max_value=61, value=15, step=2,
            key=f"profile_points_{label}",
            help="Each point is a full re-minimization over every "
                 "other free parameter, so this is the cost.",
        )

    if not st.button(
        f"Profile `{name}`", key=f"profile_run_{label}", width="stretch",
    ):
        st.caption(
            f"{int(n_points)} re-minimizations over the other "
            f"{len(free) - 1} parameter(s). Each point warm-starts "
            f"from the previous one, so this is far cheaper than "
            f"{int(n_points)} cold fits."
        )
        return

    values = np.linspace(
        centre - width * sigma, centre + width * sigma, int(n_points),
    )

    # Stay inside the prior: a profile point outside it has an
    # infinite chi2 and tells you about the box, not the likelihood.
    values = values[
        (values >= fit.prior.lower[index]) & (values <= fit.prior.upper[index])
    ]

    if values.size < 3:
        st.warning(
            "That range falls almost entirely outside this "
            "parameter's prior. Widen the prior or narrow the scan.",
            icon="⚠️",
        )
        return

    with st.spinner(f"Profiling {name} at {values.size} points..."):
        profile = fit.profile(name, values)

    delta = np.asarray(profile["delta_chi2"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(profile["values"], delta, lw=1.8, marker="o", ms=3)

    ax.axhline(1.0, ls="--", lw=1.0, color="0.5")
    ax.axhline(4.0, ls=":", lw=1.0, color="0.7")
    ax.axvline(centre, ls="-", lw=1.0, color="0.8")

    ax.set_xlabel(name)
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # The Delta chi2 = 1 crossings, which is the interval a profile
    # actually reports -- not a standard deviation of anything.
    crossings = []

    for i in range(len(delta) - 1):
        if (delta[i] - 1.0) * (delta[i + 1] - 1.0) < 0.0:
            x0, x1 = profile["values"][i], profile["values"][i + 1]
            y0, y1 = delta[i], delta[i + 1]
            crossings.append(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))

    minimum = float(profile["values"][int(np.argmin(delta))])

    if len(crossings) == 2:
        st.success(
            f"**{name} = {minimum:.4g}**  "
            f"(−{minimum - crossings[0]:.3g} / +{crossings[1] - minimum:.3g})"
            f"  — from the Δχ² = 1 crossings",
            icon="📐",
        )
    else:
        st.info(
            "Δχ² does not cross 1 on both sides of this range, so "
            "there is no two-sided interval to quote. That is a "
            "result rather than a failure: it is what a parameter "
            "against a prior edge, or one the data barely constrain, "
            "looks like. Widen the scan to see which.",
            icon="ℹ️",
        )


# ------------------------------------------------------------

def _render_fisher(fit: Fitter, label: str) -> None:
    """
    Fisher errors, and -- where a chain exists -- the ratio to what
    the chain found.

    That ratio is the whole point of showing it here. A Fisher
    matrix is a *Gaussian approximation to the posterior*: cheap
    where an MCMC is not (~2n² evaluations against millions), good
    for a near-elliptical posterior, and poor for a parameter against
    a prior edge or a plateau. The only way to know which you have is
    to compare.
    """

    import numpy as np

    if not st.button(
        "Compute the Fisher matrix", key=f"fisher_run_{label}",
        width="stretch",
    ):
        st.caption(
            f"About {2 * fit.ndim ** 2} likelihood evaluations -- "
            f"seconds, against the chain above."
        )
        return

    with st.spinner("Differentiating chi2 at the best fit..."):
        fisher = fit.fisher()

    rows = []

    summary = fit.result.mcmc.summary if fit.result.mcmc else {}

    for name, value, error in zip(
        fisher["free_params"], fisher["theta"], fisher["errors"],
    ):
        row = {
            "Parameter": name,
            "Best fit": float(value),
            "Fisher σ": float(error),
        }

        entry = summary.get(name)

        if entry:
            mcmc_sigma = 0.5 * (entry["plus"] + entry["minus"])
            row["MCMC σ"] = float(mcmc_sigma)
            row["ratio"] = (
                float(error / mcmc_sigma) if mcmc_sigma else float("nan")
            )

        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if "ratio" in rows[0]:

        ratios = np.array([r["ratio"] for r in rows], dtype=float)

        worst = float(np.max(np.abs(np.log(ratios[np.isfinite(ratios)]))))

        if worst < 0.22:  # within ~25% either way
            st.success(
                "The Gaussian approximation holds here -- every error "
                "bar within ~25% of the chain's.",
                icon="✅",
            )
        else:
            st.warning(
                "At least one Fisher error bar differs from the "
                "chain's by more than 25%. That is the approximation "
                "failing rather than the chain being wrong: the "
                "posterior is not elliptical in that direction. "
                "Quote the chain, or a profile likelihood.",
                icon="⚠️",
            )


# ------------------------------------------------------------

def _render_evidence(fits: list[Fitter], labels: list[str]) -> None:
    """
    Bayesian evidence by nested sampling, and the Bayes factor
    between two models.

    Kept behind a button and a cost estimate because this is the one
    thing in the app that can run for minutes: nested sampling
    explores the whole prior volume rather than the posterior peak,
    which is exactly what makes it able to compare models at all.
    """

    try:
        from CosmoFit.stats.nested import run_nested
        from CosmoFit.stats import evidence as evidence_mod
    except ModuleNotFoundError:
        st.warning(
            "Nested sampling needs **dynesty**, an optional "
            "dependency:\n\n```\npip install 'cosmofit[evidence]'\n```",
            icon="📦",
        )
        return

    st.caption(
        "The evidence integrates the likelihood over the *prior*, so "
        "a Bayes factor is a statement about the priors as much as "
        "about the models -- an extra parameter that does nothing is "
        "penalised by the volume it was given. That is the Occam "
        "factor AIC and BIC only approximate, and it is why the three "
        "can disagree."
    )

    n_live = st.number_input(
        "Live points", min_value=100, max_value=2000, value=400, step=100,
        key="evidence_nlive",
        help="More is a tighter ln Z and a longer run. 400 is enough "
             "for the two- to four-parameter fits this app runs.",
    )

    if not st.button("Run nested sampling", width="stretch"):
        st.caption(
            f"Roughly 10⁴–10⁵ likelihood evaluations per model, "
            f"for {len(fits)} model(s). Minutes, not seconds."
        )
        return

    results = {}

    progress = st.progress(0.0, text="Sampling...")

    for i, (label, fit) in enumerate(zip(labels, fits)):

        progress.progress(
            i / len(fits), text=f"Nested sampling {label} ({i + 1}/{len(fits)})",
        )

        results[label] = run_nested(
            fit.logpost, fit.prior, fit.free_params,
            n_live=int(n_live), progress=False,
        )

    progress.empty()

    st.session_state["evidence_results"] = results

    st.dataframe(
        pd.DataFrame([
            {
                "model": label,
                "ln Z": result.log_evidence,
                "± ": result.log_evidence_error,
                "k": len(result.free_params),
                "evaluations": result.n_evaluations,
            }
            for label, result in results.items()
        ]),
        hide_index=True, width="stretch",
    )

    if len(results) >= 2:

        st.markdown("**Bayes factors**, against the first model:")

        null_label = labels[0]

        rows = []

        for label in labels[1:]:

            factor = evidence_mod.bayes_factor(
                results[label], results[null_label],
            )

            rows.append({
                "model": label,
                "vs": null_label,
                "ln B": factor["ln_B"],
                "±": factor["ln_B_error"],
                "verdict": factor["interpretation"],
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.caption(
            "Positive `ln B` favours the model in the first column. "
            "The labels are Kass & Raftery's."
        )


# ------------------------------------------------------------

def _render_tension(fits: list[Fitter], labels: list[str]) -> None:
    """
    How far apart two posteriors are, by two definitions that
    disagree exactly when it matters.

    The Gaussian one is the number everybody quotes. The
    sample-based one makes no Gaussian assumption -- it pairs the two
    sets of samples at random and asks where zero falls in the
    distribution of the difference -- so a skewed or double-peaked
    posterior, which is where "how many sigma" quietly stops meaning
    anything, still gets an honest answer.
    """

    import numpy as np

    from CosmoFit.stats import tension as tension_mod

    with_chains = [
        (label, fit) for label, fit in zip(labels, fits)
        if fit.sampler is not None
    ]

    if len(with_chains) < 2:
        st.info(
            "Two fits with chains are needed to compare posteriors. "
            "Add a second model above.",
            icon="ℹ️",
        )
        return

    shared = set(with_chains[0][1].free_params)

    for _, fit in with_chains[1:]:
        shared &= set(fit.free_params)

    if not shared:
        st.info(
            "These fits share no free parameter, so there is nothing "
            "to compare them on.",
            icon="ℹ️",
        )
        return

    col1, col2, col3 = st.columns(3)

    names = [label for label, _ in with_chains]

    with col1:
        first = st.selectbox("First", options=names, key="tension_a")

    with col2:
        second = st.selectbox(
            "Second", options=[n for n in names if n != first],
            key="tension_b",
        )

    with col3:
        parameter = st.selectbox(
            "Parameter", options=sorted(shared), key="tension_param",
        )

    fit_a = dict(with_chains)[first]
    fit_b = dict(with_chains)[second]

    samples_a = fit_a.flat_samples()[:, fit_a.free_params.index(parameter)]
    samples_b = fit_b.flat_samples()[:, fit_b.free_params.index(parameter)]

    summary_a = fit_a.summary()[parameter]
    summary_b = fit_b.summary()[parameter]

    gaussian = tension_mod.gaussian_tension(
        summary_a["median"], 0.5 * (summary_a["plus"] + summary_a["minus"]),
        summary_b["median"], 0.5 * (summary_b["plus"] + summary_b["minus"]),
    )

    sampled = tension_mod.sample_tension(samples_a, samples_b)

    cols = st.columns(2)

    cols[0].metric(
        "Gaussian", f"{gaussian['n_sigma']:.2f}σ",
        help="Assumes both posteriors are Gaussian and independent.",
    )
    cols[1].metric(
        "Sample-based", f"{sampled['n_sigma']:.2f}σ",
        help="No Gaussian assumption: where zero falls in the "
             "distribution of the paired difference.",
    )

    st.caption(
        f"{parameter}:  {first} = {summary_a['median']:.4g} "
        f"(+{summary_a['plus']:.3g}/−{summary_a['minus']:.3g})   ·   "
        f"{second} = {summary_b['median']:.4g} "
        f"(+{summary_b['plus']:.3g}/−{summary_b['minus']:.3g})"
    )

    if abs(gaussian["n_sigma"] - sampled["n_sigma"]) > 0.5:
        st.warning(
            "The two disagree by more than half a sigma, which means "
            "at least one of these posteriors is not Gaussian. Quote "
            "the sample-based number.",
            icon="⚠️",
        )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    difference = (
        np.random.default_rng(0).choice(samples_a, size=100000)
        - np.random.default_rng(1).choice(samples_b, size=100000)
    )

    ax.hist(difference, bins=120, histtype="step", lw=1.6)
    ax.axvline(0.0, color="0.3", lw=1.4)

    ax.set_xlabel(f"{parameter}  ({first} − {second})")
    ax.set_ylabel("posterior samples")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ------------------------------------------------------------

def _render_figure(fig, base_name: str, fmt_label: str, key: str) -> None:
    """
    Render a matplotlib figure plus a download button exporting it
    in `fmt_label` (a key of `PLOT_EXPORT_FORMATS`) -- the browser's
    own save dialog is what lets the user pick *where* it goes; this
    is only responsible for *what format* it goes there as.
    """

    st.pyplot(fig, width="stretch")

    ext, mime = PLOT_EXPORT_FORMATS[fmt_label]

    buf = io.BytesIO()
    fig.savefig(buf, format=ext, bbox_inches="tight")

    st.download_button(
        f"⬇️ {fmt_label}",
        data=buf.getvalue(),
        file_name=f"{base_name}.{ext}",
        mime=mime,
        key=key,
        width="stretch",
    )

    plt.close(fig)


# ============================================================
# Page
# ============================================================

st.set_page_config(page_title="CosmoFit", page_icon="🌌", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1200px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, version_col = st.columns([6, 1])
with title_col:
    st.title("🌌 CosmoFit")
    st.caption(
        "Cosmological parameter estimation, no code required -- "
        "built on the CosmoFit Python library."
    )
with version_col:
    st.write("")
    st.caption(f"v{__version__}")

# ------------------------------------------------------------
# Sidebar: datasets, MCMC settings (shared across every model)
# ------------------------------------------------------------

with st.sidebar:

    st.markdown("### 📊 Datasets")
    st.caption("Shared by every model below -- comparisons need the same data.")

    # ------------------------------------------------------
    # Presets
    # ------------------------------------------------------
    #
    # Fourteen checkboxes with five conflict rules between them is a
    # lot to face cold. A preset writes the whole configuration --
    # datasets, versions, compute_rd -- into session state and lets
    # the widgets below pick it up, so it is a starting point that
    # can then be edited, not a mode.

    preset_choice = st.selectbox(
        "Start from a preset",
        options=["— custom —", *DATASET_PRESETS],
        key="dataset_preset",
        help="A ready-made combination for a specific question. "
             "Applying one overwrites the ticks below; you can change "
             "them afterwards.",
    )

    if preset_choice != "— custom —":

        preset = DATASET_PRESETS[preset_choice]

        st.caption(preset["note"])

        # No `st.rerun()`: this block runs *above* the checkboxes it
        # writes to, so the values land before those widgets are
        # created and are picked up on this same pass. A rerun here
        # would be a second render for no gain -- and a button whose
        # click state outlives the rerun (as it does under
        # `streamlit.testing`) turns it into an infinite loop.
        if st.button("Apply preset", width="stretch"):
            for key in DATASET_REGISTRY:
                st.session_state[f"ds_{key}"] = key in preset["datasets"]
            for key, version in (preset.get("versions") or {}).items():
                st.session_state[f"dsver_{key}"] = version
            st.session_state["compute_rd"] = bool(preset.get("compute_rd"))

    # ------------------------------------------------------
    # The checkboxes, grouped by probe
    # ------------------------------------------------------

    selected_datasets = []
    dataset_versions = {}

    # Seed each checkbox's default exactly once. Passing `value=`
    # *and* writing the same key from a preset is the one thing
    # Streamlit explicitly warns about -- the two disagree on which
    # is authoritative, and the widget silently keeps the wrong one.
    # `setdefault` leaves session state alone once it exists, so the
    # widget below owns its value from then on and a preset can
    # overwrite it freely.
    for _key in DATASET_REGISTRY:
        st.session_state.setdefault(f"ds_{_key}", _key in ("cc", "desi"))

    for group_label, keys in DATASET_GROUPS:

        with st.expander(group_label, expanded=group_label.startswith(("📏", "🌀"))):

            for key in keys:

                if key not in DATASET_REGISTRY:
                    continue

                info = DATASET_INFO.get(key, {})

                ticked = st.checkbox(
                    DATASET_LABELS.get(key, key),
                    key=f"ds_{key}",
                    help=(
                        f"Measures {info.get('observable', '?')} · "
                        f"{info.get('n', '?')} · z = {info.get('z', '?')}"
                    ),
                )

                if info:
                    st.caption(
                        f"**{info['observable']}** · {info['n']} · "
                        f"z = {info['z']}"
                    )
                    st.caption(info["what"])
                    st.caption(f"🎯 Constrains: {info['constrains']}")

                    versions = available_versions(key)
                    if len(versions) > 1:
                        dataset_versions[key] = st.selectbox(
                            "Version", options=versions,
                            key=f"dsver_{key}",
                            label_visibility="collapsed",
                            disabled=not ticked,
                        )

                    st.caption(
                        f"📄 {dataset_reference(key, dataset_versions.get(key))}"
                    )

                if ticked:
                    selected_datasets.append(key)

                st.divider()

    selected_set = set(selected_datasets)

    if selected_set:
        st.caption(
            f"**{len(selected_set)} dataset(s) selected:** "
            + ", ".join(DATASET_LABELS.get(k, k) for k in selected_datasets)
        )
    else:
        st.caption("_No datasets selected._")

    for pair, reason in INCOMPATIBLE_PAIRS:
        if pair <= selected_set:
            st.warning(
                f"**{' + '.join(DATASET_LABELS.get(k, k) for k in sorted(pair))}**: "
                f"{reason}",
                icon="⚠️",
            )

    for key, note in SLOW_DATASETS.items():
        if key in selected_set:
            st.warning(f"**{DATASET_LABELS.get(key, key)}** -- {note}", icon="🐢")

    # ------------------------------------------------------
    # Sound horizon
    # ------------------------------------------------------

    st.markdown("### 🌀 Sound horizon $r_d$")

    with st.container(border=True):

        compute_rd = st.checkbox(
            "Compute $r_d$ instead of fitting it",
            key="compute_rd",
            help="Validated against CAMB's rdrag to 5e-5.",
        )

        if compute_rd:
            st.caption(
                "r_d is derived from ω_b, ω_cb, N_eff and Σm_ν by "
                "integrating the sound speed through the drag epoch. "
                "**H₀ becomes measurable** -- but through Ω_b, which "
                "BAO cannot pin down alone, so free Ω_b and tick the "
                "BBN prior. `rd` is dropped from the free parameters "
                "automatically."
            )
        else:
            st.caption(
                "r_d is a free nuisance parameter, so BAO constrains "
                "only the product H₀·r_d and **cannot measure H₀**. "
                "That is the safe default -- it assumes nothing about "
                "the early universe."
            )

    st.markdown("### 🌱 Amplitude $\\sigma_8$")

    with st.container(border=True):

        _camb_cmb = {"planck_lite", "planck_lensing", "planck_lowe",
                     "act_lensing"}

        derive_sigma8 = st.checkbox(
            "Derive $\\sigma_8$ from the CMB instead of fitting it",
            key="derive_sigma8",
            disabled=not (selected_set & _camb_cmb),
            help="Only available with a from-scratch CMB dataset, "
                 "since there is nothing else to derive it from.",
        )

        if not (selected_set & _camb_cmb):
            st.caption(
                "σ₈ is a free parameter. Tick a from-scratch CMB "
                "dataset above to make deriving it possible."
            )
        elif derive_sigma8:
            st.caption(
                "σ₈ comes from `ln10¹⁰A_s` through the transfer "
                "function, and the growth machinery normalizes with "
                "it. **This is what makes the S₈ tension askable**: "
                "the CMB's prediction meets the lensing measurement "
                "instead of a free parameter absorbing the gap. "
                "`sigma8` is dropped from the free parameters "
                "automatically."
            )
        else:
            st.caption(
                "σ₈ is free *and* the CMB fixes an amplitude of its "
                "own — two unrelated numbers for one quantity. Fine "
                "if σ₈ is left fixed; otherwise tick the box."
            )

    st.markdown("### ⚙️ MCMC settings")

    with st.container(border=True):

        col_a, col_b = st.columns(2)
        with col_a:
            nwalkers = st.number_input(
                "Walkers", min_value=8, value=48, step=2,
                help="At least 2x the number of free parameters you "
                     "tick below, or the fit will fail to start.",
            )
            burnin = st.number_input("Burn-in", min_value=0, value=500, step=50)
        with col_b:
            nsteps = st.number_input("Steps", min_value=50, value=3000, step=50)
            seed = st.number_input("Seed", min_value=0, value=42, step=1)

        auto_processes = st.checkbox(
            "Use all available CPU cores", value=True,
            help="Let CosmoFit decide how many worker processes to "
                 "use (every core this session is allowed to run on, "
                 "but only when the run is long enough to be worth "
                 "it). Safe with a Custom model -- it detects that "
                 "case and stays single-process instead of failing.",
        )

        if auto_processes:
            n_processes = "auto"
        else:
            n_processes = int(st.number_input(
                "Parallel processes", min_value=1,
                max_value=usable_cpu_count(), value=1, step=1,
                help="Evaluate walkers across multiple CPU cores. "
                     "Only works for built-in models -- a model you "
                     "built here, from an expression or from an "
                     "action, exists only in this session and cannot "
                     "be sent to a worker process. CosmoFit detects "
                     "that and stays single-process rather than "
                     "failing.",
            ))

        best_fit_restarts = int(st.number_input(
            "Best-fit restarts", min_value=0, max_value=32, value=0, step=1,
            help="After the chain, the best fit is found by an "
                 "optimizer, and an optimizer converges into whichever "
                 "basin it started in. Restarts draw that many further "
                 "starting points from the prior and keep the best "
                 "result. Worth setting when a model fits *worse* than "
                 "the one it contains as a special case -- an "
                 "impossible answer, and how this was found. Costs one "
                 "extra optimization each; nothing during sampling.",
        ))

    st.markdown("### 💾 Saved chains")

    with st.container(border=True):

        reuse_chains = st.checkbox(
            "Save chains and reuse them", value=True,
            help="Write each model's MCMC chain to disk as it is "
                 "sampled, and reuse it next time instead of "
                 "sampling it again. Add a second model and the "
                 "first one comes back instantly; raise Steps and "
                 "only the extra steps are sampled; close the app "
                 "and it's all still there. Changing a model, its "
                 "datasets, free parameters, priors, walkers or "
                 "seed makes a different fit, which gets its own "
                 "file -- so nothing is ever silently reused when "
                 "it shouldn't be.",
        )

        chain_dir = st.text_input(
            "Folder", value="chains", disabled=not reuse_chains,
            help="Where the .h5 chain files go, relative to "
                 "wherever the app was started. Delete files in "
                 "here to force a fresh run.",
        )

# ------------------------------------------------------------
# Reference: everything the sidebar and model panels say, in one
# browsable place
# ------------------------------------------------------------
#
# The per-widget notes answer "what is this one?" while you are
# looking at it. This answers "what is there, and which should I
# pick?" -- a different question, and one you want to answer before
# ticking anything.

with st.expander("📖 Guide — what every dataset and model is"):

    guide_datasets, guide_models, guide_workflow = st.tabs(
        ["Datasets", "Models", "How to use this"]
    )

    with guide_datasets:

        st.caption(
            "A fit is usually built by taking **one** entry from each "
            "family, not by ticking everything -- several pairs "
            "measure the same sky or the same supernovae and must not "
            "be combined."
        )

        rows = []
        for group_label, keys in DATASET_GROUPS:
            for key in keys:
                if key not in DATASET_REGISTRY:
                    continue
                info = DATASET_INFO.get(key, {})
                rows.append({
                    "Family": group_label,
                    "Dataset": DATASET_LABELS.get(key, key),
                    "Measures": info.get("observable", ""),
                    "Size": info.get("n", ""),
                    "Redshift": info.get("z", ""),
                    "Constrains": info.get("constrains", ""),
                    "Reference": dataset_reference(key),
                })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**Do not combine**")
        for pair, reason in INCOMPATIBLE_PAIRS:
            st.caption(
                f"❌ **{' + '.join(DATASET_LABELS.get(k, k) for k in sorted(pair))}** "
                f"— {reason}"
            )

    with guide_models:

        st.caption(
            "What a model *changes* decides what can be done with it. "
            "Only models with a w(z) can be handed to a Boltzmann code "
            "for the full CMB spectra; only models that modify gravity "
            "predict a growth history differing from GR's at the same "
            "expansion history."
        )

        rows = []
        for group_label, names in MODEL_GROUPS:
            for name in names:
                info = MODEL_INFO.get(name, {})
                caps = _model_capabilities(BUILTIN_MODELS[name])
                rows.append({
                    "Family": group_label,
                    "Model": name,
                    "Extra parameters": info.get("params", ""),
                    "Reduces to": info.get("reduces") or "—",
                    "Own w(z)": "✅" if caps["w"] else "—",
                    "Modifies growth": "✅" if caps["mu"] else "—",
                    "CMB spectra": "✅" if caps["cmb"] else "🚫",
                    "Reference": info.get("ref", ""),
                })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.caption(
            "**Own w(z)** means the model exposes an equation of "
            "state of its own, which is what the w(z) figure plots "
            "and what a Boltzmann code needs. ΛCDM is marked '—' "
            "because w = -1 is a constant it never has to compute, "
            "not because it lacks one. **Modifies growth** means the "
            "model overrides μ(a,k); everything else grows structure "
            "exactly as GR does at the same expansion history."
        )

        st.caption(
            "**PEDE** and **DGP** are worth noting: both have exactly "
            "ΛCDM's parameter count and a completely different "
            "expansion history, so an AIC/BIC comparison against ΛCDM "
            "carries identical penalties and a χ² difference is purely "
            "a difference in fit."
        )

    with guide_workflow:

        st.markdown(
            """
**1. Pick the data.** Start from a preset in the sidebar, then
adjust. The warnings under each family are not decoration -- combining
two datasets that share supernovae or sky understates every error bar
in the result, with no other symptom.

**2. Pick the model, and free the right parameters.** Every model
shows what its extra parameters mean and which values reduce it to
ΛCDM. A model whose own parameters are left fixed is not being
tested -- the app says so under the parameter table.

**3. Check the parameter table.** By default it shows only the
parameters this particular fit uses. `rd` appears only with BAO,
`sigma8` only with growth data, `n_s`/`tau` only with the full CMB
spectra. A parameter that is free but unconstrained returns a
posterior identical to its prior, which looks exactly like a
measurement.

**4. Run, then read the χ² breakdown first.** A total χ² says a fit
is bad without saying where. The per-dataset table on the **Best fit**
tab is what turns that into "the local H₀ measurement contributes 24
of it, on one data point" -- which is the entire content of the Hubble
tension.

**5. Check convergence before believing the posterior.** The MCMC tab
says outright whether the chain is long enough. With chain saving on,
raising **Steps** and re-running only costs the extra steps.

**6. Compare.** Add a second model to get AIC/BIC, a likelihood-ratio
test where the two are nested, and every figure with both curves
overlaid.
            """
        )

# ------------------------------------------------------------
# Main panel: models to compare
# ------------------------------------------------------------

st.markdown("## 🧮 Models to compare")
st.caption(
    "Configure one model to fit it on its own, or add more to compare "
    "them side by side -- statistically (AIC/BIC/a likelihood-ratio "
    "test) and on the same plots."
)

st.session_state.setdefault("n_models", 1)

add_col, remove_col, _ = st.columns([1, 1, 4])
with add_col:
    if st.session_state["n_models"] < MAX_MODELS:
        if st.button("➕ Add model to compare", width="stretch"):
            st.session_state["n_models"] += 1
with remove_col:
    if st.session_state["n_models"] > 1:
        if st.button("➖ Remove last model", width="stretch"):
            st.session_state["n_models"] -= 1

n_models = st.session_state["n_models"]

model_classes = []
model_free_params = []
model_initial = []
model_bounds = []
build_error = None

for i in range(n_models):

    label = "Model 1 (primary)" if i == 0 else f"Model {i + 1}"

    with st.container(border=True):

        st.markdown(f"**{label}**")

        # Options carry their family as a prefix, so the dropdown
        # says what kind of object each entry is instead of listing
        # seventeen names with nothing to separate ΛCDM from f(Q).
        model_options = []
        option_to_model = {}
        for group_label, names in MODEL_GROUPS:
            for name in names:
                display = f"{name}  ·  {group_label}"
                model_options.append(display)
                option_to_model[display] = name
        for extra in (CUSTOM_CHOICE, ACTION_CHOICE):
            display = f"{extra}  ·  Not in the library"
            model_options.append(display)
            option_to_model[display] = extra

        model_display = st.selectbox(
            "Cosmology", options=model_options,
            key=f"model_choice_{i}", label_visibility="collapsed",
        )
        model_choice = option_to_model[model_display]

        if model_choice in MODEL_EQUATIONS:
            st.latex(MODEL_EQUATIONS[model_choice])

        info = MODEL_INFO.get(model_choice)
        if info:
            st.markdown(info["what"])
            cols = st.columns(2)
            cols[0].caption(f"**Extra parameters:** {info['params']}")
            cols[1].caption(
                f"**Reduces to:** {info['reduces']}" if info["reduces"]
                else "**Reduces to:** — (not a ΛCDM extension)"
            )
            st.caption(f"📄 {info['ref']}")

        if model_choice in BACKGROUND_DEGENERATE_MODELS:
            st.warning(BACKGROUND_DEGENERATE_MODELS[model_choice], icon="⚠️")

        if model_choice == CUSTOM_CHOICE:

            col1, col2 = st.columns([1, 2])
            with col1:
                st.text_input(
                    "Model name", value=f"Custom{i + 1}", key=f"custom_name_{i}",
                )
            with col2:
                st.text_input(
                    "E(z) expression", key=f"custom_E_{i}",
                    placeholder="sqrt(Omega_m*(1+z)**3 + (1-Omega_m)*(1+z)**(3*(1+w0))*(1+beta*z))",
                    help=(
                        "Available: z, every standard parameter "
                        "(H0, Omega_m, Omega_k, w0, wa, rd, MB, Omega_b, "
                        "A_s, alpha), any extra parameters defined below, "
                        "and sqrt/exp/log/log10/sin/cos/tan/sinh/cosh/"
                        "tanh/abs/sign/where/minimum/maximum/pi/e."
                    ),
                )

            with st.expander("Advanced (w(z), dE/dz, mu(a,k), extra parameters)"):

                col3, col4 = st.columns(2)
                with col3:
                    st.text_input(
                        "w(z) expression (optional)", key=f"custom_w_{i}",
                        help="For the w(z) plot only -- not needed to fit.",
                    )
                with col4:
                    st.text_input(
                        "dE/dz expression (optional)", key=f"custom_dEdz_{i}",
                        help=(
                            "For the deceleration-parameter plot only. "
                            "If left blank, a numerical derivative of "
                            "E(z) is used automatically."
                        ),
                    )

                st.text_input(
                    "mu(a,k) expression (optional)", key=f"custom_mu_{i}",
                    placeholder="1 + 3*beta",
                    help=(
                        "Effective gravitational coupling G_eff/G_N for "
                        "growth of structure (the 'fsigma8'/'s8' "
                        "datasets and the growth/compare_growth plots). "
                        "Available: a (scale factor), k (wavenumber in "
                        "h/Mpc, or unused if your model is scale-"
                        "independent), plus every parameter. If left "
                        "blank, mu=1 everywhere (standard GR growth) -- "
                        "correct unless your model modifies gravity "
                        "itself, not just the expansion history."
                    ),
                )

                st.text_area(
                    "Extra parameters (one per line, optional)",
                    key=f"custom_extra_params_{i}", height=80,
                    placeholder="beta = 0.0, -2.0, 2.0, $\\beta$",
                    help="name = default, lower, upper[, label]",
                )

        elif model_choice == ACTION_CHOICE:

            if not HAVE_THEORY:

                st.warning(
                    "Deriving a model from an action needs **sympy**, "
                    "an optional dependency:\n\n"
                    "```\npip install 'cosmofit[theory]'\n```\n\n"
                    "Nothing else in the library needs it -- models "
                    "written directly as `E(z)` (including **Custom** "
                    "above) work without it.",
                    icon="📦",
                )

            else:

                st.caption(
                    "Give the **action**, not the answer. `CosmoFit.theory` "
                    "writes FLRW with an explicit lapse, reduces the action "
                    "to a point-like Lagrangian, varies the lapse to get the "
                    "Friedmann *constraint*, and solves it for `E(z)`. What "
                    "comes back is an ordinary model that every dataset and "
                    "plot here already understands."
                )

                # Above the widgets it writes to, for the same reason
                # the dataset preset is: the values land before those
                # widgets are created and are picked up on this pass.
                preset_choice = st.selectbox(
                    "Start from a worked example",
                    options=list(ACTION_PRESETS),
                    key=f"action_preset_{i}",
                    help="Each is a real model. Applying one overwrites "
                         "the boxes below; edit them afterwards.",
                )

                preset = ACTION_PRESETS.get(preset_choice)

                if preset and st.button(
                    "Load this action", key=f"action_load_{i}", width="stretch",
                ):
                    st.session_state[f"action_gravity_{i}"] = preset["gravity"]
                    st.session_state[f"action_geometry_{i}"] = preset["geometry"]
                    st.session_state[f"action_params_{i}"] = preset["params"]
                    st.session_state[f"action_closure_{i}"] = preset["closure"]
                    st.session_state[f"action_growth_{i}"] = preset["growth"]
                    st.session_state[f"action_fields_{i}"] = preset["fields"]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.text_input(
                        "Model name", value=f"Action{i + 1}",
                        key=f"action_name_{i}",
                    )

                with col2:
                    st.selectbox(
                        "Geometry",
                        options=["auto", *GEOMETRIES],
                        key=f"action_geometry_{i}",
                        help=(
                            "Curvature (`R`), torsion (`T`) or "
                            "non-metricity (`Q`) -- three formulations "
                            "that agree in General Relativity and part "
                            "company once `f` is deformed. `auto` reads "
                            "it off whichever scalar appears below. The "
                            "sign convention is fixed by requiring an "
                            "undeformed `f` to reproduce GR exactly."
                        ),
                    )

                st.text_input(
                    "Gravitational Lagrangian  f", key=f"action_gravity_{i}",
                    placeholder="R - 2*Lam",
                    help=(
                        "An expression in the geometry scalar (`R`, `T` "
                        "or `Q`), the standard cosmological parameters, "
                        "any parameter declared below, and any scalar "
                        "field. `R0`/`T0`/`Q0` are the scalar's value "
                        "today, which is how the f(Q) literature writes "
                        "its models. Just `R` is General Relativity."
                    ),
                )

                st.text_area(
                    "Parameters (one per line)", key=f"action_params_{i}",
                    height=80,
                    placeholder="Lam = 2.1, 0.0, 6.0, $\\Lambda$",
                    help="name = default, lower, upper[, label]",
                )

                col3, col4 = st.columns([1, 1])

                with col3:
                    st.text_input(
                        "Closure parameter (optional)",
                        key=f"action_closure_{i}",
                        placeholder="Lam",
                        help=(
                            "The one parameter fixed by requiring "
                            "`E(0) = 1` rather than fitted. In ΛCDM this "
                            "is what makes `Omega_de0 = 1 - Omega_m`. "
                            "Leave blank if the action satisfies the "
                            "condition on its own -- it is checked "
                            "either way, and an action that predicts "
                            "`E(0) != 1` is refused rather than fitted, "
                            "since it would get every distance wrong by "
                            "a constant factor while looking healthy."
                        ),
                    )

                with col4:
                    st.selectbox(
                        "Growth of structure",
                        options=["gr", "quasi_static"],
                        key=f"action_growth_{i}",
                        help=(
                            "`gr` leaves μ = 1. `quasi_static` asks for "
                            "the sub-horizon result -- μ = 1/f' for a "
                            "deformed f(T)/f(Q), or the scalar-tensor "
                            "form for a field coupled to curvature. It "
                            "is an *additional* physical assumption on "
                            "top of the action, which is why it has to "
                            "be asked for. Only matters if you fit "
                            "`fsigma8` or `s8`."
                        ),
                    )

                st.selectbox(
                    "f(R) integration direction",
                    options=["backward", "forward"],
                    key=f"action_background_{i}",
                    help=(
                        "Only used by a general `f(R)` -- anything "
                        "non-linear in `R`. `backward` integrates "
                        "from today outwards, which is right while "
                        "the scalaron's oscillating mode decays "
                        "into the past, as it does for "
                        "`R + alpha*R**2`. `forward` integrates "
                        "from deep in matter domination towards "
                        "today, and is what the \"disappearing "
                        "cosmological constant\" family needs -- "
                        "Hu-Sawicki, Starobinsky 2007, Tsujikawa, "
                        "the arctan models. Backwards those reach "
                        "only z ~ 1.2 however carefully `R_0` is "
                        "chosen, because the mode grows the other "
                        "way. Two things swap over with `forward`: "
                        "`R_0` stops being a parameter and becomes "
                        "derived, and a closure parameter becomes "
                        "**required**, since `E(0) = 1` is then a "
                        "condition to satisfy rather than where "
                        "the integration starts."
                    ),
                )

                with st.expander("Scalar fields, fluids, and z_init"):

                    st.text_area(
                        "Scalar fields (one per line, optional)",
                        key=f"action_fields_{i}", height=80,
                        placeholder="phi = X - V0*exp(-lam*phi)",
                        help=(
                            "`name = L(X, name)`, with `X` the kinetic "
                            "scalar -- `X - V(phi)` is quintessence, any "
                            "other `L(X, phi)` is k-essence. A field's "
                            "name is also in scope in the Lagrangian "
                            "above, which is how scalar-tensor gravity "
                            "is written: `(1 + xi*phi**2)*R` couples the "
                            "field to curvature rather than adding it on "
                            "top of GR. A field action is *integrated* "
                            "rather than solved redshift by redshift, so "
                            "it is slower (~40 ms a point against ~150 "
                            "µs) and needs a closure parameter."
                        ),
                    )

                    col5, col6 = st.columns([1, 1])

                    with col5:
                        st.multiselect(
                            "Fluids",
                            options=list(STANDARD_FLUIDS),
                            default=["matter"],
                            key=f"action_fluids_{i}",
                            help="What else is in the universe besides "
                                 "whatever the action itself provides.",
                        )

                    with col6:
                        st.number_input(
                            "z_init (fields only)",
                            min_value=10.0, max_value=100000.0,
                            value=3000.0, step=500.0,
                            key=f"action_zinit_{i}",
                            help=(
                                "Where a field's initial conditions are "
                                "set, and the earliest redshift the "
                                "model can be asked about. Early rather "
                                "than today on purpose: integrating "
                                "*backwards* from a field at rest now "
                                "turns Hubble friction into "
                                "anti-friction and gives a "
                                "kinetic-dominated past that is not a "
                                "universe. The growth ODE starts at "
                                "z = 9999, so raise this above that to "
                                "fit `fsigma8`."
                            ),
                        )

        try:
            model_cls = _build_model_class(i, model_choice)
        except Exception as exc:
            st.info(f"{label}: not ready yet -- {exc}")
            build_error = True
            model_classes.append(None)
            model_free_params.append(None)
            model_initial.append(None)
            model_bounds.append(None)
            continue

        model_classes.append(model_cls)

        # ------------------------------------------------------
        # The equation the action turned into
        # ------------------------------------------------------
        #
        # Worth showing rather than hiding: for most of these
        # actions the Friedmann equation is the thing somebody
        # would otherwise have spent an afternoon deriving, and
        # seeing it is how you check the library understood the
        # Lagrangian you meant.

        if model_choice == ACTION_CHOICE and HAVE_THEORY:

            with st.expander("The Friedmann equation this derives"):

                try:
                    import sympy as _sp

                    action = _action_and_model(*_action_widgets(i))[0]
                    expr, E2, _z = action.constraint()

                    st.caption(
                        "Varying the lapse gives a *constraint* rather "
                        "than an evolution equation -- that is what "
                        "makes it the Friedmann equation. It vanishes "
                        "on-shell; `E2` is the squared dimensionless "
                        "Hubble rate."
                    )

                    st.latex(_sp.latex(_sp.Eq(expr, 0)))

                    if action.is_fourth_order:
                        st.caption(
                            "Nonlinear in `R`, so this is a "
                            "**fourth-order** theory: it is reduced by "
                            "a Lagrange multiplier and integrated, and "
                            "carries `R_0`, the Ricci scalar today -- "
                            "an initial condition General Relativity "
                            "does not have."
                        )

                    elif action.fields:
                        st.caption(
                            f"Carries {len(action.fields)} dynamical "
                            f"field(s), so the history is **integrated** "
                            f"from z = {action.z_init:g} with `E(0) = 1` "
                            f"as a shooting condition, not solved "
                            f"redshift by redshift."
                        )

                except Exception as exc:
                    st.caption(f"Could not render it: {exc}")

        # ------------------------------------------------------
        # What this model can be asked to do
        # ------------------------------------------------------

        caps = _model_capabilities(model_cls)

        badge_cols = st.columns(3)
        badge_cols[0].caption(
            ("✅ has **w(z)**" if caps["w"] else "➖ no w(z)")
            + " — needed for the w(z) plot"
        )
        badge_cols[1].caption(
            ("✅ modifies **growth**" if caps["mu"]
             else "➖ standard GR growth")
            + " — μ(a,k)"
        )
        badge_cols[2].caption(
            "✅ **CMB spectra** computable" if caps["cmb"]
            else "🚫 no full CMB spectra"
        )

        params_cls = getattr(model_cls, "PARAMS_CLASS", None)
        parameter_set = params_cls.parameter_set()
        defaults = params_cls.defaults()

        relevant = _relevant_parameters(
            model_choice, model_cls, selected_datasets, compute_rd,
            derive_sigma8,
        )

        # A model that *derives* a parameter rather than accepting
        # it -- ADE fixes Omega_m from its early-time condition --
        # must not have it ticked by default. `Fitter` warns about
        # it, but a default that starts in the state the library
        # warns about is a poor default: the posterior you would get
        # back is the prior, and it would look like a measurement.
        derived_params = set(getattr(model_cls, "DERIVED_PARAMS", ()) or ())

        if derived_params:
            st.caption(
                "🔒 **Derived, not fitted:** "
                + ", ".join(f"`{name}`" for name in sorted(derived_params))
                + " — this model computes it from its own parameters, "
                "so sampling it would return the prior. Left un-ticked "
                "below."
            )

        param_rows = []
        for p in parameter_set:
            lo, hi = p.bounds if p.bounds else (None, None)
            # H0/Omega_m (and any custom parameter without an
            # explicit "default") have no dataclass default --
            # falling back to 0.0 would sit outside their prior
            # bounds and produce a degenerate walker cloud (every
            # walker clipped to the same edge). The bounds midpoint
            # is always inside the prior instead.
            if p.name in defaults:
                initial_value = float(defaults[p.name])
            elif lo is not None and hi is not None:
                initial_value = float((lo + hi) / 2.0)
            else:
                initial_value = 0.0
            param_rows.append({
                "Parameter": p.name,
                "Label": p.label,
                "Free": (
                    p.name in ("H0", "Omega_m")
                    and p.name not in derived_params
                ),
                "Initial": initial_value,
                "Lower": lo,
                "Upper": hi,
                "_relevant": p.name in relevant,
            })

        all_rows = pd.DataFrame(param_rows)

        with st.expander("Parameters", expanded=True):

            hide_irrelevant = st.checkbox(
                "Show only the parameters this fit uses",
                value=True, key=f"relevant_only_{i}",
                help="The parameter container is shared by every "
                     "model, so it carries every parameter any of "
                     "them needs. Which ones actually do something "
                     "depends on the model *and* the datasets -- rd "
                     "means nothing without BAO, sigma8 nothing "
                     "without growth data. Untick to see them all.",
            )

            shown = all_rows[all_rows["_relevant"]] if hide_irrelevant else all_rows

            if hide_irrelevant and len(shown) < len(all_rows):
                st.caption(
                    f"Showing {len(shown)} of {len(all_rows)} — "
                    f"{len(all_rows) - len(shown)} hidden because "
                    f"nothing in this fit depends on them."
                )

            edited_df = st.data_editor(
                shown.drop(columns=["_relevant"]),
                hide_index=True,
                width="stretch",
                disabled=["Parameter", "Label"],
                column_config={
                    "Free": st.column_config.CheckboxColumn("Fit this parameter?"),
                    "Initial": st.column_config.NumberColumn(format="%.5g"),
                    "Lower": st.column_config.NumberColumn(format="%.5g"),
                    "Upper": st.column_config.NumberColumn(format="%.5g"),
                },
                key=f"param_editor_{i}",
            )

        free = edited_df.loc[edited_df["Free"], "Parameter"].tolist()
        model_free_params.append(free)

        # Hidden parameters still have to reach the Fitter -- they
        # are inert, not absent, and `Fitter` requires a value for
        # every field of the container. Start from every default and
        # let the edited rows override.
        initial = {
            row["Parameter"]: row["Initial"]
            for _, row in all_rows.iterrows()
        }
        initial.update(dict(zip(edited_df["Parameter"], edited_df["Initial"])))
        model_initial.append(initial)

        bounds = {
            row["Parameter"]: (row["Lower"], row["Upper"])
            for _, row in all_rows.iterrows()
            if pd.notna(row["Lower"]) and pd.notna(row["Upper"])
        }
        bounds.update({
            row["Parameter"]: (row["Lower"], row["Upper"])
            for _, row in edited_df.iterrows()
            if pd.notna(row["Lower"]) and pd.notna(row["Upper"])
        })
        model_bounds.append(bounds)

        st.caption(
            f"**{len(free)} free parameter(s)**"
            + (f" — {', '.join(free)}" if free else "")
        )

        for icon, message in _fit_warnings(
            model_choice, model_cls, selected_datasets, free, compute_rd,
        ):
            if icon == "🚫":
                st.error(message, icon=icon)
            elif icon == "⚠️":
                st.warning(message, icon=icon)
            else:
                st.caption(f"{icon} {message}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

run_clicked = st.button("🚀 Run Fit", type="primary", width="stretch")

if run_clicked:

    if not selected_datasets:
        st.error("Select at least one dataset.", icon="🚫")
    elif build_error:
        st.error("Fix the model(s) marked above before running.", icon="🚫")
    elif any(not fp for fp in model_free_params):
        st.error("Tick at least one free parameter for every model.", icon="🚫")
    else:
        progress_bar = st.progress(0.0, text="Starting...")
        fits = []

        try:
            for i in range(n_models):

                free_params = model_free_params[i]

                if derive_sigma8 and "sigma8" in free_params:
                    free_params = [n for n in free_params if n != "sigma8"]
                    st.info(
                        f"Model {i + 1}: dropped `sigma8` from the free "
                        f"parameters -- it is being derived, not fitted.",
                        icon="ℹ️",
                    )

                if compute_rd and "rd" in free_params:
                    # The GUI's parameter table lists every parameter
                    # the model has, and `rd` is ticked by default for
                    # BAO fits -- so silently un-tick it rather than
                    # raising at people who ticked a box in one place
                    # and a checkbox in another.
                    free_params = [n for n in free_params if n != "rd"]
                    st.info(
                        f"Model {i + 1}: dropped `rd` from the free "
                        f"parameters -- it is being computed, not fitted.",
                        icon="ℹ️",
                    )

                # `Fitter` warns about combinations that will run,
                # finish, and mean nothing -- a free `m_nu` with no
                # CMB that can see it, a non-minimally coupled model
                # meeting growth data with mu still 1, a parameter
                # the model derives rather than accepts. Those go to
                # `warnings`, which in a Streamlit app means stderr,
                # which means nowhere. Catching them here surfaces
                # every guard the library has and every one it grows
                # later, without the GUI having to restate any of
                # them.
                import warnings as _warnings

                with _warnings.catch_warnings(record=True) as caught:

                    _warnings.simplefilter("always", UserWarning)

                    fit = Fitter(
                        model=model_classes[i],
                        datasets=selected_datasets,
                        free_params=free_params,
                        initial=model_initial[i],
                        bounds=model_bounds[i],
                        dataset_kwargs={
                            key: {"version": version}
                            for key, version in dataset_versions.items()
                            if key in selected_datasets
                        } or None,
                        compute_rd=compute_rd,
                        derive_sigma8=derive_sigma8,
                    )

                for entry in caught:
                    st.warning(
                        f"**Model {i + 1}:** {entry.message}", icon="⚠️",
                    )

                model_label = f"Model {i + 1}"
                last_shown = {"pct": -1}

                def _on_step(step, total, elapsed, _bar=progress_bar,
                             _last=last_shown, _i=i, _n=n_models, _label=model_label):
                    pct = int(100 * step / total)
                    if pct == _last["pct"] and step != total:
                        return
                    _last["pct"] = pct
                    rate = step / elapsed if elapsed > 0 else 0.0
                    overall = (_i + pct / 100.0) / _n
                    _bar.progress(
                        overall,
                        text=(
                            f"Fitting {_label} ({_i + 1}/{_n}) -- {pct}% "
                            f"({step}/{total} steps, {rate:.1f} it/s)"
                        ),
                    )

                # One file per distinct fit (`chain_id` hashes the
                # model, datasets, free parameters and priors, plus
                # the two run settings a stored chain can't change
                # halfway through). Same configuration as last
                # time -> that file is picked up and nothing is
                # re-sampled; anything changed -> a different file,
                # sampled fresh, with the old one left alone.
                save = None
                if reuse_chains and chain_dir.strip():
                    save = ChainFile(
                        os.path.join(
                            chain_dir.strip(),
                            f"{fit.chain_id(nwalkers=int(nwalkers), seed=int(seed))}.h5",
                        )
                    )

                fit.run_mcmc(
                    nwalkers=int(nwalkers), nsteps=int(nsteps),
                    burnin=int(burnin), seed=int(seed), progress=False,
                    n_processes=n_processes, callback=_on_step,
                    save=save,
                )
                fit.best_fit(restarts=int(best_fit_restarts))

                # A fully-cached model never runs a step, so
                # `_on_step` never fires -- move the bar on itself,
                # or it sits at the previous model's position.
                progress_bar.progress(
                    (i + 1) / n_models,
                    text=f"Fitted {model_label} ({i + 1}/{n_models})",
                )

                fits.append(fit)

            progress_bar.empty()

            plain_labels, plot_labels = _fit_labels(fits)
            st.session_state["fits"] = fits
            st.session_state["fit_labels"] = plain_labels
            st.session_state["fit_plot_labels"] = plot_labels

            reused = sum(isinstance(f.sampler, StoredSampler) for f in fits)

            if reused:
                st.toast(
                    f"Fit complete -- {reused} of {len(fits)} read "
                    f"straight from a saved chain.",
                    icon="✅",
                )
            else:
                st.toast("Fit complete.", icon="✅")

        except Exception as exc:
            st.session_state.pop("fits", None)
            st.session_state.pop("fit_labels", None)
            st.session_state.pop("fit_plot_labels", None)
            progress_bar.empty()
            st.error(f"Fit failed: {exc}", icon="🚫")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

fits = st.session_state.get("fits")
fit_labels = st.session_state.get("fit_labels")
fit_plot_labels = st.session_state.get("fit_plot_labels")

st.divider()

if fits:

    st.markdown("## 📊 Results")
    st.caption(
        f"{', '.join(DATASET_LABELS.get(d, d) for d in fits[0].dataset_names)}  ·  "
        + "  ·  ".join(
            f"**{label}** ({', '.join(f.free_params)})"
            for label, f in zip(fit_labels, fits)
        )
    )

    multi = len(fits) >= 2

    tab_names = (["⚖️ Comparison"] if multi else []) + [
        "📈 Best fit", "🎲 MCMC posterior", "🔬 Inference", "🖼️ Plots",
    ]
    tabs = st.tabs(tab_names)
    tab_iter = iter(tabs)

    if multi:

        with next(tab_iter):

            rows = []
            for label, f in zip(fit_labels, fits):
                bf = f.result.best_fit
                rows.append({
                    "model": label,
                    "chi2": bf.chi2,
                    "k (free params)": bf.ndim,
                    "AIC": bf.aic(),
                    "BIC": bf.bic(),
                    "converged": (
                        f.result.mcmc.convergence["converged"]
                        if f.result.mcmc else None
                    ),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            # Likelihood-ratio test: only well-defined for exactly two
            # models where one's free parameters are a strict subset
            # of the other's (a genuine nested special case) -- can't
            # be inferred for 3+ models or non-nested pairs.
            if len(fits) == 2:

                free_sets = [set(f.free_params) for f in fits]

                if free_sets[0] < free_sets[1]:
                    null_i, alt_i = 0, 1
                elif free_sets[1] < free_sets[0]:
                    null_i, alt_i = 1, 0
                else:
                    null_i = None

                if null_i is not None:

                    comparison = model_comparison.compare_models(
                        name_null=fit_labels[null_i],
                        chi2_null=fits[null_i].best_fit_chi2,
                        k_null=fits[null_i].ndim,
                        name_alt=fit_labels[alt_i],
                        chi2_alt=fits[alt_i].best_fit_chi2,
                        k_alt=fits[alt_i].ndim,
                        n_data=fits[alt_i].n_data,
                    )
                    lrt = comparison["likelihood_ratio_test"]

                    st.markdown(
                        f"**Likelihood-ratio test** ({fit_labels[null_i]} vs. "
                        f"{fit_labels[alt_i]}, nested at "
                        f"{', '.join(free_sets[alt_i] - free_sets[null_i])}=fixed): "
                        f"Δχ² = {lrt['delta_chi2']:.2f} (Δk={lrt['delta_k']}), "
                        f"p = {lrt['p_value']:.3f}, "
                        f"**{lrt['sigma']:.2f}σ** preference for "
                        f"{fit_labels[alt_i]}."
                    )
                else:
                    st.caption(
                        "No likelihood-ratio test shown: neither model's "
                        "free parameters are a strict subset of the "
                        "other's, so they aren't nested."
                    )

    with next(tab_iter):
        if multi:
            choice = st.selectbox("Model", options=fit_labels, key="bestfit_model_choice")
            _render_best_fit(fits[fit_labels.index(choice)])
        else:
            _render_best_fit(fits[0])

    with next(tab_iter):
        if multi:
            choice = st.selectbox("Model", options=fit_labels, key="posterior_model_choice")
            _render_posterior(fits[fit_labels.index(choice)])
        else:
            _render_posterior(fits[0])

    with next(tab_iter):

        st.caption(
            "Four questions an MCMC posterior on its own does not "
            "answer. Each is a separate calculation and each is "
            "behind its own button, because two of them cost real "
            "time."
        )

        profile_tab, fisher_tab, evidence_tab, tension_tab = st.tabs([
            "Profile likelihood", "Fisher matrix",
            "Bayesian evidence", "Tension",
        ])

        with profile_tab:
            if multi:
                choice = st.selectbox(
                    "Model", options=fit_labels, key="profile_model_choice",
                )
                chosen = fits[fit_labels.index(choice)]
            else:
                choice, chosen = fit_labels[0], fits[0]

            _render_profile(chosen, choice)

        with fisher_tab:
            if multi:
                choice = st.selectbox(
                    "Model", options=fit_labels, key="fisher_model_choice",
                )
                chosen = fits[fit_labels.index(choice)]
            else:
                choice, chosen = fit_labels[0], fits[0]

            _render_fisher(chosen, choice)

        with evidence_tab:
            _render_evidence(fits, fit_labels)

        with tension_tab:
            _render_tension(fits, fit_labels)

    with next(tab_iter):

        download_format = st.radio(
            "Download format", options=list(PLOT_EXPORT_FORMATS),
            horizontal=True, key="download_format",
            help="Applies to every '⬇️' button below. SVG/PDF are vector "
                 "(best for papers); PNG is raster (best for slides/sharing).",
        )

        if multi:

            compare_options = _available_compare_plots(fits)
            chosen = st.multiselect(
                "Choose comparison plots (all models overlaid)",
                options=compare_options,
                default=compare_options[:2],
                format_func=lambda name: COMPARE_PLOT_LABELS.get(name, name),
            )

            plot_cols = st.columns(2)
            for idx, name in enumerate(chosen):
                with plot_cols[idx % 2]:
                    try:
                        fig = getattr(fits[0].plots, name)(
                            other_fits=fits[1:], labels=fit_plot_labels,
                        )
                        _render_figure(fig, name, download_format, key=f"dl_compare_{name}")
                    except Exception as exc:
                        st.error(
                            f"Could not render "
                            f"'{COMPARE_PLOT_LABELS.get(name, name)}': {exc}",
                            icon="🚫",
                        )

            with st.expander("Single-model plots"):
                choice = st.selectbox("Model", options=fit_labels, key="plots_model_choice")
                single_fit = fits[fit_labels.index(choice)]
                single_options = _available_plots(single_fit)
                single_chosen = st.multiselect(
                    "Choose plots", options=single_options,
                    format_func=lambda name: PLOT_LABELS.get(name, name),
                    key="single_plots_multiselect",
                )
                single_cols = st.columns(2)
                for idx, name in enumerate(single_chosen):
                    with single_cols[idx % 2]:
                        try:
                            fig = getattr(single_fit.plots, name)()
                            _render_figure(
                                fig, f"{choice}_{name}", download_format,
                                key=f"dl_single_{choice}_{name}",
                            )
                        except Exception as exc:
                            st.error(
                                f"Could not render '{PLOT_LABELS.get(name, name)}': {exc}",
                                icon="🚫",
                            )

        else:

            plot_options = _available_plots(fits[0])
            chosen = st.multiselect(
                "Choose plots to render",
                options=plot_options,
                default=plot_options[:2],
                format_func=lambda name: PLOT_LABELS.get(name, name),
            )

            plot_cols = st.columns(2)
            for idx, name in enumerate(chosen):
                with plot_cols[idx % 2]:
                    try:
                        fig = getattr(fits[0].plots, name)()
                        _render_figure(fig, name, download_format, key=f"dl_{name}")
                    except Exception as exc:
                        st.error(
                            f"Could not render '{PLOT_LABELS.get(name, name)}': {exc}",
                            icon="🚫",
                        )

    st.download_button(
        "⬇️ Download result(s) (JSON)",
        data=json.dumps(
            {label: f.result.to_dict() for label, f in zip(fit_labels, fits)},
            indent=2,
            # Same coercion `FitResult.save_json` uses -- a chain read
            # back from HDF5 reports numpy counters, which plain
            # `json.dumps` refuses.
            default=_json_default,
        ),
        file_name="cosmofit_result.json",
        mime="application/json",
    )

else:
    st.info(
        "👆 Configure one or more models above, then click **Run Fit**.",
        icon="🌌",
    )
