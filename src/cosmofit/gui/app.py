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
in this package.

This module is the page itself, run by Streamlit top to bottom on
every interaction; what it draws with lives beside it
(:mod:`~gui.reference`, :mod:`~gui.helpers`, :mod:`~gui.render`).
Start it with::

    pip install "cosmofit[gui]"
    cosmofit gui

Local use only: the custom-model expression box below is evaluated
with ``eval()`` (builtins stripped, only whitelisted numpy math and
the model's own parameter names reach it -- see
``cosmofit.cosmology.custom._compile_expression``). That is a
reasonable trust boundary for a tool you run on your own machine,
not for a public, multi-tenant deployment.
"""

from __future__ import annotations

import json
import os

# Before anything imports pyplot -- the library does, on import -- so
# figures render off-screen in the server process.
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from cosmofit import (
    __version__,
    Fitter,
)
from cosmofit import available_versions, dataset_reference
from cosmofit.stats import DATASET_REGISTRY, model_comparison
from cosmofit.stats.chains import ChainFile, StoredSampler
from cosmofit.stats.results import _json_default
from cosmofit.stats.fitter import usable_cpu_count

from cosmofit.gui.reference import (
    ACTION_CHOICE,
    ACTION_PRESETS,
    BACKGROUND_DEGENERATE_MODELS,
    BUILTIN_MODELS,
    COMPARE_PLOT_LABELS,
    CUSTOM_CHOICE,
    DATASET_GROUPS,
    DATASET_INFO,
    DATASET_LABELS,
    DATASET_PRESETS,
    GEOMETRIES,
    HAVE_THEORY,
    INCOMPATIBLE_PAIRS,
    MAX_MODELS,
    MODEL_EQUATIONS,
    MODEL_GROUPS,
    MODEL_INFO,
    PLOT_EXPORT_FORMATS,
    PLOT_LABELS,
    SLOW_DATASETS,
    STANDARD_FLUIDS,
)
from cosmofit.gui.helpers import (
    _action_and_model,
    _action_widgets,
    _build_model_class,
    _fit_warnings,
    _model_capabilities,
    _relevant_parameters,
)
from cosmofit.gui.render import (
    _available_compare_plots,
    _available_plots,
    _configuration,
    _equivalent_script,
    _fit_labels,
    _render_best_fit,
    _render_evidence,
    _render_figure,
    _render_fisher,
    _render_posterior,
    _render_profile,
    _render_tension,
    _restore_configuration,
)


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

    # ------------------------------------------------------
    # Saving and restoring the whole configuration
    # ------------------------------------------------------
    #
    # Placed above every widget it writes to, so a restored value
    # is picked up on this same pass -- the ordering the dataset
    # presets already rely on.

    with st.expander("💾 Configuration"):

        uploaded = st.file_uploader(
            "Restore a saved configuration",
            type=["json"],
            key="config_upload",
            help="A file saved below, from this session or someone "
                 "else's. Datasets, models, parameters and MCMC "
                 "settings come back; results do not.",
        )

        if uploaded is not None and not st.session_state.get(
            "_config_applied_" + uploaded.name, False
        ):
            try:
                applied = _restore_configuration(json.loads(uploaded.getvalue()))

                # Once per file. Streamlit re-runs the whole script
                # on every interaction and the uploader keeps
                # returning the same file, so without this the
                # configuration would be re-applied on every click
                # and no edit made afterwards would ever stick.
                st.session_state["_config_applied_" + uploaded.name] = True

                st.success(f"Restored {applied} settings.")

            except Exception as exc:
                st.error(f"Could not read that file: {exc}", icon="🚫")

        st.download_button(
            "⬇️ Save this configuration",
            data=json.dumps(_configuration(), indent=2),
            file_name="cosmofit_configuration.json",
            mime="application/json",
            key="dl_config",
            width="stretch",
            help="Everything on this page that says *what to run* -- "
                 "not what was run. Hand it to someone else and they "
                 "get your setup, not your results.",
        )

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
                key="mcmc_nwalkers",
                help="At least 2x the number of free parameters you "
                     "tick below, or the fit will fail to start.",
            )
            burnin = st.number_input(
                "Burn-in", min_value=0, value=500, step=50,
                key="mcmc_burnin",
            )
        with col_b:
            nsteps = st.number_input(
                "Steps", min_value=50, value=3000, step=50,
                key="mcmc_nsteps",
            )
            seed = st.number_input(
                "Seed", min_value=0, value=42, step=1,
                key="mcmc_seed",
            )

        auto_processes = st.checkbox(
            "Use all available CPU cores", value=True,
            key="mcmc_auto_processes",
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
                    "Give the **action**, not the answer. `cosmofit.theory` "
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

    with st.expander("🐍 The same fit in Python"):

        st.caption(
            "This app is a way into the CosmoFit library, not a "
            "replacement for it. Copy this to move the same fit into "
            "a notebook, a batch job, or version control -- it is "
            "built from the values this run used, so it cannot drift "
            "away from what you just did."
        )

        try:
            snippet = _equivalent_script(
                model_choice=model_choice,
                model_names=[
                    getattr(cls, "MODEL_NAME", None) or cls.__name__
                    for cls in model_classes
                ],
                action_specs=[
                    st.session_state.get(f"_action_spec_{i}")
                    for i in range(len(model_classes))
                ],
                datasets=list(selected_datasets),
                dataset_versions=dict(dataset_versions),
                free_params=list(free_params),
                initial=dict(model_initial[0]),
                bounds=dict(model_bounds[0] or {}),
                compute_rd=bool(compute_rd),
                derive_sigma8=bool(derive_sigma8),
                nwalkers=int(nwalkers),
                nsteps=int(nsteps),
                burnin=int(burnin),
                seed=int(seed),
            )

            st.code(snippet, language="python")

            st.download_button(
                "⬇️ Download as .py",
                data=snippet,
                file_name="cosmofit_run.py",
                mime="text/x-python",
                key="dl_script",
            )

        except Exception as exc:
            st.caption(f"Could not build the snippet: {exc}")

    with st.expander("🔗 The posterior samples"):

        st.caption(
            "The chain is the expensive thing this page produced -- "
            "the summary above is a handful of numbers derived from "
            "it. Take the samples away to make your own contours, "
            "re-marginalise, or combine them with something else."
        )

        for label, fit in zip(fit_labels, fits):

            try:
                samples = fit.flat_samples()

            except Exception as exc:
                st.caption(f"{label}: no chain to export ({exc})")
                continue

            frame = pd.DataFrame(samples, columns=list(fit.free_params))

            st.download_button(
                f"⬇️ {label} — {len(frame):,} samples (CSV)",
                data=frame.to_csv(index=False),
                file_name=f"cosmofit_chain_{label.replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"dl_chain_{label}",
                help="Post-burn-in, flattened across walkers. One row "
                     "per sample, one column per free parameter.",
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
