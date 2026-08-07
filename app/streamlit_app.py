"""
CosmoFit -- graphical interface.

A thin Streamlit layer over the public ``CosmoFit`` API
(``Fitter``, ``FitPlotter``, ``define_model`` / ``model_from_expression``):
tick which datasets to fit, configure one or more models (built-in or
your own ``E(z)`` expression), choose which parameters are free, and
run an MCMC fit + look at the resulting plots and model-comparison
statistics -- no code required. Everything here is a consumer of
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
    Fitter,
    model_from_expression,
    CCLikelihood,
    DESILikelihood,
    SDSSBAOLikelihood,
    PantheonLikelihood,
    DESSN5YRLikelihood,
    PlanckLikelihood,
)
from CosmoFit.stats import DATASET_REGISTRY, model_comparison, cpl_diagnostics


# ============================================================
# Static reference data
# ============================================================

BUILTIN_MODELS = {
    "LCDM": LCDM,
    "WCDM": WCDM,
    "CPL": CPL,
    "JBP": JBP,
    "BA": BA,
    "GCG": GCG,
}

DATASET_LABELS = {
    "cc": "Cosmic Chronometers (CC)",
    "desi": "DESI 2024 BAO",
    "sdss_bao": "SDSS BAO (BOSS DR12 + eBOSS DR16)",
    "pantheon": "Pantheon+ (SNe Ia)",
    "des_sn5yr": "DES-SN5YR (SNe Ia)",
    "planck": "Planck 2018 (CMB distance priors)",
}

#: Dataset pairs that double-count data if combined -- see README.
INCOMPATIBLE_PAIRS = [
    ({"desi", "sdss_bao"}, "DESI and SDSS BAO target much of the same sky; combining them double-counts structure."),
    ({"pantheon", "des_sn5yr"}, "DES-SN5YR's low-z sample overlaps Pantheon+; combining them double-counts those supernovae."),
]

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
    "GCG": r"p = -\dfrac{A}{\rho^{\alpha}}",
}

#: Single-model figures (Fitter.plots.<name>()).
PLOT_LABELS = {
    "chain": "MCMC chain (trace plot)",
    "corner": "Corner plot",
    "hubble_diagram": "Hubble diagram (Pantheon+)",
    "des_hubble_diagram": "Hubble diagram (DES-SN5YR)",
    "hz": "H(z) diagram (CC)",
    "bao_distances": "BAO distances (DESI)",
    "sdss_bao_distances": "BAO distances (SDSS)",
    "planck_residuals": "Planck residuals (pull plot)",
    "w_of_z": "w(z) evolution",
    "deceleration": "Deceleration parameter q(z)",
}

#: Model-comparison figures (Fitter.plots.compare_<name>(other_fits=...)).
COMPARE_PLOT_LABELS = {
    "compare_hz": "H(z) diagram (CC)",
    "compare_hubble_diagram": "Hubble diagram (Pantheon+)",
    "compare_des_hubble_diagram": "Hubble diagram (DES-SN5YR)",
    "compare_bao_distances": "BAO distances (DESI)",
    "compare_sdss_bao_distances": "BAO distances (SDSS)",
    "compare_w_of_z": "w(z) evolution",
    "compare_deceleration": "Deceleration parameter q(z)",
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


# ============================================================
# Helpers
# ============================================================

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

def _build_model_class(slot: int, model_choice: str):
    """
    Resolve one model slot's widgets into a ``Cosmology`` subclass,
    raising a clear error for an incomplete/invalid custom-model
    definition.
    """

    if model_choice != "Custom":
        return BUILTIN_MODELS[model_choice]

    name = st.session_state.get(f"custom_name_{slot}", "").strip() or f"Custom{slot + 1}"
    E_expr = st.session_state.get(f"custom_E_{slot}", "").strip()

    if not E_expr:
        raise ValueError("Enter an E(z) expression for the custom model.")

    w_expr = st.session_state.get(f"custom_w_{slot}", "").strip() or None
    dEdz_expr = st.session_state.get(f"custom_dEdz_{slot}", "").strip() or None
    extra_text = st.session_state.get(f"custom_extra_params_{slot}", "")

    extra_params = _parse_extra_params(extra_text)

    return model_from_expression(
        name,
        E=E_expr,
        extra_params=extra_params,
        w=w_expr,
        dEdz=dEdz_expr,
    )


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
        (CCLikelihood, "hz"),
        (DESILikelihood, "bao_distances"),
        (SDSSBAOLikelihood, "sdss_bao_distances"),
        (PlanckLikelihood, "planck_residuals"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in fit.likelihoods):
            methods.append(method)

    if hasattr(fit.cosmology, "w"):
        methods.append("w_of_z")

    methods.append("deceleration")

    return methods


# ------------------------------------------------------------

def _available_compare_plots(anchor_fit: Fitter) -> list[str]:
    """
    Which ``compare_*`` methods apply, based on the anchor (first)
    fit's datasets -- same logic as `_available_plots`, mapped to
    the comparison-plot names. `compare_w_of_z`/`compare_deceleration`
    are always valid (every model has an E(z), and models without
    their own w(z) fall back to the w=-1 line -- see
    `FitPlotter.compare_w_of_z`).
    """

    methods = []

    likelihood_plots = [
        (PantheonLikelihood, "compare_hubble_diagram"),
        (DESSN5YRLikelihood, "compare_des_hubble_diagram"),
        (CCLikelihood, "compare_hz"),
        (DESILikelihood, "compare_bao_distances"),
        (SDSSBAOLikelihood, "compare_sdss_bao_distances"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in anchor_fit.likelihoods):
            methods.append(method)

    methods.append("compare_w_of_z")
    methods.append("compare_deceleration")

    return methods


# ------------------------------------------------------------

def _render_best_fit(fit: Fitter) -> None:

    result = fit.result

    if result.best_fit is None:
        st.caption("No best-fit result.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("χ²", f"{result.best_fit.chi2:.3f}")
    m2.metric("AIC", f"{result.best_fit.aic():.2f}")
    m3.metric("BIC", f"{result.best_fit.bic():.2f}")

    st.dataframe(
        pd.DataFrame(
            {"parameter": list(result.best_fit.params),
             "value": list(result.best_fit.params.values())}
        ),
        hide_index=True,
        use_container_width=True,
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
        use_container_width=True,
    )

    if result.mcmc.convergence["converged"]:
        st.success(
            "Converged -- chain length exceeds 50x the "
            "autocorrelation time.", icon="✅",
        )
    else:
        st.warning(
            "Not converged yet -- consider more steps before "
            "trusting this posterior.", icon="⚠️",
        )

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

            lcdm_distance = cpl_diagnostics.mahalanobis_from_lcdm(
                samples["w0"], samples["wa"],
            )
            st.write(
                f"**LCDM point** (w0, wa) = (-1, 0) is "
                f"**{lcdm_distance['distance']:.2f}σ** from this "
                f"posterior's mean."
            )


# ------------------------------------------------------------

def _render_figure(fig, base_name: str, fmt_label: str, key: str) -> None:
    """
    Render a matplotlib figure plus a download button exporting it
    in `fmt_label` (a key of `PLOT_EXPORT_FORMATS`) -- the browser's
    own save dialog is what lets the user pick *where* it goes; this
    is only responsible for *what format* it goes there as.
    """

    st.pyplot(fig, use_container_width=True)

    ext, mime = PLOT_EXPORT_FORMATS[fmt_label]

    buf = io.BytesIO()
    fig.savefig(buf, format=ext, bbox_inches="tight")

    st.download_button(
        f"⬇️ {fmt_label}",
        data=buf.getvalue(),
        file_name=f"{base_name}.{ext}",
        mime=mime,
        key=key,
        use_container_width=True,
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

    with st.container(border=True):
        selected_datasets = []
        for key in DATASET_REGISTRY:
            default = key in ("cc", "desi")
            if st.checkbox(DATASET_LABELS.get(key, key), value=default, key=f"ds_{key}"):
                selected_datasets.append(key)

    selected_set = set(selected_datasets)
    for pair, reason in INCOMPATIBLE_PAIRS:
        if pair <= selected_set:
            st.warning(f"{' + '.join(sorted(pair))}: {reason}", icon="⚠️")

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

        n_processes = st.number_input(
            "Parallel processes", min_value=1,
            max_value=max(1, os.cpu_count() or 1), value=1, step=1,
            help="Evaluate walkers across multiple CPU cores for a "
                 "real (if less than linear) speedup. Only works for "
                 "built-in models, not a Custom one -- leave at 1 if "
                 "any model below is Custom.",
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
        if st.button("➕ Add model to compare", use_container_width=True):
            st.session_state["n_models"] += 1
with remove_col:
    if st.session_state["n_models"] > 1:
        if st.button("➖ Remove last model", use_container_width=True):
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

        model_choice = st.selectbox(
            "Cosmology", options=[*BUILTIN_MODELS.keys(), "Custom"],
            key=f"model_choice_{i}", label_visibility="collapsed",
        )

        if model_choice in MODEL_EQUATIONS:
            st.latex(MODEL_EQUATIONS[model_choice])

        if model_choice == "Custom":

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

            with st.expander("Advanced (w(z), dE/dz, extra parameters)"):

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

                st.text_area(
                    "Extra parameters (one per line, optional)",
                    key=f"custom_extra_params_{i}", height=80,
                    placeholder="beta = 0.0, -2.0, 2.0, $\\beta$",
                    help="name = default, lower, upper[, label]",
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

        params_cls = getattr(model_cls, "PARAMS_CLASS", None)
        parameter_set = params_cls.parameter_set()
        defaults = params_cls.defaults()

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
                "Free": p.name in ("H0", "Omega_m"),
                "Initial": initial_value,
                "Lower": lo,
                "Upper": hi,
            })

        with st.expander("Parameters", expanded=True):
            edited_df = st.data_editor(
                pd.DataFrame(param_rows),
                hide_index=True,
                use_container_width=True,
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
        model_initial.append(dict(zip(edited_df["Parameter"], edited_df["Initial"])))
        model_bounds.append({
            row["Parameter"]: (row["Lower"], row["Upper"])
            for _, row in edited_df.iterrows()
            if pd.notna(row["Lower"]) and pd.notna(row["Upper"])
        })

        st.caption(
            f"{len(free)} free parameter(s)" + (f" -- {', '.join(free)}" if free else "")
        )

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

run_clicked = st.button("🚀 Run Fit", type="primary", use_container_width=True)

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

                fit = Fitter(
                    model=model_classes[i],
                    datasets=selected_datasets,
                    free_params=model_free_params[i],
                    initial=model_initial[i],
                    bounds=model_bounds[i],
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

                fit.run_mcmc(
                    nwalkers=int(nwalkers), nsteps=int(nsteps),
                    burnin=int(burnin), seed=int(seed), progress=False,
                    n_processes=int(n_processes), callback=_on_step,
                )
                fit.best_fit()

                fits.append(fit)

            progress_bar.empty()

            model_names = [f.model_cls.__name__ for f in fits]
            st.session_state["fits"] = fits
            st.session_state["fit_labels"] = _dedupe_labels(model_names)
            st.toast("Fit complete.", icon="✅")

        except Exception as exc:
            st.session_state.pop("fits", None)
            st.session_state.pop("fit_labels", None)
            progress_bar.empty()
            st.error(f"Fit failed: {exc}", icon="🚫")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

fits = st.session_state.get("fits")
fit_labels = st.session_state.get("fit_labels")

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
        "📈 Best fit", "🎲 MCMC posterior", "🖼️ Plots",
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
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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

        download_format = st.radio(
            "Download format", options=list(PLOT_EXPORT_FORMATS),
            horizontal=True, key="download_format",
            help="Applies to every '⬇️' button below. SVG/PDF are vector "
                 "(best for papers); PNG is raster (best for slides/sharing).",
        )

        if multi:

            compare_options = _available_compare_plots(fits[0])
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
                            other_fits=fits[1:], labels=fit_labels,
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
        ),
        file_name="cosmofit_result.json",
        mime="application/json",
    )

else:
    st.info(
        "👆 Configure one or more models above, then click **Run Fit**.",
        icon="🌌",
    )
