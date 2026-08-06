"""
CosmoFit -- graphical interface.

A thin Streamlit layer over the public ``CosmoFit`` API
(``Fitter``, ``FitPlotter``, ``define_model`` / ``model_from_expression``):
tick which datasets to fit, pick a built-in model or write your own
``E(z)`` as a plain expression, choose which parameters are free,
and run an MCMC fit + look at the resulting plots -- no code
required. Everything here is a consumer of ``CosmoFit``'s existing
public API; no fitting/plotting logic lives in this file.

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

import json

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
from CosmoFit.stats import DATASET_REGISTRY


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

#: LaTeX preview shown next to the model picker -- background
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

def _build_model_class(model_choice: str):
    """
    Resolve the current sidebar selection into a ``Cosmology``
    subclass, raising a clear error for an incomplete/invalid
    custom-model definition.
    """

    if model_choice != "Custom":
        return BUILTIN_MODELS[model_choice]

    name = st.session_state.get("custom_name", "").strip() or "CustomModel"
    E_expr = st.session_state.get("custom_E", "").strip()

    if not E_expr:
        raise ValueError("Enter an E(z) expression for the custom model.")

    w_expr = st.session_state.get("custom_w", "").strip() or None
    dEdz_expr = st.session_state.get("custom_dEdz", "").strip() or None
    extra_text = st.session_state.get("custom_extra_params", "")

    extra_params = _parse_extra_params(extra_text)

    return model_from_expression(
        name,
        E=E_expr,
        extra_params=extra_params,
        w=w_expr,
        dEdz=dEdz_expr,
    )


# ------------------------------------------------------------

def _available_plots(fit: Fitter) -> list[str]:
    """
    Which ``fit.plots.<name>()`` methods make sense for this fit's
    model/dataset combination.
    """

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
# Sidebar: datasets, model, MCMC settings
# ------------------------------------------------------------

with st.sidebar:

    st.markdown("### 📊 Datasets")

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

    st.markdown("### 🧮 Model")

    with st.container(border=True):

        model_choice = st.selectbox(
            "Cosmology", options=[*BUILTIN_MODELS.keys(), "Custom"],
            label_visibility="collapsed",
        )

        if model_choice in MODEL_EQUATIONS:
            st.latex(MODEL_EQUATIONS[model_choice])

        if model_choice == "Custom":

            st.text_input("Model name", value="MyModel", key="custom_name")
            st.text_area(
                "E(z) expression", key="custom_E", height=80,
                placeholder="sqrt(Omega_m*(1+z)**3 + (1-Omega_m)*(1+z)**(3*(1+w0))*(1+beta*z))",
                help=(
                    "Available: z, every standard parameter "
                    "(H0, Omega_m, Omega_k, w0, wa, rd, MB, Omega_b, A_s, "
                    "alpha), any extra parameters defined below, and "
                    "sqrt/exp/log/log10/sin/cos/tan/sinh/cosh/tanh/abs/"
                    "sign/where/minimum/maximum/pi/e."
                ),
            )

            with st.expander("Advanced (w(z), dE/dz)"):
                st.text_area(
                    "w(z) expression (optional)", key="custom_w", height=60,
                    help="For the w(z) plot only -- not needed to fit.",
                )
                st.text_area(
                    "dE/dz expression (optional)", key="custom_dEdz", height=60,
                    help=(
                        "For the deceleration-parameter plot only. If left "
                        "blank, a numerical (finite-difference) derivative "
                        "of E(z) is used automatically."
                    ),
                )

            st.text_area(
                "Extra parameters (one per line, optional)",
                key="custom_extra_params", height=80,
                placeholder="beta = 0.0, -2.0, 2.0, $\\beta$",
                help="name = default, lower, upper[, label]",
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

# ------------------------------------------------------------
# Main panel: parameters
# ------------------------------------------------------------

try:
    model_cls = _build_model_class(model_choice)
except Exception as exc:
    st.info(f"Custom model not ready yet: {exc}")
    st.stop()

params_cls = getattr(model_cls, "PARAMS_CLASS", None)
parameter_set = params_cls.parameter_set()
defaults = params_cls.defaults()

st.markdown("## 🎛️ Parameters")
st.caption("Tick which parameters to fit; edit initial values or bounds directly in the table.")

param_rows = []
for p in parameter_set:
    lo, hi = p.bounds if p.bounds else (None, None)
    # H0/Omega_m (and any custom parameter without an explicit
    # "default") have no dataclass default -- falling back to 0.0
    # would sit outside their prior bounds and produce a degenerate
    # walker cloud (every walker clipped to the same edge). The
    # bounds midpoint is always inside the prior instead.
    if p.name in defaults:
        initial = float(defaults[p.name])
    elif lo is not None and hi is not None:
        initial = float((lo + hi) / 2.0)
    else:
        initial = 0.0
    param_rows.append({
        "Parameter": p.name,
        "Label": p.label,
        "Free": p.name in ("H0", "Omega_m"),
        "Initial": initial,
        "Lower": lo,
        "Upper": hi,
    })

param_df = pd.DataFrame(param_rows)

with st.container(border=True):
    edited_df = st.data_editor(
        param_df,
        hide_index=True,
        use_container_width=True,
        disabled=["Parameter", "Label"],
        column_config={
            "Free": st.column_config.CheckboxColumn("Fit this parameter?"),
            "Initial": st.column_config.NumberColumn(format="%.5g"),
            "Lower": st.column_config.NumberColumn(format="%.5g"),
            "Upper": st.column_config.NumberColumn(format="%.5g"),
        },
        key="param_editor",
    )

free_params = edited_df.loc[edited_df["Free"], "Parameter"].tolist()
initial = dict(zip(edited_df["Parameter"], edited_df["Initial"]))
bounds = {
    row["Parameter"]: (row["Lower"], row["Upper"])
    for _, row in edited_df.iterrows()
    if pd.notna(row["Lower"]) and pd.notna(row["Upper"])
}

n_free = len(free_params)
st.caption(
    f"{n_free} free parameter(s) selected"
    + (f" -- {', '.join(free_params)}" if n_free else "")
)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

run_clicked = st.button("🚀 Run Fit", type="primary", use_container_width=True)

if run_clicked:

    if not selected_datasets:
        st.error("Select at least one dataset.", icon="🚫")
    elif not free_params:
        st.error("Tick at least one parameter to fit.", icon="🚫")
    else:
        progress_bar = None
        try:
            fit = Fitter(
                model=model_cls,
                datasets=selected_datasets,
                free_params=free_params,
                initial=initial,
                bounds=bounds,
            )

            progress_bar = st.progress(0.0, text="Starting MCMC...")
            # Re-rendering the bar on every single step (up to
            # thousands of steps) would flood the browser with
            # updates for no visible benefit -- only push at whole
            # percentage points (or the final step).
            last_shown = {"pct": -1}

            def _on_step(step, total, elapsed, _bar=progress_bar, _last=last_shown):
                pct = int(100 * step / total)
                if pct == _last["pct"] and step != total:
                    return
                _last["pct"] = pct
                rate = step / elapsed if elapsed > 0 else 0.0
                _bar.progress(
                    step / total,
                    text=f"Running MCMC -- {pct}% ({step}/{total} steps, {rate:.1f} it/s)",
                )

            fit.run_mcmc(
                nwalkers=int(nwalkers), nsteps=int(nsteps),
                burnin=int(burnin), seed=int(seed), progress=False,
                callback=_on_step,
            )
            progress_bar.empty()

            with st.spinner("Finding best fit..."):
                fit.best_fit()

            st.session_state["fit"] = fit
            st.toast("Fit complete.", icon="✅")

        except Exception as exc:
            st.session_state.pop("fit", None)
            if progress_bar is not None:
                progress_bar.empty()
            st.error(f"Fit failed: {exc}", icon="🚫")

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

fit = st.session_state.get("fit")

st.divider()

if fit is not None:

    st.markdown("## 📊 Results")
    st.caption(
        f"**{fit.model_cls.__name__}** · "
        f"{', '.join(DATASET_LABELS.get(d, d) for d in fit.dataset_names)} · "
        f"free: {', '.join(fit.free_params)}"
    )

    result = fit.result

    tab_best, tab_mcmc, tab_plots = st.tabs(
        ["📈 Best fit", "🎲 MCMC posterior", "🖼️ Plots"]
    )

    with tab_best:

        if result.best_fit is not None:

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
        else:
            st.caption("No best-fit result.")

    with tab_mcmc:

        if result.mcmc is not None:

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
        else:
            st.caption("No MCMC run.")

    with tab_plots:

        plot_options = _available_plots(fit)
        chosen = st.multiselect(
            "Choose plots to render",
            options=plot_options,
            default=plot_options[:2],
            format_func=lambda name: PLOT_LABELS.get(name, name),
            label_visibility="collapsed",
        )

        plot_cols = st.columns(2)
        for i, name in enumerate(chosen):
            with plot_cols[i % 2]:
                try:
                    fig = getattr(fit.plots, name)()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as exc:
                    st.error(
                        f"Could not render '{PLOT_LABELS.get(name, name)}': {exc}",
                        icon="🚫",
                    )

    st.download_button(
        "⬇️ Download result (JSON)",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="cosmofit_result.json",
        mime="application/json",
    )

else:
    st.info(
        "👈 Configure datasets/model/parameters on the left, then click **Run Fit**.",
        icon="🌌",
    )
