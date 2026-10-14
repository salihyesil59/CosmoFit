"""
The Results tab: model comparison, the best fit, the posterior, the
inference a posterior alone does not answer, the figures, and the fit
taken away as Python, YAML, samples or JSON.

The panels whose controls only change *what is looked at* -- which
model, which figure, a profile's range -- are fragments: touching them
reruns that panel, not the page.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cosmofit.gui.reference import (
    COMPARE_PLOT_LABELS,
    DATASET_LABELS,
    PLOT_EXPORT_FORMATS,
    PLOT_LABELS,
)
from cosmofit.gui.state import go
from cosmofit.gui.style import step_header


def render(plan) -> None:

    step_header("results")

    fits = st.session_state.get("fits")

    if not fits:
        st.info(
            "No fit yet. Set one up -- a Quick start on the Start tab is "
            "the shortest way -- and press **Run Fit** on the Run tab.",
            icon="🌌",
        )
        st.button("🚀 Go to Run", on_click=go, args=("run",), type="primary")
        return

    from cosmofit.gui import render as draw

    labels = st.session_state["fit_labels"]
    plot_labels = st.session_state["fit_plot_labels"]

    flash = st.session_state.pop("_flash", None)

    if flash:
        st.toast(flash, icon="✅")
        st.success(flash, icon="✅")

    for kind, text in st.session_state.get("fit_messages", ()):
        if kind == "warning":
            st.warning(text, icon="⚠️")
        else:
            st.info(text, icon="ℹ️")

    st.caption(
        f"{', '.join(DATASET_LABELS.get(d, d) for d in fits[0].dataset_names)}  ·  "
        + "  ·  ".join(
            f"**{label}** ({', '.join(f.free_params)})"
            for label, f in zip(labels, fits)
        )
    )

    multi = len(fits) >= 2

    names = (["⚖️ Comparison"] if multi else []) + [
        "📈 Best fit", "🎲 MCMC posterior", "🔬 Inference", "🖼️ Plots",
    ]

    tabs = iter(st.tabs(names))

    if multi:
        with next(tabs):
            st.caption(
                "Lower AIC/BIC is better; a difference below ~2 decides "
                "nothing, above ~6 is strong. The likelihood-ratio test "
                "applies only when one model is the other with "
                "parameters fixed."
            )
            _comparison(fits, labels)

    with next(tabs):
        st.caption(
            "The single best point, and how much of its χ² each dataset "
            "carries -- a dataset far above the others is where a "
            "tension lives."
        )
        _per_model(fits, labels, "bestfit_model_choice", draw._render_best_fit)

    with next(tabs):
        st.caption(
            "Medians and 68% intervals of the posterior. Check the "
            "convergence verdict before quoting any of them."
        )
        _per_model(fits, labels, "posterior_model_choice", draw._render_posterior)

    with next(tabs):
        _inference(fits, labels)

    with next(tabs):
        _plots(fits, labels, plot_labels)

    st.markdown("### Take it with you")

    _take_away(fits, labels)


# ------------------------------------------------------------

def _comparison(fits, labels) -> None:

    from cosmofit.gui import render as draw
    from cosmofit.stats import model_comparison

    rows = []

    for label, f in zip(labels, fits):
        bf = draw._result(f).best_fit
        rows.append({
            "model": label,
            "chi2": bf.chi2,
            "k (free params)": bf.ndim,
            "AIC": bf.aic(),
            "BIC": bf.bic(),
            "converged": (
                draw._result(f).mcmc.convergence["converged"]
                if draw._result(f).mcmc else None
            ),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Likelihood-ratio test: only well-defined for exactly two models
    # where one's free parameters are a strict subset of the other's
    # (a genuine nested special case) -- it can't be inferred for 3+
    # models or non-nested pairs.
    if len(fits) != 2:
        return

    free_sets = [set(f.free_params) for f in fits]

    if free_sets[0] < free_sets[1]:
        null_i, alt_i = 0, 1
    elif free_sets[1] < free_sets[0]:
        null_i, alt_i = 1, 0
    else:
        st.caption(
            "No likelihood-ratio test shown: neither model's "
            "free parameters are a strict subset of the "
            "other's, so they aren't nested."
        )
        return

    comparison = model_comparison.compare_models(
        name_null=labels[null_i],
        chi2_null=fits[null_i].best_fit_chi2,
        k_null=fits[null_i].ndim,
        name_alt=labels[alt_i],
        chi2_alt=fits[alt_i].best_fit_chi2,
        k_alt=fits[alt_i].ndim,
        n_data=fits[alt_i].n_data,
    )
    lrt = comparison["likelihood_ratio_test"]

    st.markdown(
        f"**Likelihood-ratio test** ({labels[null_i]} vs. "
        f"{labels[alt_i]}, nested at "
        f"{', '.join(free_sets[alt_i] - free_sets[null_i])}=fixed): "
        f"Δχ² = {lrt['delta_chi2']:.2f} (Δk={lrt['delta_k']}), "
        f"p = {lrt['p_value']:.3f}, "
        f"**{lrt['sigma']:.2f}σ** preference for "
        f"{labels[alt_i]}."
    )


@st.fragment
def _per_model(fits, labels, key, draw_one) -> None:
    """One model's panel, with a picker when there are several."""

    if len(fits) >= 2:
        choice = st.selectbox("Model", options=labels, key=key)
        draw_one(fits[labels.index(choice)])
    else:
        draw_one(fits[0])


def _inference(fits, labels) -> None:

    from cosmofit.gui import render as draw

    st.caption(
        "Four questions an MCMC posterior on its own does not "
        "answer. Each is a separate calculation behind its own "
        "button, because two of them cost real time."
    )

    profile_tab, fisher_tab, evidence_tab, tension_tab = st.tabs([
        "Profile likelihood", "Fisher matrix",
        "Bayesian evidence", "Tension",
    ])

    with profile_tab:
        st.caption(
            "χ² minimised over every other parameter at each value of "
            "one. Honest where the posterior runs into a prior edge."
        )
        _chosen(fits, labels, "profile_model_choice", draw._render_profile)

    with fisher_tab:
        st.caption(
            "The Gaussian approximation to the posterior, from the "
            "curvature of χ² at the best fit -- and how far it is from "
            "the chain's errors."
        )
        _chosen(fits, labels, "fisher_model_choice", draw._render_fisher)

    with evidence_tab:
        _fragment(draw._render_evidence, fits, labels)

    with tension_tab:
        st.caption(
            "How many σ apart two models put the same parameter -- by "
            "a Gaussian formula and by the samples themselves."
        )
        _fragment(draw._render_tension, fits, labels)


@st.fragment
def _chosen(fits, labels, key, draw_one) -> None:
    """`_per_model`, for a panel that also wants the model's label."""

    if len(fits) >= 2:
        choice = st.selectbox("Model", options=labels, key=key)
    else:
        choice = labels[0]

    draw_one(fits[labels.index(choice)], choice)


@st.fragment
def _fragment(draw_all, fits, labels) -> None:

    draw_all(fits, labels)


@st.fragment
def _plots(fits, labels, plot_labels) -> None:

    from cosmofit.gui import render as draw

    st.caption(
        "Pick figures to draw; each is drawn once and kept, so "
        "switching back and forth is instant. ⬇️ saves it in the "
        "chosen format."
    )

    download_format = st.radio(
        "Download format", options=list(PLOT_EXPORT_FORMATS),
        horizontal=True, key="download_format",
        help="Applies to every '⬇️' button below. SVG/PDF are vector "
             "(best for papers); PNG is raster (best for slides/sharing).",
    )

    if len(fits) >= 2:

        compare_options = draw._available_compare_plots(fits)

        chosen = st.multiselect(
            "Choose comparison plots (all models overlaid)",
            options=compare_options,
            default=compare_options[:2],
            format_func=lambda name: COMPARE_PLOT_LABELS.get(name, name),
        )

        _grid(
            chosen,
            lambda name: (lambda: getattr(fits[0].plots, name)(
                other_fits=fits[1:], labels=plot_labels,
            )),
            COMPARE_PLOT_LABELS, download_format, prefix="compare",
        )

        with st.expander("Single-model plots"):

            choice = st.selectbox("Model", options=labels, key="plots_model_choice")
            single = fits[labels.index(choice)]

            single_chosen = st.multiselect(
                "Choose plots", options=draw._available_plots(single),
                format_func=lambda name: PLOT_LABELS.get(name, name),
                key="single_plots_multiselect",
            )

            _grid(
                single_chosen,
                lambda name: (lambda: getattr(single.plots, name)()),
                PLOT_LABELS, download_format, prefix=f"{choice}",
            )

    else:

        plot_options = draw._available_plots(fits[0])

        chosen = st.multiselect(
            "Choose plots to render",
            options=plot_options,
            default=plot_options[:2],
            format_func=lambda name: PLOT_LABELS.get(name, name),
        )

        _grid(
            chosen,
            lambda name: (lambda: getattr(fits[0].plots, name)()),
            PLOT_LABELS, download_format, prefix="single",
        )


def _grid(names, maker, titles, download_format, prefix) -> None:
    """Figures two to a row, each drawn once (`_render_cached_figure`)."""

    from cosmofit.gui import render as draw

    columns = st.columns(2)

    for index, name in enumerate(names):
        with columns[index % 2]:
            try:
                draw._render_cached_figure(
                    maker(name), f"{prefix}:{name}",
                    f"{prefix}_{name}".replace(":", "_"),
                    download_format, key=f"dl_{prefix}_{name}",
                )
            except Exception as exc:
                st.error(
                    f"Could not render '{titles.get(name, name)}': {exc}",
                    icon="🚫",
                )


# ------------------------------------------------------------

def _take_away(fits, labels) -> None:

    from cosmofit.gui import render as draw
    from cosmofit.stats.results import _json_default

    last = st.session_state.get("last_run") or {}

    with st.expander("🐍 The same fit in Python"):

        st.caption(
            "This app is a way into the CosmoFit library, not a "
            "replacement for it. Copy this to move the same fit into "
            "a notebook, a batch job, or version control -- it is "
            "built from the values this run used, so it cannot drift "
            "away from what you just did."
        )

        try:
            snippet = draw._equivalent_script(**last)

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

        try:
            yaml_input = draw._equivalent_input(
                fits[0], last["sampler"], dict(last["sampler_options"]),
                int(last["seed"]),
            )
        except Exception as exc:
            yaml_input = None
            st.caption(f"Could not write the YAML input: {exc}")

        if yaml_input is not None:

            st.markdown(
                "Or as an input for the command line -- "
                "`cosmofit run cosmofit_run.yaml` (model 1):"
            )

            st.code(yaml_input, language="yaml")

            st.download_button(
                "⬇️ Download as .yaml",
                data=yaml_input,
                file_name="cosmofit_run.yaml",
                mime="text/yaml",
                key="dl_yaml",
            )

    with st.expander("🔗 The posterior samples"):

        st.caption(
            "The chain is the expensive thing this page produced -- "
            "the summary above is a handful of numbers derived from "
            "it. Take the samples away to make your own contours, "
            "re-marginalise, or combine them with something else."
        )

        for label, fit in zip(labels, fits):

            try:
                samples = fit.flat_samples()

            except Exception as exc:
                st.caption(f"{label}: no chain to export ({exc})")
                continue

            # The CSV is written when the button is pressed, not on
            # every rerun of the page.
            def _csv(samples=samples, columns=tuple(fit.free_params)):
                return pd.DataFrame(samples, columns=list(columns)).to_csv(index=False)

            st.download_button(
                f"⬇️ {label} — {len(samples):,} samples (CSV)",
                data=_csv,
                file_name=f"cosmofit_chain_{label.replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"dl_chain_{label}",
                help="Post-burn-in, flattened across walkers. One row "
                     "per sample, one column per free parameter.",
            )

    # Read here: the callable below runs on another thread, where
    # session state is out of reach.
    results = {label: draw._result(f) for label, f in zip(labels, fits)}

    st.download_button(
        "⬇️ Download result(s) (JSON)",
        data=lambda: json.dumps(
            {label: result.to_dict() for label, result in results.items()},
            indent=2,
            # Same coercion `FitResult.save_json` uses -- a chain read
            # back from HDF5 reports numpy counters, which plain
            # `json.dumps` refuses.
            default=_json_default,
        ),
        file_name="cosmofit_result.json",
        mime="application/json",
    )
