"""
The Guide tab: everything the other tabs say, in one browsable place.

The notes beside each control answer "what is this one?" while it is
being looked at. This answers "what is there, and which should I
pick?" -- a different question, and one to answer before ticking
anything.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cosmofit import dataset_reference
from cosmofit.stats import DATASET_REGISTRY

from cosmofit.gui.reference import (
    BUILTIN_MODELS,
    DATASET_GROUPS,
    DATASET_INFO,
    DATASET_LABELS,
    GLOSSARY,
    INCOMPATIBLE_PAIRS,
    MODEL_GROUPS,
    MODEL_INFO,
    TROUBLESHOOTING,
    WORKFLOW,
)
from cosmofit.gui.helpers import _model_capabilities


@st.cache_data(show_spinner=False)
def _dataset_table() -> pd.DataFrame:
    """Every dataset, one row each -- built once per process."""

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

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _model_table() -> pd.DataFrame:
    """Every built-in model, one row each -- built once per process."""

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

    return pd.DataFrame(rows)


def render(plan) -> None:

    st.markdown("## 📖 Guide")
    st.caption("What every dataset and model is, the words the page uses, and what to do when something goes wrong.")

    workflow, datasets, models, glossary, trouble = st.tabs([
        "How to use this", "Datasets", "Models", "Glossary", "Troubleshooting",
    ])

    with workflow:
        st.markdown(WORKFLOW)

    with datasets:

        st.caption(
            "A fit is usually built by taking **one** entry from each "
            "family, not by ticking everything -- several pairs "
            "measure the same sky or the same supernovae and must not "
            "be combined."
        )

        st.dataframe(_dataset_table(), hide_index=True, width="stretch")

        st.markdown("**Do not combine**")
        for pair, reason in INCOMPATIBLE_PAIRS:
            st.caption(
                f"❌ **{' + '.join(DATASET_LABELS.get(k, k) for k in sorted(pair))}** "
                f"— {reason}"
            )

    with models:

        st.caption(
            "What a model *changes* decides what can be done with it. "
            "Only models with a w(z) can be handed to a Boltzmann code "
            "for the full CMB spectra; only models that modify gravity "
            "predict a growth history differing from GR's at the same "
            "expansion history."
        )

        st.dataframe(_model_table(), hide_index=True, width="stretch")

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

    with glossary:

        columns = st.columns(2)

        for n, (term, meaning) in enumerate(GLOSSARY.items()):
            columns[n % 2].markdown(f"**{term}** — {meaning}")

    with trouble:

        for symptom, fix in TROUBLESHOOTING:
            with st.expander(symptom):
                st.markdown(fix)
