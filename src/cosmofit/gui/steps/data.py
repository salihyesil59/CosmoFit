"""
The Data tab: presets, the datasets by probe family, and the two
choices about the early universe that change what they measure.
"""

from __future__ import annotations

import functools

import streamlit as st

from cosmofit import dataset_reference
from cosmofit.stats import DATASET_REGISTRY

from cosmofit.gui.reference import (
    DATASET_GROUPS,
    DATASET_INFO,
    DATASET_LABELS,
    DATASET_PRESETS,
    SLOW_DATASETS,
)
from cosmofit.gui.state import apply_preset, versions
from cosmofit.gui.style import next_button, step_header


#: The datasets a from-scratch CMB computation comes with -- the only
#: ones σ₈ can be derived from.
_CAMB_CMB = {"planck_lite", "planck_lensing", "planck_lowe", "act_lensing"}


@functools.lru_cache(maxsize=None)
def _reference(key: str, version: str | None) -> str:

    return dataset_reference(key, version)


def _details(key: str, version: str | None) -> str:
    """A dataset's tooltip: what it is, what it pins down, its source."""

    info = DATASET_INFO.get(key)

    if not info:
        return _reference(key, version)

    return (
        f"{info['what']}\n\n"
        f"🎯 **Constrains:** {info['constrains']}\n\n"
        f"📄 {_reference(key, version)}"
    )


def render(plan) -> None:

    step_header("data")

    _presets(plan)

    st.markdown("### Datasets")

    left, right = st.columns([5, 2], gap="large")

    with left:
        st.caption(
            "Grouped by what they measure. A fit is usually built from "
            "**one** dataset per family -- hover ⓘ for what each is and "
            "what it pins down."
        )
        _families()

    with right, st.container(border=True):
        _summary(plan)

    st.markdown("### Physics options")
    st.caption(
        "Two assumptions about the early universe. Both are off by "
        "default, which assumes the least."
    )

    rd_column, sigma8_column = st.columns(2)

    with rd_column, st.container(border=True, height="stretch"):
        _sound_horizon(plan)

    with sigma8_column, st.container(border=True, height="stretch"):
        _amplitude(plan)

    st.write("")
    next_button("model", "Next: choose the model", key="next_from_data")


# ------------------------------------------------------------

def _presets(plan) -> None:
    """
    Ready-made combinations, as cards. Fourteen checkboxes with
    seventeen conflict rules between them is a lot to face cold; a
    preset writes the datasets, versions and r_d choice and the
    checkboxes below pick them up, so it is a starting point that can
    then be edited, not a mode.
    """

    st.markdown("### Start from a preset")

    selected = set(plan.datasets)

    names = list(DATASET_PRESETS)

    for row in range(0, len(names), 3):

        columns = st.columns(3)

        for column, name in zip(columns, names[row:row + 3]):

            preset = DATASET_PRESETS[name]
            in_use = set(preset["datasets"]) == selected

            with column, st.container(border=True, height="stretch"):

                st.markdown(f"**{name}**" + ("  ✅" if in_use else ""))
                st.caption(preset["note"])

                st.button(
                    "In use" if in_use else "Use this preset",
                    key=f"preset_{names.index(name)}",
                    width="stretch",
                    disabled=in_use,
                    on_click=apply_preset, args=(name,),
                )


def _families() -> None:

    for group_label, keys in DATASET_GROUPS:

        keys = [key for key in keys if key in DATASET_REGISTRY]

        ticked_here = sum(bool(st.session_state.get(f"ds_{key}")) for key in keys)

        title = group_label + (f"  ·  {ticked_here} selected" if ticked_here else "")

        with st.expander(title, expanded=bool(ticked_here)):

            for key in keys:

                info = DATASET_INFO.get(key, {})
                version = st.session_state.get(f"dsver_{key}")

                ticked = st.checkbox(
                    DATASET_LABELS.get(key, key),
                    key=f"ds_{key}",
                    help=_details(key, version),
                    persist_state="session",
                )

                if info:
                    st.caption(
                        f"{info['observable']} · {info['n']} · z = {info['z']}"
                    )

                if len(versions(key)) > 1:
                    st.selectbox(
                        "Version", options=versions(key),
                        key=f"dsver_{key}",
                        label_visibility="collapsed",
                        disabled=not ticked,
                        persist_state="session",
                    )


def _summary(plan) -> None:
    """What is ticked, and what is wrong with it -- next to the ticks."""

    st.markdown("**Your selection**")

    if not plan.datasets:
        st.caption("Nothing yet. Pick a preset above or tick a dataset.")
        return

    for key in plan.datasets:
        version = plan.versions.get(key)
        st.markdown(
            f"- {DATASET_LABELS.get(key, key)}"
            + (f" · `{version}`" if version else "")
        )

    for pair, reason in plan.conflicts:
        st.warning(
            f"**{' + '.join(DATASET_LABELS.get(k, k) for k in sorted(pair))}**: "
            f"{reason}",
            icon="⚠️",
        )

    for key, note in SLOW_DATASETS.items():
        if key in plan.datasets:
            st.warning(f"**{DATASET_LABELS.get(key, key)}** -- {note}", icon="🐢")

    if not plan.conflicts:
        st.caption("✅ No two of these share data.")


def _sound_horizon(plan) -> None:

    st.markdown("**🌀 Sound horizon $r_d$**")

    compute_rd = st.checkbox(
        "Compute $r_d$ instead of fitting it",
        key="compute_rd",
        help="Validated against CAMB's rdrag to 5e-5.",
        persist_state="session",
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


def _amplitude(plan) -> None:

    st.markdown("**🌱 Amplitude $\\sigma_8$**")

    cmb = bool(set(plan.datasets) & _CAMB_CMB)

    derive_sigma8 = st.checkbox(
        "Derive $\\sigma_8$ from the CMB instead of fitting it",
        key="derive_sigma8",
        disabled=not cmb,
        help="Only available with a from-scratch CMB dataset, "
             "since there is nothing else to derive it from.",
        persist_state="session",
    )

    if not cmb:
        st.caption(
            "σ₈ is a free parameter. Tick a from-scratch CMB "
            "dataset to make deriving it possible."
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
