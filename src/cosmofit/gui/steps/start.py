"""
The first tab: what the page does, where a fit stands, and three
analyses that run with one click.
"""

from __future__ import annotations

import streamlit as st

from cosmofit.gui.reference import QUICK_STARTS, STEPS
from cosmofit.gui.state import apply_quick_start, go, step_status
from cosmofit.gui.style import badge


#: The four steps of a fit, as the Start tab lays them out.
_STEP_CARDS = [
    ("data", "Choose the datasets the models are fitted to -- or start "
             "from a preset."),
    ("model", "Pick one cosmology, or several to compare, and tick which "
              "parameters to fit."),
    ("run", "Clear the checklist, choose how long to sample, and run."),
    ("results", "Best fit, posterior, model comparison, figures, and the "
                "same fit as Python."),
]


def render(plan) -> None:

    st.markdown(
        "Fit cosmological models to real data -- supernovae, BAO, the "
        "CMB, cosmic chronometers, growth of structure -- and compare "
        "them, **without writing code**. Every number on this page comes "
        "from the CosmoFit library, and every fit can be taken away as a "
        "Python script or a `cosmofit run` input."
    )

    # --------------------------------------------------------
    # Quick start
    # --------------------------------------------------------

    st.markdown("### ⚡ Quick start")
    st.caption(
        "New here? Pick one. It fills in every tab and takes you to "
        "**Run** -- press the button there and you have a result. Change "
        "anything afterwards; nothing is locked."
    )

    columns = st.columns(len(QUICK_STARTS))

    for column, (key, quick) in zip(columns, QUICK_STARTS.items()):

        with column, st.container(border=True, height="stretch"):

            st.markdown(f"**{quick['title']}**")
            st.caption(quick["subtitle"])
            st.write(quick["what"])

            st.button(
                "Set it up", key=f"quick_{key}", width="stretch",
                type="primary" if key == "first_fit" else "secondary",
                on_click=apply_quick_start, args=(key,),
            )

    # --------------------------------------------------------
    # The steps, and where this fit stands on each
    # --------------------------------------------------------

    st.markdown("### 🧭 Or build your own, step by step")
    st.caption(
        "The tabs above, left to right. The badge says where each one "
        "stands for the current setup."
    )

    status = step_status(plan)

    columns = st.columns(len(_STEP_CARDS))

    for number, (column, (step, text)) in enumerate(zip(columns, _STEP_CARDS), start=1):

        with column, st.container(border=True, height="stretch"):

            st.markdown(
                f"**{number}. {STEPS[step]}** &nbsp; {badge(status[step])}",
                unsafe_allow_html=True,
            )
            st.caption(text)

            st.button(
                "Open", key=f"open_{step}", width="stretch",
                on_click=go, args=(step,),
            )

    # --------------------------------------------------------
    # Three things worth knowing before the first fit
    # --------------------------------------------------------

    st.markdown("### 💡 Worth knowing")

    tips = st.columns(3)

    tips[0].markdown(
        "**Every ⓘ explains.** Hover over the small question marks next "
        "to any control for what it does and when to change it."
    )
    tips[1].markdown(
        "**Nothing is sampled twice.** Chains are saved to disk and read "
        "back: add a model, and the fits already done return instantly."
    )
    tips[2].markdown(
        "**Your setup travels.** Save it from the sidebar as a file, "
        "and restore it here or hand it to someone else."
    )

    st.button(
        "📖 Read the guide: datasets, models, glossary, troubleshooting",
        key="open_guide", on_click=go, args=("guide",), type="tertiary",
    )
