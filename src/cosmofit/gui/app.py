"""
CosmoFit -- graphical interface.

A thin Streamlit layer over the public ``cosmofit`` API: pick the
datasets, configure one or more models (built-in, your own ``E(z)``
expression, or derived from an action), choose which parameters are
free, run the fit, and read the plots and model-comparison statistics
-- no code required. Chains are saved to disk and reused, so adding a
model (or reopening the app) doesn't re-sample the fits that haven't
changed. No fitting or plotting logic lives in this package.

The page is a row of tabs in the order a fit is built -- Start, Data,
Model, Run, Results, Guide -- and only the open one is drawn
(:mod:`~gui.steps`). What they configure lives in session state and is
read back as one plan (:mod:`~gui.state`), so the Run tab knows the
datasets without the Data tab being on screen. Start it with::

    pip install "cosmofit[gui]"
    cosmofit gui

Local use only: the custom-model expression box is evaluated with
``eval()`` (builtins stripped, only whitelisted numpy math and the
model's own parameter names reach it -- see
``cosmofit.cosmology.custom._compile_expression``). That is a
reasonable trust boundary for a tool you run on your own machine,
not for a public, multi-tenant deployment.
"""

from __future__ import annotations

import json
import sys

# Before anything imports pyplot -- the library does, on import -- so
# figures render off-screen in the server process.
import matplotlib
matplotlib.use("Agg")

import streamlit as st


# ============================================================
# Page
# ============================================================
#
# The title goes out before the library is imported. Streamlit sends
# what the script draws as it draws it, and importing CosmoFit is
# the one slow thing on a cold start -- seconds of numpy, scipy and
# matplotlib -- so the page shows itself first and loads after.

st.set_page_config(page_title="CosmoFit", page_icon="🌌", layout="wide")

st.markdown(
    '<div class="cf-hero"><h1>🌌 CosmoFit</h1>'
    "<p>Cosmological parameter estimation, no code required.</p></div>",
    unsafe_allow_html=True,
)

if "cosmofit.gui.state" not in sys.modules:
    with st.spinner("Loading the library…"):
        import cosmofit.gui.state  # noqa: F401

from cosmofit import __version__  # noqa: E402

from cosmofit.gui import state  # noqa: E402
from cosmofit.gui.reference import STEPS  # noqa: E402
from cosmofit.gui.steps import data, guide, model, results, run, start  # noqa: E402
from cosmofit.gui.style import BADGES, CSS, badge  # noqa: E402


st.markdown(CSS, unsafe_allow_html=True)

state.seed_defaults()

# A finished run asks for the Results tab; the tabs' key can only be
# written before they are drawn, which is here.
_goto = st.session_state.pop("_goto", None)

if _goto:
    state.go(_goto)


# ------------------------------------------------------------
# Sidebar: saving the setup, and where it stands
# ------------------------------------------------------------

with st.sidebar:

    # Above the plan, so a restored configuration is what it is built
    # from on this same pass.
    with st.expander("💾 Save or restore this setup"):

        uploaded = st.file_uploader(
            "Restore a saved setup",
            type=["json"],
            key="config_upload",
            help="A file saved below, from this session or someone "
                 "else's. Datasets, models, parameters and sampler "
                 "settings come back; results do not.",
        )

        if uploaded is not None and not st.session_state.get(
            "_config_applied_" + uploaded.name, False
        ):
            try:
                applied = state._restore_configuration(json.loads(uploaded.getvalue()))

                # Once per file. Streamlit re-runs the whole script on
                # every interaction and the uploader keeps returning
                # the same file, so without this the configuration
                # would be re-applied on every click and no edit made
                # afterwards would ever stick.
                st.session_state["_config_applied_" + uploaded.name] = True

                state.seed_defaults()

                st.success(f"Restored {applied} settings.")

            except Exception as exc:
                st.error(f"Could not read that file: {exc}", icon="🚫")

    plan = state.current_plan()

    status = state.step_status(plan)

    st.markdown("### Your fit")

    for step in ("data", "model", "run", "results"):

        label_column, button_column = st.columns([3, 2], vertical_alignment="center")

        label_column.markdown(
            f"{STEPS[step]}<br>{badge(status[step])}", unsafe_allow_html=True,
        )
        button_column.button(
            "Open", key=f"side_{step}", width="stretch",
            on_click=state.go, args=(step,),
        )

    _next = next(
        (step for step in ("data", "model", "run", "results")
         if status[step] in ("todo", "block")),
        None,
    )

    if _next:
        st.caption(f"**Next:** {STEPS[_next]} -- {BADGES[status[_next]][0]}.")

    st.divider()

    st.download_button(
        "⬇️ Save this setup",
        data=json.dumps(state._configuration(), indent=2),
        file_name="cosmofit_configuration.json",
        mime="application/json",
        key="dl_config",
        width="stretch",
        help="Everything on this page that says *what to run* -- "
             "not what was run. Hand it to someone else and they "
             "get your setup, not your results.",
    )

    st.button(
        "📖 Guide", key="side_guide", width="stretch",
        on_click=state.go, args=("guide",),
    )

    st.caption(f"CosmoFit v{__version__}")


# ------------------------------------------------------------
# The tabs
# ------------------------------------------------------------
#
# Only the open tab is drawn: switching reruns the page, but the page
# is then one tab rather than six, and a tab with a corner plot on it
# no longer costs anything while another is open.

_PAGES = {
    "start": start, "data": data, "model": model,
    "run": run, "results": results, "guide": guide,
}

tabs = st.tabs(
    list(STEPS.values()), key="step",
    default=st.session_state["open_step"], on_change=state.remember_step,
)

for tab, page in zip(tabs, _PAGES.values()):
    with tab:
        if tab.open:
            page.render(plan)
