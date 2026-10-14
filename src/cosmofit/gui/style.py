"""
The page's look, and the few pieces every tab draws the same way: a
step's header, a status badge, the button to the next step.
"""

from __future__ import annotations

import streamlit as st

from cosmofit.gui.reference import STEPS, STEP_INTROS
from cosmofit.gui.state import go


CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1240px; }
[data-testid="stMetricValue"] { font-size: 1.5rem; }

/* The workflow tabs: larger, so they read as the page's navigation
   rather than one more control on it. */
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }

.cf-hero h1 { margin-bottom: 0.1rem; }
.cf-hero p { font-size: 1.08rem; opacity: 0.85; margin-top: 0; }

.cf-badge {
    display: inline-block;
    padding: 0.05rem 0.55rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.cf-ok    { background: rgba(46, 160, 67, 0.15);  color: #2ea043; }
.cf-warn  { background: rgba(210, 153, 34, 0.18); color: #b7811d; }
.cf-block { background: rgba(209, 73, 91, 0.15);  color: #d1495b; }
.cf-todo  { background: rgba(128, 128, 128, 0.15); color: #808080; }
</style>
"""


#: Badge text and class per status (see `state.step_status`).
BADGES = {
    "ok": ("✓ ready", "cf-ok"),
    "warn": ("! check", "cf-warn"),
    "block": ("✗ needs attention", "cf-block"),
    "todo": ("not yet", "cf-todo"),
}


#: The same, for a checklist line.
CHECK_ICONS = {"ok": "✅", "warn": "⚠️", "block": "🚫"}


def badge(status: str) -> str:
    """A status badge, as HTML for ``st.markdown``."""

    text, css = BADGES[status]

    return f'<span class="cf-badge {css}">{text}</span>'


def step_header(step: str) -> None:
    """A step's title, what to do on it, and why."""

    what, why = STEP_INTROS[step]

    st.markdown(f"## {STEPS[step]}")
    st.info(f"**{what}**\n\n{why}", icon="🧭")


def next_button(step: str, text: str, key: str) -> None:
    """A full-width button to the next step."""

    st.button(
        f"{text}  →", key=key, type="primary", width="stretch",
        on_click=go, args=(step,),
    )
