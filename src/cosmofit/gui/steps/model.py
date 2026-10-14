"""
The Model tab: one slot per model to compare -- built in, from an
``E(z)`` expression, or derived from an action -- each with the
parameter table that says what is fitted.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cosmofit.gui.reference import (
    ACTION_CHOICE,
    ACTION_PRESETS,
    BACKGROUND_DEGENERATE_MODELS,
    CUSTOM_CHOICE,
    HAVE_THEORY,
    MAX_MODELS,
    MODEL_EQUATIONS,
    MODEL_INFO,
    theory,
)
from cosmofit.gui.helpers import (
    _action_and_model,
    _action_widgets,
    _model_capabilities,
)
from cosmofit.gui.state import (
    MODEL_OPTIONS,
    add_model,
    apply_param_edits,
    editor_key,
    remove_model,
)
from cosmofit.gui.style import next_button, step_header


def render(plan) -> None:

    step_header("model")

    n_models = len(plan.slots)

    add_column, remove_column, note_column = st.columns([1, 1, 3])

    add_column.button(
        "➕ Add model to compare", width="stretch",
        disabled=n_models >= MAX_MODELS, on_click=add_model,
        help=f"Up to {MAX_MODELS}. Every model is fitted to the same "
             f"data, and the results compare them side by side.",
    )
    remove_column.button(
        "➖ Remove last model", width="stretch",
        disabled=n_models <= 1, on_click=remove_model,
    )
    note_column.caption(
        "One model fits on its own; two or more add AIC/BIC, a "
        "likelihood-ratio test where they are nested, and every figure "
        "with all of them overlaid."
    )

    if n_models == 1:
        _slot(plan.slots[0], plan)
    else:
        tabs = st.tabs([f"Model {slot.index + 1}" for slot in plan.slots])
        for tab, slot in zip(tabs, plan.slots):
            with tab:
                _slot(slot, plan)

    st.write("")
    next_button("run", "Next: check and run", key="next_from_model")


# ------------------------------------------------------------

def _slot(slot, plan) -> None:

    i = slot.index

    with st.container(border=True):

        st.markdown(f"**{slot.label}**" + (" (primary)" if i == 0 and len(plan.slots) > 1 else ""))

        st.selectbox(
            "Cosmology", options=MODEL_OPTIONS,
            key=f"model_choice_{i}", label_visibility="collapsed",
            persist_state="session",
            help="Grouped by what the model changes. The last two entries "
                 "build a model the library does not ship.",
        )

        choice = slot.choice

        if choice in MODEL_EQUATIONS:
            st.latex(MODEL_EQUATIONS[choice])

        info = MODEL_INFO.get(choice)

        if info:
            st.markdown(info["what"])
            cols = st.columns(2)
            cols[0].caption(f"**Extra parameters:** {info['params']}")
            cols[1].caption(
                f"**Reduces to:** {info['reduces']}" if info["reduces"]
                else "**Reduces to:** — (not a ΛCDM extension)"
            )
            st.caption(f"📄 {info['ref']}")

        if choice in BACKGROUND_DEGENERATE_MODELS:
            st.warning(BACKGROUND_DEGENERATE_MODELS[choice], icon="⚠️")

        if choice == CUSTOM_CHOICE:
            _custom_widgets(i)
        elif choice == ACTION_CHOICE:
            _action_widgets_ui(i)

        if slot.error:
            st.info(f"{slot.label}: not ready yet -- {slot.error}")
            return

        if choice == ACTION_CHOICE and HAVE_THEORY:
            _derived_equation(i)

        _capabilities(slot)

        _parameters(slot)


# ------------------------------------------------------------

def _capabilities(slot) -> None:
    """What this model can be asked to do."""

    caps = _model_capabilities(slot.model_cls)

    badge_cols = st.columns(3)
    badge_cols[0].caption(
        ("✅ has **w(z)**" if caps["w"] else "➖ no w(z)")
        + " — needed for the w(z) plot"
    )
    badge_cols[1].caption(
        ("✅ modifies **growth**" if caps["mu"] else "➖ standard GR growth")
        + " — μ(a,k)"
    )
    badge_cols[2].caption(
        "✅ **CMB spectra** computable" if caps["cmb"]
        else "🚫 no full CMB spectra"
    )


def _parameters(slot) -> None:

    i = slot.index

    labels = {
        p.name: p.label for p in slot.model_cls.PARAMS_CLASS.parameter_set()
    }

    # A model that *derives* a parameter rather than accepting it --
    # ADE fixes Omega_m from its early-time condition -- must not have
    # it ticked by default: the posterior would be the prior, and it
    # would look like a measurement.
    derived = set(getattr(slot.model_cls, "DERIVED_PARAMS", ()) or ())

    if derived:
        st.caption(
            "🔒 **Derived, not fitted:** "
            + ", ".join(f"`{name}`" for name in sorted(derived))
            + " — this model computes it from its own parameters, "
            "so sampling it would return the prior. Left un-ticked "
            "below."
        )

    with st.expander("Parameters", expanded=True):

        st.caption(
            "**Free** — fit it: sampled between **Lower** and **Upper** "
            "(a flat prior). Unticked, it is held at **Initial**. Free "
            "parameters also start the optimizer at Initial."
        )

        hide_irrelevant = st.checkbox(
            "Show only the parameters this fit uses",
            key=f"relevant_only_{i}",
            persist_state="session",
            help="The parameter container is shared by every "
                 "model, so it carries every parameter any of "
                 "them needs. Which ones actually do something "
                 "depends on the model *and* the datasets -- rd "
                 "means nothing without BAO, sigma8 nothing "
                 "without growth data. Untick to see them all.",
        )

        rows = slot.rows

        shown = [
            row for row in rows
            if not hide_irrelevant or row["Parameter"] in slot.relevant
        ]

        if hide_irrelevant and len(shown) < len(rows):
            st.caption(
                f"Showing {len(shown)} of {len(rows)} — "
                f"{len(rows) - len(shown)} hidden because "
                f"nothing in this fit depends on them."
            )

        frame = pd.DataFrame(
            [
                {
                    "Parameter": row["Parameter"],
                    "Label": labels.get(row["Parameter"], ""),
                    **{k: row[k] for k in ("Free", "Initial", "Lower", "Upper")},
                }
                for row in shown
            ],
            columns=["Parameter", "Label", "Free", "Initial", "Lower", "Upper"],
        )

        key = editor_key(i)

        st.data_editor(
            frame,
            hide_index=True,
            width="stretch",
            disabled=["Parameter", "Label"],
            column_config={
                "Free": st.column_config.CheckboxColumn(
                    "Fit this parameter?",
                    help="Ticked: sampled within its bounds. Unticked: "
                         "held at Initial.",
                ),
                "Initial": st.column_config.NumberColumn(
                    format="%.5g", help="Fixed value, or where the "
                                        "optimizer starts.",
                ),
                "Lower": st.column_config.NumberColumn(
                    format="%.5g", help="Prior lower bound.",
                ),
                "Upper": st.column_config.NumberColumn(
                    format="%.5g", help="Prior upper bound.",
                ),
            },
            key=key,
            on_change=apply_param_edits,
            args=(i, [row["Parameter"] for row in shown], key),
        )

    free = slot.free

    st.caption(
        f"**{len(free)} free parameter(s)**"
        + (f" — {', '.join(free)}" if free else " — tick at least one.")
    )

    for icon, message in slot.warnings:
        if icon == "🚫":
            st.error(message, icon=icon)
        elif icon == "⚠️":
            st.warning(message, icon=icon)
        else:
            st.caption(f"{icon} {message}")


# ------------------------------------------------------------

def _custom_widgets(i: int) -> None:

    col1, col2 = st.columns([1, 2])

    with col1:
        st.text_input(
            "Model name", key=f"custom_name_{i}", persist_state="session",
        )

    with col2:
        st.text_input(
            "E(z) expression", key=f"custom_E_{i}", persist_state="session",
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
                persist_state="session",
                help="For the w(z) plot only -- not needed to fit.",
            )

        with col4:
            st.text_input(
                "dE/dz expression (optional)", key=f"custom_dEdz_{i}",
                persist_state="session",
                help=(
                    "For the deceleration-parameter plot only. "
                    "If left blank, a numerical derivative of "
                    "E(z) is used automatically."
                ),
            )

        st.text_input(
            "mu(a,k) expression (optional)", key=f"custom_mu_{i}",
            persist_state="session",
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
            persist_state="session",
            placeholder="beta = 0.0, -2.0, 2.0, $\\beta$",
            help="name = default, lower, upper[, label]",
        )


def _load_action_preset(i: int) -> None:
    """Write the chosen worked example into slot ``i``'s boxes."""

    preset = ACTION_PRESETS.get(st.session_state.get(f"action_preset_{i}"))

    if not preset:
        return

    for field in ("gravity", "geometry", "params", "closure", "growth", "fields"):
        st.session_state[f"action_{field}_{i}"] = preset[field]


def _action_widgets_ui(i: int) -> None:

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
        return

    module = theory()

    st.caption(
        "Give the **action**, not the answer. `cosmofit.theory` "
        "writes FLRW with an explicit lapse, reduces the action "
        "to a point-like Lagrangian, varies the lapse to get the "
        "Friedmann *constraint*, and solves it for `E(z)`. What "
        "comes back is an ordinary model that every dataset and "
        "plot here already understands."
    )

    st.selectbox(
        "Start from a worked example",
        options=list(ACTION_PRESETS),
        key=f"action_preset_{i}",
        help="Each is a real model. Loading one overwrites the boxes "
             "below; edit them afterwards.",
    )

    st.button(
        "Load this action", key=f"action_load_{i}", width="stretch",
        disabled=not ACTION_PRESETS.get(st.session_state.get(f"action_preset_{i}")),
        on_click=_load_action_preset, args=(i,),
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.text_input(
            "Model name", key=f"action_name_{i}", persist_state="session",
        )

    with col2:
        st.selectbox(
            "Geometry",
            options=["auto", *module.GEOMETRIES],
            key=f"action_geometry_{i}",
            persist_state="session",
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
        persist_state="session",
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
        height=80, persist_state="session",
        placeholder="Lam = 2.1, 0.0, 6.0, $\\Lambda$",
        help="name = default, lower, upper[, label]",
    )

    col3, col4 = st.columns([1, 1])

    with col3:
        st.text_input(
            "Closure parameter (optional)",
            key=f"action_closure_{i}", persist_state="session",
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
            key=f"action_growth_{i}", persist_state="session",
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
        key=f"action_background_{i}", persist_state="session",
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
            key=f"action_fields_{i}", height=80, persist_state="session",
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
                options=list(module.STANDARD_FLUIDS),
                key=f"action_fluids_{i}", persist_state="session",
                help="What else is in the universe besides "
                     "whatever the action itself provides.",
            )

        with col6:
            st.number_input(
                "z_init (fields only)",
                min_value=10.0, max_value=100000.0, step=500.0,
                key=f"action_zinit_{i}", persist_state="session",
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


def _derived_equation(i: int) -> None:
    """
    The Friedmann equation the action turned into.

    Worth showing rather than hiding: for most of these actions it is
    the thing somebody would otherwise have spent an afternoon
    deriving, and seeing it is how you check the library understood
    the Lagrangian you meant.
    """

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
