"""
Turning the sidebar's choices into a fit: which parameters matter,
what to warn about, and the model class -- built-in, from an
expression, or from an action.
"""

from __future__ import annotations

import streamlit as st

from cosmofit import (
    model_from_expression,
)

from cosmofit.gui.reference import (
    ACTION_CHOICE,
    ALWAYS_RELEVANT,
    Action,
    BUILTIN_MODELS,
    COMPUTE_RD_PARAMS,
    CUSTOM_CHOICE,
    DATASET_PARAMS,
    HAVE_THEORY,
    MODEL_STANDARD_PARAMS,
)


# ============================================================
# Helpers
# ============================================================

def _model_capabilities(model_cls) -> dict:
    """
    What a model can be asked to do, read off the class itself
    rather than hardcoded -- so a custom model gets honest badges
    too.

    ``mu`` is checked for an *override*: every model inherits
    ``Cosmology.mu`` returning 1 (standard GR growth), and only a
    model that replaces it predicts a growth history differing from
    GR's at the same background.
    """

    from cosmofit.cosmology.core.base import Cosmology
    from cosmofit.cosmology.boltzmann import supports_cmb_spectra

    cmb_ok, cmb_reason = supports_cmb_spectra(model_cls)

    return {
        "w": hasattr(model_cls, "w"),
        "mu": getattr(model_cls, "mu", None) is not Cosmology.mu,
        "cmb": cmb_ok,
        "cmb_reason": cmb_reason,
        "extra": list(getattr(model_cls, "EXTRA_PARAMS", {}) or {}),
    }


# ------------------------------------------------------------

def _relevant_parameters(model_choice, model_cls, datasets, compute_rd,
                         derive_sigma8=False) -> set:
    """
    Which parameters actually do something in *this* fit.

    The union of what the model's E(z) uses, what its own
    ``EXTRA_PARAMS`` add, and what the ticked datasets require. A
    model the user built -- from an expression or from an action --
    is opaque, since it could reference anything, so everything is
    reported relevant rather than guessing and hiding something it
    needs.
    """

    if model_choice in (CUSTOM_CHOICE, ACTION_CHOICE):
        params_cls = getattr(model_cls, "PARAMS_CLASS", None)
        return set(params_cls.names()) if params_cls else set()

    relevant = set(ALWAYS_RELEVANT)

    relevant |= MODEL_STANDARD_PARAMS.get(model_choice, set())

    relevant |= set(getattr(model_cls, "EXTRA_PARAMS", {}) or {})

    for name in datasets:
        relevant |= DATASET_PARAMS.get(name, set())

    if compute_rd:
        # rd stops being a parameter and the densities behind it
        # start being ones.
        relevant.discard("rd")
        relevant |= COMPUTE_RD_PARAMS

    if derive_sigma8:
        # Same trade: sigma8 stops being sampled, ln1e10As carries
        # the amplitude instead.
        relevant.discard("sigma8")
        relevant |= {"ln1e10As"}

    return relevant


# ------------------------------------------------------------

def _fit_warnings(model_choice, model_cls, datasets, free_params,
                  compute_rd) -> list[tuple[str, str]]:
    """
    Model-and-dataset combinations worth flagging *before* the run,
    as ``(icon, message)`` pairs.

    Two kinds. Some are outright errors that would only surface as a
    stack trace minutes in (the CMB spectra with a modified-gravity
    model). The rest are quieter: fits that will run, finish, and
    produce a posterior for a parameter the data cannot constrain --
    which looks exactly like a real result.
    """

    warnings = []

    caps = _model_capabilities(model_cls)

    selected = set(datasets)

    if "planck_lite" in selected and not caps["cmb"]:
        warnings.append((
            "🚫",
            f"**{model_choice}** cannot be used with the full CMB "
            f"spectra: {caps['cmb_reason']} Use the compressed "
            f"distance priors instead, or change the model.",
        ))

    # Extra parameters left fixed: the model reduces to something
    # simpler and the fit is not testing what it looks like it is.
    idle_extra = [
        name for name in caps["extra"] if name not in free_params
    ]
    if idle_extra:
        warnings.append((
            "ℹ️",
            f"**{', '.join(idle_extra)}** left fixed. "
            f"{model_choice}'s own parameter(s) are not being fit, so "
            f"this is a fit of whatever it reduces to at those "
            f"values -- tick them under **Parameters** to actually "
            f"test the model.",
        ))

    # Growth-only models against background-only data.
    background_only = not (selected & {"fsigma8", "s8"})
    if caps["mu"] and background_only:
        warnings.append((
            "⚠️",
            f"**{model_choice}** modifies gravity, and its signature "
            f"is in how structure *grows*. With no growth dataset "
            f"ticked, only its expansion history is being tested -- "
            f"add **fσ₈** and/or **S₈**.",
        ))

    # sigma8 free but nothing measures it.
    if "sigma8" in free_params and background_only:
        warnings.append((
            "⚠️",
            "**σ₈** is free but no dataset constrains it -- its "
            "posterior will be its prior. Tick fσ₈ or S₈, or untick "
            "σ₈.",
        ))

    # BAO with rd free and nothing else to anchor H0.
    if compute_rd and not (selected & {"desi", "sdss_bao", "bao_lowz"}):
        warnings.append((
            "ℹ️",
            "**Compute r_d** is on but no BAO dataset is ticked. "
            "r_d only enters through BAO, so this changes nothing.",
        ))

    if compute_rd and "Omega_b" not in free_params:
        warnings.append((
            "ℹ️",
            "With r_d computed, **Ω_b** is what carries H₀ -- leaving "
            "it fixed pins r_d to one value and throws away the "
            "reason to compute it. Free Ω_b and tick the BBN prior.",
        ))

    if (selected & {"planck_lite", "planck_lensing", "act_lensing",
                    "planck_lowe"}) and "sigma8" in free_params:
        warnings.append((
            "⚠️",
            "**σ₈ is defined twice here.** The CMB datasets compute "
            "it from `ln10¹⁰A_s` through the transfer function, "
            "while fσ₈/S₈ are compared against the *free* σ₈ you "
            "are sampling — and nothing makes the two agree. Tick "
            "**Derive σ₈ from the CMB** in the sidebar, or fix σ₈.",
        ))

    if "planck_lite" in selected:
        missing = [
            name for name in ("ln1e10As", "n_s", "tau_reio")
            if name not in free_params
        ]
        if missing:
            warnings.append((
                "⚠️",
                f"The full CMB spectra depend on "
                f"**{', '.join(missing)}**, which are fixed. The fit "
                f"will run but is conditioning on those values rather "
                f"than measuring them.",
            ))
        if "tau" not in selected and "tau_reio" in free_params:
            warnings.append((
                "⚠️",
                "**τ** is free but the τ prior dataset is not ticked. "
                "plik_lite covers ℓ ≥ 30 only, where τ is degenerate "
                "with the primordial amplitude -- both posteriors will "
                "be unconstrained.",
            ))

    return warnings


# ------------------------------------------------------------

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

def _parse_fields(text: str) -> dict:
    """
    Parse the scalar-field box:

        phi = X - V0*exp(-lam*phi)

    into the ``fields`` dict :class:`cosmofit.theory.Action` takes:
    field name -> Lagrangian density ``L(X, phi)``, with ``X`` the
    kinetic scalar. More than one line is more than one field.
    """

    fields = {}

    for line_no, raw in enumerate(text.splitlines(), start=1):

        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"Field line {line_no}: expected "
                f"'name = L(X, name)', got {raw!r}"
            )

        name, lagrangian = line.split("=", 1)

        name = name.strip()
        lagrangian = lagrangian.strip()

        if not name or not lagrangian:
            raise ValueError(
                f"Field line {line_no}: both a name and a Lagrangian "
                f"are needed, got {raw!r}"
            )

        fields[name] = lagrangian

    return fields


# ------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=32)
def _action_and_model(
    name: str,
    gravity: str,
    geometry: str | None,
    fluids: tuple,
    params_text: str,
    closure: str,
    growth: str,
    fields_text: str,
    z_init: float,
    background: str,
):
    """
    Build an :class:`~cosmofit.theory.Action` and the model class it
    derives, returning both.

    Cached on the definition itself, because Streamlit re-runs the
    whole script on every widget interaction and this is the one
    genuinely expensive thing in it. Reducing the action, varying the
    lapse and solving the constraint is symbolic work -- and a field
    action additionally compiles an integrator -- where everything
    else on this page is a dictionary lookup.

    Both objects come back from one call so the page can show the
    Friedmann equation that was derived without deriving it twice.
    """

    action = Action(
        gravity,
        geometry=geometry,
        fluids=tuple(fluids),
        params=_parse_extra_params(params_text),
        closure=closure or None,
        growth=growth,
        fields=_parse_fields(fields_text) or None,
        z_init=float(z_init),
        background=background,
    )

    return action, action.build(name)


# ------------------------------------------------------------

def _action_widgets(slot: int) -> tuple:
    """
    This slot's action widgets, as the argument tuple
    :func:`_action_and_model` is keyed on.

    Reading them in one place is what keeps the cache key and the
    thing built from it in step -- a second reader that forgot one
    widget would return a stale model for a changed definition,
    which is exactly the failure a cache is good at hiding.
    """

    gravity = st.session_state.get(f"action_gravity_{slot}", "").strip()

    if not gravity:
        raise ValueError(
            "Enter a gravitational Lagrangian -- `R - 2*Lam` is "
            "General Relativity with a cosmological constant."
        )

    geometry = st.session_state.get(f"action_geometry_{slot}", "auto")

    return (
        (
            st.session_state.get(f"action_name_{slot}", "").strip()
            or f"Action{slot + 1}"
        ),
        gravity,
        None if geometry == "auto" else geometry,
        tuple(
            st.session_state.get(f"action_fluids_{slot}", None) or ("matter",)
        ),
        st.session_state.get(f"action_params_{slot}", ""),
        st.session_state.get(f"action_closure_{slot}", "").strip(),
        st.session_state.get(f"action_growth_{slot}", "gr"),
        st.session_state.get(f"action_fields_{slot}", ""),
        float(st.session_state.get(f"action_zinit_{slot}", 3000.0)),
        st.session_state.get(f"action_background_{slot}", "backward"),
    )


# ------------------------------------------------------------

def _build_model_class(slot: int, model_choice: str):
    """
    Resolve one model slot's widgets into a ``Cosmology`` subclass,
    raising a clear error for an incomplete/invalid custom-model
    definition.
    """

    if model_choice == ACTION_CHOICE:

        if not HAVE_THEORY:
            raise ValueError(
                "Deriving a model from an action needs sympy: "
                "pip install 'cosmofit[theory]'."
            )

        widgets = _action_widgets(slot)

        # Kept so the "same fit in Python" panel can reproduce the
        # action rather than guess at it. Recorded from the very
        # tuple the model is built from, so the two cannot disagree.
        (
            _name, _gravity, _geometry, _fluids, _params, _closure,
            _growth, _fields, _z_init, _background,
        ) = widgets

        st.session_state[f"_action_spec_{slot}"] = {
            "name": _name,
            "gravity": _gravity,
            "geometry": _geometry,
            "fluids": list(_fluids),
            "params": _parse_extra_params(_params),
            "closure": _closure or None,
            "growth": _growth,
            "fields": _parse_fields(_fields) or None,
            "background": _background,
        }

        return _action_and_model(*widgets)[1]

    if model_choice != CUSTOM_CHOICE:
        return BUILTIN_MODELS[model_choice]

    name = st.session_state.get(f"custom_name_{slot}", "").strip() or f"Custom{slot + 1}"
    E_expr = st.session_state.get(f"custom_E_{slot}", "").strip()

    if not E_expr:
        raise ValueError("Enter an E(z) expression for the custom model.")

    w_expr = st.session_state.get(f"custom_w_{slot}", "").strip() or None
    dEdz_expr = st.session_state.get(f"custom_dEdz_{slot}", "").strip() or None
    mu_expr = st.session_state.get(f"custom_mu_{slot}", "").strip() or None
    extra_text = st.session_state.get(f"custom_extra_params_{slot}", "")

    extra_params = _parse_extra_params(extra_text)

    return model_from_expression(
        name,
        E=E_expr,
        extra_params=extra_params,
        w=w_expr,
        dEdz=dEdz_expr,
        mu=mu_expr,
    )
