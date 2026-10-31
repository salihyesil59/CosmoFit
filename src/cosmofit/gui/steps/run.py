"""
The Run tab: the checklist that stands between a plan and a run worth
waiting for, how long to sample, and the run itself.
"""

from __future__ import annotations

import math
import os
import uuid
import warnings

import pandas as pd
import streamlit as st

from cosmofit.gui.reference import DATASET_LABELS, SAMPLER_PROFILES, SAMPLERS, STEPS
from cosmofit.gui.state import apply_profile, go
from cosmofit.gui.style import CHECK_ICONS, fastcmb_toggle, step_header


def render(plan) -> None:

    step_header("run")

    left, right = st.columns([3, 2], gap="large")

    with left:
        _checklist(plan)

    with right, st.container(border=True):
        _what_will_run(plan)

    st.markdown("### How long to sample")

    _sampler_settings(plan)

    st.write("")

    clicked = st.button(
        "🚀 Run Fit", type="primary", width="stretch",
        disabled=not plan.ready,
        help=None if plan.ready else "Clear the 🚫 items in the checklist first.",
    )

    if clicked:
        _run(plan)


# ------------------------------------------------------------

def _checklist(plan) -> None:

    st.markdown("### Pre-flight checklist")

    items = plan.checklist()

    if plan.ready:
        st.caption(
            "Nothing blocks this run. ⚠️ lines will run but are worth a "
            "look first."
            if any(item["state"] == "warn" for item in items)
            else "All clear."
        )
    else:
        st.caption("🚫 lines would make the run fail. Each has a way to fix it.")

    for n, item in enumerate(items):

        text_column, fix_column = st.columns([5, 1], vertical_alignment="center")

        text_column.markdown(f"{CHECK_ICONS[item['state']]} {item['text']}")

        if item["state"] != "ok" and item["step"] != "run":
            fix_column.button(
                "Fix →", key=f"fix_{n}", width="stretch",
                on_click=go, args=(item["step"],),
                help=f"Open {STEPS[item['step']]}",
            )


def _what_will_run(plan) -> None:

    st.markdown("**What will run**")

    st.caption("Data")
    st.markdown(
        ", ".join(DATASET_LABELS.get(k, k) for k in plan.datasets) or "_none_"
    )

    if plan.compute_rd or plan.derive_sigma8:
        st.caption(
            " · ".join(
                text for flag, text in (
                    (plan.compute_rd, "r_d computed"),
                    (plan.derive_sigma8, "σ₈ derived from the CMB"),
                ) if flag
            )
        )

    if plan.fastcmb:
        st.caption("CMB spectra: fastcmb, emulating CAMB")

    st.caption("Models")

    for slot in plan.slots:
        if slot.error:
            st.markdown(f"- {slot.label}: _not ready_")
        else:
            st.markdown(
                f"- **{slot.name}** — {', '.join(slot.free) or '_nothing free_'}"
            )


def _sampler_settings(plan) -> None:

    sampler_label = st.radio(
        "Sampler",
        options=list(SAMPLERS),
        key="mcmc_sampler",
        horizontal=True,
        persist_state="session",
        help="Both run on the library's 2.0 core. Adaptive "
             "Metropolis-Hastings runs a few independent chains, "
             "learns the posterior's covariance as it goes, and "
             "stops once the chains agree (Gelman-Rubin R - 1). "
             "emcee's ensemble moves many walkers together and "
             "stops at 50 autocorrelation times.",
    )

    adaptive = SAMPLERS[sampler_label] == "mcmc"

    if adaptive:

        profile = st.radio(
            "Profile",
            options=list(SAMPLER_PROFILES),
            key="run_profile",
            horizontal=True,
            on_change=apply_profile,
            persist_state="session",
            help="Sets Chains, Stop at R − 1 and Max steps below; "
                 "change any of them under Advanced.",
        )

        settings = SAMPLER_PROFILES[profile]["settings"]

        if all(st.session_state[k] == v for k, v in settings.items()):
            st.caption(SAMPLER_PROFILES[profile]["note"])
        else:
            st.caption(
                f"Edited under Advanced: {st.session_state['mcmc_chains']} "
                f"chains, R − 1 < {st.session_state['mcmc_rminus1']:g}, at "
                f"most {st.session_state['mcmc_nsteps']} steps. Picking a "
                f"profile resets them."
            )

    if plan.needs_camb:
        fastcmb_toggle(plan)

    with st.expander("Advanced sampler settings"):

        col_a, col_b = st.columns(2)

        if adaptive:

            with col_a:
                n_chains = st.number_input(
                    "Chains", min_value=2, max_value=16, step=1,
                    key="mcmc_chains", persist_state="session",
                    help="Independent chains, compared with each other "
                         "to decide when to stop.",
                )
                st.number_input(
                    "Stop at R − 1", min_value=0.001, max_value=0.5,
                    step=0.005, format="%.3f",
                    key="mcmc_rminus1", persist_state="session",
                    help="How closely the chains must agree. 0.01 is "
                         "the usual target; 0.05 for a first look.",
                )

            with col_b:
                st.number_input(
                    "Max steps", min_value=50, step=1000,
                    key="mcmc_nsteps", persist_state="session",
                    help="Steps per chain before giving up on "
                         "convergence -- a ceiling, not a target.",
                )
                st.number_input(
                    "Burn-in (fraction)", min_value=0.0, max_value=0.9,
                    step=0.05, key="mcmc_burnin", persist_state="session",
                    help="Of each chain's steps, discarded from the "
                         "start before summarizing.",
                )

            if st.session_state["mcmc_processes"] > int(n_chains):
                st.session_state["mcmc_processes"] = int(n_chains)

            st.number_input(
                "Processes", min_value=1, max_value=int(n_chains), step=1,
                key="mcmc_processes", persist_state="session",
                help="Run the chains in separate processes. Starting one "
                     "rebuilds the model and reloads its data, which "
                     "takes seconds -- worth it for a Boltzmann-code "
                     "likelihood, not for distances alone. A model built "
                     "here (from an expression or an action) exists only "
                     "in this session and always runs in one.",
            )

        else:

            with col_a:
                st.number_input(
                    "Walkers", min_value=8, step=2,
                    key="mcmc_nwalkers", persist_state="session",
                    help="At least 2x the number of free parameters, or "
                         "the fit will fail to start.",
                )

            with col_b:
                st.number_input(
                    "Max steps", min_value=50, step=1000,
                    key="mcmc_nsteps", persist_state="session",
                    help="Steps per walker before giving up on "
                         "convergence -- a ceiling, not a target.",
                )

        col_c, col_d = st.columns(2)

        with col_c:
            st.number_input(
                "Seed", min_value=0, step=1, key="mcmc_seed",
                persist_state="session",
                help="Same seed, same configuration: the same chains.",
            )

        with col_d:
            st.number_input(
                "Best-fit restarts", min_value=0, max_value=32, step=1,
                key="mcmc_restarts", persist_state="session",
                help="Before sampling, the best fit is found by an "
                     "optimizer -- the chains start there -- and an "
                     "optimizer converges into whichever basin it started "
                     "in. Restarts draw that many further starting points "
                     "from the prior and keep the best result. Worth setting "
                     "when a model fits *worse* than the one it contains as "
                     "a special case -- an impossible answer, and how this "
                     "was found. Costs one extra optimization each.",
            )

    with st.container(border=True):

        save = st.checkbox(
            "Save chains and reuse them", key="chains_save",
            persist_state="session",
            help="Write each model's chains to disk as they are "
                 "sampled, in getdist's format, and reuse them next "
                 "time. Add a second model and the first one comes "
                 "back instantly; tighten R − 1 or raise Max steps "
                 "and the saved chains continue rather than restart; "
                 "close the app and it's all still there. Changing a "
                 "model, its datasets, free parameters, priors, the "
                 "sampler, its chains or walkers, or the seed makes a "
                 "different fit, which gets its own folder -- so "
                 "nothing is ever silently reused when it shouldn't be.",
        )

        st.text_input(
            "Folder", key="chains_folder", disabled=not save,
            persist_state="session",
            help="Where each fit's folder of chains goes, relative to "
                 "wherever the app was started. Delete a fit's folder "
                 "to force a fresh run.",
        )


# ------------------------------------------------------------

class _LivePanel:
    """
    What a run shows while it runs: a progress bar, the stage, and at
    every check of the chains their numbers and a convergence chart.
    """

    def __init__(self, n_models: int):

        self.n_models = n_models
        self.bar = st.progress(0.0, text="Starting...")
        self.metrics = st.empty()
        self.chart = st.empty()
        self.history = []

    def stage(self, i: int, done: float, text: str) -> None:
        self.bar.progress(min(1.0, (i + done) / self.n_models), text=text)

    def new_model(self) -> None:
        self.history = []
        self.metrics.empty()
        self.chart.empty()

    def check(self, entry: dict, i: int, label: str) -> None:
        """One check of the chains (the core sampler's callback)."""

        steps, seconds = entry["steps"], entry.get("seconds", 0.0)
        ceiling = max(entry["max_samples"], 1)

        if "Rminus1_stop" in entry:
            value, target = entry["Rminus1"], entry["Rminus1_stop"]
            state = f"R − 1 = {value:.3g}, stopping at {target:g}"
            metric = ("R − 1", f"{value:.3g}", f"target {target:g}")
            point = math.log10(value) if 0 < value < math.inf else None
            series = ("log₁₀(R − 1)", "target", math.log10(target))
        else:
            tau, factor = entry["tau"], entry["tau_factor"]
            ratio = steps / tau if tau and math.isfinite(tau) else None
            state = f"τ = {tau:.3g}"
            metric = ("Steps / τ", f"{ratio:.1f}" if ratio else "—", f"target {factor:g}")
            point = ratio
            series = ("steps / τ", "target", factor)

        self.stage(i, min(1.0, steps / ceiling),
                   f"Sampling {label} ({i + 1}/{self.n_models}) -- {steps} steps, {state}")

        cells = [
            ("Steps", f"{steps:,}", f"of at most {ceiling:,}"),
            metric,
            ("Acceptance", f"{entry['acceptance']:.0%}", None),
            ("Elapsed", _duration(seconds), _remaining(steps, ceiling, seconds)),
        ]

        calls = entry.get("theory_calls")

        if calls:
            total = calls["camb"] + calls["emulated"]
            cells.append((
                "Emulated", f"{calls['emulated'] / max(total, 1):.0%}",
                f"CAMB ran {calls['camb']:,} times",
            ))

        with self.metrics.container(border=True):
            for column, (name, value, note) in zip(st.columns(len(cells)), cells):
                column.metric(name, value)
                if note:
                    column.caption(note)

        if point is not None:
            self.history.append((steps, point))

        if len(self.history) >= 2:
            name, target_name, target = series
            frame = pd.DataFrame(
                {name: [p for _, p in self.history], target_name: target},
                index=pd.Index([s for s, _ in self.history], name="steps"),
            )
            self.chart.line_chart(frame, height=220, color=["#d1495b", "#808080"])

    def clear(self) -> None:
        self.bar.empty()
        self.metrics.empty()
        self.chart.empty()


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def _remaining(steps: int, ceiling: int, seconds: float) -> str | None:
    """At most how long is left: the steps to the ceiling, at this pace."""

    if steps <= 0 or seconds <= 0 or steps >= ceiling:
        return None

    return f"≤ {_duration((ceiling - steps) * seconds / steps)} left"


def _emulator_note(label: str, calls: dict | None) -> str | None:
    """What fastcmb did in a run, for the run's notes."""

    if not calls:
        return None

    total = calls["camb"] + calls["emulated"]

    if total == 0:
        return None

    note = (f"{label}: fastcmb answered {calls['emulated'] / total:.1%} of "
            f"{total:,} points; CAMB ran {calls['camb']:,} times")

    if calls.get("screened"):
        note += f", and {calls['screened']:,} proposals were rejected from the emulator alone"

    return note + "."


def _run(plan) -> None:
    """
    Fit every model, then open the Results tab.

    What the run says along the way -- a library warning, a parameter
    dropped because it is computed -- is kept with the results rather
    than drawn here, since the page moves to the Results tab when it
    is done and would otherwise take the messages with it.
    """

    from cosmofit import Fitter
    from cosmofit.compat import best_fit_on_core, fast_info, sample_on_core

    from cosmofit.gui.helpers import _picklable
    from cosmofit.gui.render import _fit_labels

    panel = _LivePanel(len(plan.slots))

    fits = []
    messages = []
    emulated = []   # per model, whether fastcmb stood in for CAMB

    n_models = len(plan.slots)

    try:
        for slot in plan.slots:

            i = slot.index
            model_label = slot.label
            free_params = list(slot.free)

            if plan.derive_sigma8 and "sigma8" in free_params:
                free_params.remove("sigma8")
                messages.append((
                    "info",
                    f"{model_label}: dropped `sigma8` from the free "
                    f"parameters -- it is being derived, not fitted.",
                ))

            if plan.compute_rd and "rd" in free_params:
                # The table may have `rd` ticked from a BAO fit before
                # the box was ticked; un-tick it rather than raise at
                # someone who ticked a box in one place and a checkbox
                # in another.
                free_params.remove("rd")
                messages.append((
                    "info",
                    f"{model_label}: dropped `rd` from the free "
                    f"parameters -- it is being computed, not fitted.",
                ))

            # `Fitter` warns about combinations that will run, finish,
            # and mean nothing -- a free `m_nu` with no CMB that can see
            # it, a parameter the model derives rather than accepts.
            # Those go to `warnings`, which in a Streamlit app means
            # stderr, which means nowhere. Catching them here surfaces
            # every guard the library has and every one it grows later,
            # without the GUI having to restate any of them.
            with warnings.catch_warnings(record=True) as caught:

                warnings.simplefilter("always", UserWarning)

                fit = Fitter(
                    model=slot.model_cls,
                    datasets=plan.datasets,
                    free_params=free_params,
                    initial=slot.initial,
                    bounds=slot.bounds,
                    dataset_kwargs={
                        key: {"version": version}
                        for key, version in plan.versions.items()
                    } or None,
                    compute_rd=plan.compute_rd,
                    derive_sigma8=plan.derive_sigma8,
                )

            for entry in caught:
                messages.append(("warning", f"**Model {i + 1}:** {entry.message}"))

            panel.new_model()
            panel.stage(i, 0.0, f"Best fit for {model_label} ({i + 1}/{n_models})")

            options = dict(plan.sampler_options)

            # fastcmb: the native theories with an emulator in front of
            # CAMB, for the best fit and the chains alike. Its CAMB
            # points are kept beside the chains, one file per model,
            # so the processes share them and the next fit starts
            # from them.
            info = None

            if plan.fastcmb:
                training = None
                if plan.chain_dir:
                    folder = os.path.join(plan.chain_dir, "fastcmb")
                    os.makedirs(folder, exist_ok=True)
                    training = os.path.join(folder, slot.name)
                try:
                    info = fast_info(fit, training_file=training)
                except ValueError as exc:
                    messages.append(("info", f"{model_label}: CAMB without fastcmb -- {exc}"))
                if info is not None and training is None and options.get("processes", 1) > 1:
                    options["processes"] = 1
                    messages.append((
                        "info",
                        f"{model_label}: one process -- with fastcmb, parallel "
                        f"chains share the emulator's CAMB points through the "
                        f"chains folder, and saving chains is off.",
                    ))

            emulated.append(info is not None)

            # The best fit first: the chains start there, which is
            # where the posterior is.
            if info is None:
                fit.best_fit(restarts=plan.restarts)
            else:
                best_fit_on_core(fit, info, restarts=plan.restarts, seed=plan.seed)

            panel.stage(
                i, 0.0,
                f"Best fit for {model_label}: χ² = {fit.best_fit_result.fun:.2f}. "
                f"Starting the chains...",
            )

            last_check = {}

            def _on_check(entry, _i=i, _label=model_label, _last=last_check):
                _last.update(entry)
                panel.check(entry, _i, _label)

            if options.get("processes", 1) > 1 and not _picklable(slot.model_cls):
                options["processes"] = 1
                messages.append((
                    "info",
                    f"{model_label} was built in this session, so its "
                    f"chains run in one process.",
                ))

            # One folder per distinct fit (`chain_id` hashes the model,
            # datasets, free parameters and priors, plus the sampler
            # settings a run cannot change halfway through). Same
            # configuration as last time -> its chains are read back,
            # and continued only if they had not met their stopping
            # rule; anything changed -> a new folder.
            output = None
            if plan.chain_dir:
                identity = {
                    k: v for k, v in options.items() if k in ("chains", "walkers")
                }
                if info is not None:
                    identity["engine"] = "fastcmb"
                output = os.path.join(
                    plan.chain_dir,
                    fit.chain_id(sampler=plan.sampler_name, seed=plan.seed, **identity),
                    "run",
                )

            sample_on_core(
                fit, plan.sampler_name, options, output=output, seed=plan.seed,
                callback=_on_check, burn_in=plan.burn_in, info=info,
            )

            note = _emulator_note(model_label, last_check.get("theory_calls"))

            if note:
                messages.append(("info", note))

            panel.stage(i, 1.0, f"Fitted {model_label} ({i + 1}/{n_models})")

            fits.append((fit, free_params))

    except Exception as exc:
        for key in ("fits", "fit_labels", "fit_plot_labels", "last_run"):
            st.session_state.pop(key, None)
        panel.clear()
        for kind, text in messages:
            getattr(st, kind)(text)
        st.error(f"Fit failed: {exc}", icon="🚫")
        return

    panel.clear()

    fitted = [fit for fit, _ in fits]

    plain_labels, plot_labels = _fit_labels(fitted)

    st.session_state["fits"] = fitted
    st.session_state["fit_labels"] = plain_labels
    st.session_state["fit_plot_labels"] = plot_labels
    st.session_state["fit_messages"] = messages
    st.session_state["fits_token"] = uuid.uuid4().hex
    st.session_state["_figures"] = {}
    st.session_state["_results"] = {}

    # What was run, for the Python and YAML it is offered as -- not
    # what the widgets say by the time someone opens that panel.
    first = plan.slots[0]
    st.session_state["last_run"] = dict(
        model_choice=first.choice,
        model_names=[
            getattr(slot.model_cls, "MODEL_NAME", None) or slot.model_cls.__name__
            for slot in plan.slots
        ],
        action_specs=[
            st.session_state.get(f"_action_spec_{slot.index}")
            for slot in plan.slots
        ],
        datasets=list(plan.datasets),
        dataset_versions=dict(plan.versions),
        free_params=list(fits[0][1]),
        initial=dict(first.initial),
        bounds=dict(first.bounds),
        compute_rd=plan.compute_rd,
        derive_sigma8=plan.derive_sigma8,
        sampler=plan.sampler_name,
        sampler_options=dict(plan.sampler_options),
        burn_in=plan.burn_in,
        seed=plan.seed,
        fastcmb=bool(emulated and emulated[0]),
    )

    reused = sum(bool(getattr(f.sampler, "reused", False)) for f in fitted)

    st.session_state["_flash"] = (
        f"Fit complete -- {reused} of {len(fitted)} read straight from a "
        f"saved chain."
        if reused else "Fit complete."
    )

    # The tabs already exist on this pass, so their key cannot be
    # written now; the next pass opens Results before drawing them.
    st.session_state["_goto"] = "results"

    st.rerun()
