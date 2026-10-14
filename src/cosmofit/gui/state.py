"""
What the page is configured to do, read from session state alone.

The page is split into tabs and only the open one is drawn, so the Run
tab has to know the datasets and models without the widgets that set
them being on screen. Every widget therefore keeps its value with
``persist_state="session"``, its default is seeded here once
(`seed_defaults`), and `current_plan` turns the lot into the fit to
run -- the one place the tabs, the sidebar and the run loop all read.

The parameter tables are the exception, because ``st.data_editor``
neither persists nor can be written through session state: each
model's table lives under ``params_{i}`` as plain rows, the editor is
drawn from them, and its edits are folded back by `apply_param_edits`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import streamlit as st

from cosmofit import __version__, available_versions
from cosmofit.stats import DATASET_REGISTRY

from cosmofit.gui.reference import (
    ACTION_CHOICE,
    CUSTOM_CHOICE,
    BUILTIN_MODELS,
    DATASET_LABELS,
    DATASET_PRESETS,
    INCOMPATIBLE_PAIRS,
    MODEL_GROUPS,
    QUICK_STARTS,
    SAMPLER_PROFILES,
    SAMPLERS,
    STEPS,
)
from cosmofit.gui.helpers import (
    _build_model_class,
    _fit_warnings,
    _relevant_parameters,
)


# ============================================================
# Model options
# ============================================================
#
# Options carry their family as a suffix, so the dropdown says what
# kind of object each entry is instead of listing twenty names with
# nothing to separate ΛCDM from f(Q).

MODEL_OPTIONS = [
    f"{name}  ·  {group}" for group, names in MODEL_GROUPS for name in names
] + [f"{extra}  ·  Not in the library" for extra in (CUSTOM_CHOICE, ACTION_CHOICE)]


#: Dropdown entry -> model name (or `CUSTOM_CHOICE`/`ACTION_CHOICE`).
OPTION_TO_MODEL = {option: option.split("  ·  ")[0] for option in MODEL_OPTIONS}


def model_option(name: str) -> str:
    """The dropdown entry for a model name."""

    return next(o for o, n in OPTION_TO_MODEL.items() if n == name)


# ============================================================
# Defaults
# ============================================================

@functools.lru_cache(maxsize=None)
def versions(key: str) -> tuple:
    """A dataset's versions, the first being its default."""

    return tuple(available_versions(key))


def _slot_defaults(i: int) -> dict:

    return {
        f"model_choice_{i}": MODEL_OPTIONS[0],
        f"relevant_only_{i}": True,
        f"custom_name_{i}": f"Custom{i + 1}",
        f"custom_E_{i}": "",
        f"custom_w_{i}": "",
        f"custom_dEdz_{i}": "",
        f"custom_mu_{i}": "",
        f"custom_extra_params_{i}": "",
        f"action_name_{i}": f"Action{i + 1}",
        f"action_geometry_{i}": "auto",
        f"action_gravity_{i}": "",
        f"action_params_{i}": "",
        f"action_closure_{i}": "",
        f"action_growth_{i}": "gr",
        f"action_background_{i}": "backward",
        f"action_fields_{i}": "",
        f"action_fluids_{i}": ["matter"],
        f"action_zinit_{i}": 3000.0,
    }


def defaults() -> dict:
    """Every widget's starting value, by key."""

    values = {
        "open_step": STEPS["start"],
        "n_models": 1,
        "compute_rd": False,
        "derive_sigma8": False,
        "mcmc_sampler": next(iter(SAMPLERS)),
        "run_profile": "Standard",
        "mcmc_chains": 4,
        "mcmc_rminus1": 0.01,
        "mcmc_nsteps": 20000,
        "mcmc_burnin": 0.3,
        "mcmc_processes": 1,
        "mcmc_nwalkers": 32,
        "mcmc_seed": 42,
        "mcmc_restarts": 0,
        "chains_save": True,
        "chains_folder": "chains",
    }

    for key in DATASET_REGISTRY:
        values[f"ds_{key}"] = key in ("cc", "desi")
        if len(versions(key)) > 1:
            values[f"dsver_{key}"] = versions(key)[0]

    return values


def seed_defaults() -> None:
    """
    Write each default into session state, once.

    A widget given both ``value=`` and a session-state value is the
    one thing Streamlit warns about -- the two disagree on which is
    authoritative. Seeding here and passing no ``value=`` leaves the
    widget owning its value from then on, and a preset free to
    overwrite it.
    """

    state = st.session_state

    for key, value in defaults().items():
        state.setdefault(key, value)

    for i in range(state["n_models"]):
        for key, value in _slot_defaults(i).items():
            state.setdefault(key, value)


# ============================================================
# Saving and restoring the configuration
# ============================================================

#: Session-state keys that describe *what to run*, as opposed to
#: what was run or which tab is open. Saved and restored by prefix
#: rather than by an exhaustive list, so a widget added later to an
#: existing family is carried without anyone remembering to come
#: back here.
#:
#: Deliberately excluded: `dl_*` (download button state),
#: `*_model_choice` and `*_multiselect` (which result is being
#: looked at), `profile_run_`/`fisher_run_` (whether an expensive
#: extra has been asked for) and `step` (the open tab). Restoring
#: those would reproduce someone else's *view* rather than their
#: configuration.
_CONFIG_PREFIXES = (
    "ds_", "dsver_", "compute_rd", "derive_sigma",
    "model_choice_", "n_models",
    "action_", "custom_",
    "mcmc_", "params_", "relevant_only_",
)


_CONFIG_EXCLUDE = ("action_load_", "action_preset_", "dataset_preset")


def _configuration() -> dict:
    """
    The current configuration, as something that can be written to
    a file and read back.

    A run here can take fourteen dataset ticks, three models with
    their own parameters, and a set of MCMC settings. Closing the
    tab loses all of it, and there is no way to hand it to someone
    else -- which for a tool whose whole point is reproducible
    fitting is a gap worth closing.
    """

    return {
        "version": __version__,
        "state": {
            key: value
            for key, value in st.session_state.items()
            if isinstance(key, str)
            and key.startswith(_CONFIG_PREFIXES)
            and not key.startswith(_CONFIG_EXCLUDE)
            and isinstance(value, (str, int, float, bool, list, type(None)))
        },
    }


def _restore_configuration(payload: dict) -> int:
    """
    Write a saved configuration back into session state.

    Returns how many keys were applied.

    Values are written rather than widgets set, and this runs
    *above* the widgets it writes to, so they are picked up on the
    same pass -- the same ordering the dataset presets rely on. A
    `st.rerun()` here would be a second render for no gain.

    Unknown keys are dropped rather than trusted: a file from a
    later version can name a widget this one does not have, and
    Streamlit raises on an unexpected key only once the widget
    tries to use it, which would be a crash far from the cause.
    """

    state = (payload or {}).get("state") or {}

    applied = 0

    for key, value in state.items():

        if not isinstance(key, str):
            continue

        if not key.startswith(_CONFIG_PREFIXES):
            continue

        if key.startswith(_CONFIG_EXCLUDE):
            continue

        st.session_state[key] = value

        applied += 1

    # The tables come back as rows; a fresh editor has to be drawn
    # from them rather than keep the edits it was holding.
    for i in range(int(st.session_state.get("n_models", 1))):
        _bump_editor(i)

    return applied


# ============================================================
# Parameter tables
# ============================================================

_EDITABLE = ("Free", "Initial", "Lower", "Upper")


def _default_rows(model_cls, free=None) -> list[dict]:
    """
    A model's parameter table, before anyone has touched it.

    H0 and Omega_m are free (or `free`, when given), except a
    parameter the model *derives* rather than accepts -- ADE fixes
    Omega_m from its early-time condition, so sampling it would
    return the prior and look exactly like a measurement.

    H0/Omega_m (and any custom parameter without an explicit
    "default") have no dataclass default -- falling back to 0.0 would
    sit outside their prior bounds and produce a degenerate walker
    cloud (every walker clipped to the same edge). The bounds
    midpoint is always inside the prior instead.
    """

    params_cls = model_cls.PARAMS_CLASS
    values = params_cls.defaults()
    derived = set(getattr(model_cls, "DERIVED_PARAMS", ()) or ())
    free = set(free) if free is not None else {"H0", "Omega_m"}

    rows = []

    for p in params_cls.parameter_set():

        lo, hi = p.bounds if p.bounds else (None, None)

        if p.name in values:
            initial = float(values[p.name])
        elif lo is not None and hi is not None:
            initial = float((lo + hi) / 2.0)
        else:
            initial = 0.0

        rows.append({
            "Parameter": p.name,
            "Free": p.name in free and p.name not in derived,
            "Initial": initial,
            "Lower": None if lo is None else float(lo),
            "Upper": None if hi is None else float(hi),
        })

    return rows


def param_rows(i: int, choice: str, model_cls) -> list[dict]:
    """
    Model ``i``'s table, kept in session state across tabs and reruns.

    A different model starts a fresh table. The same model -- a Custom
    one whose expression gained a parameter, say -- keeps every row it
    had and adds the new ones at their defaults, so an edit is never
    lost to an unrelated change.
    """

    state = st.session_state

    stored = state.get(f"params_{i}")

    if stored is None or state.get(f"params_model_{i}") != choice:
        rows = _default_rows(model_cls)
    else:
        previous = {row["Parameter"]: row for row in stored}
        rows = [
            previous.get(row["Parameter"], row)
            for row in _default_rows(model_cls)
        ]

    if rows != stored:
        state[f"params_{i}"] = rows
        state[f"params_model_{i}"] = choice

    return rows


def editor_key(i: int) -> str:
    """The parameter editor's current key; see `apply_param_edits`."""

    return f"param_editor_{i}_{st.session_state.get(f'_param_rev_{i}', 0)}"


def _bump_editor(i: int) -> None:

    st.session_state[f"_param_rev_{i}"] = st.session_state.get(f"_param_rev_{i}", 0) + 1


def apply_param_edits(i: int, shown: list[str], key: str) -> None:
    """
    Fold the editor's edits into model ``i``'s rows (an ``on_change``
    callback).

    The editor reports edits by row position in what it was shown,
    which is why `shown` -- those rows' names, in order -- comes
    along. Its key then moves on, so the next pass draws a fresh
    editor from the updated rows instead of replaying the edits on top
    of them.
    """

    edits = (st.session_state.get(key) or {}).get("edited_rows", {})

    rows = [dict(row) for row in st.session_state.get(f"params_{i}", [])]
    index = {row["Parameter"]: row for row in rows}

    for position, change in edits.items():

        name = shown[int(position)]

        for column, value in change.items():
            if column == "Free":
                index[name][column] = bool(value)
            elif column in _EDITABLE:
                index[name][column] = None if value is None else float(value)

    st.session_state[f"params_{i}"] = rows

    _bump_editor(i)


# ============================================================
# The plan
# ============================================================

@dataclass
class Slot:
    """One model slot, resolved."""

    index: int
    choice: str
    model_cls: object = None
    error: str | None = None
    rows: list = field(default_factory=list)
    relevant: set = field(default_factory=set)
    warnings: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Model {self.index + 1}"

    @property
    def name(self) -> str:
        """What to call it in a tab or a checklist line."""

        if self.model_cls is not None and self.choice in (CUSTOM_CHOICE, ACTION_CHOICE):
            return getattr(self.model_cls, "MODEL_NAME", None) or self.model_cls.__name__
        return self.choice

    @property
    def free(self) -> list[str]:
        return [row["Parameter"] for row in self.rows if row["Free"]]

    @property
    def initial(self) -> dict:
        return {row["Parameter"]: row["Initial"] for row in self.rows}

    @property
    def bounds(self) -> dict:
        return {
            row["Parameter"]: (row["Lower"], row["Upper"])
            for row in self.rows
            if row["Lower"] is not None and row["Upper"] is not None
        }


@dataclass
class Plan:
    """Everything a run needs, from session state."""

    datasets: list
    versions: dict
    compute_rd: bool
    derive_sigma8: bool
    slots: list
    sampler_name: str
    sampler_options: dict
    burn_in: float
    seed: int
    restarts: int
    chain_dir: str | None

    @property
    def conflicts(self) -> list:
        """Ticked pairs that double-count data, with why."""

        selected = set(self.datasets)
        return [(pair, reason) for pair, reason in INCOMPATIBLE_PAIRS if pair <= selected]

    def checklist(self) -> list[dict]:
        """
        What stands between this plan and a meaningful run, as items
        ``{state, text, step}``: ``state`` is ``"ok"``, ``"warn"`` (it
        will run, look at it first) or ``"block"`` (it would fail).
        ``step`` is the tab that fixes it.
        """

        items = []

        if self.datasets:
            items.append(dict(
                state="ok", step="data",
                text=f"{len(self.datasets)} dataset(s) selected",
            ))
        else:
            items.append(dict(
                state="block", step="data", text="No dataset is selected",
            ))

        for pair, _ in self.conflicts:
            items.append(dict(
                state="warn", step="data",
                text=" + ".join(DATASET_LABELS.get(k, k) for k in sorted(pair))
                + " double-count data",
            ))

        for slot in self.slots:

            if slot.error:
                items.append(dict(
                    state="block", step="model",
                    text=f"{slot.label} is not ready: {slot.error}",
                ))
                continue

            if not slot.free:
                items.append(dict(
                    state="block", step="model",
                    text=f"{slot.label} ({slot.name}) has no free parameter",
                ))
                continue

            items.append(dict(
                state="ok", step="model",
                text=f"{slot.label} ({slot.name}) fits "
                     f"{len(slot.free)} parameter(s): {', '.join(slot.free)}",
            ))

            for icon, message in slot.warnings:
                if icon in ("🚫", "⚠️"):
                    items.append(dict(
                        state="block" if icon == "🚫" else "warn",
                        step="model",
                        text=f"{slot.label}: {message}",
                    ))

            walkers = self.sampler_options.get("walkers")
            if walkers is not None and walkers < 2 * len(slot.free):
                items.append(dict(
                    state="block", step="run",
                    text=f"{slot.label}: emcee needs at least "
                         f"{2 * len(slot.free)} walkers for "
                         f"{len(slot.free)} free parameters, not {walkers}",
                ))

        return items

    @property
    def ready(self) -> bool:
        return not any(item["state"] == "block" for item in self.checklist())


def current_plan() -> Plan:
    """Read the whole configuration out of session state."""

    state = st.session_state

    datasets = [key for key in DATASET_REGISTRY if state.get(f"ds_{key}")]

    chosen_versions = {
        key: state[f"dsver_{key}"]
        for key in datasets
        if f"dsver_{key}" in state
    }

    compute_rd = bool(state["compute_rd"])
    derive_sigma8 = bool(state["derive_sigma8"])

    slots = []

    for i in range(state["n_models"]):

        choice = OPTION_TO_MODEL.get(state[f"model_choice_{i}"], "LCDM")
        slot = Slot(index=i, choice=choice)

        try:
            slot.model_cls = _build_model_class(i, choice)
        except Exception as exc:
            slot.error = str(exc)
            slots.append(slot)
            continue

        slot.rows = param_rows(i, choice, slot.model_cls)
        slot.relevant = _relevant_parameters(
            choice, slot.model_cls, datasets, compute_rd, derive_sigma8,
        )
        slot.warnings = _fit_warnings(
            choice, slot.model_cls, datasets, slot.free, compute_rd,
        )

        slots.append(slot)

    sampler_name = SAMPLERS[state["mcmc_sampler"]]

    if sampler_name == "mcmc":
        sampler_options = {
            "chains": int(state["mcmc_chains"]),
            "Rminus1_stop": float(state["mcmc_rminus1"]),
            "max_samples": int(state["mcmc_nsteps"]),
            "processes": int(state["mcmc_processes"]),
        }
        burn_in = float(state["mcmc_burnin"])
    else:
        sampler_options = {
            "walkers": int(state["mcmc_nwalkers"]),
            "max_samples": int(state["mcmc_nsteps"]),
        }
        burn_in = 0.0

    folder = str(state["chains_folder"]).strip()

    return Plan(
        datasets=datasets,
        versions=chosen_versions,
        compute_rd=compute_rd,
        derive_sigma8=derive_sigma8,
        slots=slots,
        sampler_name=sampler_name,
        sampler_options=sampler_options,
        burn_in=burn_in,
        seed=int(state["mcmc_seed"]),
        restarts=int(state["mcmc_restarts"]),
        chain_dir=folder if state["chains_save"] and folder else None,
    )


def step_status(plan: Plan) -> dict:
    """
    ``"ok"``, ``"warn"``, ``"block"`` or ``"todo"`` per step, for the
    badges on the Start tab and in the sidebar.
    """

    worst = {}

    for item in plan.checklist():
        rank = {"ok": 0, "warn": 1, "block": 2}[item["state"]]
        if rank >= {"ok": 0, "warn": 1, "block": 2}.get(worst.get(item["step"]), -1):
            worst[item["step"]] = item["state"]

    fitted = bool(st.session_state.get("fits"))

    if not plan.ready:
        run = "block"
    elif fitted:
        run = "ok"
    else:
        run = "warn" if worst.get("run") == "warn" else "todo"

    return {
        "data": worst.get("data", "todo"),
        "model": worst.get("model", "todo"),
        "run": run,
        "results": "ok" if fitted else "todo",
    }


# ============================================================
# Callbacks: moving between tabs, and writing a whole setup
# ============================================================
#
# Callbacks run before the script does, so whatever they write is in
# place when the widgets reading it are created -- including the tabs'
# own key, which cannot be written once the tabs exist.

def go(step: str) -> None:
    """Open a tab (``step`` is a key of `STEPS`)."""

    st.session_state["step"] = STEPS[step]
    st.session_state["open_step"] = STEPS[step]


def remember_step() -> None:
    """
    Note which tab is open (the tabs' ``on_change``).

    The tabs are drawn with this as their default, so the open tab
    survives a rerun that does not report the tabs' own state -- as
    a script driven headlessly, by ``AppTest``, does not.
    """

    st.session_state["open_step"] = st.session_state["step"]


def apply_preset(name: str) -> None:
    """Write a dataset preset's datasets, versions and r_d choice."""

    preset = DATASET_PRESETS[name]

    for key in DATASET_REGISTRY:
        st.session_state[f"ds_{key}"] = key in preset["datasets"]

    for key, version in (preset.get("versions") or {}).items():
        st.session_state[f"dsver_{key}"] = version

    st.session_state["compute_rd"] = bool(preset.get("compute_rd"))


def apply_profile() -> None:
    """Write the chosen sampling profile's settings."""

    profile = SAMPLER_PROFILES.get(st.session_state.get("run_profile"))

    if profile:
        st.session_state.update(profile["settings"])


def apply_quick_start(key: str) -> None:
    """
    Write a whole analysis -- datasets, one model, its free
    parameters, a sampling profile -- and open the Run tab.
    """

    quick = QUICK_STARTS[key]

    apply_preset(quick["preset"])

    st.session_state["derive_sigma8"] = False
    st.session_state["n_models"] = 1

    st.session_state["model_choice_0"] = model_option(quick["model"])
    st.session_state["relevant_only_0"] = True
    st.session_state["params_0"] = _default_rows(
        BUILTIN_MODELS[quick["model"]], free=quick["free"],
    )
    st.session_state["params_model_0"] = quick["model"]
    _bump_editor(0)

    st.session_state["mcmc_sampler"] = next(iter(SAMPLERS))
    st.session_state["run_profile"] = quick["profile"]
    apply_profile()

    go("run")


def add_model() -> None:

    st.session_state["n_models"] += 1


def remove_model() -> None:

    st.session_state["n_models"] = max(1, st.session_state["n_models"] - 1)
