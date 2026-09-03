"""
The graphical interface, driven headlessly.

A GUI is the one part of this project where a mistake is invisible
from the outside: the library's tests all pass while the app shows a
stack trace, or -- worse -- shows a plausible number computed from the
wrong widget. Streamlit's own ``AppTest`` runs the script in-process
and exposes the resulting element tree, so the flows a user actually
takes can be exercised the same way any other code path is.

What is checked here is deliberately behavioural rather than
cosmetic. Not "is there a checkbox" but: does a preset write the
configuration it claims to, does a fit run end to end, does the
warning that a model cannot be used with a dataset actually appear
before someone waits an hour to find out.

Fits are kept to a few hundred likelihood evaluations. They are not
testing the sampler -- that has its own tests -- only that the wiring
from widget to ``Fitter`` to rendered table holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("streamlit", reason="GUI extra not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402


APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _fresh(timeout: float = 300.0) -> AppTest:
    """
    A booted app with chain saving off and a short chain configured.

    Chain saving is disabled so a test run never writes ``chains/``
    into whatever directory pytest was started from -- and so one
    test cannot hand its samples to another and hide a regression.
    """

    app = AppTest.from_file(APP, default_timeout=timeout)

    app.run()

    for checkbox in app.checkbox:
        if checkbox.label.startswith("Save chains"):
            checkbox.set_value(False)

    app.run()

    for number in app.number_input:
        if number.label == "Steps":
            number.set_value(60)
        elif number.label == "Burn-in":
            number.set_value(10)
        elif number.label == "Walkers":
            number.set_value(8)

    app.run()

    return app


def _select_model(app: AppTest, name: str) -> AppTest:
    """
    Pick a model by name. The dropdown shows ``"CPL · Dark energy on
    top of GR"``, so entries are matched by prefix rather than
    equality.
    """

    picker = [s for s in app.selectbox if s.label == "Cosmology"][0]

    option = next(o for o in picker.options if o.startswith(f"{name} "))

    return picker.set_value(option).run()


def _apply_preset(app: AppTest, name: str) -> AppTest:

    picker = [s for s in app.selectbox if s.label == "Start from a preset"][0]

    picker.set_value(name).run()

    button = [b for b in app.button if b.label == "Apply preset"][0]

    return button.click().run()


def _parameter_table(app: AppTest):
    """
    The editable parameter table for the first model.

    ``st.data_editor`` has no accessor of its own on ``AppTest`` --
    it is reported alongside ``st.dataframe`` -- so it is picked out
    by the one column only it has.
    """

    tables = [
        f.value for f in app.dataframe
        if "Parameter" in getattr(f.value, "columns", [])
    ]

    assert tables, "no parameter table rendered"

    return tables[0]


def _messages(app: AppTest) -> str:
    """
    Every error, warning, info and caption the page is currently
    showing, as one searchable string.

    `st.info` is in there because that is what an incomplete model
    definition renders as -- "not ready yet -- ..." is guidance
    rather than a failure, and leaving it out made this helper
    silently blind to a whole class of message.
    """

    parts = [e.value for e in app.error]
    parts += [w.value for w in app.warning]
    parts += [i.value for i in app.info]
    parts += [c.value for c in app.caption]

    return "\n".join(parts)


# ============================================================
# It boots
# ============================================================

def test_app_starts_without_exceptions():

    app = AppTest.from_file(APP, default_timeout=300)

    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]

    assert any(b.label == "🚀 Run Fit" for b in app.button)


# ============================================================
# Presets
# ============================================================

def test_preset_writes_the_configuration_it_describes():
    """
    A preset is only useful if it actually sets what it says. This
    one is the BAO+BBN measurement, so it must tick DESI *and* the
    BBN prior, select DR2, turn on the computed sound horizon, and
    untick everything else.
    """

    app = _apply_preset(_fresh(), "DESI DR2 + BBN → H₀ without the CMB")

    assert not app.exception, [str(e.value) for e in app.exception]

    state = app.session_state

    assert state["ds_desi"] is True
    assert state["ds_omega_b"] is True
    assert state["compute_rd"] is True
    assert state["dsver_desi"] == "desi2025"

    # ...and left the rest alone rather than adding to whatever was
    # already ticked.
    assert state["ds_cc"] is False
    assert state["ds_pantheon"] is False


def test_applying_a_preset_does_not_loop():
    """
    The preset button writes widget state directly instead of
    calling ``st.rerun()``.

    It used to rerun, which is the obvious Streamlit idiom and is
    also unnecessary here -- the button sits above the widgets it
    writes to, so the values land on the same pass. It was worse
    than unnecessary: a button whose click state survives the rerun
    re-fires and the script never terminates. That is exactly what
    this test would hit, as a timeout.
    """

    app = _apply_preset(_fresh(timeout=60.0), "Late-time background (default)")

    assert not app.exception

    assert app.session_state["ds_cc"] is True
    assert app.session_state["ds_desi"] is True


# ============================================================
# A fit, end to end
# ============================================================

def test_fit_runs_and_renders_its_tables():
    """
    Widget → Fitter → rendered result, for the combination the
    dark-energy results are argued with.
    """

    app = _apply_preset(_fresh(timeout=600.0), "The Hubble tension, both sides")

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [e.value for e in app.error]

    frames = [f.value for f in app.dataframe]

    # The per-dataset chi2 breakdown -- the table that turns a total
    # chi2 into a statement about which dataset is in tension.
    breakdown = [f for f in frames if "dataset" in f.columns]

    assert breakdown, "no per-dataset chi2 table rendered"

    table = breakdown[0]

    assert set(table["dataset"]) == {"DESI", "Planck", "H0"}

    assert (table["χ²"] >= 0).all()

    # The local H0 measurement disagrees with CMB-anchored data at
    # ~5 sigma, so it must dominate the breakdown on a single point.
    h0_row = table[table["dataset"] == "H0"].iloc[0]

    assert h0_row["N"] == 1
    assert h0_row["χ²"] > 9.0, (
        "the H0 dataset should carry a large chi2 here -- if it does "
        "not, the breakdown is not being evaluated at the best fit"
    )


def test_derived_quantities_are_rendered():
    """
    z_t, q0 and r_d were reachable only from Python before; they are
    posteriors, not best-fit values, and the app now shows them.
    """

    app = _apply_preset(_fresh(timeout=600.0), "Late-time background (default)")

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]

    frames = [f.value for f in app.dataframe]

    derived = [f for f in frames if "quantity" in f.columns]

    assert derived, "no derived-quantities table rendered"

    labels = " ".join(derived[0]["quantity"])

    assert "q₀" in labels
    assert "z_t" in labels
    assert "r_d" in labels


# ============================================================
# The guidance itself
# ============================================================

def test_parameter_table_hides_what_the_fit_does_not_use():
    """
    The shared parameter container carries every parameter any model
    needs -- twenty-odd of them. For an LCDM fit against CC and BAO,
    all but a handful are inert, and showing them does not offer
    flexibility, it hides which ones matter.
    """

    app = _apply_preset(_fresh(), "Late-time background (default)")

    shown = set(_parameter_table(app)["Parameter"])

    # Needed: the two free ones, curvature, and the BAO sound horizon.
    assert {"H0", "Omega_m", "Omega_k", "rd"} <= shown

    # Not needed: no supernovae, no growth data, no CMB spectra.
    assert "sigma8" not in shown
    assert "tau_reio" not in shown
    assert "n_s" not in shown


def test_parameter_table_follows_the_datasets():
    """
    Relevance is a property of the *fit*, not the model: ticking a
    growth dataset must make sigma8 appear for the same LCDM.
    """

    app = _apply_preset(_fresh(), "Growth of structure (tests modified gravity)")

    shown = set(_parameter_table(app)["Parameter"])

    assert "sigma8" in shown


def test_incompatible_datasets_are_flagged():

    app = _fresh()

    app.session_state["ds_pantheon"] = True
    app.session_state["ds_des_sn5yr"] = True

    app.run()

    assert "double-count" in _messages(app).lower()


def test_model_that_cannot_do_cmb_spectra_says_so_before_running():
    """
    A modified-Friedmann model with the full CMB spectra raises
    inside ``Fitter``. Finding that out from a stack trace after
    clicking Run is not acceptable when it can be known from the two
    dropdowns.
    """

    app = _apply_preset(_fresh(), "Full CMB from scratch (slow)")

    app = _select_model(app, "DGP")

    messages = _messages(app)

    assert "cannot be used with the full CMB spectra" in messages


def test_modified_gravity_without_growth_data_is_flagged():
    """
    f(R) Hu-Sawicki's background *is* LCDM's by construction, so a
    background-only fit of it constrains nothing about the model --
    which the app has to say, because the fit will otherwise run,
    converge, and produce a posterior.
    """

    app = _apply_preset(_fresh(), "Late-time background (default)")

    app = _select_model(app, "FRHuSawicki")

    messages = _messages(app)

    assert "growth" in messages.lower()
    assert "f_R0" in messages


def test_unfitted_extra_parameters_are_flagged():
    """
    A model whose own parameters are left fixed is a fit of whatever
    it reduces to at those values -- not of the model.
    """

    app = _select_model(_apply_preset(_fresh(), "Late-time background (default)"), "GEDE")

    messages = _messages(app)

    assert "Delta" in messages
    assert "left fixed" in messages


def test_guide_lists_every_dataset_and_model():
    """
    The reference tables are generated from the same dictionaries the
    widgets use, so a dataset or model added to the library without a
    note would show up here as a blank row rather than silently.
    """

    app = _fresh(timeout=60.0)

    frames = [f.value for f in app.dataframe]

    dataset_guide = [f for f in frames if "Measures" in f.columns]
    model_guide = [f for f in frames if "Reduces to" in f.columns]

    assert dataset_guide, "dataset guide table missing"
    assert model_guide, "model guide table missing"

    from CosmoFit.stats.fitter import DATASET_REGISTRY

    assert len(dataset_guide[0]) == len(DATASET_REGISTRY)

    assert (dataset_guide[0]["Measures"].str.len() > 0).all()
    assert (dataset_guide[0]["Reference"].str.len() > 0).all()

    from CosmoFit.cosmology.models import __all__ as model_names

    assert len(model_guide[0]) == len(model_names)

    assert (model_guide[0]["Reference"].str.len() > 0).all()


# ============================================================
# Models from an action
# ============================================================
#
# The GUI's second route to a model the library does not ship. The
# first (`Custom`) takes an `E(z)` -- the result of a derivation
# somebody did by hand. This one takes the input instead, and the
# whole point is that the user never writes `E(z)` at all.

requires_sympy = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sympy") is None,
    reason="sympy not installed (optional 'theory' extra)",
)


def _select_action(app: AppTest) -> AppTest:

    return _select_model(app, "From an action")


def _load_action_preset(app: AppTest, name: str) -> AppTest:

    picker = [
        s for s in app.selectbox if s.label == "Start from a worked example"
    ][0]

    picker.set_value(name).run()

    button = [b for b in app.button if b.label == "Load this action"][0]

    return button.click().run()


@requires_sympy
def test_the_action_route_is_offered():

    app = _fresh()

    picker = [s for s in app.selectbox if s.label == "Cosmology"][0]

    assert any(o.startswith("From an action ") for o in picker.options)

    app = _select_action(app)

    assert not app.exception, [str(e.value) for e in app.exception]

    labels = [t.label for t in app.text_input] + [
        t.label for t in app.text_area
    ]

    assert "Gravitational Lagrangian  f" in labels
    assert "Parameters (one per line)" in labels


@requires_sympy
def test_a_preset_writes_the_action_it_names():
    """
    Same contract as the dataset presets: it has to set what it says,
    and it has to leave the page in a state that builds.
    """

    app = _load_action_preset(
        _select_action(_fresh()),
        "Power-law f(T)  ·  Bengochea & Ferraro (2009)",
    )

    assert not app.exception, [str(e.value) for e in app.exception]

    state = app.session_state

    assert state["action_gravity_0"] == "T + A0*(-T)**b"
    assert state["action_geometry_0"] == "teleparallel"
    assert state["action_closure_0"] == "A0"
    assert state["action_growth_0"] == "quasi_static"
    assert "A0 =" in state["action_params_0"]
    assert "b =" in state["action_params_0"]


@requires_sympy
def test_an_empty_action_says_what_is_missing_rather_than_crashing():
    """
    The dropdown can be selected before anything has been typed into
    it, which is the state every user is in for the first second.
    """

    app = _select_action(_fresh())

    assert not app.exception, [str(e.value) for e in app.exception]

    assert "gravitational Lagrangian" in _messages(app)


@requires_sympy
def test_the_derived_friedmann_equation_is_shown():
    """
    The reason to give an action rather than an `E(z)`: the equation
    is derived for you, and seeing it is how you check the library
    understood the Lagrangian you meant.
    """

    app = _load_action_preset(
        _select_action(_fresh()),
        "Exponential f(Q)  ·  reproduces FQExponential",
    )

    assert not app.exception, [str(e.value) for e in app.exception]

    rendered = "\n".join(latex.value for latex in app.latex)

    # The transcendental f(Q) constraint carries an exponential of
    # lambda over E^2; a background-only expansion would not.
    assert "lam" in rendered
    assert "E_{2}" in rendered or "E2" in rendered


@requires_sympy
def test_a_model_derived_from_an_action_fits_end_to_end():
    """
    Widget → `Action` → `Cosmology` subclass → `Fitter` → rendered
    table, with no `E(z)` typed anywhere.

    ΛCDM from `R - 2*Lam` is the one to run: it is the cheapest
    action here, and the library already has a hand-written ΛCDM to
    be wrong against if the derivation goes astray.
    """

    app = _fresh(timeout=900.0)

    app = _apply_preset(app, "Late-time background (default)")

    app = _load_action_preset(
        _select_action(app), "General Relativity + Λ (rederives ΛCDM)",
    )

    assert not app.exception, [str(e.value) for e in app.exception]

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [e.value for e in app.error]

    frames = [f.value for f in app.dataframe]

    breakdown = [f for f in frames if "dataset" in f.columns]

    assert breakdown, "no per-dataset chi2 table rendered"

    assert (breakdown[0]["χ²"] >= 0).all()

    # A derived ΛCDM is still ΛCDM: the fit has to land somewhere a
    # background fit lands, not merely finish.
    fits = app.session_state["fits"]

    assert len(fits) == 1

    assert 60.0 < fits[0].result.best_fit.params["H0"] < 80.0

    assert 0.1 < fits[0].result.best_fit.params["Omega_m"] < 0.6


# ============================================================
# The inference tab
# ============================================================
#
# Profile likelihoods, Fisher matrices, Bayesian evidence and
# tension statistics all landed in the library three releases before
# the GUI had any of them. Each is behind its own button because two
# of them cost real time, which is also what makes them awkward to
# test: the button has to be found and clicked, and the thing it
# renders has to be checked, not merely the absence of an exception.


def _fitted(datasets_preset: str = "Late-time background (default)") -> AppTest:
    """A single ΛCDM fit, run, with the results tabs rendered."""

    app = _apply_preset(_fresh(timeout=900.0), datasets_preset)

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]

    return app


def _click(app: AppTest, label: str) -> AppTest:

    matches = [b for b in app.button if b.label == label]

    assert matches, f"no button labelled {label!r}"

    return matches[0].click().run()


def test_the_inference_tab_exists_after_a_fit():

    app = _fitted()

    names = [name for tab in app.tabs for name in [tab.label]]

    assert any("Inference" in str(name) for name in names), names


def test_the_fisher_matrix_is_computed_and_compared_to_the_chain():
    """
    The comparison is the point of showing it. A Fisher matrix is a
    Gaussian approximation to the posterior -- cheap where an MCMC is
    not, good for a near-elliptical posterior, poor against a prior
    edge -- and the only way to know which you have is the ratio.
    """

    app = _click(_fitted(), "Compute the Fisher matrix")

    assert not app.exception, [str(e.value) for e in app.exception]

    frames = [f.value for f in app.dataframe]

    fisher = [f for f in frames if "Fisher σ" in f.columns]

    assert fisher, "no Fisher table rendered"

    table = fisher[0]

    assert set(table["Parameter"]) == {"H0", "Omega_m"}

    assert (table["Fisher σ"] > 0).all()

    # The chain ran, so the comparison columns have to be there.
    assert "MCMC σ" in table.columns
    assert "ratio" in table.columns

    # Short chains are noisy, but not *that* noisy: an order of
    # magnitude apart would mean the two are not measuring the same
    # thing at all.
    assert ((table["ratio"] > 0.2) & (table["ratio"] < 5.0)).all()


def test_a_profile_likelihood_runs_and_reports_an_interval():
    """
    Widget → `Fitter.profile` → a Δχ² curve and the crossings that
    are the interval a profile actually reports. Not a standard
    deviation of anything, which is the reason it is offered
    separately from the posterior.
    """

    app = _fitted()

    points = [
        n for n in app.number_input if n.label == "Points"
    ]

    assert points, "the profile point count is missing"

    app = points[0].set_value(7).run()

    app = _click(app, "Profile `H0`")

    assert not app.exception, [str(e.value) for e in app.exception]

    assert not app.error, [e.value for e in app.error]

    # Either an interval or the explicit statement that there is not
    # one -- both are results, and silence would not be. (`AppTest`
    # has no accessor for a `st.pyplot` figure, so the verdict is
    # what there is to check.)
    messages = _messages(app) + "".join(m.value for m in app.success)

    assert "Δχ²" in messages, messages[-400:]


def test_tension_needs_two_posteriors_and_says_so():
    """
    With one model there is nothing to compare, and the tab has to
    say that rather than render an empty figure or raise.
    """

    app = _fitted()

    assert "Two fits with chains" in _messages(app)


def test_tension_between_two_models_reports_both_definitions():
    """
    The Gaussian number is the one everybody quotes; the
    sample-based one makes no Gaussian assumption. Showing both is
    the point -- they disagree exactly when the first has stopped
    meaning anything.
    """

    app = _apply_preset(_fresh(timeout=900.0), "Late-time background (default)")

    # A second model, so there are two posteriors to compare.
    add = [b for b in app.button if "Add" in b.label]

    assert add, [b.label for b in app.button]

    app = add[0].click().run()

    pickers = [s for s in app.selectbox if s.label == "Cosmology"]

    assert len(pickers) == 2, "second model slot did not appear"

    option = next(o for o in pickers[1].options if o.startswith("WCDM "))

    app = pickers[1].set_value(option).run()

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]

    labels = [m.label for m in app.metric]

    assert "Gaussian" in labels
    assert "Sample-based" in labels

    values = {m.label: m.value for m in app.metric}

    for key in ("Gaussian", "Sample-based"):
        assert values[key].endswith("σ"), values[key]
        assert float(values[key].removesuffix("σ")) >= 0.0


def test_best_fit_restarts_are_offered_and_used():
    """
    `restarts` exists because a model once fit *worse* than the one
    it contains as a special case -- an impossible answer, and the
    optimizer converging into the wrong basin. It was reachable only
    from Python.
    """

    app = _fresh()

    restarts = [
        n for n in app.number_input if n.label == "Best-fit restarts"
    ]

    assert restarts, "the restarts control is missing"

    assert restarts[0].value == 0

    app = restarts[0].set_value(1).run()

    app = _apply_preset(app, "Late-time background (default)")

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]

    assert app.session_state["fits"][0].result.best_fit is not None


# ============================================================
# The library's own guards, made visible
# ============================================================


def test_a_derived_parameter_is_not_ticked_by_default():
    """
    ADE fixes `Omega_m` from its early-time condition rather than
    accepting it, so sampling it returns the prior -- and would look
    exactly like a measurement.

    `Fitter` warns about that, but a default that starts in the
    state the library warns about is a poor default. The table has
    to arrive with the box already clear.
    """

    app = _select_model(_fresh(), "ADE")

    assert not app.exception, [str(e.value) for e in app.exception]

    table = _parameter_table(app)

    row = table[table["Parameter"] == "Omega_m"]

    assert not row.empty, "Omega_m missing from the parameter table"

    assert not bool(row.iloc[0]["Free"]), (
        "ADE derives Omega_m -- it must not be ticked by default"
    )

    # H0 is not derived, so it is still ticked.
    assert bool(table[table["Parameter"] == "H0"].iloc[0]["Free"])

    assert "Derived, not fitted" in _messages(app)


def test_a_library_warning_reaches_the_page():
    """
    `Fitter` warns through `warnings`, which in a Streamlit app goes
    to stderr -- which is to say nowhere. The run loop catches them
    and renders them, so every guard the library has, and every one
    it grows later, is visible without the GUI restating any of it.

    Driven with a conflicting dataset pair because that is the one
    `Fitter` warning reachable from the widgets alone. The page also
    flags conflicts *before* a run, from its own table, so what is
    asserted here is the caught-and-rendered form specifically --
    prefixed with the model it came from, which only the run loop
    produces.
    """

    app = _fresh(timeout=900.0)

    for key in ("ds_sdss_bao", "ds_fsigma8"):
        for checkbox in app.checkbox:
            if checkbox.key == key:
                app = checkbox.set_value(True).run()
                break

    for key in ("ds_desi", "ds_pantheon", "ds_planck"):
        for checkbox in app.checkbox:
            if checkbox.key == key and checkbox.value:
                app = checkbox.set_value(False).run()
                break

    app = [b for b in app.button if b.label == "🚀 Run Fit"][0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]

    shown = "\n".join(w.value for w in app.warning)

    assert "**Model 1:**" in shown, shown

    assert "sdss_bao" in shown and "fsigma8" in shown


# ============================================================
# The admissibility gate, and the second integration direction
# ============================================================
#
# Both were added to the library after this app was written, so
# neither was reachable from it: a general f(R) could only be
# integrated backwards, which the whole "disappearing cosmological
# constant" family cannot survive, and nothing showed whether a
# fitted model was a theory worth having fitted.


def test_the_integration_direction_is_offered():
    """
    Without this control the GUI can only build the f(R) models
    that integrate backwards -- which excludes Hu-Sawicki,
    Starobinsky 2007, Tsujikawa and the arctan family.
    """

    app = _select_action(_fresh())

    pickers = [
        s for s in app.selectbox
        if s.label == "f(R) integration direction"
    ]

    assert pickers, "the direction control is missing"

    assert set(pickers[0].options) == {"backward", "forward"}

    assert pickers[0].value == "backward", (
        "backward has to stay the default: it is what every model "
        "already in the library uses, and forward additionally "
        "requires a closure parameter"
    )


def test_choosing_forward_reaches_the_builder():
    """
    A control that is displayed but not wired is worse than no
    control. Setting it must change what the app builds, which is
    checked by the app running clean afterwards rather than raising
    a signature mismatch between the widget tuple and the builder.
    """

    app = _select_action(_fresh())

    picker = [
        s for s in app.selectbox
        if s.label == "f(R) integration direction"
    ][0]

    app = picker.set_value("forward").run()

    assert not app.exception, [str(e.value) for e in app.exception]


def test_a_model_with_no_gate_shows_no_verdict():
    """
    Most of the library is dark energy on top of General
    Relativity, where neither question arises. Showing an empty
    panel there would be noise.
    """

    from CosmoFit import LCDM

    assert not hasattr(LCDM, "viability")

    assert not hasattr(LCDM, "screening")


def test_the_gate_asks_the_two_questions_separately():
    """
    The panel is driven by the model's own methods, so this checks
    the pair it renders rather than the pixels: a model can be a
    consistent theory and still be excluded by local tests, and
    `FRHuSawicki` at a large enough amplitude is exactly that.
    """

    from CosmoFit import FRHuSawicki

    model = FRHuSawicki(
        FRHuSawicki.PARAMS_CLASS(H0=70.0, Omega_m=0.3, f_R0=-1e-2, n=1.0)
    )

    assert model.viability()["ok"], "the theory itself is fine"

    screening = model.screening()

    assert not screening["ok"], "and it is excluded anyway"

    assert screening["deviation"] / screening["bound"] > 1e3
