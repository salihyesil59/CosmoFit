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
    Every error, warning and caption the page is currently showing,
    as one searchable string.
    """

    parts = [e.value for e in app.error]
    parts += [w.value for w in app.warning]
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
