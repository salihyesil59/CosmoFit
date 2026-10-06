"""
Drawing a finished fit: the best fit, the convergence gate, the
posterior, profiles, Fisher errors, evidence and tension, and the
saved configuration and its equivalent script.
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

from cosmofit import (
    __version__,
    Fitter,
    CCLikelihood,
    DESILikelihood,
    SDSSBAOLikelihood,
    BAOLowZLikelihood,
    PantheonLikelihood,
    DESSN5YRLikelihood,
    Union3Likelihood,
    PlanckLikelihood,
    PlanckLiteLikelihood,
    PlanckLensingLikelihood,
    ACTDR6LensingLikelihood,
    FSigma8Likelihood,
    EBOSSELGLikelihood,
    EBOSSLyaLikelihood,
)
from cosmofit.stats import cpl_diagnostics
from cosmofit.compat import evidence_on_core, fisher_on_core, profile_on_core

from cosmofit.gui.reference import (
    PLOT_EXPORT_FORMATS,
)


# ------------------------------------------------------------

def _fit_labels(fits: list[Fitter]) -> tuple[list[str], list[str]]:
    """
    Two index-aligned label lists for a set of fits: plain text for
    the UI (dropdowns, tables, JSON keys) and LaTeX for figure
    legends.

    Both are needed because they go to renderers with different
    abilities. Streamlit shows a string literally, so a selectbox
    option must read ``wCDM``, not ``$w$CDM``; matplotlib renders
    ``$...$`` as mathtext, so a legend should read ``ΛCDM`` rather
    than the ASCII spelling of the class name. Disambiguating
    suffixes (two fits of the same model) are applied to the plain
    names and then copied onto the LaTeX ones, so the two lists
    never disagree about which fit is "(2)".
    """

    plain = _dedupe_labels([f.model_cls.plain_name() for f in fits])

    latex = [
        f.model_cls.plot_label() + label[len(f.model_cls.plain_name()):]
        for f, label in zip(fits, plain)
    ]

    return plain, latex


# ------------------------------------------------------------

def _dedupe_labels(labels: list[str]) -> list[str]:
    """``["CPL", "CPL"]`` -> ``["CPL (1)", "CPL (2)"]``; leaves
    already-unique labels untouched."""

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    seen = {}
    out = []
    for label in labels:
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} ({seen[label]})")

    return out


# ------------------------------------------------------------

def _available_plots(fit: Fitter) -> list[str]:
    """Which single-model ``fit.plots.<name>()`` methods apply."""

    methods = []

    if fit.sampler is not None:
        methods += ["chain", "corner"]

    likelihood_plots = [
        (PantheonLikelihood, "hubble_diagram"),
        (DESSN5YRLikelihood, "des_hubble_diagram"),
        (Union3Likelihood, "union3_hubble_diagram"),
        (CCLikelihood, "hz"),
        (DESILikelihood, "bao_distances"),
        (SDSSBAOLikelihood, "sdss_bao_distances"),
        (BAOLowZLikelihood, "lowz_bao_distances"),
        (PlanckLikelihood, "planck_residuals"),
        (PlanckLiteLikelihood, "cmb_spectra"),
        (PlanckLensingLikelihood, "cmb_lensing"),
        (ACTDR6LensingLikelihood, "cmb_lensing"),
        (FSigma8Likelihood, "growth"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in fit.likelihoods):
            methods.append(method)

    # The two released likelihood *surfaces*, which go down different
    # branches of the same method -- a 1-D curve and a 2-D contour
    # set. Worth offering because they are the figure that shows why
    # a Gaussian summary of these two would be wrong.
    for cls in (EBOSSELGLikelihood, EBOSSLyaLikelihood):
        if any(isinstance(lk, cls) for lk in fit.likelihoods):
            methods.append("eboss_surface")
            break

    if hasattr(fit.cosmology, "w"):
        methods.append("w_of_z")

    # The w0-wa plane is a 2D posterior, so it needs both parameters
    # sampled -- not just present in the model.
    if fit.sampler is not None and {"w0", "wa"} <= set(fit.free_params):
        methods.append("w0_wa_plane")

    methods.append("deceleration")

    return methods


# ------------------------------------------------------------

def _available_compare_plots(fits: list[Fitter]) -> list[str]:
    """
    Which ``compare_*`` methods apply, based on the anchor (first)
    fit's datasets -- same logic as `_available_plots`, mapped to
    the comparison-plot names. `compare_w_of_z`/`compare_deceleration`
    are always valid (every model has an E(z), and models without
    their own w(z) fall back to the w=-1 line -- see
    `FitPlotter.compare_w_of_z`).

    `compare_w0_wa_plane` is the exception that needs *every* fit,
    not just the anchor: it overlays one posterior per model, so a
    single model without w0/wa free has nothing to contribute to it.
    """

    anchor_fit = fits[0]

    methods = []

    likelihood_plots = [
        (PantheonLikelihood, "compare_hubble_diagram"),
        (DESSN5YRLikelihood, "compare_des_hubble_diagram"),
        (CCLikelihood, "compare_hz"),
        (DESILikelihood, "compare_bao_distances"),
        (SDSSBAOLikelihood, "compare_sdss_bao_distances"),
        (FSigma8Likelihood, "compare_growth"),
    ]

    for cls, method in likelihood_plots:
        if any(isinstance(lk, cls) for lk in anchor_fit.likelihoods):
            methods.append(method)

    methods.append("compare_w_of_z")

    if all(
        fit.sampler is not None and {"w0", "wa"} <= set(fit.free_params)
        for fit in fits
    ):
        methods.append("compare_w0_wa_plane")

    methods.append("compare_deceleration")

    return methods


# ------------------------------------------------------------

def _render_best_fit(fit: Fitter) -> None:

    result = fit.result

    if result.best_fit is None:
        st.caption("No best-fit result.")
        return

    chi2 = result.best_fit.chi2
    dof = fit.n_data - result.best_fit.ndim

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("χ²", f"{chi2:.2f}")
    m2.metric(
        "χ²/dof", f"{chi2 / dof:.3f}" if dof > 0 else "—",
        help=f"{fit.n_data} data points − {result.best_fit.ndim} free "
             f"parameters = {dof} degrees of freedom. Around 1 is a "
             f"good fit; well above 1 means the model cannot describe "
             f"the data, well below usually means the error bars are "
             f"conservative.",
    )
    m3.metric(
        "AIC", f"{result.best_fit.aic():.2f}",
        help="χ² + 2k. Lower is better; a difference below ~2 is "
             "not evidence either way.",
    )
    m4.metric(
        "BIC", f"{result.best_fit.bic():.2f}",
        help="χ² + k·ln(n). Penalizes extra parameters harder than "
             "AIC does, so it favours simpler models more strongly.",
    )

    st.dataframe(
        pd.DataFrame(
            {"parameter": list(result.best_fit.params),
             "value": list(result.best_fit.params.values())}
        ),
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # Which dataset is the fit actually struggling with?
    # --------------------------------------------------------
    #
    # A single total chi2 says a fit is bad without saying where.
    # The per-dataset breakdown is what turns "chi2 = 640" into
    # "the local H0 measurement is contributing 23 of it, on one
    # data point" -- which is the whole content of a tension.

    st.markdown("**χ² by dataset**")

    rows = []
    for likelihood in fit.likelihoods:
        summary = likelihood.summary()
        n = summary["n_data"]
        rows.append({
            "dataset": summary["name"],
            "N": n,
            "χ²": summary["chi2"],
            "χ²/N": summary["chi2"] / n if n else None,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "χ²": st.column_config.NumberColumn(format="%.2f"),
            "χ²/N": st.column_config.ProgressColumn(
                "χ²/N", format="%.2f", min_value=0.0, max_value=4.0,
                help="Per-point χ². A dataset sitting far above the "
                     "others is the one in tension with the rest.",
            ),
        },
    )

    st.caption(
        "Evaluated at the best-fit point. These sum to the total χ² "
        "above; a single dataset carrying a disproportionate share is "
        "where a tension lives."
    )

    _render_gate(fit)


# ------------------------------------------------------------

#: Session-state keys that describe *what to run*, as opposed to
#: what was run or which tab is open. Saved and restored by prefix
#: rather than by an exhaustive list, so a widget added later to an
#: existing family is carried without anyone remembering to come
#: back here.
#:
#: Deliberately excluded: `dl_*` (download button state),
#: `*_model_choice` and `*_multiselect` (which result is being
#: looked at), and `profile_run_`/`fisher_run_` (whether an
#: expensive extra has been asked for). Restoring those would
#: reproduce someone else's *view* rather than their configuration.
_CONFIG_PREFIXES = (
    "ds_", "dsver_", "compute_rd", "derive_sigma",
    "model_choice_", "n_models",
    "action_", "custom_",
    "mcmc_", "param_editor_", "relevant_only_",
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

    return applied


# ------------------------------------------------------------

def _equivalent_script(
    model_choice: str,
    model_names: list[str],
    action_specs: list[dict | None],
    datasets: list[str],
    dataset_versions: dict,
    free_params: list[str],
    initial: dict,
    bounds: dict,
    compute_rd: bool,
    derive_sigma8: bool,
    sampler: str,
    sampler_options: dict,
    burn_in: float,
    seed: int,
) -> str:
    """
    The Python that reproduces what the page is configured to do.

    This library is a Python library; the app is a way into it, not
    a replacement for it. Anyone who explores here and then wants
    the fit in a notebook, in a batch job, or under version control
    has otherwise to reconstruct the `Fitter` call by reading the
    widgets back -- which is exactly the sort of transcription that
    goes wrong quietly.

    Built from the same values the run uses rather than from a
    template, so a snippet that disagrees with the run is a bug
    here rather than a difference the reader has to notice.
    """

    lines = ["from cosmofit import Fitter", "from cosmofit.core import run"]

    action_specs = [spec for spec in action_specs if spec]

    if action_specs:
        lines.append("from cosmofit.theory import Action")
    elif model_names:
        lines.append(f"from cosmofit import {', '.join(sorted(set(model_names)))}")

    lines.append("")

    for index, spec in enumerate(action_specs):

        arguments = [repr(spec["gravity"])]

        for key in (
            "geometry", "fluids", "params", "closure",
            "growth", "fields", "background",
        ):
            value = spec.get(key)

            if value in (None, "", (), {}, "gr", "backward"):
                continue

            arguments.append(f"{key}={value!r}")

        lines.append(f"model_{index + 1} = Action(")
        lines.extend(f"    {argument}," for argument in arguments)
        lines.append(f").build({spec['name']!r})")
        lines.append("")

    versions = {
        key: {"version": version}
        for key, version in (dataset_versions or {}).items()
        if key in datasets
    }

    for index, name in enumerate(model_names or ["model"]):

        target = (
            f"model_{index + 1}"
            if index < len(action_specs)
            else name
        )

        lines.append("fit = Fitter(")
        lines.append(f"    model={target},")
        lines.append(f"    datasets={datasets!r},")
        lines.append(f"    free_params={free_params!r},")
        lines.append(f"    initial={initial!r},")

        if bounds:
            lines.append(f"    bounds={bounds!r},")

        if versions:
            lines.append(f"    dataset_kwargs={versions!r},")

        if compute_rd:
            lines.append("    compute_rd=True,")

        if derive_sigma8:
            lines.append("    derive_sigma8=True,")

        lines.append(")")
        lines.append("")
        lines.append("# The same fit as an input for the 2.0 core, which samples it.")
        lines.append("info = fit.to_info(exact=True)")
        lines.append(f"info['sampler'] = {{{sampler!r}: {dict(sampler_options, seed=seed)!r}}}")
        lines.append("info['output'] = 'chains/cosmofit_run'")
        lines.append("")
        lines.append("_, sampler = run(info)")
        lines.append("")

        if sampler == "mcmc":
            lines.append(f"products = sampler.products(skip={burn_in!r})")
            lines.append(
                "print('R - 1 =', products['Rminus1'], "
                "'converged:', products['converged'])"
            )
        else:
            lines.append("products = sampler.products()")
            lines.append(
                "print('tau =', products['tau'], 'converged:', products['converged'])"
            )

        # One model is the common case and reads best as a script;
        # several would need a loop, and guessing how someone wants
        # to hold them is worse than showing the first and saying so.
        if len(model_names) > 1:
            lines.append("")
            lines.append(
                f"# Model 1 of {len(model_names)}. The others differ "
                f"only in `model=`."
            )
        break

    return "\n".join(lines)


def _equivalent_input(fit: Fitter, sampler: str, sampler_options: dict, seed: int):
    """
    The fit as a YAML input for ``cosmofit run`` -- or ``None`` for a
    model built in this session, which a file cannot name.
    """

    import yaml

    from cosmofit.compat import _model_reference

    if not isinstance(_model_reference(fit.model_cls), str):
        return None

    info = fit.to_info(exact=True)
    info["sampler"] = {sampler: dict(sampler_options, seed=int(seed))}
    info["output"] = "chains/cosmofit_run"

    return yaml.safe_dump(info, sort_keys=False)


# ------------------------------------------------------------

def _render_gate(fit: Fitter) -> None:
    """
    Whether the fitted model is a theory worth having fitted.

    Two questions, kept apart because they have different answers.
    ``viability()`` asks whether the theory is *consistent* -- no
    ghost graviton, no tachyonic scalaron, a positive effective
    coupling. ``screening()`` asks whether it is *allowed*, which
    is a statement about local gravity tests and not about the
    theory's health. A model can pass one and fail the other, and
    the arctan f(R) does exactly that: it fits as well as LCDM and
    is excluded by the Solar System by four orders of magnitude.

    Only shown for models that answer -- most of the library is
    dark energy on top of General Relativity, where neither
    question arises.
    """

    cosmology = getattr(fit, "cosmology", None)

    has_viability = hasattr(cosmology, "viability")
    has_screening = hasattr(cosmology, "screening")

    if not (has_viability or has_screening):
        return

    st.markdown("#### Is this a theory worth fitting?")

    columns = st.columns(2)

    with columns[0]:

        if has_viability:
            try:
                verdict = cosmology.viability()

                if verdict["ok"]:
                    st.success("**Consistent** — no ghost, no tachyon.")
                else:
                    st.error(
                        "**Not consistent.** "
                        + " ".join(verdict["reasons"])
                    )

            except Exception as exc:
                st.warning(f"Could not check consistency: {exc}")

    with columns[1]:

        if has_screening:
            try:
                verdict = cosmology.screening()

                over = verdict["deviation"] / verdict["bound"]

                if verdict["ok"]:
                    st.success(
                        f"**Allowed by local tests** — "
                        f"|f_R0| = {verdict['deviation']:.3g}, "
                        f"within the {verdict['bound']:.0e} bound."
                    )
                else:
                    st.error(
                        f"**Excluded by local tests** — "
                        f"|f_R0| = {verdict['deviation']:.3g}, "
                        f"{over:.3g}× the Solar System bound of "
                        f"{verdict['bound']:.0e}."
                    )

            except Exception as exc:
                st.warning(f"Could not check screening: {exc}")

    st.caption(
        "Consistency and exclusion are different questions and are "
        "asked separately, because a model can pass one and fail the "
        "other. The screening number is the *linear* estimate: "
        "chameleon screening is non-linear, so failing it means "
        "\"excluded unless screening rescues it\" rather than a proof."
    )


# ------------------------------------------------------------

def _render_posterior(fit: Fitter) -> None:

    result = fit.result

    if result.mcmc is None:
        st.caption("No MCMC run.")
        return

    st.dataframe(
        pd.DataFrame([
            {"parameter": name, "median": s["median"],
             "+": s["plus"], "-": s["minus"]}
            for name, s in result.mcmc.summary.items()
        ]),
        hide_index=True,
        width="stretch",
    )

    convergence = result.mcmc.convergence
    rule = convergence.get("stopping_rule")
    saved = getattr(fit.sampler, "output", None)

    if rule is None:
        reached = "chain length exceeds 50x the autocorrelation time"
        missed = "the chain is shorter than 50 autocorrelation times"
    elif rule["target"] is not None:
        reached = f"R − 1 = {rule['value']:.3g}, below {rule['target']:g}"
        missed = f"R − 1 = {rule['value']:.3g}, still above {rule['target']:g}"
    else:
        reached = f"{rule['rule']}, with τ = {rule['value']:.3g}"
        missed = f"τ = {rule['value']:.3g} is not settled at {rule['rule']}"

    if convergence["converged"]:
        st.success(f"Converged -- {reached}.", icon="✅")
    elif saved:
        st.warning(
            f"Not converged yet -- {missed}. Raise Max steps and run "
            f"again before trusting this posterior: the saved chains "
            f"continue from where they stopped.", icon="⚠️",
        )
    else:
        st.warning(
            f"Not converged yet -- {missed}. Raise Max steps before "
            f"trusting this posterior.", icon="⚠️",
        )

    if saved:
        st.caption(
            f"Chains saved under `{saved}` in getdist's format "
            f"({result.mcmc.nsteps} steps x {result.mcmc.nwalkers} "
            f"{'chains' if rule and rule['target'] is not None else 'walkers'})."
        )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------
    #
    # `stats.derived` pushes every posterior sample back through the
    # model's own E(z), so these carry real error bars rather than
    # being evaluated once at the best fit. Nothing in the GUI
    # surfaced them before, which meant the acceleration transition
    # redshift -- a headline number in most dark-energy papers --
    # was reachable only from Python.

    with st.expander("Derived quantities (z_t, q₀, r_d)", expanded=False):

        st.caption(
            "Each posterior sample is pushed back through this "
            "model's own E(z) and dE/dz, so these are proper "
            "posteriors, not the best-fit value with no uncertainty."
        )

        try:
            from cosmofit.stats import derived

            rows = []

            q0 = derived.summarize(derived.deceleration_today(fit))
            rows.append({
                "quantity": "q₀ (deceleration today)",
                "median": q0["median"],
                "+": q0["plus"], "−": q0["minus"],
            })

            z_t = derived.summarize(derived.transition_redshift(fit))
            rows.append({
                "quantity": "z_t (acceleration begins)",
                "median": z_t["median"],
                "+": z_t["plus"], "−": z_t["minus"],
            })

            r_d = derived.summarize(derived.sound_horizon(fit))
            rows.append({
                "quantity": "r_d [Mpc], from the densities",
                "median": r_d["median"],
                "+": r_d["plus"], "−": r_d["minus"],
            })

            st.dataframe(
                pd.DataFrame(rows), hide_index=True,
                width="stretch",
                column_config={
                    "median": st.column_config.NumberColumn(format="%.4g"),
                    "+": st.column_config.NumberColumn(format="%.3g"),
                    "−": st.column_config.NumberColumn(format="%.3g"),
                },
            )

            if q0["median"] < 0:
                st.caption(
                    f"q₀ < 0: the expansion is accelerating today, and "
                    f"began doing so at z ≈ {z_t['median']:.2f}."
                )
            else:
                st.caption(
                    "q₀ > 0: this fit does **not** have an "
                    "accelerating universe today."
                )

            if z_t.get("n_undefined"):
                st.caption(
                    f"{z_t['n_undefined']} sample(s) never cross "
                    f"q = 0 in the search range and are excluded."
                )

            if not fit.compute_rd:
                st.caption(
                    f"r_d here is what the early-universe physics "
                    f"*predicts* for these densities; the fit used "
                    f"the free parameter "
                    f"({fit.result.mcmc.summary['rd']['median']:.2f} Mpc) "
                    f"instead. A disagreement between the two is the "
                    f"standard signature of new physics before "
                    f"recombination."
                    if "rd" in fit.free_params else
                    "r_d here is what the early-universe physics "
                    "predicts for these densities; the fit used the "
                    "fixed `rd` value instead."
                )

        except Exception as exc:
            st.caption(f"Could not compute derived quantities: {exc}")

    # CPL-family diagnostics (w(z)=-1 crossing redshift, direction,
    # distance from the LCDM point) -- only meaningful when both w0
    # and wa were actually fit.
    if "w0" in fit.free_params and "wa" in fit.free_params:

        with st.expander("w0-wa posterior diagnostics"):

            samples = fit.samples_dict()

            z_cross, frac_cross = cpl_diagnostics.crossing_redshift(
                samples["w0"], samples["wa"],
            )

            st.write(
                f"**w(z) = -1 crossing:** {frac_cross:.1%} of samples "
                f"cross in z in [0, 2.5]"
                + (
                    f", at z = {z_cross.mean():.2f} (mean) if they do"
                    if len(z_cross) else ""
                )
            )

            if len(z_cross) > 0:
                direction = cpl_diagnostics.crossing_direction(
                    samples["w0"], samples["wa"],
                )
                st.write(
                    f"Quintessence → phantom: "
                    f"{direction['quintessence_to_phantom']:.1%}  ·  "
                    f"Phantom → quintessence: "
                    f"{direction['phantom_to_quintessence']:.1%}"
                )

            regions = cpl_diagnostics.region_fractions(
                samples["w0"], samples["wa"],
            )
            st.write(
                "**Dark-energy region** (see the w0-wa plane plot): "
                + "  ·  ".join(
                    f"{name}: {fraction:.1%}"
                    for name, fraction in regions.items()
                )
            )

            lcdm_distance = cpl_diagnostics.mahalanobis_from_lcdm(
                samples["w0"], samples["wa"],
            )
            # Report `sigma`, not `distance`: the Mahalanobis
            # distance D is not a number of sigma in 2D (D^2 follows
            # chi2 with 2 d.o.f.), and quoting it as one overstates
            # the tension -- see `mahalanobis_from_lcdm`.
            st.write(
                f"**LCDM point** (w0, wa) = (-1, 0) is excluded at "
                f"**{lcdm_distance['sigma']:.2f}σ** "
                f"({lcdm_distance['confidence_level']:.1%} confidence; "
                f"Mahalanobis distance D = "
                f"{lcdm_distance['distance']:.2f}, which is *not* a "
                f"number of sigma in 2D)."
            )


# ------------------------------------------------------------

def _render_profile(fit: Fitter, label: str) -> None:
    """
    Profile likelihood: chi2 minimized over every *other* free
    parameter, at each fixed value of one.

    The honest tool where Wilks' theorem does not apply -- a
    parameter against a prior edge, or a surface with a plateau,
    where the marginal posterior and the profile say different
    things and the marginal is the one that smooths a real feature
    away.
    """

    import numpy as np

    free = list(fit.free_params)

    if not free:
        st.info("This fit has no free parameters to profile.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        name = st.selectbox(
            "Parameter", options=free, key=f"profile_param_{label}",
        )

    index = free.index(name)

    centre = float(fit.result.best_fit.params[name])

    sigma = None

    if fit.result.mcmc is not None:
        entry = fit.result.mcmc.summary.get(name)
        if entry:
            sigma = 0.5 * (entry["plus"] + entry["minus"])

    if not sigma or not np.isfinite(sigma) or sigma <= 0.0:
        lo, hi = fit.prior.lower[index], fit.prior.upper[index]
        sigma = 0.05 * (hi - lo)

    with col2:
        width = st.number_input(
            "Half-width (σ)", min_value=1.0, max_value=8.0, value=3.0,
            step=0.5, key=f"profile_width_{label}",
            help="How far either side of the best fit to scan, in "
                 "units of this parameter's own posterior width.",
        )

    with col3:
        n_points = st.number_input(
            "Points", min_value=5, max_value=61, value=15, step=2,
            key=f"profile_points_{label}",
            help="Each point is a full re-minimization over every "
                 "other free parameter, so this is the cost.",
        )

    if not st.button(
        f"Profile `{name}`", key=f"profile_run_{label}", width="stretch",
    ):
        st.caption(
            f"{int(n_points)} re-minimizations over the other "
            f"{len(free) - 1} parameter(s). Each point warm-starts "
            f"from the previous one, so this is far cheaper than "
            f"{int(n_points)} cold fits."
        )
        return

    values = np.linspace(
        centre - width * sigma, centre + width * sigma, int(n_points),
    )

    # Stay inside the prior: a profile point outside it has an
    # infinite chi2 and tells you about the box, not the likelihood.
    values = values[
        (values >= fit.prior.lower[index]) & (values <= fit.prior.upper[index])
    ]

    if values.size < 3:
        st.warning(
            "That range falls almost entirely outside this "
            "parameter's prior. Widen the prior or narrow the scan.",
            icon="⚠️",
        )
        return

    with st.spinner(f"Profiling {name} at {values.size} points..."):
        profile = profile_on_core(fit, name, values)

    delta = np.asarray(profile["delta_chi2"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(profile["values"], delta, lw=1.8, marker="o", ms=3)

    ax.axhline(1.0, ls="--", lw=1.0, color="0.5")
    ax.axhline(4.0, ls=":", lw=1.0, color="0.7")
    ax.axvline(centre, ls="-", lw=1.0, color="0.8")

    ax.set_xlabel(name)
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # The Delta chi2 = 1 crossings, which is the interval a profile
    # actually reports -- not a standard deviation of anything.
    crossings = []

    for i in range(len(delta) - 1):
        if (delta[i] - 1.0) * (delta[i + 1] - 1.0) < 0.0:
            x0, x1 = profile["values"][i], profile["values"][i + 1]
            y0, y1 = delta[i], delta[i + 1]
            crossings.append(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))

    minimum = float(profile["values"][int(np.argmin(delta))])

    if len(crossings) == 2:
        st.success(
            f"**{name} = {minimum:.4g}**  "
            f"(−{minimum - crossings[0]:.3g} / +{crossings[1] - minimum:.3g})"
            f"  — from the Δχ² = 1 crossings",
            icon="📐",
        )
    else:
        st.info(
            "Δχ² does not cross 1 on both sides of this range, so "
            "there is no two-sided interval to quote. That is a "
            "result rather than a failure: it is what a parameter "
            "against a prior edge, or one the data barely constrain, "
            "looks like. Widen the scan to see which.",
            icon="ℹ️",
        )


# ------------------------------------------------------------

def _render_fisher(fit: Fitter, label: str) -> None:
    """
    Fisher errors, and -- where a chain exists -- the ratio to what
    the chain found.

    That ratio is the whole point of showing it here. A Fisher
    matrix is a *Gaussian approximation to the posterior*: cheap
    where an MCMC is not (~2n² evaluations against millions), good
    for a near-elliptical posterior, and poor for a parameter against
    a prior edge or a plateau. The only way to know which you have is
    to compare.
    """

    import numpy as np

    if not st.button(
        "Compute the Fisher matrix", key=f"fisher_run_{label}",
        width="stretch",
    ):
        st.caption(
            f"About {2 * fit.ndim ** 2} likelihood evaluations -- "
            f"seconds, against the chain above."
        )
        return

    with st.spinner("Differentiating chi2 at the best fit..."):
        fisher = fisher_on_core(fit)

    rows = []

    summary = fit.result.mcmc.summary if fit.result.mcmc else {}

    for name, value, error in zip(
        fisher["free_params"], fisher["theta"], fisher["errors"],
    ):
        row = {
            "Parameter": name,
            "Best fit": float(value),
            "Fisher σ": float(error),
        }

        entry = summary.get(name)

        if entry:
            mcmc_sigma = 0.5 * (entry["plus"] + entry["minus"])
            row["MCMC σ"] = float(mcmc_sigma)
            row["ratio"] = (
                float(error / mcmc_sigma) if mcmc_sigma else float("nan")
            )

        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if "ratio" in rows[0]:

        ratios = np.array([r["ratio"] for r in rows], dtype=float)

        worst = float(np.max(np.abs(np.log(ratios[np.isfinite(ratios)]))))

        if worst < 0.22:  # within ~25% either way
            st.success(
                "The Gaussian approximation holds here -- every error "
                "bar within ~25% of the chain's.",
                icon="✅",
            )
        else:
            st.warning(
                "At least one Fisher error bar differs from the "
                "chain's by more than 25%. That is the approximation "
                "failing rather than the chain being wrong: the "
                "posterior is not elliptical in that direction. "
                "Quote the chain, or a profile likelihood.",
                icon="⚠️",
            )


# ------------------------------------------------------------

def _render_evidence(fits: list[Fitter], labels: list[str]) -> None:
    """
    Bayesian evidence by nested sampling, and the Bayes factor
    between two models.

    Kept behind a button and a cost estimate because this is the one
    thing in the app that can run for minutes: nested sampling
    explores the whole prior volume rather than the posterior peak,
    which is exactly what makes it able to compare models at all.
    """

    from cosmofit.stats import evidence as evidence_mod

    try:
        import dynesty  # noqa: F401
    except ModuleNotFoundError:
        st.warning(
            "Nested sampling needs **dynesty**, an optional "
            "dependency:\n\n```\npip install 'cosmofit[evidence]'\n```",
            icon="📦",
        )
        return

    st.caption(
        "The evidence integrates the likelihood over the *prior*, so "
        "a Bayes factor is a statement about the priors as much as "
        "about the models -- an extra parameter that does nothing is "
        "penalised by the volume it was given. That is the Occam "
        "factor AIC and BIC only approximate, and it is why the three "
        "can disagree."
    )

    n_live = st.number_input(
        "Live points", min_value=100, max_value=2000, value=400, step=100,
        key="evidence_nlive",
        help="More is a tighter ln Z and a longer run. 400 is enough "
             "for the two- to four-parameter fits this app runs.",
    )

    if not st.button("Run nested sampling", width="stretch"):
        st.caption(
            f"Roughly 10⁴–10⁵ likelihood evaluations per model, "
            f"for {len(fits)} model(s). Minutes, not seconds."
        )
        return

    results = {}

    progress = st.progress(0.0, text="Sampling...")

    for i, (label, fit) in enumerate(zip(labels, fits)):

        progress.progress(
            i / len(fits), text=f"Nested sampling {label} ({i + 1}/{len(fits)})",
        )

        results[label] = evidence_on_core(fit, nlive=int(n_live))

    progress.empty()

    st.session_state["evidence_results"] = results

    st.dataframe(
        pd.DataFrame([
            {
                "model": label,
                "ln Z": result.log_evidence,
                "± ": result.log_evidence_error,
                "k": len(result.free_params),
                "evaluations": result.n_evaluations,
            }
            for label, result in results.items()
        ]),
        hide_index=True, width="stretch",
    )

    if len(results) >= 2:

        st.markdown("**Bayes factors**, against the first model:")

        null_label = labels[0]

        rows = []

        for label in labels[1:]:

            factor = evidence_mod.bayes_factor(
                results[label], results[null_label],
            )

            rows.append({
                "model": label,
                "vs": null_label,
                "ln B": factor["ln_B"],
                "±": factor["ln_B_error"],
                "verdict": factor["interpretation"],
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.caption(
            "Positive `ln B` favours the model in the first column. "
            "The labels are Kass & Raftery's."
        )


# ------------------------------------------------------------

def _render_tension(fits: list[Fitter], labels: list[str]) -> None:
    """
    How far apart two posteriors are, by two definitions that
    disagree exactly when it matters.

    The Gaussian one is the number everybody quotes. The
    sample-based one makes no Gaussian assumption -- it pairs the two
    sets of samples at random and asks where zero falls in the
    distribution of the difference -- so a skewed or double-peaked
    posterior, which is where "how many sigma" quietly stops meaning
    anything, still gets an honest answer.
    """

    import numpy as np

    from cosmofit.stats import tension as tension_mod

    with_chains = [
        (label, fit) for label, fit in zip(labels, fits)
        if fit.sampler is not None
    ]

    if len(with_chains) < 2:
        st.info(
            "Two fits with chains are needed to compare posteriors. "
            "Add a second model above.",
            icon="ℹ️",
        )
        return

    shared = set(with_chains[0][1].free_params)

    for _, fit in with_chains[1:]:
        shared &= set(fit.free_params)

    if not shared:
        st.info(
            "These fits share no free parameter, so there is nothing "
            "to compare them on.",
            icon="ℹ️",
        )
        return

    col1, col2, col3 = st.columns(3)

    names = [label for label, _ in with_chains]

    with col1:
        first = st.selectbox("First", options=names, key="tension_a")

    with col2:
        second = st.selectbox(
            "Second", options=[n for n in names if n != first],
            key="tension_b",
        )

    with col3:
        parameter = st.selectbox(
            "Parameter", options=sorted(shared), key="tension_param",
        )

    fit_a = dict(with_chains)[first]
    fit_b = dict(with_chains)[second]

    samples_a = fit_a.flat_samples()[:, fit_a.free_params.index(parameter)]
    samples_b = fit_b.flat_samples()[:, fit_b.free_params.index(parameter)]

    summary_a = fit_a.summary()[parameter]
    summary_b = fit_b.summary()[parameter]

    gaussian = tension_mod.gaussian_tension(
        summary_a["median"], 0.5 * (summary_a["plus"] + summary_a["minus"]),
        summary_b["median"], 0.5 * (summary_b["plus"] + summary_b["minus"]),
    )

    sampled = tension_mod.sample_tension(samples_a, samples_b)

    cols = st.columns(2)

    cols[0].metric(
        "Gaussian", f"{gaussian['n_sigma']:.2f}σ",
        help="Assumes both posteriors are Gaussian and independent.",
    )
    cols[1].metric(
        "Sample-based", f"{sampled['n_sigma']:.2f}σ",
        help="No Gaussian assumption: where zero falls in the "
             "distribution of the paired difference.",
    )

    st.caption(
        f"{parameter}:  {first} = {summary_a['median']:.4g} "
        f"(+{summary_a['plus']:.3g}/−{summary_a['minus']:.3g})   ·   "
        f"{second} = {summary_b['median']:.4g} "
        f"(+{summary_b['plus']:.3g}/−{summary_b['minus']:.3g})"
    )

    if abs(gaussian["n_sigma"] - sampled["n_sigma"]) > 0.5:
        st.warning(
            "The two disagree by more than half a sigma, which means "
            "at least one of these posteriors is not Gaussian. Quote "
            "the sample-based number.",
            icon="⚠️",
        )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    difference = (
        np.random.default_rng(0).choice(samples_a, size=100000)
        - np.random.default_rng(1).choice(samples_b, size=100000)
    )

    ax.hist(difference, bins=120, histtype="step", lw=1.6)
    ax.axvline(0.0, color="0.3", lw=1.4)

    ax.set_xlabel(f"{parameter}  ({first} − {second})")
    ax.set_ylabel("posterior samples")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ------------------------------------------------------------

def _render_figure(fig, base_name: str, fmt_label: str, key: str) -> None:
    """
    Render a matplotlib figure plus a download button exporting it
    in `fmt_label` (a key of `PLOT_EXPORT_FORMATS`) -- the browser's
    own save dialog is what lets the user pick *where* it goes; this
    is only responsible for *what format* it goes there as.
    """

    st.pyplot(fig, width="stretch")

    ext, mime = PLOT_EXPORT_FORMATS[fmt_label]

    buf = io.BytesIO()
    fig.savefig(buf, format=ext, bbox_inches="tight")

    st.download_button(
        f"⬇️ {fmt_label}",
        data=buf.getvalue(),
        file_name=f"{base_name}.{ext}",
        mime=mime,
        key=key,
        width="stretch",
    )

    plt.close(fig)
