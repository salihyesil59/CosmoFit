"""
CosmoFit -- CPL (w0waCDM) MCMC Analysis (4-dataset, publication-quality)

Plain-script version of ``cpl_mcmc_tfd42.ipynb``, for a genuinely
long, publication-scale run. The notebook remains the reference for
reading through the analysis interactively; this script is for
actually running it (it prints as it goes and writes every figure to
disk instead of rendering inline).

Multi-core behaviour is the same either way. An earlier version of
this note claimed multiprocessing "doesn't work inside a live Jupyter
kernel, not yet root-caused" -- that was a misdiagnosis. Measured
inside a real kernel, a worker pool parallelizes just as well as it
does in a plain script. What was really happening: the per-evaluation
cost was dominated by a *triangular solve* against the Pantheon+
covariance, a sequential recurrence that neither BLAS threads nor
extra worker processes can speed up much, so the speedup was
underwhelming everywhere -- easy to read as "multiprocessing is
broken in notebooks". That solve is now a threaded mat-vec (see
``data.covariance.DenseCovariance``), so the MCMC saturates every
core even at ``n_processes=1``.

Usage
-----
    python examples/cpl_mcmc_tfd42.py

Output
------
Everything below is printed to stdout as it runs (redirect to a file
if you want it saved: ``python examples/cpl_mcmc_tfd42.py | tee
cpl_mcmc_tfd42.log``). Every figure is saved as an SVG into
``examples/cpl_mcmc_tfd42_figures/`` (created if it doesn't exist)
instead of being displayed -- there's no notebook cell to render them
inline in. Open them afterwards in a browser or image viewer.

Runtime: CPU-bound, on the order of several minutes to tens of
minutes depending on core count (see ``N_PROCESSES`` below).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from CosmoFit import LCDM, CPL, Fitter, PlanckLikelihood
from CosmoFit.stats import model_comparison, cpl_diagnostics

FIGURE_DIR = Path(__file__).parent / "cpl_mcmc_tfd42_figures"

#: "auto" lets CosmoFit size the worker pool itself -- every core
#: this process is *allowed* to use (which is not the same as
#: `os.cpu_count()` inside a container, cgroup, or SLURM job), and
#: only when the run is long enough to earn back the startup cost.
#: Override with an explicit integer to leave headroom for other
#: work on the same machine.
N_PROCESSES = "auto"


def savefig(fit_or_fig, name, method=None, **kwargs):
    """
    Call ``fit.plots.<method>(**kwargs)`` (or use a figure already
    returned by one) and save it as
    ``cpl_mcmc_tfd42_figures/<name>.svg``.
    """

    path = FIGURE_DIR / f"{name}.svg"

    if method is not None:
        fig = getattr(fit_or_fig.plots, method)(save_path=path, **kwargs)
    else:
        fig = fit_or_fig
        fig.savefig(path, bbox_inches="tight")

    print(f"  saved {path}")

    import matplotlib.pyplot as plt
    plt.close(fig)


def main():

    FIGURE_DIR.mkdir(exist_ok=True)

    print(f"Using n_processes={N_PROCESSES}")
    print(f"Figures will be saved to {FIGURE_DIR}/\n")

    # ------------------------------------------------------------
    # Building the CPL fit
    # ------------------------------------------------------------

    print("=" * 70)
    print("Building the CPL fit (CC + DESI + Pantheon+ + Planck)")
    print("=" * 70)

    fit = Fitter(
        model=CPL,
        datasets=["cc", "desi", "pantheon", "planck"],
        free_params=["H0", "Omega_m", "w0", "wa", "rd", "Omega_b"],
        initial={
            "H0": 67.4,
            "Omega_m": 0.315,
            "w0": -1.0,
            "wa": 0.0,
            "rd": 147.1,
            "Omega_b": 0.0493,
        },
    )

    print(fit)
    print(f"Total data points: {fit.n_data}")
    print("chi2 breakdown at initial point:", fit.chi2_breakdown())

    # ------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Running the MCMC (nwalkers=64, nsteps=12000)")
    print("=" * 70)

    fit.run_mcmc(
        nwalkers=64, nsteps=12000, burnin=2000, seed=42,
        progress=True, n_processes=N_PROCESSES,
    )

    # ------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Convergence diagnostics")
    print("=" * 70)

    conv = fit.convergence()
    for name, tau in conv["tau"].items():
        print(f"  {name:>10s}:  tau = {tau:6.1f}   "
              f"n_eff = {conv['n_effective'][name]:7.1f}")
    print(f"\nconverged (n_steps >= 50*tau for every parameter): "
          f"{conv['converged']}")

    # ------------------------------------------------------------
    # Posterior summary and best fit
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Posterior summary")
    print("=" * 70)

    for name, stats_ in fit.summary().items():
        print(f"  {name:>10s} = {stats_['median']:8.4f}  "
              f"+{stats_['plus']:.4f} / -{stats_['minus']:.4f}")

    fit.best_fit()

    print("\nBest-fit parameters (maximum likelihood):")
    for name, value in fit.best_fit_params.items():
        print(f"  {name:>10s} = {value:.4f}")
    print(f"\nchi2_min = {fit.best_fit_chi2:.2f}  "
          f"({fit.n_data} data points, {fit.ndim} free parameters)")

    # ------------------------------------------------------------
    # MCMC diagnostics + standard figures
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Figures")
    print("=" * 70)

    savefig(fit, "chain", method="chain")
    savefig(fit, "corner", method="corner", truths=[
        fit.best_fit_params["H0"], fit.best_fit_params["Omega_m"],
        fit.best_fit_params["w0"], fit.best_fit_params["wa"],
        fit.best_fit_params["rd"], fit.best_fit_params["Omega_b"],
    ])
    savefig(fit, "hubble_diagram", method="hubble_diagram")
    savefig(fit, "hz", method="hz")
    savefig(fit, "bao_distances", method="bao_distances")
    savefig(fit, "w_of_z", method="w_of_z")
    savefig(fit, "deceleration", method="deceleration")

    # ------------------------------------------------------------
    # CPL-specific posterior diagnostics
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CPL-specific posterior diagnostics")
    print("=" * 70)

    samples = fit.samples_dict()

    z_cross, frac_cross = cpl_diagnostics.crossing_redshift(
        samples["w0"], samples["wa"],
    )
    print(f"w(z)=-1 crossing: {frac_cross:.1%} of samples cross in "
          f"[0, 2.5], at z = {np.median(z_cross):.2f} (median) if they do")

    if len(z_cross) > 0:
        direction = cpl_diagnostics.crossing_direction(
            samples["w0"], samples["wa"],
        )
        print(f"  quintessence -> phantom: "
              f"{direction['quintessence_to_phantom']:.1%}")
        print(f"  phantom -> quintessence: "
              f"{direction['phantom_to_quintessence']:.1%}")

    lcdm_distance = cpl_diagnostics.mahalanobis_from_lcdm(
        samples["w0"], samples["wa"],
    )
    print(f"\nLCDM point (w0, wa) = (-1, 0) is "
          f"{lcdm_distance['distance']:.2f} sigma from the CPL "
          f"posterior mean {tuple(np.round(lcdm_distance['mean'], 3))}")

    # ------------------------------------------------------------
    # Model comparison: CPL vs. flat LCDM
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Model comparison: CPL vs. flat LCDM")
    print("=" * 70)

    fit_lcdm = Fitter(
        model=LCDM,
        datasets=["cc", "desi", "pantheon", "planck"],
        free_params=["H0", "Omega_m", "rd", "Omega_b"],
        initial={"H0": 67.4, "Omega_m": 0.315, "rd": 147.1, "Omega_b": 0.0493},
    )

    fit_lcdm.run_mcmc(
        nwalkers=32, nsteps=8000, burnin=1500, seed=42,
        progress=True, n_processes=N_PROCESSES,
    )
    fit_lcdm.best_fit()

    print(f"\nLCDM  chi2_min = {fit_lcdm.best_fit_chi2:.2f}  "
          f"({fit_lcdm.ndim} free parameters)")
    print(f"CPL   chi2_min = {fit.best_fit_chi2:.2f}  "
          f"({fit.ndim} free parameters)")

    comparison = model_comparison.compare_models(
        name_null="LCDM", chi2_null=fit_lcdm.best_fit_chi2, k_null=fit_lcdm.ndim,
        name_alt="CPL", chi2_alt=fit.best_fit_chi2, k_alt=fit.ndim,
        n_data=fit.n_data,
    )
    for name in ("LCDM", "CPL"):
        row = comparison[name]
        print(f"  {name:>5s}:  chi2={row['chi2']:8.2f}   "
              f"AIC={row['AIC']:8.2f}   BIC={row['BIC']:8.2f}")

    print(f"\ndelta_AIC (LCDM - CPL) = {comparison['delta_AIC']:+.2f}   "
          f"(positive favors CPL)")
    print(f"delta_BIC (LCDM - CPL) = {comparison['delta_BIC']:+.2f}")

    lrt = comparison["likelihood_ratio_test"]
    print(f"\nLikelihood-ratio test: delta_chi2={lrt['delta_chi2']:.2f} "
          f"(delta_k={lrt['delta_k']}), p={lrt['p_value']:.3f}, "
          f"{lrt['sigma']:.2f} sigma preference for CPL over LCDM")

    # ------------------------------------------------------------
    # CPL vs. LCDM comparison figures
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CPL vs. LCDM comparison figures")
    print("=" * 70)

    savefig(fit, "compare_hz", method="compare_hz",
            other_fits=fit_lcdm, labels=["CPL", "LCDM"])
    savefig(fit, "compare_hubble_diagram", method="compare_hubble_diagram",
            other_fits=fit_lcdm, labels=["CPL", "LCDM"])
    savefig(fit, "compare_w_of_z", method="compare_w_of_z",
            other_fits=fit_lcdm, labels=["CPL", "LCDM"])
    savefig(fit, "compare_deceleration", method="compare_deceleration",
            other_fits=fit_lcdm, labels=["CPL", "LCDM"])

    # ------------------------------------------------------------
    # Planck residuals
    # ------------------------------------------------------------

    savefig(fit, "planck_residuals", method="planck_residuals")

    # ------------------------------------------------------------
    # Save the numeric results too, not just the figures
    # ------------------------------------------------------------

    result_path = FIGURE_DIR.parent / "cpl_mcmc_tfd42_result.json"
    fit.result.save_json(result_path)
    print(f"\nSaved {result_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
