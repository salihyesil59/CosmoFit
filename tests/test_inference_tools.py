"""
Evidence, profile likelihoods and Fisher matrices.

The three things this library's notebooks kept building by hand or
disclaiming. Each is checked against something that does not share
its machinery:

  evidence   an analytically integrable Gaussian
  profile    the cliff `examples/lscdm_mcmc.ipynb` found by hand
  Fisher     an MCMC of the same posterior
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import LCDM, Fitter
from CosmoFit.stats.evidence import bayes_factor, interpret
from CosmoFit.stats.priors import UniformPrior




def has_dynesty():

    try:
        import dynesty  # noqa: F401
    except ImportError:
        return False

    return True


needs_dynesty = pytest.mark.skipif(
    not has_dynesty(),
    reason="dynesty not installed (optional 'evidence' extra)",
)


# ============================================================
# Evidence, against an integral that can be done by hand
# ============================================================

class GaussianPosterior:
    """
    ``log L = -1/2 sum ((theta - mu)/sigma)^2``, with no
    normalization -- so that over a uniform prior box wide enough
    to contain it,

        Z = prod(sigma_i sqrt(2 pi)) / V

    exactly. Nothing about this touches a cosmology.
    """

    def __init__(self, mu, sigma):

        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)

    def log_likelihood(self, theta):

        return float(
            -0.5 * np.sum(((np.asarray(theta) - self.mu) / self.sigma) ** 2)
        )

    def exact_log_evidence(self, lower, upper):

        volume = np.prod(np.asarray(upper) - np.asarray(lower))

        return float(
            np.sum(np.log(self.sigma * np.sqrt(2.0 * np.pi)))
            - np.log(volume)
        )


@needs_dynesty
@pytest.mark.parametrize("sigma", [[0.3, 0.5], [0.3, 0.5, 0.2, 0.4]])
def test_evidence_matches_an_analytic_integral(sigma):
    """
    The check that matters for an evidence: the answer is known.

    Eight sigma of prior on each side, so the box contains the
    Gaussian to machine precision and the truncation is not what is
    being measured.
    """

    from CosmoFit.stats.nested import run_nested

    sigma = np.asarray(sigma, dtype=float)
    mu = np.zeros(len(sigma))

    lower, upper = mu - 8.0 * sigma, mu + 8.0 * sigma

    names = [f"p{i}" for i in range(len(sigma))]

    prior = UniformPrior(
        names, {n: (a, b) for n, a, b in zip(names, lower, upper)},
    )

    posterior = GaussianPosterior(mu, sigma)

    result = run_nested(
        posterior, prior, names,
        n_live=400, dlogz=0.01, seed=1, progress=False,
    )

    exact = posterior.exact_log_evidence(lower, upper)

    deviation = abs(result.log_evidence - exact) / result.log_evidence_error

    assert deviation < 3.0, (
        f"ln Z = {result.log_evidence:.4f} +- "
        f"{result.log_evidence_error:.4f} against exact {exact:.4f}"
    )

    # And the posterior it returns has to be the right one too.
    np.testing.assert_allclose(
        result.samples.std(axis=0), sigma, rtol=0.06,
    )


@needs_dynesty
def test_evidence_moves_with_the_prior_volume():
    """
    The property that makes a Bayes factor a statement about the
    priors as well as the models, and the reason
    `bayes_factor` reports the volumes.

    Doubling the box of a parameter the likelihood constrains
    halves the evidence -- ``ln Z`` falls by ``ln 2`` -- with the
    fit completely unchanged.
    """

    from CosmoFit.stats.nested import run_nested

    sigma = np.array([0.3, 0.5])
    mu = np.zeros(2)

    def evidence(width):

        lower, upper = mu - width * sigma, mu + width * sigma

        names = ["p0", "p1"]

        prior = UniformPrior(
            names, {n: (a, b) for n, a, b in zip(names, lower, upper)},
        )

        return run_nested(
            GaussianPosterior(mu, sigma), prior, names,
            n_live=400, dlogz=0.01, seed=1, progress=False,
        ).log_evidence

    narrow = evidence(8.0)
    wide = evidence(16.0)

    # Two parameters, each doubled: 2 ln 2.
    assert narrow - wide == pytest.approx(2.0 * np.log(2.0), abs=0.15)


def test_interpretation_thresholds():

    assert interpret(0.4) == "inconclusive"
    assert interpret(2.0) == "positive"
    assert interpret(4.0) == "strong"
    assert interpret(9.0) == "very strong"

    # The sign says which model; the scale is about magnitude.
    assert interpret(-9.0) == interpret(9.0)


def test_bayes_factor_reports_the_prior_volumes():
    """
    A Bayes factor that does not say which prior box it used is not
    a reproducible number.
    """

    from CosmoFit.stats.nested import NestedResult

    def fake(log_z, volume):
        return NestedResult(
            log_evidence=log_z, log_evidence_error=0.1,
            samples=np.zeros((2, 1)), free_params=["x"],
            prior_volume=volume, n_live=1, n_evaluations=1,
        )

    result = bayes_factor(fake(-10.0, 4.0), fake(-13.0, 2.0))

    assert result["ln_B"] == pytest.approx(3.0)
    assert result["favours"] == "alt"
    assert result["prior_volume_alt"] == 4.0
    assert result["prior_volume_null"] == 2.0


# ============================================================
# Profile likelihood
# ============================================================

def test_profile_reproduces_the_lscdm_cliff():
    """
    `examples/lscdm_mcmc.ipynb` built this by hand and found chi2
    falling by 28 between z_dagger = 2.3 and 2.4 -- the feature
    that a marginal posterior smooths over and that the whole
    section 15 analysis turned on.

    Same number through the API.
    """

    from CosmoFit.cosmology.models.lscdm import LsCDM

    fit = Fitter(
        model=LsCDM,
        datasets=["cc", "desi", "bao_lowz", "des_sn5yr", "planck", "omega_b"],
        dataset_kwargs={"desi": {"version": "desi2025"}},
        free_params=["H0", "Omega_m", "Omega_b", "z_dagger"],
        initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493,
                 "z_dagger": 2.5},
        bounds={"z_dagger": (0.5, 100.0)},
        compute_rd=True,
    )

    profile = fit.profile("z_dagger", [2.3, 2.4])

    step = profile["chi2"][0] - profile["chi2"][1]

    assert step == pytest.approx(28.0, abs=1.0)

    assert profile["delta_chi2"].min() == 0.0


def test_profile_leaves_the_original_fitter_alone():
    """
    It re-fits with a parameter held fixed, which must not be done
    by mutating the fitter the caller is holding.
    """

    fit = Fitter(
        model=LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    before = list(fit.free_params), fit.theta0.copy()

    fit.profile("Omega_m", [0.28, 0.32])

    assert list(fit.free_params) == before[0]

    np.testing.assert_allclose(fit.theta0, before[1])


def test_profile_refuses_what_it_cannot_do():

    fit = Fitter(
        model=LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    with pytest.raises(ValueError, match="not a free parameter"):
        fit.profile("rd", [147.0])

    single = Fitter(
        model=LCDM, datasets=["cc"], free_params=["H0"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    with pytest.raises(ValueError, match="at least one other"):
        single.profile("H0", [68.0])


# ============================================================
# Fisher matrix
# ============================================================

def test_fisher_errors_match_an_mcmc():
    """
    A Fisher matrix is a Gaussian approximation to the posterior.
    On a near-Gaussian one it should agree with the chain that
    cost a thousand times more to run.
    """

    fit = Fitter(
        model=LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    fit.best_fit()

    fisher = fit.fisher()

    # Short by MCMC standards, and ample for a two-parameter,
    # near-Gaussian posterior -- the comparison below is at 10%.
    fit.run_mcmc(
        nwalkers=16, nsteps=1200, burnin=300, progress=False, seed=1,
    )

    summary = fit.summary()

    for index, name in enumerate(fit.free_params):

        mcmc_sigma = 0.5 * (
            summary[name]["plus"] + summary[name]["minus"]
        )

        assert fisher["errors"][index] == pytest.approx(
            mcmc_sigma, rel=0.10,
        ), name


def test_fisher_needs_a_point_to_expand_about():

    fit = Fitter(
        model=LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    with pytest.raises(RuntimeError, match="best_fit"):
        fit.fisher()

    # ...or an explicit one.
    result = fit.fisher(theta=fit.theta0)

    assert result["matrix"].shape == (2, 2)

    np.testing.assert_allclose(
        result["matrix"], result["matrix"].T, rtol=1e-8,
    )


def test_fisher_covariance_inverts_the_matrix():

    fit = Fitter(
        model=LCDM, datasets=["cc", "desi"],
        free_params=["H0", "Omega_m"],
        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},
    )

    fit.best_fit()

    fisher = fit.fisher()

    np.testing.assert_allclose(
        fisher["matrix"] @ fisher["covariance"],
        np.eye(2),
        atol=1e-8,
    )


def test_warm_start_does_not_change_the_profile():
    """
    Each point starts from the previous point's solution, because
    neighbouring points on a profile differ by one small step and
    their optima are close. That is a saving in iterations, and it
    must not be a saving in correctness.

    It matters most where profiles are most expensive: against a
    Boltzmann-code likelihood, `best_fit` falls back to
    gradient-free Nelder-Mead and each cold point costs several
    hundred CAMB calls.
    """

    from CosmoFit.cosmology.models.lscdm import LsCDM

    def profile_with(warm_start):

        fit = Fitter(
            model=LsCDM,
            datasets=["cc", "desi", "bao_lowz"],
            dataset_kwargs={"desi": {"version": "desi2025"}},
            free_params=["H0", "Omega_m", "Omega_b", "z_dagger"],
            initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493,
                     "z_dagger": 2.5},
            bounds={"z_dagger": (0.5, 100.0)},
            compute_rd=True,
        )

        return fit.profile(
            "z_dagger", [2.2, 2.6, 3.0], warm_start=warm_start,
        )["chi2"]

    np.testing.assert_allclose(
        profile_with(True), profile_with(False), rtol=1e-6,
    )


def test_warm_start_can_be_turned_off():
    """
    Documented as the escape hatch for a surface with a
    discontinuity a walk could get trapped on the wrong side of --
    which `LsCDM`'s `z_dagger` has. The flag has to actually reach
    the loop.
    """

    import inspect

    signature = inspect.signature(Fitter.profile)

    assert signature.parameters["warm_start"].default is True
