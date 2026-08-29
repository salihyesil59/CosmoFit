"""
High-level fitting interface.

``Fitter`` is the "few lines of code" entry point CosmoFit is
built around: pick a cosmological model, pick which datasets
to combine, pick which parameters are free, and get an MCMC
posterior, a best-fit point, and the usual diagnostic plots
back out.

Example
-------
>>> from CosmoFit import CPL, Fitter
>>>
>>> fit = Fitter(
...     model=CPL,
...     datasets=["cc", "desi", "pantheon"],
...     free_params=["H0", "Omega_m", "w0", "wa", "rd"],
...     initial={"H0": 67.4, "Omega_m": 0.315,
...              "w0": -1.0, "wa": 0.0, "rd": 147.1},
... )
>>> fit.run_mcmc(nwalkers=48, nsteps=6500, burnin=1000)
>>> fit.best_fit()
>>> fit.summary()
>>> fit.plots.corner()

Add ``save=`` to that MCMC call and the chain is written to
disk as it is sampled, and picked back up instead of re-sampled
the next time the same code runs::

>>> fit.run_mcmc(nwalkers=48, nsteps=6500, burnin=1000,
...              save="chains/cpl.h5")

...or reopened later without setting the fit up again at all::

>>> fit = Fitter.from_chain("chains/cpl.h5")

See :mod:`stats.chains`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Callable, Sequence

import numpy as np

from CosmoFit.typing import PathLike, Redshift

if TYPE_CHECKING:
    # `dynesty` is an optional extra and `stats.nested` imports it,
    # so this cannot be a runtime import -- but a type checker needs
    # the name for `run_nested`'s return.
    from CosmoFit.stats.nested import NestedResult

from CosmoFit.cosmology.core.base import Cosmology
from CosmoFit.cosmology.core.parameters import CosmologyParameters

from CosmoFit.likelihoods.cc import CCLikelihood
from CosmoFit.likelihoods.desi import DESILikelihood
from CosmoFit.likelihoods.sdss_bao import SDSSBAOLikelihood
from CosmoFit.likelihoods.sdss_bao import SDSSFullShapeLikelihood
from CosmoFit.likelihoods.eboss_dr16 import EBOSSELGLikelihood
from CosmoFit.likelihoods.eboss_dr16 import EBOSSELGFullShapeLikelihood
from CosmoFit.likelihoods.eboss_dr16 import EBOSSLyaLikelihood
from CosmoFit.likelihoods.bao_lowz import BAOLowZLikelihood
from CosmoFit.likelihoods.pantheon import PantheonLikelihood
from CosmoFit.likelihoods.des_sn5yr import DESSN5YRLikelihood
from CosmoFit.likelihoods.union3 import Union3Likelihood
from CosmoFit.likelihoods.planck import PlanckLikelihood
from CosmoFit.likelihoods.planck_lite import PlanckLiteLikelihood
from CosmoFit.likelihoods.planck_lensing import PlanckLensingLikelihood
from CosmoFit.likelihoods.planck_lowe import PlanckLowEELikelihood
from CosmoFit.likelihoods.act_lensing import ACTDR6LensingLikelihood
from CosmoFit.likelihoods.priors import (
    H0Likelihood,
    OmegaBLikelihood,
    TauLikelihood,
)
from CosmoFit.likelihoods.fsigma8 import FSigma8Likelihood
from CosmoFit.likelihoods.s8 import S8Likelihood
from CosmoFit.likelihoods.joint import JointLikelihood

from CosmoFit.plots import FitPlotter

from .priors import UniformPrior
from .posterior import LogPosterior
from .sampler import EnsembleSampler
from .results import FitResult, BestFitResult, MCMCResult
from .chains import (
    ChainFile,
    DEFAULT_CHAIN_NAME,
    StoredSampler,
    build_metadata,
    compare_signatures,
    signature_id,
)


# ============================================================
# Dataset registry
# ============================================================

#: Maps a short dataset name (as passed in ``datasets=[...]``)
#: to the likelihood class that implements it.
DATASET_REGISTRY = {
    "cc": CCLikelihood,
    "desi": DESILikelihood,
    "sdss_bao": SDSSBAOLikelihood,
    "sdss_fsbao": SDSSFullShapeLikelihood,
    "eboss_elg": EBOSSELGLikelihood,
    "eboss_elg_fs": EBOSSELGFullShapeLikelihood,
    "eboss_lya": EBOSSLyaLikelihood,
    "bao_lowz": BAOLowZLikelihood,
    "pantheon": PantheonLikelihood,
    "des_sn5yr": DESSN5YRLikelihood,
    "union3": Union3Likelihood,
    "planck": PlanckLikelihood,
    "planck_lite": PlanckLiteLikelihood,
    "planck_lensing": PlanckLensingLikelihood,
    "planck_lowe": PlanckLowEELikelihood,
    "act_lensing": ACTDR6LensingLikelihood,
    "fsigma8": FSigma8Likelihood,
    "s8": S8Likelihood,
    "h0": H0Likelihood,
    "omega_b": OmegaBLikelihood,
    "tau": TauLikelihood,
}


#: Pairs of datasets that must not appear in the same fit, and why.
#:
#: Every entry is a case where two datasets are not independent
#: measurements but the same sky, the same supernovae, or the same
#: number twice -- so combining them multiplies a likelihood by
#: (part of) itself, which understates the uncertainty and biases
#: the result without producing any visible symptom. Each was
#: previously documented only in the relevant likelihood's
#: docstring, where nothing checked it. :class:`Fitter` now warns.
CONFLICTING_DATASETS = {

    ("desi", "sdss_bao"):
        "DESI covers much of the same sky and the same structure "
        "BOSS/eBOSS did; they are not independent.",

    ("desi", "sdss_fsbao"):
        "DESI covers much of the same sky and the same structure "
        "BOSS/eBOSS did; they are not independent.",

    ("sdss_bao", "sdss_fsbao"):
        "The same BOSS/eBOSS galaxies analysed two ways -- the BAO "
        "peak alone, and the full anisotropic shape with the growth "
        "rate. `sdss_fsbao` contains these BAO measurements.",

    ("sdss_fsbao", "fsigma8"):
        "The `fsigma8` compilation includes BOSS and eBOSS growth "
        "measurements, which `sdss_fsbao` measures again -- this "
        "time jointly with the geometry.",

    # Not a new dataset's problem, but the same double-counting,
    # and it has been reachable all along: the BAO-only SDSS
    # dataset and the growth compilation are built from the same
    # BOSS/eBOSS galaxies, and combining them treats geometry and
    # growth from one survey as independent measurements. The
    # released BAO+FS consensus (`sdss_fsbao`) exists precisely so
    # they need not be.
    ("sdss_bao", "fsigma8"):
        "The `fsigma8` compilation includes BOSS and eBOSS growth "
        "measurements from the same galaxies `sdss_bao`'s BAO "
        "comes from, and the two are correlated (0.19 to 0.64 within "
        "a redshift bin in the released covariance). Use "
        "`sdss_fsbao`, which measures both jointly.",

    ("desi", "eboss_lya"):
        "DESI's Lyman-alpha forest sample is drawn from much of the "
        "same sky as eBOSS's and re-observes many of the same "
        "quasars, so the two BAO measurements at z ~ 2.33 are "
        "correlated. That this needs saying is not a judgement "
        "call: DESI and eBOSS publish a *joint* Lyman-alpha "
        "likelihood precisely because multiplying the separate ones "
        "is wrong.",

    ("desi", "eboss_elg"):
        "DESI's emission-line galaxy sample succeeds eBOSS's over "
        "much of the same footprint.",

    ("desi", "eboss_elg_fs"):
        "DESI's emission-line galaxy sample succeeds eBOSS's over "
        "much of the same footprint.",

    ("eboss_elg", "eboss_elg_fs"):
        "The same eBOSS DR16 galaxies analysed two ways -- an "
        "isotropic BAO scale, and the full anisotropic shape with "
        "the growth rate. Use one or the other, not both.",

    ("fsigma8", "eboss_elg_fs"):
        "The `fsigma8` compilation includes eBOSS growth-rate "
        "measurements, which this grid's f*sigma8 axis measures "
        "again.",


    ("pantheon", "des_sn5yr"):
        "DES-SN5YR's low-z anchor sample (~11% of it) is also "
        "compiled into Pantheon+.",

    ("pantheon", "union3"):
        "Union3 and Pantheon+ compile substantially the same "
        "literature supernovae.",

    ("des_sn5yr", "union3"):
        "Union3's high-redshift half overlaps the DES sample.",

    ("planck_lensing", "act_lensing"):
        "ACT's lensing map overlaps Planck's on the sky, so the two "
        "reconstructions are correlated. ACT publish a proper joint "
        "variant for this; combining the separate likelihoods "
        "overstates the joint constraint.",

    ("planck_lowe", "tau"):
        "The 'tau' Gaussian prior is a compression of exactly this "
        "low-l EE likelihood -- the same measurement twice.",

    ("planck", "planck_lite"):
        "The distance priors are a compression of exactly these "
        "bandpowers -- this is the whole Planck dataset twice, "
        "once in full and once in summary.",

}


#: How each dataset name should be *written* -- the short form used
#: in figure legends and captions, where the registry key above is
#: an identifier rather than a name a reader recognizes
#: (``"des_sn5yr"`` -> ``"DES-SN5YR"``). Short on purpose: several of
#: these are joined together to describe a fit, so this is the
#: abbreviation, not the GUI's full descriptive label.
DATASET_LABELS = {
    "cc": "CC",
    "desi": "DESI",
    "sdss_bao": "SDSS",
    "sdss_fsbao": "SDSS BAO+FS",
    "eboss_elg": "eBOSS ELG",
    "eboss_elg_fs": "eBOSS ELG (full shape)",
    "eboss_lya": r"eBOSS Ly$\alpha$",
    "bao_lowz": "6dFGS + MGS",
    "pantheon": "Pantheon+",
    "des_sn5yr": "DES-SN5YR",
    "union3": "Union3",
    "planck": "Planck",
    "planck_lite": "Planck TTTEEE",
    "planck_lensing": "Planck lensing",
    "planck_lowe": "Planck lowE",
    "act_lensing": "ACT DR6 lensing",
    "fsigma8": r"$f\sigma_8$",
    "s8": r"$S_8$",
    "h0": r"$H_0$",
    "omega_b": "BBN",
    "tau": r"$\tau$",
}


#: Datasets that compute the CMB from scratch, and therefore carry
#: their own *derived* amplitude via the Boltzmann code.
_CAMB_CMB_DATASETS = {
    "planck_lite", "planck_lensing", "planck_lowe", "act_lensing",
}

#: Datasets that use the *free* ``sigma8`` parameter through the
#: growth machinery.
_GROWTH_DATASETS = {"fsigma8", "s8"}

#: Datasets that actually solve the linear growth equation, and so
#: read the model's ``mu(a, k)``. ``"s8"`` is not among them: it
#: compares the free ``sigma8`` directly and never integrates.
_GROWTH_CALCULATOR_DATASETS = {"fsigma8", "eboss_elg_fs"}


def _warn_derived_parameters(model, free_params) -> None:
    """
    Warn when a free parameter is one the model computes for
    itself.

    Some models fix a parameter by construction rather than
    accepting it: ``ADE`` derives ``Omega_m`` from ``n_ade``,
    because its early-time condition determines the whole
    background from that one number. Sampling it produces a
    posterior for a value nothing reads -- which is to say, the
    prior back again.

    A model declares these in ``DERIVED_PARAMS``.
    """

    import warnings

    derived = getattr(model, "DERIVED_PARAMS", frozenset())

    clashing = sorted(set(free_params) & set(derived))

    if not clashing:
        return

    warnings.warn(

        f"{getattr(model, 'MODEL_NAME', model.__name__)} derives "
        f"{', '.join(clashing)} rather than accepting it, so "
        f"freeing it samples a value the model never reads -- the "
        f"posterior you get back will be its prior. Drop it from "
        f"`free_params`.",

        UserWarning,

        stacklevel=3,

    )


def _warn_blind_neutrino_mass(names, free_params, compute_rd) -> None:
    """
    Warn when ``m_nu`` is free but nothing in the fit can see it.

    Massive neutrinos change the CMB in two ways a distance ratio
    cannot express: they smooth the acoustic peaks through lensing,
    and they shift the early ISW. The **compressed** Planck priors
    carry neither. Worse, they cannot: their ``z_star`` is a fit in
    ``(omega_b, omega_cb)`` alone, calibrated by CHW19 against CAMB
    at the Planck fiducial ``Sum m_nu = 0.06 eV``. Feed it 0.8 eV
    and it returns the 0.06 eV answer, to every digit.

    So in a compressed-CMB fit ``m_nu`` reaches the likelihood
    through exactly one route -- the sound horizon, when
    ``compute_rd`` is set -- where it shifts ``r_d`` with nothing
    to push back. The sampler finds a large mass, and the number
    means nothing about neutrinos.

    Measured, on CC + DESI DR2 + low-z BAO + DES-SN5YR + compressed
    Planck + BBN: the best fit runs to ``Sum m_nu = 0.82 eV`` with
    ``delta chi2 = 5.9`` against zero, while the published bound
    from the same data with the *full* CMB is below 0.1 eV.

    Use ``"planck_lite"`` (with ``"planck_lensing"``, which is
    where most of the sensitivity is) instead: a Boltzmann code
    computes the spectra at whatever mass it is given.
    """

    import warnings

    if "m_nu" not in free_params:
        return

    seen = set(names)

    if seen & _CAMB_CMB_DATASETS:
        return

    reachable = "the sound horizon" if compute_rd else "nothing at all"

    warnings.warn(

        f"`m_nu` is a free parameter, but this fit has no CMB "
        f"likelihood that can respond to it -- it reaches "
        f"{reachable}.\n\n"

        f"The compressed Planck priors are blind to the neutrino "
        f"mass by construction: their `z_star` is a fitting formula "
        f"in (omega_b, omega_cb), calibrated at the fiducial "
        f"Sum m_nu = 0.06 eV, and returns the same z_star at 0.8 eV. "
        f"The CMB's actual sensitivity is in lensing smoothing and "
        f"the early ISW, which no distance ratio carries.\n\n"

        f"A posterior for `m_nu` from this combination is not a "
        f"neutrino-mass measurement. Add 'planck_lite' and "
        f"'planck_lensing', or fix `m_nu`.",

        UserWarning,

        stacklevel=3,

    )


def _warn_ungrounded_coupling(model, names) -> None:
    """
    Warn when growth data meet a scalar-tensor model whose ``mu``
    is still 1.

    In scalar-tensor gravity the field sets the strength of
    gravity, so ``G_eff/G_N`` is not 1 and does not stay put -- it
    moves as the field rolls. A background action does not by
    itself determine that, which is why
    ``Action(growth="quasi_static")`` is opt-in, and why the
    default leaves ``mu = 1``.

    That default is right for a minimally coupled field, where the
    scalar really is dark energy sitting on top of General
    Relativity. It is *wrong* here, and wrong silently: the fit
    runs, the growth chi-squared is finite, and the answer is
    General Relativity's growth attached to a modified background.
    Nothing downstream can tell.
    """

    import warnings

    if not getattr(model, "_NON_MINIMAL", False):
        return

    if getattr(model, "mu", None) is not Cosmology.mu:
        return

    growth = sorted(set(names) & _GROWTH_CALCULATOR_DATASETS)

    if not growth:
        return

    warnings.warn(
        f"{model.plain_name()} couples its field to the curvature "
        f"-- scalar-tensor gravity -- but reports mu = 1, so "
        f"{growth} would be fit with General Relativity's growth "
        f"on top of a modified background.\n\n"
        f"In this theory G_eff/G_N is neither 1 nor constant: it "
        f"moves as the field rolls. Rebuild the action with "
        f"growth='quasi_static' for the sub-horizon result, or "
        f"drop the growth data.",
        UserWarning,
        stacklevel=3,
    )


def _warn_inconsistent_amplitude(names, free_params) -> None:
    """
    Warn when a fit carries two unrelated definitions of the same
    amplitude.

    ``sigma8`` is a free parameter that
    :class:`~cosmology.calculators.growth.GrowthCalculator` uses to
    normalize its growth factor, and it is what the ``"fsigma8"``
    and ``"s8"`` likelihoods are compared against. It is *not* what
    a Boltzmann-computed CMB likelihood uses: there the amplitude
    comes from ``ln1e10As`` through the transfer function, and
    ``sigma8`` is derived rather than sampled.

    Put both in one fit and nothing makes the two agree. The
    sampler will happily settle the free ``sigma8`` on the growth
    data and ``ln1e10As`` on the CMB, and report a posterior for
    each -- two numbers describing one physical quantity, differing
    by however much the data pull them apart, with no error
    anywhere.

    This is a warning rather than a refusal because the combination
    is still useful with ``sigma8`` *fixed*, and because the honest
    fix (deriving ``sigma8`` from the Boltzmann code and feeding it
    to the growth calculator) is a change to what the growth
    machinery is, not a guard clause.
    """

    import warnings

    seen = set(names)

    cmb = sorted(seen & _CAMB_CMB_DATASETS)
    growth = sorted(seen & _GROWTH_DATASETS)

    if not (cmb and growth):
        return

    if "sigma8" not in free_params:
        return

    warnings.warn(

        f"'{', '.join(cmb)}' computes the CMB from scratch, so its "

        f"amplitude is set by `ln1e10As` and sigma8 is derived from "

        f"it -- while '{', '.join(growth)}' is compared against the "

        f"*free* `sigma8` parameter you are sampling. Nothing forces "

        f"the two to agree, so this fit carries two independent "

        f"amplitudes for one physical quantity.\n\n"

        f"Pass `derive_sigma8=True` and drop 'sigma8' from "

        f"free_params: the growth machinery then normalizes with "

        f"the Boltzmann code's own sigma8 and the fit carries one "

        f"amplitude, sampled as `ln1e10As`. Fixing `sigma8` or "

        f"dropping one dataset also works.",

        UserWarning,

        stacklevel=3,

    )


def _warn_conflicting_datasets(names) -> None:
    """
    Warn if a fit combines two datasets that are not independent.

    Every rule in :data:`CONFLICTING_DATASETS` was already written
    down in a likelihood docstring, where it protected nobody who
    did not go looking. The failure mode is silent -- overlapping
    data treated as independent produces a perfectly ordinary-looking
    posterior with error bars that are too small -- so it is worth
    saying out loud at the one moment the combination is chosen.

    A warning, not an error: there are legitimate reasons to build
    such a fit deliberately (quantifying how much the overlap
    matters, reproducing a published analysis that made this
    choice), and refusing outright would make those impossible.
    """

    import warnings

    seen = set(names)

    for (first, second), reason in CONFLICTING_DATASETS.items():

        if first in seen and second in seen:

            warnings.warn(

                f"Datasets '{first}' and '{second}' should not be "

                f"combined: {reason} Treating them as independent "

                f"double-counts that data, which understates the "

                f"uncertainties and can bias the result. Use one "

                f"of the two.",

                UserWarning,

                stacklevel=3,

            )


def dataset_label(names) -> str:
    """
    A fit's dataset combination as one short, readable string:
    ``["cc", "desi", "des_sn5yr"]`` -> ``"CC + DESI + DES-SN5YR"``.

    Used for figure legends (see
    :meth:`~plots.FitPlotter.w0_wa_plane`), where joining the raw
    registry keys would put Python identifiers on a published plot.
    An unknown name is passed through unchanged rather than dropped.

    Separated by ``" + "`` rather than a bare ``"+"``: one of these
    names already ends in a plus (``Pantheon+``), and the tight form
    renders it as ``Pantheon++Planck``.
    """

    return " + ".join(DATASET_LABELS.get(name, name) for name in names)


# ============================================================
# Multiprocessing worker plumbing (for Fitter.run_mcmc(n_processes=...))
# ============================================================
#
# emcee's own multiprocessing recipe (pass a `multiprocessing.Pool`
# straight to `EnsembleSampler`) re-pickles and sends the *entire*
# log-posterior object -- including every dataset's covariance
# matrix -- to a worker on every single ensemble step. For a dataset
# like Pantheon+ (a ~1600x1600 dense covariance), that's tens of MB
# per step, which comfortably outweighs the few milliseconds an
# actual likelihood evaluation takes -- i.e. naive multiprocessing
# here is *slower* than one process, not faster (measured directly:
# ~19ms to pickle a 4-dataset `LogPosterior` vs ~2ms to evaluate it).
#
# So instead, each worker builds its *own* `Fitter` once, from a
# small, cheap-to-pickle "recipe" (`Fitter._recipe()`: the model
# class, dataset names, free parameter names, and current parameter
# values -- no arrays), via the pool's `initializer`. Only a length-
# `ndim` float vector then crosses the process boundary per
# evaluation. This requires `model` to be picklable *by reference*
# (true for every built-in model, and for a module-level `Cosmology`
# subclass with `EXTRA_PARAMS` -- not for a `define_model()`/
# `model_from_expression()` class, which is built dynamically and
# has no stable import path; `Fitter.run_mcmc(n_processes=...)`
# checks for this and raises a clear error rather than an obscure
# pickling failure).

_worker_fitter = None


#: Below this estimated single-process MCMC wall time (seconds),
#: ``n_processes="auto"`` stays single-process: starting N workers
#: costs a `Fitter` rebuild each (re-reading every dataset and
#: re-factorizing its covariance), which a short run never earns
#: back. Long production chains -- the ones actually worth
#: parallelizing -- clear this by orders of magnitude.
_AUTO_PARALLEL_MIN_SECONDS = 8.0


def usable_cpu_count() -> int:
    """
    Number of CPUs this process may actually run on.

    ``os.cpu_count()`` reports the machine's total, ignoring any
    affinity mask -- so it over-reports inside a container, a cgroup,
    a SLURM/PBS allocation, or under ``taskset``. Oversizing the pool
    that way oversubscribes the cores the job really has and makes
    the MCMC *slower*. ``os.sched_getaffinity`` reports the real
    allowance where it exists (Linux, including WSL); fall back to
    ``os.cpu_count()`` elsewhere (macOS, Windows).
    """

    import os

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _fork_available() -> bool:
    """
    Whether the 'fork' start method -- the only one this library's
    worker plumbing supports (see ``Fitter._mcmc_pool``) -- exists
    on this platform.
    """

    import multiprocessing as mp

    return "fork" in mp.get_all_start_methods()


def _init_worker(recipe: dict) -> None:
    """
    Pool ``initializer``: build this worker process's own `Fitter`
    once (stored in a process-global, not returned -- `Pool`
    initializers can't have a return value collected).
    """

    global _worker_fitter

    try:
        import threadpoolctl
        # Undo whatever BLAS thread pool numpy set up before the
        # fork -- with N worker processes each also running a
        # multi-threaded BLAS, you oversubscribe the machine's
        # cores and multiprocessing can end up *slower* than one
        # process. Confirmed: without this, 8-process speedup on an
        # 8-core machine measured 1.4x; with it, 2.4x.
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    _worker_fitter = Fitter(**recipe)


def _worker_log_prob(theta):
    """Pool worker target: evaluate this worker's own `Fitter`."""

    return _worker_fitter.logpost(theta)


# ============================================================
# Rebuilding a model from a saved chain
# ============================================================

def _resolve_model(meta: dict):
    """
    Import the model class a saved chain was sampled with, from
    the name and module recorded in its metadata (see
    :meth:`Fitter.from_chain`).

    Tries the recorded module first -- so a user's own
    ``Cosmology`` subclass in ``my_models.py`` comes back too,
    not just built-in ones -- then falls back to CosmoFit's own
    model package, which covers a chain written before a model
    moved between modules.
    """

    import importlib

    name = meta.get("model")
    module = meta.get("model_module")

    if not name:
        raise ValueError(
            "The saved chain's metadata doesn't record which model "
            "it was sampled with; pass `model=` explicitly."
        )

    if module:
        try:
            found = getattr(importlib.import_module(module), name, None)
        except ImportError:
            found = None

        if found is not None:
            return found

    from CosmoFit.cosmology import models as builtin_models

    found = getattr(builtin_models, name, None)

    if found is not None:
        return found

    raise ValueError(
        f"Can't import the model {name!r} the saved chain was "
        f"sampled with (recorded module: {module!r}). A model built "
        f"at runtime -- by `define_model()`, "
        f"`model_from_expression()` or "
        f"`CosmoFit.theory.Action.build()` -- exists only in the "
        f"session that made it, so rebuild it the same way and pass "
        f"it as `Fitter.from_chain(..., model=MyModel)`."
    )


# ============================================================
# Fitter
# ============================================================

class Fitter:
    """
    Ties together a cosmological model, one or more datasets,
    a prior, and (optionally) an MCMC run and best-fit search.

    Parameters
    ----------
    model : type
        A ``Cosmology`` subclass, e.g. ``LCDM`` or ``CPL`` -- or a
        custom model built with :func:`cosmology.custom.define_model`
        (or a hand-written ``Cosmology`` subclass declaring
        ``EXTRA_PARAMS``), for testing a new model not built into
        CosmoFit.

    datasets : list[str]
        Which likelihoods to combine. Keys of
        :data:`DATASET_REGISTRY` (currently
        ``"cc"``, ``"desi"``, ``"sdss_bao"``, ``"pantheon"``,
        ``"des_sn5yr"``, ``"planck"``, ``"fsigma8"``, ``"s8"``). Do
        not combine ``"desi"``/``"sdss_bao"`` or
        ``"pantheon"``/``"des_sn5yr"`` in the same fit -- see the
        corresponding likelihood classes' docstrings for why.

    free_params : list[str]
        Names of the :class:`CosmologyParameters` fields that
        are allowed to vary (e.g.
        ``["H0", "Omega_m", "w0", "wa", "rd"]``).

    initial : dict[str, float]
        Starting values for every parameter in
        ``CosmologyParameters`` that you want to override
        (both free and fixed ones may be given here; anything
        not given falls back to the dataclass default).

    fixed : dict[str, float], optional
        Explicit values for parameters that are *not* being
        fit. Equivalent to putting them in ``initial`` and
        simply not listing them in ``free_params`` -- provided
        separately mainly for readability.

    bounds : dict[str, tuple[float, float]], optional
        Overrides for the default uniform-prior bounds
        (``model.PARAMS_CLASS.default_bounds()``, which is
        :data:`cosmology.core.parameters.DEFAULT_BOUNDS` for every
        built-in model).

    dataset_kwargs : dict[str, dict], optional
        Extra keyword arguments passed to the constructor of
        each likelihood, keyed by dataset name. E.g.
        ``{"pantheon": {"marginalize_MB": True}}``.

    compute_rd : bool, optional
        If True, the BAO sound horizon ``rd`` is *computed* from
        the physical densities (``omega_b``, ``omega_cb``,
        ``N_eff``, ``m_nu``) rather than fitted as a free nuisance
        parameter -- see
        :mod:`cosmology.calculators.sound_horizon`. Default False.

        This changes what a BAO fit can measure. With ``rd`` free,
        BAO constrains only the product ``H0 rd``, so ``H0`` is
        unconstrained by BAO alone. With ``rd`` computed, ``H0``
        becomes measurable -- but through ``Omega_b``, which BAO
        cannot pin down on its own. That is what the ``"omega_b"``
        (BBN) dataset is for, and the pair is how every published
        "BAO + BBN gives H0" constraint is produced::

            Fitter(model=LCDM,
                   datasets=["desi", "omega_b"],
                   free_params=["H0", "Omega_m", "Omega_b"],
                   compute_rd=True)

        Passing ``compute_rd=True`` with ``"rd"`` in
        ``free_params`` raises: the likelihood would ignore the
        sampled value, and its "posterior" would be its prior.
    """

    def __init__(
        self,
        model,
        datasets,
        free_params,
        initial,
        fixed=None,
        bounds=None,
        dataset_kwargs=None,
        compute_rd=False,
        derive_sigma8=False,
    ):

        self.model_cls = model
        self.free_params = list(free_params)
        self.dataset_names = list(datasets)
        self.compute_rd = bool(compute_rd)
        self.derive_sigma8 = bool(derive_sigma8)

        if self.derive_sigma8 and "sigma8" in self.free_params:

            raise ValueError(

                "`derive_sigma8=True` takes sigma8 from the "

                "Boltzmann code, so it cannot also be a free "

                "parameter -- sampling it would be sampling a value "

                "nothing reads, producing a posterior for `sigma8` "

                "that is just its prior. Drop 'sigma8' from "

                "free_params; the amplitude is carried by "

                "`ln1e10As` instead."

            )

        if self.derive_sigma8 and not (

            set(self.dataset_names) & _CAMB_CMB_DATASETS

        ):

            raise ValueError(

                f"`derive_sigma8=True` needs a CMB likelihood that "

                f"computes the spectra from scratch -- there is "

                f"nothing else to derive sigma8 from. Add one of "

                f"{sorted(_CAMB_CMB_DATASETS)}, or leave sigma8 "

                f"free."

            )

        if self.compute_rd and "rd" in self.free_params:

            raise ValueError(

                "`compute_rd=True` derives rd from the physical "

                "densities (omega_b, omega_cb, N_eff, m_nu), so it "

                "cannot also be a free parameter -- sampling it "

                "would be sampling a value the likelihood ignores, "

                "producing a posterior for `rd` that is just its "

                "prior. Drop 'rd' from free_params.\n\n"

                "With rd computed, `Omega_b` is what BAO now needs "

                "in order to say anything about H0: free it and add "

                "the 'omega_b' (BBN) dataset."

            )

        # ------------------------------------------------------
        # Build the (shared, mutable) parameter object
        #
        # `model.PARAMS_CLASS` is `CosmologyParameters` for every
        # built-in model; a custom model (`cosmology.define_model`,
        # or a `Cosmology` subclass declaring `EXTRA_PARAMS`) gets
        # its own dynamically-built superset instead -- see
        # `Cosmology.__init_subclass__`.
        # ------------------------------------------------------

        params_cls = getattr(model, "PARAMS_CLASS", CosmologyParameters)
        self.params_cls = params_cls

        values = dict(params_cls.defaults())
        values.update(initial or {})
        values.update(fixed or {})

        missing = [
            name for name in params_cls.names()
            if name not in values
        ]
        if missing:
            raise ValueError(
                f"No initial/default value for parameter(s) "
                f"{missing}. Provide them via `initial=` or "
                f"`fixed=`."
            )

        self.initial = {name: values[name] for name in self.free_params}

        # Every parameter's starting value, free and fixed alike,
        # snapshotted here because `self.params` does *not* stay
        # put: the posterior writes each proposed theta into it on
        # every single MCMC step (`LogPosterior._apply`), so after a
        # run it holds the last walker's position, not the input.
        # Saved chains record this instead -- it is what
        # `Fitter.from_chain` rebuilds an equivalent fitter from.
        self._initial_all = {
            name: values[name] for name in params_cls.names()
        }

        self.params = params_cls(
            **{name: values[name] for name in params_cls.names()}
        )

        self.cosmology = self.model_cls(self.params)

        # Set on the instance, not the class -- two fitters in one
        # session (a compute_rd comparison, say) must not share it.
        self.cosmology.compute_rd = self.compute_rd

        # ------------------------------------------------------
        # Build likelihoods
        # ------------------------------------------------------

        dataset_kwargs = dataset_kwargs or {}
        self.dataset_kwargs = dataset_kwargs

        self.likelihoods = []

        _warn_conflicting_datasets(self.dataset_names)

        _warn_inconsistent_amplitude(self.dataset_names, self.free_params)

        _warn_derived_parameters(self.model_cls, self.free_params)

        _warn_ungrounded_coupling(self.model_cls, self.dataset_names)

        _warn_blind_neutrino_mass(
            self.dataset_names, self.free_params, self.compute_rd,
        )

        for name in self.dataset_names:

            if name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset '{name}'. "
                    f"Available: {list(DATASET_REGISTRY)}"
                )

            cls = DATASET_REGISTRY[name]

            kwargs = dataset_kwargs.get(name, {})

            self.likelihoods.append(cls(self.cosmology, **kwargs))

        self.joint = JointLikelihood(*self.likelihoods)

        # Only now: `derive_sigma8` reads the Boltzmann backend, and
        # the backend is created by whichever CMB likelihood was
        # just constructed above.
        self.cosmology.derive_sigma8 = self.derive_sigma8

        # ------------------------------------------------------
        # Prior + posterior
        # ------------------------------------------------------

        self.bounds = bounds

        merged_bounds = params_cls.default_bounds()
        if bounds:
            merged_bounds.update(bounds)

        self.prior = UniformPrior(self.free_params, merged_bounds)
        self.logpost = LogPosterior(self.joint, self.cosmology, self.prior)

        self.sampler = None
        self.burnin = 0
        self.nested = None
        self.best_fit_result = None

        #: The :class:`~stats.chains.ChainFile` this fit's chain
        #: is stored in, once `run_mcmc(save=...)`, `load_chain`
        #: or `from_chain` has pointed it at one -- so "where did
        #: this posterior come from" stays answerable from the
        #: fitter itself.
        self.chain = None

        # ------------------------------------------------------
        # Plots
        # ------------------------------------------------------

        self.plots = FitPlotter(self)

    # ============================================================
    # Basic properties
    # ============================================================

    @property
    def ndim(self) -> int:
        return len(self.free_params)

    @property
    def n_data(self) -> int:
        return self.joint.n_data

    @property
    def theta0(self) -> np.ndarray:
        return np.array([self.initial[n] for n in self.free_params])

    # ============================================================
    # chi2 / likelihood at a point (no MCMC needed)
    # ============================================================

    def chi2(self, theta: Redshift | None = None) -> float:
        """
        Total chi2 at ``theta`` (defaults to the initial point).
        """

        if theta is None:
            theta = self.theta0

        return self.logpost.chi2(theta)

    def chi2_breakdown(self, theta: Redshift | None = None) -> dict:
        """
        Per-dataset chi2 at ``theta`` (defaults to the initial
        point).
        """

        if theta is not None:
            self.logpost._apply(theta)

        return {lk.name: lk.chi2() for lk in self.likelihoods}

    # ============================================================
    # MCMC
    # ============================================================

    def run_mcmc(
        self,
        nwalkers: int = 48,
        nsteps: int = 6000,
        burnin: int = 1000,
        initial_scatter: dict[str, float] | None = None,
        seed: int = 42,
        progress: bool = True,
        moves: object | None = None,
        callback: Callable[[int, int, float], None] | None = None,
        n_processes: int | str = "auto",
        save: PathLike | ChainFile | None = None,
        resume: bool | str = "auto",
    ) -> MCMCResult:
        """
        Run an ``emcee`` ensemble MCMC, via
        :class:`~stats.sampler.EnsembleSampler`.

        Parameters
        ----------
        nwalkers, nsteps : int
            Standard emcee settings. ``nwalkers`` should be at
            least ``2 * ndim``.

        burnin : int
            Number of initial steps discarded by
            :meth:`flat_samples` / :meth:`summary` by default.

        initial_scatter : dict[str, float], optional
            Per-parameter Gaussian scatter used to initialize
            the walkers around ``initial``. Defaults to 1% of
            each parameter's prior width.

        seed : int
            Seed for the walker initialization RNG.

        progress : bool
            Show an emcee progress bar.

        moves : emcee move or list of (move, weight), optional
            Passed through to ``emcee.EnsembleSampler`` (see
            :class:`~stats.sampler.EnsembleSampler`). Defaults to
            emcee's own default proposal (``StretchMove``).

        callback : callable(step, nsteps, elapsed), optional
            Called after every completed step -- see
            :meth:`~stats.sampler.EnsembleSampler.run`. Used by the
            GUI to drive a live progress bar.

        n_processes : int or "auto"
            Evaluate walkers' log-posteriors across this many worker
            processes instead of one.

            ``"auto"`` (the default) uses every CPU this process is
            allowed to run on (:func:`usable_cpu_count`, which
            respects container/cgroup/SLURM/``taskset`` affinity
            rather than blindly taking the machine's core count) --
            but only when the run is big enough to earn back the
            cost of starting the workers, and only where the 'fork'
            start method exists. Otherwise it stays at 1. Pass an
            explicit integer to override, or ``n_processes=1`` to
            force single-process.

            Each worker builds its own `Fitter` once (see the
            `stats.fitter` module docstring for why -- naive
            `pool=multiprocessing.Pool()` is often *slower* here, not
            faster), so this needs `model` to be picklable by
            reference: every built-in model works; a model built
            with `cosmology.custom.define_model()` /
            `model_from_expression()` doesn't. ``"auto"`` detects
            that case and quietly stays single-process; an explicit
            ``n_processes > 1`` raises `ValueError` instead, rather
            than failing later with an obscure pickling error.

            Speedup is real but well under `n_processes`x -- per-step
            IPC costs something, and the likelihood's dominant cost
            (a covariance mat-vec) is memory-bandwidth-bound, so
            worker processes contend for the same memory bus.
            Reliable on Linux/macOS; not available on Windows, which
            has no 'fork' (``"auto"`` falls back to 1 there).

            Note that the chain itself does *not* depend on this:
            the proposal RNG lives in this process and is seeded from
            `seed`, so a given `seed` gives bit-identical results at
            any `n_processes`.

        save : str, Path or ChainFile, optional
            Write the chain to this HDF5 file as it is sampled,
            and reuse it on the next call. This is what makes an
            analysis re-runnable without re-sampling: the same
            three lines of code cost hours the first time and
            seconds every time after, so adding a plot or a
            derived quantity later is cheap. Because every step is
            written as it is taken, an interrupted run (Ctrl-C, a
            walltime limit) also keeps everything it had already
            done -- call again to carry on from there.

            Pass a :class:`~stats.chains.ChainFile` instead of a
            path to keep several chains in one file (its
            ``name=``), e.g. one per model in a comparison.

            The file also records which posterior the chain came
            from -- model, datasets, free parameters, prior
            bounds, fixed parameter values -- and a resume that
            doesn't match on all of them is refused rather than
            silently gluing samples from two different
            distributions together.

        resume : {"auto", True, False}
            What to do when ``save`` already holds a chain.
            Ignored without ``save``.

            ``"auto"`` (default)
                Continue it if it exists, start it if it doesn't.
                ``nsteps`` is then the *total* chain length to
                end up with, already-stored steps included: a
                second call asking for the same ``nsteps`` has
                nothing left to sample and returns immediately,
                while a larger ``nsteps`` extends the chain by
                the difference. That's what makes re-running a
                script (or a notebook cell) idempotent.

            ``True``
                The same, but require the chain to already exist
                -- for "analyze what I sampled last night, don't
                quietly start a fresh 12-hour run if the file
                moved".

            ``False``
                Ignore and **discard** whatever is stored, and
                sample from scratch. The only destructive option,
                and never reached by default.

        Returns
        -------
        emcee.EnsembleSampler or stats.chains.StoredSampler
            The sampler that ran. When ``resume`` had nothing
            left to sample, the stored chain is returned instead,
            read back from the file -- either way,
            :meth:`summary`, :meth:`convergence`,
            :meth:`best_fit` and the plots work off it the same.
        """

        chain, chain_backend, steps_to_run = self._prepare_chain(
            save, resume,
            nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, seed=seed,
        )

        # A saved chain that is already `nsteps` long: the whole
        # point of `save=`. Read it back and stop -- no pool, no
        # sampler, no likelihood evaluated. Everything downstream
        # (`summary`, `convergence`, `best_fit`, the plots) reads
        # the chain through the same interface either way.
        if chain is not None and steps_to_run <= 0:

            self.sampler = chain.open()
            self.burnin = burnin
            self.chain = chain

            return self.sampler

        n_processes = self._resolve_n_processes(
            n_processes, nwalkers=nwalkers, nsteps=steps_to_run,
        )

        backend = EnsembleSampler(moves=moves)

        with self._mcmc_pool(n_processes) as pool:

            # With a pool, `emcee` ships `logpost` itself to every
            # worker on every step -- `self.logpost` is the heavy,
            # data-carrying object multiprocessing here is built to
            # avoid sending (see the module docstring above
            # `_init_worker`). `_worker_log_prob` is a cheap
            # top-level reference to *this worker's own* `Fitter`
            # instead, built once by the pool's `initializer`.
            logpost = _worker_log_prob if pool is not None else self.logpost

            sampler = backend.run(
                logpost,
                self.prior,
                self.theta0,
                nwalkers=nwalkers,
                nsteps=steps_to_run,
                initial_scatter=initial_scatter,
                seed=seed,
                progress=progress,
                callback=callback,
                pool=pool,
                backend=chain_backend,
            )

        self.sampler = sampler
        self.burnin = burnin
        self.chain = chain

        if chain is not None:
            # Re-stamp now that the run has finished, so `updated`
            # dates the chain's last *completed* step rather than
            # the moment sampling started.
            chain.write_metadata(
                self._chain_metadata(
                    previous=chain.metadata,
                    burnin=burnin, seed=seed, nwalkers=nwalkers,
                )
            )

        return sampler

    # ------------------------------------------------------------

    def _model_is_picklable(self) -> bool:
        """
        Whether `model` survives a by-reference pickle, which the
        worker plumbing requires. False for any class built at
        runtime -- `define_model()`, `model_from_expression()` or
        `CosmoFit.theory.Action.build()`.
        """

        import pickle

        try:
            pickle.dumps(self.model_cls)
        except (pickle.PicklingError, AttributeError, TypeError):
            return False

        return True

    # ------------------------------------------------------------

    def _seconds_per_eval(self, n_eval: int = 20, n_warmup: int = 20) -> float:
        """
        Steady-state wall-clock cost of one log-posterior
        evaluation, used by ``n_processes="auto"`` to decide whether
        a run is long enough to be worth parallelizing.

        The generous warmup is not just about filling caches: a
        large covariance mat-vec (see
        :class:`~data.covariance.DenseCovariance`) is threaded by
        BLAS, and BLAS spins its thread pool up lazily. The first
        ~20 evaluations are measurably erratic because of it --
        timed here at 5 ms, then 8 ms, before settling to ~0.8 ms --
        so timing only the first few overestimates the steady-state
        cost several-fold. The median (rather than the mean) of the
        timed calls then discards the occasional GC/scheduler
        outlier. The whole probe costs ~40 ms, negligible against
        any run this is used to make a decision about.
        """

        import time

        theta = self.theta0

        for _ in range(n_warmup):
            self.logpost(theta)

        timings = []

        for _ in range(n_eval):
            start = time.perf_counter()
            self.logpost(theta)
            timings.append(time.perf_counter() - start)

        return float(np.median(timings))

    # ------------------------------------------------------------

    def _resolve_n_processes(self, n_processes, nwalkers: int, nsteps: int) -> int:
        """
        Turn the public ``n_processes`` argument (an int, or
        ``"auto"``) into a concrete worker count.

        See :meth:`run_mcmc` for the user-facing description of what
        ``"auto"`` decides and why.
        """

        if n_processes != "auto":

            n_processes = int(n_processes)

            if n_processes < 1:
                raise ValueError(
                    f"n_processes must be >= 1 or \"auto\", "
                    f"got {n_processes}."
                )

            return n_processes

        # --- "auto" from here on: never raise, only decline ---

        if not _fork_available():
            return 1

        if not self._model_is_picklable():
            return 1

        n_cpu = usable_cpu_count()

        if n_cpu < 2:
            return 1

        # A pool costs one full `Fitter` rebuild per worker (every
        # dataset re-read and re-factorized). Only pay that when the
        # serial run would take long enough to earn it back.
        estimated_serial = self._seconds_per_eval() * nwalkers * nsteps

        if estimated_serial < _AUTO_PARALLEL_MIN_SECONDS:
            return 1

        # Never start more workers than there is work for: emcee
        # evaluates the ensemble one half at a time, so only
        # nwalkers/2 log-posteriors are ever in flight at once.
        return max(1, min(n_cpu, nwalkers // 2))

    # ------------------------------------------------------------

    def _recipe(self) -> dict:
        """
        This fitter's constructor arguments, cheap to pickle (no
        arrays -- just the model class, names, and current scalar
        parameter values) -- enough for a worker process to build an
        equivalent `Fitter` of its own. See `run_mcmc(n_processes=)`.
        """

        return dict(
            model=self.model_cls,
            datasets=self.dataset_names,
            free_params=self.free_params,
            initial=self.params.as_dict(),
            bounds=self.bounds,
            dataset_kwargs=self.dataset_kwargs,
            compute_rd=self.compute_rd,
            derive_sigma8=self.derive_sigma8,
        )

    # ------------------------------------------------------------

    def _mcmc_pool(self, n_processes: int):
        """
        A `multiprocessing.Pool` (as a context manager) for
        `run_mcmc(n_processes=...)`, or a no-op context manager
        yielding `None` for the default `n_processes=1` (so
        `run_mcmc` can always do ``with self._mcmc_pool(n) as
        pool:`` regardless).
        """

        from contextlib import nullcontext

        if n_processes <= 1:
            return nullcontext(None)

        import pickle
        import multiprocessing as mp

        recipe = self._recipe()

        try:
            pickle.dumps(recipe["model"])
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"n_processes={n_processes} needs `model` "
                f"({recipe['model']!r}) to be picklable by reference, "
                f"which every built-in CosmoFit model is. A model "
                f"built at runtime is not -- that covers "
                f"`define_model()`, `model_from_expression()` and "
                f"`CosmoFit.theory.Action.build()`, all of which "
                f"produce a class that exists only in the session "
                f"that made it. Use n_processes=1 for this model."
            ) from exc

        # Explicitly request "fork" rather than using whatever
        # `multiprocessing`'s *default* context is. That default
        # isn't "fork" everywhere: Python 3.14 changed it to
        # "forkserver" on Linux too (it already wasn't "fork" on
        # macOS/Windows), and "forkserver"/"spawn" need every
        # process-spawning call to happen behind an
        # `if __name__ == "__main__":` guard -- a plain script
        # without one crashes outright
        # ("RuntimeError: ... before bootstrapping phase"), and even
        # inside a Jupyter kernel (which sidesteps that crash) it
        # measured *no* speedup at all in practice, only overhead.
        # "fork" has neither problem and needs no such guard, and
        # remains available as an explicit context on Linux/macOS
        # even where it's no longer the default -- just not on
        # Windows, where it was never available.
        try:
            ctx = mp.get_context("fork")
        except ValueError as exc:
            raise ValueError(
                f"n_processes={n_processes} needs the 'fork' "
                f"multiprocessing start method, which isn't available "
                f"on this platform (e.g. Windows never has it). Use "
                f"n_processes=1 here."
            ) from exc

        return ctx.Pool(n_processes, initializer=_init_worker, initargs=(recipe,))

    # ------------------------------------------------------------

    def flat_samples(self, burnin: int | None = None) -> np.ndarray:

        if self.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        if burnin is None:
            burnin = self.burnin

        return self.sampler.get_chain(discard=burnin, flat=True)

    # ------------------------------------------------------------

    def convergence(self, burnin: int | None = None, tol: int = 50) -> dict:
        """
        MCMC convergence diagnostics, based on the integrated
        autocorrelation time tau of each free parameter's chain
        (``emcee``'s own recommended diagnostic -- see
        https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).

        A chain is considered trustworthy once it has run for at
        least ``tol`` (default 50, emcee's own default) times tau:
        shorter than that, the posterior (especially its width)
        is not yet reliable, no matter how good ``summary()``
        looks. This is *not* checked automatically anywhere else
        in ``Fitter`` -- call this explicitly after ``run_mcmc()``
        and inspect ``converged`` before trusting ``summary()`` /
        ``plots.corner()``.

        Parameters
        ----------
        burnin : int, optional
            Steps to discard before estimating tau. Defaults to
            ``self.burnin``.

        tol : int, optional
            Convergence threshold, in units of tau (chain length
            must exceed ``tol * tau`` for every parameter).

        Returns
        -------
        dict with keys:
            tau : dict[str, float]
                Autocorrelation time per free parameter.
            n_used : int
                Number of post-burnin steps the estimate is based on.
            n_effective : dict[str, float]
                Effective number of independent samples per
                parameter (n_used * nwalkers / tau).
            converged : bool
                Whether every parameter satisfies
                ``n_used >= tol * tau``.
        """

        if self.sampler is None:
            raise RuntimeError("Call run_mcmc() first.")

        if burnin is None:
            burnin = self.burnin

        chain = self.sampler.get_chain(discard=burnin)
        n_used, nwalkers, _ = chain.shape

        # `discard=burnin`, matching `n_used` above: tau must be
        # estimated on the *same* post-burn-in chain the convergence
        # test and `n_effective` are then reported for. Estimating it
        # on the full chain instead (`discard=0`) folds in the
        # walkers' initial transient -- a one-way drift from the
        # starting ball toward the posterior, which looks like an
        # enormously long correlation time -- and so inflates tau
        # while `n_used` still counts only post-burn-in steps. The
        # two are then inconsistent: measured on a 3000-step,
        # 1500-burn-in CPL chain, tau came out ~45% too large
        # (118 vs 78) and `n_effective` ~35% too small, i.e. the
        # reported effective sample size understated the real one.
        #
        # `quiet=True`: report the (unreliable) estimate instead of
        # raising when the chain is too short to trust it -- that
        # is exactly the situation `converged` below is meant to
        # flag to the caller.
        tau = self.sampler.get_autocorr_time(discard=burnin, quiet=True)

        tau_dict = dict(zip(self.free_params, map(float, tau)))

        n_effective = {
            name: float(n_used * nwalkers / t)
            for name, t in tau_dict.items()
        }

        converged = all(n_used >= tol * t for t in tau_dict.values())

        return {
            "tau": tau_dict,
            "n_used": int(n_used),
            "n_effective": n_effective,
            "converged": converged,
        }

    # ------------------------------------------------------------

    def samples_dict(self, burnin: int | None = None) -> dict:
        """
        Flat posterior samples as a dict of 1D arrays, keyed by
        parameter name.
        """

        flat = self.flat_samples(burnin=burnin)

        return {
            name: flat[:, i]
            for i, name in enumerate(self.free_params)
        }

    # ------------------------------------------------------------

    def summary(self, burnin: int | None = None) -> dict:
        """
        Posterior median +/- 68% interval for every free
        parameter.
        """

        flat = self.flat_samples(burnin=burnin)

        result = {}

        for i, name in enumerate(self.free_params):

            q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])

            result[name] = {
                "median": float(q50),
                "plus": float(q84 - q50),
                "minus": float(q50 - q16),
            }

        return result

    # ============================================================
    # Saved chains
    # ============================================================

    def _chain_signature(self) -> dict:
        """
        What has to match for a saved chain to belong to *this*
        fit: everything that defines the posterior being sampled.

        Not a checksum of the run (walkers, steps, seed, burn-in
        are all free to change between sessions -- that's the
        point of resuming) but of the *distribution*. Continue a
        chain under a different model, dataset combination, free
        parameter list, prior bound or fixed parameter value and
        the result is one array of samples drawn from two
        different posteriors, with nothing in the file to say so.
        :func:`~stats.chains.compare_signatures` compares two of
        these; :meth:`_check_chain_compatible` refuses the resume.
        """

        return {
            "model": self.model_cls.__name__,
            "datasets": list(self.dataset_names),
            "free_params": list(self.free_params),
            "bounds": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(
                    self.prior.names, self.prior.lower, self.prior.upper,
                )
            },
            "fixed": {
                name: value
                for name, value in self._initial_all.items()
                if name not in self.free_params
            },
            "dataset_kwargs": self.dataset_kwargs or {},
            "compute_rd": bool(self.compute_rd),
            "derive_sigma8": bool(self.derive_sigma8),
        }

    # ------------------------------------------------------------

    def chain_id(self, **extra) -> str:
        """
        A filename-safe name identifying this exact fit, e.g.
        ``"CPL_3f9a1c04"`` -- for saving chains without inventing
        a name for each one.

        Two fitters over the same posterior produce the same id
        in any session on any machine; change the model, the
        datasets, the free parameters, a prior bound or a fixed
        value and the id changes with it. Saving under it
        therefore reuses a chain exactly when reusing it is
        correct, and starts a new file (rather than colliding
        with, or overwriting, the old one) when it isn't::

            fit.run_mcmc(
                nsteps=6000,
                save=f"chains/{fit.chain_id(nwalkers=48)}.h5",
            )

        Parameters
        ----------
        **extra
            Folded into the id, for anything that should force a
            separate file without being part of the posterior --
            ``nwalkers`` and ``seed`` are the useful ones, since
            a stored chain can't change either one halfway
            through.
        """

        return (
            f"{self.model_cls.__name__}_"
            f"{signature_id(self._chain_signature(), extra)}"
        )

    # ------------------------------------------------------------

    def _chain_metadata(self, previous=None, *, burnin, seed, nwalkers) -> dict:
        """
        The full metadata block written next to a saved chain:
        the signature above, plus the run settings and the model's
        import path (which :meth:`from_chain` needs to rebuild an
        equivalent fitter from the file alone).
        """

        return build_metadata(
            self._chain_signature(),
            previous=previous,
            model_module=self.model_cls.__module__,
            initial=dict(self._initial_all),
            burnin=int(burnin),
            seed=int(seed),
            nwalkers=int(nwalkers),
        )

    # ------------------------------------------------------------

    def _check_chain_compatible(self, chain: ChainFile, nwalkers=None) -> None:
        """
        Refuse a stored chain that wasn't sampled from this
        fitter's posterior, naming what differs.
        """

        meta = chain.metadata

        if not meta:
            raise ValueError(
                f"{chain.path} already holds a chain, but no CosmoFit "
                f"metadata describing it -- there is no way to tell "
                f"which model, datasets or priors it was sampled with, "
                f"so it can't safely be continued or reused. Save to a "
                f"different path, or pass `resume=False` to discard "
                f"what's there and sample from scratch."
            )

        differences = compare_signatures(meta, self._chain_signature())

        if differences:
            raise ValueError(
                "The saved chain was sampled from a different "
                "posterior than this fit:\n  - "
                + "\n  - ".join(differences)
                + f"\n({chain.path}). Samples from two different "
                f"posteriors must not be merged into one chain -- "
                f"save this fit to a different path, or pass "
                f"`resume=False` to discard the stored chain and "
                f"sample from scratch."
            )

        shape = chain.shape

        if shape is not None:

            stored_nwalkers, stored_ndim = shape

            if stored_ndim != self.ndim:
                raise ValueError(
                    f"The saved chain has {stored_ndim} free "
                    f"parameter(s), this fit has {self.ndim} "
                    f"({chain.path})."
                )

            if nwalkers is not None and stored_nwalkers != nwalkers:
                raise ValueError(
                    f"The saved chain was sampled with "
                    f"{stored_nwalkers} walkers, but this run asks "
                    f"for {nwalkers} ({chain.path}). A chain can only "
                    f"be continued with the ensemble it was sampled "
                    f"with -- pass nwalkers={stored_nwalkers}, or save "
                    f"to a different path."
                )

    # ------------------------------------------------------------

    def _prepare_chain(self, save, resume, *, nwalkers, nsteps, burnin, seed):
        """
        Turn ``run_mcmc``'s ``save``/``resume`` into the three
        things the run itself needs: the :class:`ChainFile` (or
        None), the emcee backend to write through (or None), and
        how many steps are actually left to sample.

        Also writes the metadata *before* sampling starts, so a
        run that never finishes still leaves behind a file that
        says what it was.
        """

        if save is None:
            return None, None, nsteps

        if resume not in ("auto", True, False):
            raise ValueError(
                f'resume must be "auto", True or False, got {resume!r}.'
            )

        if resume != "auto":
            # `True`/`False` are the meaningful values, but 1/0
            # arrive here too and `is False` would quietly miss
            # them.
            resume = bool(resume)

        chain = save if isinstance(save, ChainFile) else ChainFile(save)

        stored_steps = chain.iteration

        if resume is False:

            # The one destructive path in this method, and only
            # ever taken because the caller asked for it by name.
            if stored_steps:
                chain.reset(nwalkers, self.ndim)

            previous = None
            steps_to_run = nsteps

        elif stored_steps == 0:

            if resume is True:
                raise FileNotFoundError(
                    f"resume=True, but there is no saved chain to "
                    f"resume at {chain.path} (chain '{chain.name}'). "
                    f'Use resume="auto" to start one there if it '
                    f"doesn't exist yet."
                )

            previous = None
            steps_to_run = nsteps

        else:

            self._check_chain_compatible(chain, nwalkers=nwalkers)

            # `nsteps` is the *total* length to end up with, so
            # re-running an unchanged script is a no-op instead of
            # doubling the chain. See `run_mcmc`'s `resume`.
            previous = chain.metadata
            steps_to_run = nsteps - stored_steps

        chain.write_metadata(
            self._chain_metadata(
                previous=previous,
                burnin=burnin, seed=seed, nwalkers=nwalkers,
            )
        )

        return chain, chain.backend(), steps_to_run

    # ------------------------------------------------------------

    def load_chain(
        self,
        path: PathLike | ChainFile,
        name: str = DEFAULT_CHAIN_NAME,
        burnin: int | None = None,
    ) -> StoredSampler:
        """
        Attach a chain saved by an earlier
        ``run_mcmc(save=...)`` to this fitter, without sampling
        anything.

        Everything that reads the posterior -- :meth:`summary`,
        :meth:`convergence`, :meth:`samples_dict`,
        :meth:`best_fit`, :attr:`result`, ``plots.corner()``, the
        derived-quantity posteriors -- then works exactly as it
        would have straight after the original run.

        This is the explicit version of what
        ``run_mcmc(save=..., resume=True)`` does implicitly; use
        it when you want to be sure nothing can start sampling,
        or to point an existing fitter at a chain stored under a
        different filename.

        Parameters
        ----------
        path : str, Path or ChainFile
            The saved chain.

        name : str
            Which chain inside the file, if it holds more than
            one. Ignored when ``path`` is a
            :class:`~stats.chains.ChainFile`.

        burnin : int, optional
            Steps to discard. Defaults to the burn-in recorded
            when the chain was sampled.

        Returns
        -------
        stats.chains.StoredSampler

        Raises
        ------
        FileNotFoundError
            If there is no such chain.
        ValueError
            If it was sampled from a different posterior than
            this fitter's -- see :meth:`_chain_signature`.
        """

        chain = path if isinstance(path, ChainFile) else ChainFile(path, name=name)

        if not chain.exists:
            raise FileNotFoundError(
                f"No chain '{chain.name}' in {chain.path}."
            )

        self._check_chain_compatible(chain)

        stored = chain.open()

        self.sampler = stored
        self.burnin = int(stored.burnin if burnin is None else burnin)
        self.chain = chain

        return stored

    # ------------------------------------------------------------

    @classmethod
    def from_chain(
        cls,
        path,
        name: str = DEFAULT_CHAIN_NAME,
        model=None,
        burnin=None,
        **overrides,
    ):
        """
        Rebuild a fitter from a saved chain and attach that
        chain: the "open last week's results and keep working"
        entry point.

        The chain file records the model, datasets, free
        parameters, prior bounds and fixed parameter values it
        was sampled with, which is exactly the fitter's
        constructor -- so nothing has to be re-typed, and it
        can't drift out of sync with the samples.

        The datasets *are* re-read (that's what makes
        :meth:`best_fit`, per-dataset chi2 and the model-vs-data
        plots work); no likelihood is evaluated and no sampling
        happens. For posterior summaries alone, skip the fitter
        entirely -- :func:`stats.chains.open_chain` reads the
        same file with no data loading at all.

        Parameters
        ----------
        path : str, Path or ChainFile
            The saved chain.

        name : str
            Which chain inside the file, if it holds more than
            one.

        model : type, optional
            The model class, when it can't be imported back from
            the name and module recorded in the file -- which is
            the case for a model built by
            :func:`cosmology.custom.define_model` /
            ``model_from_expression`` (it exists only in the
            session that defined it). Rebuild the same model and
            pass it here.

        burnin : int, optional
            Steps to discard. Defaults to the burn-in recorded
            when the chain was sampled.

        **overrides
            Passed on to the constructor, overriding what the
            file recorded (e.g. ``dataset_kwargs=``).

        Examples
        --------
        >>> fit = Fitter.from_chain("chains/cpl.h5")
        >>> fit.summary()
        >>> fit.plots.corner()
        >>> fit.best_fit()          # datasets are live again
        """

        chain = path if isinstance(path, ChainFile) else ChainFile(path, name=name)

        if not chain.exists:
            raise FileNotFoundError(
                f"No chain '{chain.name}' in {chain.path}."
            )

        meta = chain.metadata

        if not meta:
            raise ValueError(
                f"{chain.path} holds a chain but no CosmoFit metadata, "
                f"so there is nothing to rebuild a Fitter from (it was "
                f"probably written by emcee directly). Construct the "
                f"Fitter yourself and use `fit.load_chain(...)`, or "
                f"read the samples with "
                f"`CosmoFit.stats.chains.open_chain(...)`."
            )

        model_cls = model if model is not None else _resolve_model(meta)

        kwargs = dict(
            model=model_cls,
            datasets=meta["datasets"],
            free_params=meta["free_params"],
            initial=meta.get("initial") or meta.get("fixed", {}),
            bounds={
                key: tuple(value)
                for key, value in (meta.get("bounds") or {}).items()
            },
            dataset_kwargs=meta.get("dataset_kwargs") or None,
            # Absent from chains written before rd could be
            # computed, where it was necessarily False.
            compute_rd=bool(meta.get("compute_rd", False)),
            derive_sigma8=bool(meta.get("derive_sigma8", False)),
        )
        kwargs.update(overrides)

        fit = cls(**kwargs)

        fit.load_chain(chain, burnin=burnin)

        return fit

    # ============================================================
    # Best fit
    # ============================================================

    # ============================================================
    # Evidence, profiles and curvature
    # ============================================================

    def _respawn(self, free_params, overrides=None):
        """
        An equivalent fitter with a different set of free
        parameters -- same model, same data, same everything else.

        Used by :meth:`profile`, which has to re-fit with one
        parameter held fixed. Built from the stored construction
        arguments rather than by mutating this fitter, so the
        original is left exactly as the caller had it.
        """

        initial = dict(self._initial_all)

        initial.update(overrides or {})

        return type(self)(
            model=self.model_cls,
            datasets=list(self.dataset_names),
            free_params=list(free_params),
            initial=initial,
            bounds=self.bounds,
            dataset_kwargs=self.dataset_kwargs or None,
            compute_rd=self.compute_rd,
            derive_sigma8=self.derive_sigma8,
        )

    # ------------------------------------------------------------

    def run_nested(self, n_live: int = 500, dlogz: float = 0.05,
                   seed: int | None = 42,
                   progress: bool = True, **kwargs) -> NestedResult:
        """
        Integrate the posterior by nested sampling, giving the
        Bayesian evidence ``ln Z`` as well as samples.

        Needs ``dynesty``: ``pip install "cosmofit[evidence]"``.

        This is the tool for the comparisons where a
        likelihood-ratio test is not defined -- ``LsCDM`` against
        ``LCDM``, which is reached only as ``z_dagger -> infinity``,
        or ``DGP``, which is not nested at all. See
        :mod:`stats.evidence`, and read its note on prior
        sensitivity before quoting a Bayes factor.

        Parameters
        ----------
        n_live : int, optional
            Live points; the accuracy knob. The evidence error
            scales roughly as ``sqrt(information / n_live)``.
        dlogz : float, optional
            Stopping criterion on the remaining evidence.
        seed : int, optional
        progress : bool, optional
        **kwargs
            Forwarded to ``dynesty.NestedSampler``.

        Returns
        -------
        stats.nested.NestedResult
        """

        from CosmoFit.stats.nested import run_nested as _run

        self.nested = _run(
            self.logpost,
            self.prior,
            self.free_params,
            n_live=n_live,
            dlogz=dlogz,
            seed=seed,
            progress=progress,
            **kwargs,
        )

        return self.nested

    # ------------------------------------------------------------

    def profile(self, name: str, values: Redshift, restarts: int = 0,
                seed: int = 0, progress: bool = False,
                warm_start: bool = True) -> dict:
        """
        Profile likelihood: ``chi2`` minimized over every *other*
        free parameter, at each fixed value of ``name``.

        The honest tool when Wilks' theorem does not apply. It is
        also how a boundary-limited parameter should be reported --
        ``examples/05-case-studies/lscdm_mcmc.ipynb`` found a 28-unit cliff in
        ``z_dagger`` this way, which a marginal posterior smoothed
        over.

        Parameters
        ----------
        name : str
            A free parameter of this fitter.
        values : array_like
            Values to fix it at.
        restarts : int, optional
            Passed to :meth:`best_fit` at each point. Worth setting
            where the surface has more than one basin -- which is
            exactly the situation a profile is usually
            investigating.
        seed : int, optional
        progress : bool, optional
        warm_start : bool, optional
            Start each point from the previous point's solution
            rather than from ``initial``. Neighbouring points on a
            profile differ by one small step in one parameter, so
            their optima are close, and the optimizer arrives in a
            fraction of the iterations.

            It matters most exactly where profiles are most
            expensive: with a Boltzmann-code likelihood,
            ``best_fit`` falls back to gradient-free Nelder-Mead
            because CAMB's numerical noise defeats a finite
            difference, and that costs several hundred evaluations
            per point from cold.

            Turn it off to make each point independent -- worth
            doing if the surface has a discontinuity the walk could
            get trapped on the wrong side of, which is not
            hypothetical here: ``LsCDM``'s ``z_dagger`` has one.
            Ordering the values from the far side, or passing
            ``restarts``, is the other way.

        Returns
        -------
        dict
            ``values``, ``chi2``, ``delta_chi2`` (from the profile
            minimum), and ``params`` -- the re-optimized values of
            the other parameters at each point.
        """

        if name not in self.free_params:

            raise ValueError(
                f"'{name}' is not a free parameter of this fitter "
                f"({self.free_params}); there is nothing to profile."
            )

        others = [p for p in self.free_params if p != name]

        if not others:

            raise ValueError(
                f"Profiling '{name}' needs at least one other free "
                f"parameter to minimize over."
            )

        values = np.atleast_1d(np.asarray(values, dtype=float))

        chi2 = np.empty(len(values))

        params = []

        carried = {}

        for index, value in enumerate(values):

            sub = self._respawn(others, {name: float(value), **carried})

            sub.best_fit(restarts=restarts, seed=seed)

            chi2[index] = sub.best_fit_chi2

            params.append(sub.best_fit_params)

            if warm_start:
                carried = dict(sub.best_fit_params)

            if progress:
                print(
                    f"  {name} = {value:g}  chi2 = {chi2[index]:.4f}",
                    flush=True,
                )

        return {
            "name": name,
            "values": values,
            "chi2": chi2,
            "delta_chi2": chi2 - chi2.min(),
            "params": params,
        }

    # ------------------------------------------------------------

    def fisher(
        self,
        steps: Redshift | None = None,
        theta: Redshift | None = None,
    ) -> dict:
        """
        Fisher matrix: the curvature of ``chi2`` at the best fit,
        ``F_ij = (1/2) d2 chi2 / d theta_i d theta_j``.

        Central differences throughout -- a forward difference
        would inherit a first-derivative error, and at a minimum
        the first derivative is what is supposed to vanish.

        Cheap where an MCMC is not: ``~2 n^2`` likelihood
        evaluations. That is the whole reason it exists here --
        ``examples/05-case-studies/s8_tension_cmb.ipynb`` needed parameter errors
        from a fit whose every evaluation is a CAMB call, where a
        converged chain is about thirteen hours.

        It is a *Gaussian approximation to the posterior*, good for
        Planck's near-elliptical LCDM contours and poor for a
        parameter against a prior edge or a posterior with a
        plateau. Check it against something before trusting it.

        Parameters
        ----------
        steps : array_like, optional
            Finite-difference step per parameter. Defaults to
            1e-3 of each prior width, which is small enough for the
            quadratic approximation and large enough not to be
            eaten by an analytic likelihood's own noise. **Set it
            by hand for a Boltzmann-code likelihood**, where the
            noise floor is much higher -- roughly a third of each
            parameter's expected uncertainty.
        theta : array_like, optional
            Point to expand about. Defaults to the best fit, which
            must therefore have been found.

        Returns
        -------
        dict
            ``matrix``, its inverse as ``covariance``, ``errors``
            (the square roots of the diagonal), ``theta`` and
            ``steps``.
        """

        if theta is None:

            if self.best_fit_result is None:

                raise RuntimeError(
                    "fisher() expands about the best fit; call "
                    "best_fit() first, or pass theta= explicitly."
                )

            theta = np.array(
                [self.best_fit_params[k] for k in self.free_params],
                dtype=float,
            )

        theta = np.asarray(theta, dtype=float)

        if steps is None:
            steps = 1.0e-3 * self._prior_width()

        steps = np.broadcast_to(
            np.asarray(steps, dtype=float), theta.shape,
        ).copy()

        n = len(theta)

        chi2 = self.logpost.chi2

        centre = chi2(theta)

        matrix = np.zeros((n, n))

        for i in range(n):

            shift = np.zeros(n)
            shift[i] = steps[i]

            matrix[i, i] = (
                chi2(theta + shift) - 2.0 * centre + chi2(theta - shift)
            ) / steps[i] ** 2 / 2.0

        for i in range(n):
            for j in range(i + 1, n):

                si, sj = np.zeros(n), np.zeros(n)
                si[i], sj[j] = steps[i], steps[j]

                mixed = (
                    chi2(theta + si + sj) - chi2(theta + si - sj)
                    - chi2(theta - si + sj) + chi2(theta - si - sj)
                )

                matrix[i, j] = matrix[j, i] = (
                    mixed / (4.0 * steps[i] * steps[j]) / 2.0
                )

        covariance = np.linalg.inv(matrix)

        return {
            "matrix": matrix,
            "covariance": covariance,
            "errors": np.sqrt(np.diag(covariance)),
            "theta": theta,
            "steps": steps,
            "free_params": list(self.free_params),
        }

    # ------------------------------------------------------------

    def best_fit(self, x0: Redshift | None = None,
                 bounds: Sequence[tuple[float, float]] | None = None,
                 eps: float | None = None,
                 method: str = "L-BFGS-B", restarts: int = 0,
                 seed: int | None = None) -> BestFitResult:
        """
        Maximum-likelihood point via ``scipy.optimize.minimize``
        (L-BFGS-B), starting either from ``x0``, from the
        highest-posterior MCMC sample if available, or from
        ``initial``.

        Parameters
        ----------
        x0 : array_like, optional
            Starting point.

        bounds : list of (float, float), optional
            Defaults to the prior bounds.

        eps : float or array_like, optional
            Step size for L-BFGS-B's numerical gradient. Defaults to
            SciPy's ~1.5e-8, which is right for a smooth analytic
            chi2 and **wrong for an expensive or numerically noisy
            one** -- see the note below.

        method : str, optional
            Any ``scipy.optimize.minimize`` method. The default
            L-BFGS-B is the right choice for the analytic
            likelihoods; ``"Nelder-Mead"`` is what the automatic
            rescue below falls back to.

        restarts : int, optional
            Additional starting points, drawn uniformly from the
            prior, kept if they land lower. Zero by default, because
            for a single-minimum posterior it only costs time.

            Set it when the likelihood surface can have more than
            one minimum. The case that motivated this: profiling
            ``LsCDM``'s ``z_dagger`` across the redshift of a BAO
            measurement, where the model's prediction jumps and the
            surface acquires a second, worse basin. A single start
            landed in it and returned a ``chi2`` *higher* than
            nested ``LCDM``'s -- which is impossible, since
            ``LsCDM`` contains ``LCDM`` at ``z_dagger -> inf``, and
            was the only reason the failure was noticed at all.

            No stall detection or Nelder-Mead rescue is skipped for
            the restarts; each one is a full ``best_fit`` attempt.

        seed : int, optional
            Seed for the restart draws, so a multi-start fit is
            reproducible.

        Notes
        -----
        L-BFGS-B estimates gradients by finite differences, and
        SciPy's default step is about 1.5e-8. For every likelihood
        built on interpolation tables and closed-form ``E(z)`` that
        is fine. For a likelihood that calls a Boltzmann code it is
        not: a 1.5e-8 change in ``H0`` moves chi2 by less than
        CAMB's own numerical noise, so the estimated gradient is
        zero, and **the optimizer returns the starting point
        unchanged while reporting success**.

        That is a silent wrong answer -- the caller gets their
        initial guess back labelled "best fit" -- so it is detected
        rather than left to be discovered. If the first attempt does
        not move, this warns and retries with **Nelder-Mead**, which
        needs no gradient at all and so cannot be defeated the same
        way. The better of the two results is kept.

        Measured on the full Planck 2018 CMB (652 points, six free
        parameters), starting well away from the minimum:

        ==============================  =========  ======
        attempt                            chi2     nfev
        ==============================  =========  ======
        L-BFGS-B, default step            1359.63      21
        L-BFGS-B, prior-scaled step       1091.87     161
        Powell                             997.74     218
        **Nelder-Mead**                  **993.79**    422
        ==============================  =========  ======

        For scale, chi2 at Planck's own published best fit is
        993.17 -- so only the derivative-free attempt actually finds
        the minimum, and the default one is wrong by 366.

        The rescue runs only when the first attempt failed outright,
        so nothing that already worked changes. That matters: a
        larger step is *not* uniformly better, and on some cheap
        likelihoods it converges to a slightly worse minimum than
        the default does.

        The rescue cannot help with a *local* minimum, which is a
        different failure: there the optimizer converges properly,
        reports success, and is simply in the wrong basin. Nothing
        in the result distinguishes that from the right answer, so
        it is left to the caller -- ``restarts`` is the lever.
        """

        if x0 is None:

            if self.sampler is not None:

                flat = self.flat_samples()
                log_prob = self.sampler.get_log_prob(
                    discard=self.burnin, flat=True,
                )
                x0 = flat[np.argmax(log_prob)]

            else:

                x0 = self.theta0

        if bounds is None:
            bounds = list(zip(self.prior.lower, self.prior.upper))

        x0 = np.asarray(x0, dtype=float)

        options = {} if eps is None else {"eps": eps}

        result = self._single_best_fit(x0, bounds, options, eps, method)

        if restarts:

            rng = np.random.default_rng(seed)

            for start in self._restart_points(rng, int(restarts)):

                candidate = self._single_best_fit(
                    start, bounds, options, eps, method,
                )

                if candidate.fun < result.fun:

                    result = candidate

        # Leave the cosmology *at* the best fit rather than wherever
        # the last objective evaluation happened to land. Without
        # this, reading a per-likelihood chi2 after the fit reports
        # the wrong cosmology -- silently, and with restarts on,
        # from a random draw. It cost a notebook section.
        self.logpost.chi2(result.x)

        self.best_fit_result = result

        return result

    # ------------------------------------------------------------

    def _restart_points(self, rng, wanted, budget=40):
        """
        Draw starting points uniformly from the prior, keeping only
        those where ``chi2`` is finite.

        Prior-bounded is not the same as allowed. A wide prior on
        several parameters at once contains plenty of corners where
        the model has no sensible background, or where a tabulated
        likelihood's prediction falls off its grid -- and those come
        back as ``inf``. Starting an optimizer there buys nothing:
        the objective is flat at infinity, so there is no direction
        to move in and the run is wasted.

        ``budget`` caps the draws so a prior that is mostly excluded
        cannot spin forever. Falling short is not an error -- the
        first attempt's result still stands.
        """

        lower = np.asarray(self.prior.lower, dtype=float)
        upper = np.asarray(self.prior.upper, dtype=float)

        points = []

        for _ in range(wanted * budget):

            if len(points) >= wanted:
                break

            start = rng.uniform(lower, upper)

            if np.isfinite(self.logpost.chi2(start)):

                points.append(start)

        return points

    # ------------------------------------------------------------

    def _single_best_fit(self, x0, bounds, options, eps, method):
        """
        One optimizer run from ``x0``, stall rescue included.
        """

        from scipy.optimize import minimize

        result = minimize(
            self.logpost.chi2,
            x0,
            method=method,
            bounds=bounds,
            options=options,
        )

        rescuable = eps is None and method == "L-BFGS-B"

        if rescuable and self._optimizer_stalled(result, x0, bounds):

            result = self._retry_best_fit(result, x0, bounds)

        return result

    # ------------------------------------------------------------

    def _prior_width(self) -> np.ndarray:
        """
        Width of each free parameter's prior -- the only scale the
        fitter knows for each parameter, and so the natural unit for
        "did this move?" and for a fallback gradient step.
        """

        return (

            np.asarray(self.prior.upper, dtype=float)

            - np.asarray(self.prior.lower, dtype=float)

        )

    # ------------------------------------------------------------

    def _optimizer_stalled(self, result, x0, bounds) -> bool:
        """
        Whether the optimizer finished essentially where it started.

        Measured in units of each parameter's prior width, because
        the parameters have wildly different scales (``H0`` ~ 70,
        ``Omega_b`` ~ 0.05) and an absolute threshold would be
        meaningless for one of them.
        """

        moved = np.abs(np.asarray(result.x, dtype=float) - x0)

        return bool(np.all(moved / self._prior_width() < 1.0e-6))

    # ------------------------------------------------------------

    def _retry_best_fit(self, first, x0, bounds):
        """
        Second attempt with Nelder-Mead, for the stalled case
        described in :meth:`best_fit`.

        Derivative-free by construction, so the numerical noise that
        defeated the gradient cannot defeat it. The initial simplex
        is built from the prior widths rather than left to SciPy's
        default (5% of each coordinate's *value*), because these
        parameters differ by three orders of magnitude and a
        value-relative simplex is badly shaped for them.
        """

        import warnings

        from scipy.optimize import minimize

        warnings.warn(

            "best_fit() did not move from its starting point: "

            "L-BFGS-B's default finite-difference step (~1.5e-8) is "

            "too small for this likelihood to respond to, which "

            "happens when a Boltzmann code's own numerical noise "

            "exceeds the chi2 change such a step produces. Retrying "

            "with Nelder-Mead, which needs no gradient. This is "

            "slower; pass `method=` or `eps=` to control it.",

            UserWarning,

            stacklevel=3,

        )

        width = self._prior_width()

        simplex = np.vstack(

            [x0] + [

                x0 + 0.02 * width[i] * np.eye(len(x0))[i]

                for i in range(len(x0))

            ],

        )

        # Keep the simplex inside the prior.
        lower = np.asarray(self.prior.lower, dtype=float)
        upper = np.asarray(self.prior.upper, dtype=float)

        simplex = np.clip(simplex, lower, upper)

        second = minimize(

            self.logpost.chi2,

            x0,

            method="Nelder-Mead",

            bounds=bounds,

            options={

                "initial_simplex": simplex,

                "maxfev": 2000,

                "xatol": 1e-4,

                "fatol": 1e-3,

            },

        )

        return second if second.fun < first.fun else first

    # ------------------------------------------------------------

    @property
    def best_fit_params(self) -> dict:

        if self.best_fit_result is None:
            raise RuntimeError("Call best_fit() first.")

        return dict(zip(self.free_params, self.best_fit_result.x))

    @property
    def best_fit_chi2(self) -> float:

        if self.best_fit_result is None:
            raise RuntimeError("Call best_fit() first.")

        return float(self.best_fit_result.fun)

    # ============================================================
    # Consolidated result
    # ============================================================

    @property
    def result(self) -> FitResult:
        """
        A single :class:`~stats.results.FitResult` snapshot of
        whatever this fitter currently has: the best-fit point
        (if :meth:`best_fit` has been called), the MCMC posterior
        (if :meth:`run_mcmc` has been called), or both.

        This is a read-only summary built on demand -- it does
        not replace :meth:`summary`, :meth:`convergence`,
        :attr:`best_fit_params` etc., which remain the way to get
        each piece individually. It exists for the "one object,
        printable in one line, saveable to JSON" case, e.g.::

            fit.run_mcmc(...)
            fit.best_fit()
            print(fit.result)
            fit.result.save_json("cpl_fit.json")
        """

        best_fit = None
        if self.best_fit_result is not None:
            best_fit = BestFitResult(
                params=self.best_fit_params,
                chi2=self.best_fit_chi2,
                ndim=self.ndim,
                n_data=self.n_data,
                success=bool(self.best_fit_result.success),
                message=str(self.best_fit_result.message),
            )

        mcmc = None
        if self.sampler is not None:
            mcmc = MCMCResult(
                summary=self.summary(),
                convergence=self.convergence(),
                # `int()`: with `run_mcmc(save=...)` the sampler's
                # counters come from HDF5 attributes, so emcee hands
                # back `np.int64` rather than `int` -- which
                # `json.dumps` refuses, breaking `save_json()` (and
                # the GUI's JSON download) for exactly the runs that
                # were saved to disk.
                nwalkers=int(self.sampler.nwalkers),
                nsteps=int(self.sampler.iteration),
                burnin=self.burnin,
                ndim=self.ndim,
                acceptance_fraction=float(
                    np.mean(self.sampler.acceptance_fraction)
                ),
            )

        return FitResult(
            model=self.model_cls.__name__,
            datasets=self.dataset_names,
            free_params=self.free_params,
            best_fit=best_fit,
            mcmc=mcmc,
        )

    # ------------------------------------------------------------

    def __repr__(self):

        return (
            f"Fitter(model={self.model_cls.__name__}, "
            f"datasets={self.dataset_names}, "
            f"free_params={self.free_params})"
        )
