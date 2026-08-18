"""
Saving MCMC chains to disk, and picking them back up later.

The MCMC is the expensive part of a fit -- minutes to hours.
Everything built on top of it (summaries, convergence
diagnostics, corner plots, derived-quantity posteriors, model
comparison) is seconds. With the chain living only in memory,
though, every one of those cheap steps is gated behind
re-running the expensive one: reopen the notebook, add one plot,
and the sampler starts again from step zero.

This module gives the chain a home. ``Fitter.run_mcmc(save=...)``
writes it into an HDF5 file *as the sampler runs* (via emcee's
own ``HDFBackend``, so an interrupted run keeps every step it had
already taken), together with enough CosmoFit metadata to say
which posterior it came from. A later session either resumes that
chain where it stopped or just reads it back for analysis,
without sampling at all.

Example
-------
>>> fit = Fitter(model=CPL, datasets=["cc", "desi"], ...)
>>>
>>> # First run: samples, and writes as it goes.
>>> fit.run_mcmc(nsteps=6000, burnin=1000, save="chains/cpl.h5")
>>>
>>> # Any later session, same code: nothing to re-sample.
>>> fit.run_mcmc(nsteps=6000, burnin=1000, save="chains/cpl.h5")
>>> fit.plots.corner()

Analysis-only, without rebuilding the fitter (no datasets are
read, no likelihood is ever evaluated)::

>>> from CosmoFit.stats.chains import open_chain
>>> chain = open_chain("chains/cpl.h5")
>>> chain.summary(burnin=1000)

What is actually in the file
----------------------------
Two HDF5 groups:

``mcmc``
    emcee's own layout, written by ``emcee.backends.HDFBackend``
    -- ``chain`` (nsteps, nwalkers, ndim), ``log_prob``,
    ``accepted``, and the sampler's random state. Readable by
    plain ``h5py``, or by ``emcee`` itself, with or without
    CosmoFit.

``cosmofit``
    CosmoFit's metadata: the model, datasets, free parameters,
    prior bounds, fixed parameter values, burn-in, seed, and
    versions -- a JSON string in a ``<chain name>/metadata``
    attribute on the group (keyed by chain name, so one file can
    hold several). This is what makes a saved chain
    self-describing, and what :func:`compare_signatures` checks
    before a resume, so a chain is never silently continued
    under a posterior it wasn't sampled from.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# Constants
# ============================================================

#: HDF5 group emcee's own backend writes the chain into.
DEFAULT_CHAIN_NAME = "mcmc"

#: HDF5 group CosmoFit writes its metadata into, alongside it.
META_GROUP = "cosmofit"

#: Attribute (a JSON string) holding that metadata.
META_ATTR = "metadata"

#: Metadata keys that define *which posterior* a chain belongs
#: to, and so must match for a resume to be meaningful. Anything
#: else in the metadata (burn-in, seed, timestamps, initial
#: values) is informational -- it describes the run, not the
#: distribution being sampled.
SIGNATURE_KEYS = (
    "model",
    "datasets",
    "free_params",
    "bounds",
    "fixed",
    "dataset_kwargs",
)


# ============================================================
# Helpers
# ============================================================

def _require_h5py():
    """
    Import ``h5py``, or explain how to get it.

    It is a declared dependency, so this should never fire on a
    clean install -- but an environment assembled by hand can
    easily be missing it, and the error that surfaces then comes
    from inside emcee and doesn't mention CosmoFit at all.
    """

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Saving or reading MCMC chains needs `h5py`, which "
            "doesn't seem to be installed in this environment. "
            "Install it with `pip install h5py`. Everything else in "
            "CosmoFit works without it -- only `save=`/`resume=` and "
            "`CosmoFit.stats.chains` need it."
        ) from exc

    return h5py


def _json_safe(obj):
    """
    Make ``obj`` JSON-encodable, converting numpy scalars/arrays
    (which metadata values routinely are -- prior bounds are
    ``float64``, GUI inputs are numpy scalars) to plain Python.
    """

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def _now() -> str:

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _package_version() -> str:
    """
    Installed CosmoFit version, recorded in every chain file so a
    chain found later says which version wrote it.
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cosmofit")
    except PackageNotFoundError:
        return "unknown"


def build_metadata(signature: dict, previous: dict | None = None, **extra) -> dict:
    """
    Assemble the metadata dict written next to a chain: the
    posterior's signature (see :data:`SIGNATURE_KEYS`), whatever
    run settings the caller adds, and provenance stamps.

    Parameters
    ----------
    signature : dict
        Identifies the posterior -- from
        :meth:`~stats.fitter.Fitter._chain_signature`.

    previous : dict, optional
        Metadata already in the file, when this is a resumed
        run. Only its ``created`` stamp survives, so a chain
        grown over five sessions still reports when it started.

    **extra
        Anything else worth recording (burn-in, seed, initial
        values, nwalkers ...). Informational -- not compared on
        resume.
    """

    import emcee

    meta = dict(signature)
    meta.update(extra)

    meta["cosmofit_version"] = _package_version()
    meta["emcee_version"] = emcee.__version__
    meta["created"] = (previous or {}).get("created") or _now()
    meta["updated"] = _now()

    return meta


# ============================================================
# Compatibility between a stored chain and a live fitter
# ============================================================

def compare_signatures(stored: dict, current: dict) -> list[str]:
    """
    Differences between two chain signatures (the
    :data:`SIGNATURE_KEYS` subset of two metadata dicts), as
    human-readable strings.

    An empty list means the stored chain samples the same
    posterior as the fitter that produced ``current``, and can
    be resumed or reused. Anything else must not be: continuing
    a chain under a changed model, dataset combination, free
    parameter list, prior, or fixed parameter value silently
    welds two different distributions into one set of samples.

    Parameters
    ----------
    stored, current : dict
        Metadata dicts (from :attr:`ChainFile.metadata` and
        :meth:`~stats.fitter.Fitter._chain_signature`).

    Returns
    -------
    list of str
        One entry per mismatched key, quoting both values.
    """

    differences = []

    for key in SIGNATURE_KEYS:

        # An older/foreign file simply may not carry a key. Only
        # compare what both sides actually have -- a missing key
        # is reported by the caller as "no metadata", not as a
        # mismatch against a default that was never stored.
        if key not in stored or key not in current:
            continue

        was = _json_safe(stored[key])
        now = _json_safe(current[key])

        if was == now:
            continue

        # `bounds` and `fixed` hold one entry per parameter, and
        # printing both dicts whole to report one changed number
        # buries it. Diff them entry by entry instead.
        if isinstance(was, dict) and isinstance(now, dict):
            differences.extend(_dict_differences(key, was, now))
        else:
            differences.append(f"{key}: stored {was!r}, now {now!r}")

    return differences


def _dict_differences(key: str, was: dict, now: dict) -> list[str]:
    """
    Per-entry differences between two dict-valued signature
    fields, as ``key[name]: stored X, now Y`` lines.
    """

    differences = []

    for name in sorted(set(was) | set(now)):

        if name not in now:
            differences.append(
                f"{key}[{name}]: stored {was[name]!r}, now absent"
            )
        elif name not in was:
            differences.append(
                f"{key}[{name}]: not in the stored chain, now {now[name]!r}"
            )
        elif was[name] != now[name]:
            differences.append(
                f"{key}[{name}]: stored {was[name]!r}, now {now[name]!r}"
            )

    return differences


# ============================================================
# Chain file
# ============================================================

class ChainFile:
    """
    An HDF5 file holding one MCMC chain, plus the CosmoFit
    metadata describing it.

    Wraps ``emcee.backends.HDFBackend`` -- which owns the chain
    itself and needs no help -- with the two things a chain has
    to carry to be re-usable months later: what posterior it came
    from, and how to check that against the fit in front of you.

    Normally built for you by ``Fitter.run_mcmc(save="...")``.
    Construct one directly to put several chains in one file
    (``name=``), or to inspect a file without a fitter.

    Parameters
    ----------
    path : str or Path
        The ``.h5`` file. Parent directories are created on
        write.

    name : str
        HDF5 group holding this chain, so one file can hold
        several (e.g. one per model in a comparison). Defaults
        to emcee's own ``"mcmc"``.

    Examples
    --------
    >>> chain = ChainFile("chains/models.h5", name="CPL")
    >>> chain.exists, chain.iteration
    (True, 6000)
    """

    def __init__(self, path, name: str = DEFAULT_CHAIN_NAME):

        self.path = Path(path)
        self.name = str(name)

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """
        Whether the file exists *and* already holds this chain's
        group. False for a path that was never written, and for
        a file holding only some other chain's group.
        """

        if not self.path.exists():
            return False

        h5py = _require_h5py()

        try:
            with h5py.File(self.path, "r") as f:
                return self.name in f
        except OSError:
            # Not an HDF5 file at all, or unreadable. Treat as
            # "nothing to resume from"; a later write raises with
            # h5py's own (accurate) message.
            return False

    @property
    def iteration(self) -> int:
        """
        Number of steps already stored (0 if the chain doesn't
        exist yet).
        """

        if not self.exists:
            return 0

        return int(self.backend(read_only=True).iteration)

    @property
    def shape(self) -> tuple[int, int] | None:
        """
        ``(nwalkers, ndim)`` of the stored chain, or None if
        there isn't one.
        """

        if not self.exists:
            return None

        nwalkers, ndim = self.backend(read_only=True).shape

        return (int(nwalkers), int(ndim))

    # ------------------------------------------------------------
    # emcee backend
    # ------------------------------------------------------------

    def backend(self, read_only: bool = False):
        """
        An ``emcee.backends.HDFBackend`` for this chain -- what
        gets handed to ``emcee.EnsembleSampler(backend=...)``,
        and what every read below goes through.

        Parameters
        ----------
        read_only : bool
            Open without permission to write. Use it for
            analysis, so a stray call can't append to (or reset)
            a finished chain.
        """

        _require_h5py()

        from emcee.backends import HDFBackend

        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        return HDFBackend(str(self.path), name=self.name, read_only=read_only)

    def reset(self, nwalkers: int, ndim: int) -> None:
        """
        Throw away this chain's stored steps and start it empty
        at ``(nwalkers, ndim)``.

        Destructive, and deliberately not something any default
        path reaches: ``run_mcmc(save=..., resume=False)`` is the
        only caller, and only because that argument says so
        explicitly. Other chains in the same file are untouched.
        """

        self.backend().reset(nwalkers, ndim)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """
        The CosmoFit metadata stored alongside the chain (an
        empty dict for a chain written by plain emcee, or one
        that doesn't exist yet).
        """

        if not self.path.exists():
            return {}

        h5py = _require_h5py()

        try:
            with h5py.File(self.path, "r") as f:

                if META_GROUP not in f:
                    return {}

                group = f[META_GROUP]

                raw = group.attrs.get(f"{self.name}/{META_ATTR}")

                if raw is None:
                    return {}

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                return json.loads(raw)

        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def write_metadata(self, meta: dict) -> None:
        """
        Store ``meta`` (JSON-encoded) next to the chain,
        replacing whatever was there.

        Keyed by chain name inside a single ``cosmofit`` group,
        so several chains can share one file without colliding.
        """

        h5py = _require_h5py()

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.path, "a") as f:

            group = f.require_group(META_GROUP)

            group.attrs[f"{self.name}/{META_ATTR}"] = json.dumps(
                _json_safe(meta), indent=2,
            )

    # ------------------------------------------------------------
    # Reading back
    # ------------------------------------------------------------

    def open(self) -> "StoredSampler":
        """
        Read-only access to the stored chain, as a stand-in for
        the ``emcee.EnsembleSampler`` that produced it -- see
        :class:`StoredSampler`.
        """

        if not self.exists:
            raise FileNotFoundError(
                f"No chain '{self.name}' in {self.path}. "
                f"Run `Fitter.run_mcmc(save=...)` to create one."
            )

        return StoredSampler(
            self.backend(read_only=True), metadata=self.metadata,
        )

    def info(self) -> dict:
        """
        A short, printable description of the file: what's in it
        and how far it got. Cheap -- reads attributes and
        metadata, never the chain itself.
        """

        meta = self.metadata

        shape = self.shape

        return {
            "path": str(self.path),
            "name": self.name,
            "exists": self.exists,
            "nsteps": self.iteration,
            "nwalkers": None if shape is None else shape[0],
            "ndim": None if shape is None else shape[1],
            "model": meta.get("model"),
            "datasets": meta.get("datasets"),
            "free_params": meta.get("free_params"),
            "burnin": meta.get("burnin"),
            "created": meta.get("created"),
            "updated": meta.get("updated"),
            "cosmofit_version": meta.get("cosmofit_version"),
        }

    # ------------------------------------------------------------

    def __repr__(self):

        if not self.exists:
            return f"ChainFile({str(self.path)!r}, name={self.name!r}, empty)"

        shape = self.shape
        meta = self.metadata

        model = meta.get("model", "?")
        datasets = "+".join(meta.get("datasets", [])) or "?"

        return (
            f"ChainFile({str(self.path)!r}, name={self.name!r}, "
            f"model={model}, datasets={datasets}, "
            f"nsteps={self.iteration}, nwalkers={shape[0]}, "
            f"ndim={shape[1]})"
        )


# ============================================================
# Read-only sampler
# ============================================================

class StoredSampler:
    """
    A chain read back from disk, wearing the small part of the
    ``emcee.EnsembleSampler`` interface that CosmoFit actually
    asks of it.

    That's the point: ``Fitter.flat_samples()``,
    ``.summary()``, ``.convergence()``, ``.best_fit()``,
    ``plots.corner()``, the derived-quantity posteriors and
    everything else read the sampler through
    ``get_chain``/``get_log_prob``/``nwalkers``/``iteration``.
    Give them one of these instead of a live sampler and they
    work unchanged, on a chain that may have been sampled weeks
    ago in another process.

    Read-only by construction -- there is no ``run``. To add
    steps to a stored chain, go back through
    ``Fitter.run_mcmc(save=..., nsteps=<a larger total>)``.

    Parameters
    ----------
    backend : emcee.backends.Backend
        Usually an ``HDFBackend`` opened ``read_only=True``.

    metadata : dict, optional
        The CosmoFit metadata stored with the chain.
    """

    def __init__(self, backend, metadata=None):

        self.backend = backend
        self.metadata = dict(metadata or {})

    # --- shape ---------------------------------------------------

    @property
    def nwalkers(self) -> int:
        return int(self.backend.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.backend.shape[1])

    @property
    def iteration(self) -> int:
        return int(self.backend.iteration)

    # --- chain ---------------------------------------------------

    def get_chain(self, **kwargs):
        return self.backend.get_chain(**kwargs)

    def get_log_prob(self, **kwargs):
        return self.backend.get_log_prob(**kwargs)

    def get_last_sample(self):
        return self.backend.get_last_sample()

    def get_autocorr_time(self, **kwargs):
        return self.backend.get_autocorr_time(**kwargs)

    @property
    def acceptance_fraction(self):
        return self.backend.accepted / float(self.backend.iteration)

    @property
    def random_state(self):
        return self.backend.random_state

    # --- convenience, for analysis without a Fitter --------------

    @property
    def free_params(self) -> list[str]:
        """
        Parameter names, from the stored metadata. Falls back to
        ``["theta_0", ...]`` for a chain written by plain emcee,
        which carries no names.
        """

        names = self.metadata.get("free_params")

        if names:
            return list(names)

        return [f"theta_{i}" for i in range(self.ndim)]

    @property
    def burnin(self) -> int:
        """
        Burn-in recorded at sampling time, used as the default
        by the methods below.
        """

        return int(self.metadata.get("burnin", 0) or 0)

    def flat_samples(self, burnin=None) -> np.ndarray:
        """
        Post-burn-in samples, walkers flattened together --
        the same array ``Fitter.flat_samples()`` returns.
        """

        if burnin is None:
            burnin = self.burnin

        return self.get_chain(discard=int(burnin), flat=True)

    def samples_dict(self, burnin=None) -> dict:
        """
        Flat samples as a dict of 1D arrays keyed by parameter
        name.
        """

        flat = self.flat_samples(burnin=burnin)

        return {
            name: flat[:, i]
            for i, name in enumerate(self.free_params)
        }

    def summary(self, burnin=None) -> dict:
        """
        Posterior median +/- 68% interval per parameter, in the
        same shape as ``Fitter.summary()`` -- available here
        without constructing a ``Fitter`` at all, i.e. without
        reading a single dataset.
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

    # ------------------------------------------------------------

    def __repr__(self):

        model = self.metadata.get("model", "?")
        datasets = "+".join(self.metadata.get("datasets", [])) or "?"

        return (
            f"StoredSampler(model={model}, datasets={datasets}, "
            f"nsteps={self.iteration}, nwalkers={self.nwalkers}, "
            f"params={self.free_params})"
        )


# ============================================================
# Module-level convenience
# ============================================================

def signature_id(signature: dict, extra: dict | None = None, length: int = 8) -> str:
    """
    A short, stable hash of a posterior signature -- for naming
    chain files automatically, so that "the same fit" always
    lands on the same file and a changed fit lands on a new one
    instead of colliding with (or invalidating) the old.

    Stable across sessions and machines: the digest is taken over
    the JSON encoding with sorted keys, not Python's built-in
    ``hash`` (which is randomly salted per process).

    Parameters
    ----------
    signature : dict
        From :meth:`~stats.fitter.Fitter._chain_signature`.

    extra : dict, optional
        Anything else that should force a *different* file when
        it changes -- typically ``nwalkers`` and ``seed``, which
        don't change the posterior but can't be changed mid-chain
        either.

    length : int
        Hex digits to keep.
    """

    import hashlib

    payload = json.dumps(
        {"signature": _json_safe(signature), "extra": _json_safe(extra or {})},
        sort_keys=True,
    )

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def open_chain(path, name: str = DEFAULT_CHAIN_NAME) -> StoredSampler:
    """
    Read a saved chain back for analysis, without a ``Fitter``
    and without touching any dataset.

    Use this for the cheap questions -- what did the posterior
    look like, has it converged, how long did it run. Anything
    needing the likelihood again (``best_fit()``, per-dataset
    chi2, model comparison, most plots) needs the fitter too:
    see :meth:`~stats.fitter.Fitter.from_chain`.

    >>> chain = open_chain("chains/cpl.h5")
    >>> chain.summary()
    """

    return ChainFile(path, name=name).open()


def chain_info(path, name: str = DEFAULT_CHAIN_NAME) -> dict:
    """
    What's in a chain file -- model, datasets, parameters, how
    many steps -- without reading the chain itself.

    >>> chain_info("chains/cpl.h5")["nsteps"]
    6000
    """

    return ChainFile(path, name=name).info()


def list_chains(path) -> list[str]:
    """
    Names of every chain stored in ``path`` (one file can hold
    several -- see :class:`ChainFile`'s ``name``).
    """

    path = Path(path)

    if not path.exists():
        return []

    h5py = _require_h5py()

    try:
        with h5py.File(path, "r") as f:
            return sorted(k for k in f.keys() if k != META_GROUP)
    except OSError:
        return []
