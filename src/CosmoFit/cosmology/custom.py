"""
Minimal-code custom cosmological models.

CosmoFit's built-in models (LCDM, CPL, ...) are hand-written
``Cosmology`` subclasses. Testing a genuinely new model -- one not
in the literature, invented to check an idea -- against CosmoFit's
existing datasets/likelihoods/MCMC machinery shouldn't require
writing one. :func:`define_model` builds a ``Cosmology`` subclass
from a single ``E(z)`` function (everything downstream -- distances,
every likelihood, every plot except ``deceleration()``/``w_of_z()``
-- only ever needs ``E(z)``, see ``Cosmology.__init_subclass__``).

Example
-------
>>> from CosmoFit import define_model, Fitter
>>> import numpy as np
>>>
>>> MyModel = define_model(
...     "MyModel",
...     E=lambda p, z: np.sqrt(
...         p["Omega_m"] * (1 + z) ** 3
...         + (1 - p["Omega_m"]) * (1 + z) ** (3 * (1 + p["w0"])) * (1 + p["beta"] * z)
...     ),
...     extra_params={"beta": {"default": 0.0, "bounds": (-2.0, 2.0), "label": r"$\\beta$"}},
... )
>>>
>>> fit = Fitter(
...     model=MyModel,
...     datasets=["cc", "desi"],
...     free_params=["H0", "Omega_m", "w0", "beta"],
...     initial={"H0": 67.4, "Omega_m": 0.315, "w0": -1.0, "beta": 0.0},
... )
>>> fit.run_mcmc(nwalkers=48, nsteps=3000, burnin=500)
>>> fit.best_fit()
>>> fit.plots.corner()

``E`` (and, if given, ``w``/``dEdz``/``Omega_de``) receives ``(p, z)``:
``p`` is a plain ``dict`` of every current parameter value (standard
ones -- ``H0``, ``Omega_m``, ``Omega_k``, ``w0``, ``wa``, ... -- plus
any ``extra_params``), and ``z`` is already an ``ndarray``. It must
be written with ``numpy`` operations (it is called on a whole
redshift grid at once, not per-point) and return an ``ndarray`` the
same shape as ``z``.

If ``dEdz`` isn't supplied, a central-finite-difference fallback is
installed so ``fitter.plots.deceleration()`` works without deriving
one by hand -- direct ``Cosmology`` subclasses don't get this
fallback (they keep the base class's strict
``NotImplementedError``, same as every built-in model).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.core.base import Cosmology


# ============================================================
# Function wrapping
# ============================================================

def _wrap(func):
    """
    Wrap a user ``f(params_dict, z) -> ndarray`` into a
    ``Cosmology`` method ``self, z -> ndarray``.
    """

    def method(self, z):

        z = np.asarray(z, dtype=float)

        return np.asarray(func(self.params.as_dict(), z), dtype=float)

    return method


# ------------------------------------------------------------

def _numerical_dEdz(self, z, h: float = 1e-4):
    """
    Central-finite-difference fallback for ``dEdz``, used when
    :func:`define_model` isn't given one explicitly. Only feeds
    ``background.q()`` (the deceleration-parameter plot); fitting
    and every other plot only need ``E(z)``.
    """

    z = np.asarray(z, dtype=float)

    return (self.E(z + h) - self.E(z - h)) / (2.0 * h)


# ============================================================
# define_model
# ============================================================

def define_model(
    name: str,
    E,
    *,
    extra_params: dict | None = None,
    w=None,
    dEdz=None,
    Omega_de=None,
) -> type:
    """
    Build a new :class:`~cosmology.core.base.Cosmology` subclass
    from a plain ``E(z)`` function.

    Parameters
    ----------
    name : str
        Model name (``model.MODEL_NAME`` and the generated class's
        ``__name__``).

    E : callable(params: dict, z: ndarray) -> ndarray
        Dimensionless Hubble parameter. Required -- this alone is
        enough to fit the model against every CosmoFit dataset and
        produce every plot except :meth:`~plots.FitPlotter.w_of_z`
        / :meth:`~plots.FitPlotter.deceleration`.

    extra_params : dict[str, dict], optional
        New parameters this model needs beyond the standard set
        (``H0``, ``Omega_m``, ``Omega_k``, ``w0``, ``wa``, ``rd``,
        ``MB``, ``Omega_b``, ``A_s``, ``alpha``). Maps name -> spec
        dict with keys ``"default"`` (float, default 0.0),
        ``"bounds"`` (``(lower, upper)``, needed only if the
        parameter will be fit rather than fixed, unless bounds are
        instead passed to ``Fitter(bounds=...)``), and ``"label"``
        (str, optional, LaTeX label for corner plots). Names must
        not collide with the standard set.

    w : callable(params, z) -> ndarray, optional
        Dark-energy equation of state, for
        :meth:`~plots.FitPlotter.w_of_z`. Not required for fitting.

    dEdz : callable(params, z) -> ndarray, optional
        Derivative of ``E(z)``, for
        :meth:`~plots.FitPlotter.deceleration` /
        ``background.q()``. Defaults to a numerical (central
        finite-difference) approximation of ``E`` if omitted.

    Omega_de : callable(params, z) -> ndarray, optional
        Dark-energy density parameter, for ``background.Omega_de()``.

    Returns
    -------
    type
        A new ``Cosmology`` subclass, usable directly as
        ``Fitter(model=..., ...)``.
    """

    extra_params = dict(extra_params or {})

    attrs = {
        "MODEL_NAME": name,
        "EXTRA_PARAMS": extra_params,
        "E": _wrap(E),
    }

    if w is not None:
        attrs["w"] = _wrap(w)

    if dEdz is not None:
        attrs["dEdz"] = _wrap(dEdz)
    else:
        attrs["dEdz"] = _numerical_dEdz

    if Omega_de is not None:
        attrs["Omega_de"] = _wrap(Omega_de)

    return type(name, (Cosmology,), attrs)


# ============================================================
# model_from_expression
# ============================================================

#: Whitelisted names available inside a model expression, in
#: addition to `z` and the model's own parameters. Deliberately
#: small -- elementwise numpy math only, nothing that reads from or
#: writes to the outside world.
_SAFE_NAMESPACE = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "abs": np.abs,
    "sign": np.sign,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "pi": np.pi,
    "e": np.e,
}


def _compile_expression(expr: str):
    """
    Compile a Python expression string into an
    ``f(params: dict, z: ndarray) -> ndarray`` callable, for
    :func:`model_from_expression`.

    Evaluated with ``eval()`` but with builtins removed and only
    :data:`_SAFE_NAMESPACE`'s elementwise numpy functions, the
    model's own current parameter values, and ``z`` available as
    names -- no imports, attribute access, or builtins reach the
    expression. This is a convenience for trusted, local use (e.g.
    a GUI text box run on your own machine), not a hardened sandbox
    for untrusted/publicly-submitted input.
    """

    code = compile(expr, "<model expression>", "eval")

    def f(params, z):

        namespace = dict(_SAFE_NAMESPACE)
        namespace.update(params)
        namespace["z"] = z

        return eval(code, {"__builtins__": {}}, namespace)

    return f


# ------------------------------------------------------------

def model_from_expression(
    name: str,
    E: str,
    *,
    extra_params: dict | None = None,
    w: str | None = None,
    dEdz: str | None = None,
    Omega_de: str | None = None,
) -> type:
    """
    Same as :func:`define_model`, but ``E``/``w``/``dEdz``/
    ``Omega_de`` are given as Python expression **strings** (e.g.
    ``"sqrt(Omega_m*(1+z)**3 + (1-Omega_m)*(1+z)**(3*(1+w0)))"``)
    instead of callables -- convenient for text-entry UIs (see the
    Streamlit app under ``app/``), where asking for a Python
    function isn't practical. Each expression sees ``z`` and every
    current parameter value (standard ones plus any
    ``extra_params``) as plain names; see :func:`_compile_expression`
    for exactly what else is available.

    Example
    -------
    >>> MyModel = model_from_expression(
    ...     "MyModel",
    ...     E="sqrt(Omega_m*(1+z)**3 + (1-Omega_m)*(1+z)**(3*(1+w0))*(1+beta*z))",
    ...     extra_params={"beta": {"default": 0.0, "bounds": (-2.0, 2.0)}},
    ... )
    """

    return define_model(
        name,
        E=_compile_expression(E),
        extra_params=extra_params,
        w=_compile_expression(w) if w else None,
        dEdz=_compile_expression(dEdz) if dEdz else None,
        Omega_de=_compile_expression(Omega_de) if Omega_de else None,
    )
