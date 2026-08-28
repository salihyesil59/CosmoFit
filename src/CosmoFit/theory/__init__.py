"""
Build a cosmological model from an action rather than from an
already-solved expansion history.

Every model in :mod:`cosmology.models` encodes the *result* of a
derivation somebody did by hand: an ``E(z)`` typed in from a
paper. :func:`cosmology.custom.define_model` lowers the barrier to
adding one, but not the work -- it still wants ``E(z)``.

This subpackage takes the input instead of the output. Give it a
gravitational action on an FLRW metric, and it reduces the action
to a point-like Lagrangian, varies the lapse to get the Friedmann
constraint, solves that constraint for ``E(z)``, and hands back an
ordinary :class:`~cosmology.core.base.Cosmology` subclass that
every dataset, likelihood, sampler and plot in this library
already knows how to use.

>>> from CosmoFit import Fitter
>>> from CosmoFit.theory import Action
>>>
>>> model = Action(
...     "T + alpha * (-T)**b",
...     geometry="teleparallel",
...     params={
...         "alpha": {"default": 1.0, "bounds": (0.0, 20.0)},
...         "b": {"default": 0.1, "bounds": (-3.0, 0.99)},
...     },
...     closure="alpha",
... ).build("PowerLawFT")
>>>
>>> fit = Fitter(
...     model=model,
...     datasets=["cc", "desi"],
...     free_params=["H0", "Omega_m", "b"],
... )
>>> fit.best_fit()

An action can also carry dynamical scalar fields, in which case
the expansion history is integrated rather than solved pointwise:

>>> quintessence = Action(
...     "R",
...     fields={"phi": "X - V0*exp(-lam*phi)"},
...     params={
...         "V0": {"default": 2.1, "bounds": (0.05, 50.0)},
...         "lam": {"default": 0.5, "bounds": (0.0, 1.7)},
...     },
...     closure="V0",
... ).build("ExponentialQuintessence")

A general ``f(R)`` is fourth-order and needs its own reduction,
which happens automatically:

>>> starobinsky = Action(
...     "R - 2*Lam + alpha_fr*R**2",
...     params={
...         "Lam": {"default": 2.1, "bounds": (0.0, 6.0)},
...         "alpha_fr": {"default": 1e-3, "bounds": (1e-6, 1.0)},
...     },
... ).build("Starobinsky")

See :mod:`~theory.minisuperspace` for the reduction itself,
:mod:`~theory.curvature` for the Lagrange-multiplier route a
general ``f(R)`` takes and why it integrates *backwards*,
:mod:`~theory.solve` for how a transcendental constraint is solved
on the physical branch, and :mod:`~theory.fields` for why a
field's initial conditions are set early rather than today.
"""

try:
    import sympy as _sympy

except ModuleNotFoundError as _exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "CosmoFit.theory derives Friedmann equations symbolically "
        "and needs sympy, which is an optional dependency: "
        "pip install 'cosmofit[theory]'. Nothing else in the "
        "library requires it -- models written directly as E(z) "
        "(cosmology.models, cosmology.custom.define_model) work "
        "without it."
    ) from _exc

from CosmoFit.theory.action import Action, Fluid, STANDARD_FLUIDS

from CosmoFit.theory.curvature import is_higher_order

from CosmoFit.theory.minisuperspace import (
    GEOMETRIES,
    Minisuperspace,
    field_equations,
    friedmann_constraint,
    reduce_order,
)

__all__ = [
    "Action",
    "Fluid",
    "STANDARD_FLUIDS",
    "is_higher_order",
    "GEOMETRIES",
    "Minisuperspace",
    "field_equations",
    "friedmann_constraint",
    "reduce_order",
]
