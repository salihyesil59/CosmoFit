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

See :mod:`~theory.minisuperspace` for the reduction itself and
what it can and cannot do (a general ``f(R)`` is fourth-order and
is refused rather than approximated), and :mod:`~theory.solve` for
how a transcendental constraint is solved on the physical branch.
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
    "GEOMETRIES",
    "Minisuperspace",
    "field_equations",
    "friedmann_constraint",
    "reduce_order",
]
