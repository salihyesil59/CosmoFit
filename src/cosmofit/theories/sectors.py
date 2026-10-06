"""
Every dark sector by name.

In an input, ``background: {dark_energy: cpl}`` -- or, for a sector with
options of its own, ``dark_energy: {name: legacy, model: ...}``. The old
model class names (``"CPL"``, ``"RunningVacuum"``, ``"FQExponential"``,
...) are accepted too, case-insensitively.
"""

from __future__ import annotations

from .dark_energy import DARK_ENERGY
from .dark_sector import DarkSector
from .holographic import ADE, HDE, RDE
from .legacy_expansion import LegacyExpansion
from .modified_gravity import DGP, Cardassian, FQExponential, FRTLinear, FTPowerLaw
from .running_vacuum import RunningVacuum


__all__ = ["DARK_SECTORS", "get_dark_sector"]


DARK_SECTORS = {
    **DARK_ENERGY,
    **{
        cls.name: cls for cls in (
            DGP, Cardassian, FQExponential, FTPowerLaw, FRTLinear,
            HDE, ADE, RDE, RunningVacuum, LegacyExpansion,
        )
    },
}

#: Old class names, and other spellings people reach for.
_ALIASES = {
    "lcdm": "lambda", "cosmological_constant": "lambda",
    "logarithmicde": "logarithmic", "log": "logarithmic",
    "lscdm": "lscdm", "ide": "ide", "runningvacuum": "rvm",
    "fqexponential": "fq_exponential", "ftpowerlaw": "ft_power_law",
    "frtlinear": "frt_linear", "frhusawicki": "hu_sawicki",
}


def get_dark_sector(spec) -> DarkSector:
    """
    A dark sector from a name, ``{"name": ..., **options}``, an instance,
    or a :class:`DarkSector` subclass.
    """

    if isinstance(spec, DarkSector):
        return spec

    if isinstance(spec, type) and issubclass(spec, DarkSector):
        return spec()

    options = {}

    if isinstance(spec, dict):
        options = dict(spec)
        spec = options.pop("name", "lambda")

    key = str(spec).lower()
    key = _ALIASES.get(key, key)

    if key not in DARK_SECTORS:
        raise ValueError(
            f"Unknown dark sector {spec!r}; known: {sorted(DARK_SECTORS)}."
        )

    return DARK_SECTORS[key](**options)
