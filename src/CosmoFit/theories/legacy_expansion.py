"""
Any old-style model as a dark sector: models built from an action,
with ``define_model``, or by subclassing ``Cosmology``.

The wrapped class supplies ``E(z)`` itself, without radiation -- it
never had any -- so this works with ``radiation: false`` only. Its
parameters beyond the background's own (``H0``, ``Omega_m``,
``Omega_k``, ...) become parameters of the background, defaulting to
the class's defaults::

    theory:
      background:
        radiation: false
        dark_energy: {name: legacy, model: "mypackage.models:MyModel"}
"""

from __future__ import annotations

from .dark_sector import DarkSector


__all__ = ["LegacyExpansion"]


#: Parameters the background itself provides, never forwarded as the
#: sector's own.
_BACKGROUND_NAMES = {"H0", "Omega_m", "Omega_b", "Omega_k", "N_eff", "m_nu"}


class LegacyExpansion(DarkSector):
    """
    Options
    -------
    model : str or type
        A ``Cosmology`` subclass, a built-in model name, or an import
        path.
    """

    name = "legacy"
    radiation_ok = False

    def __init__(self, model="LCDM"):

        from CosmoFit.core.legacy import _resolve_model

        self.model_cls = _resolve_model(model)

        params_cls = self.model_cls.PARAMS_CLASS

        self.defaults = {
            name: value for name, value in params_cls.defaults().items()
            if name not in _BACKGROUND_NAMES
        }

        self.flat_only = bool(getattr(self.model_cls, "FLAT_ONLY", False))

        self._cache = {}

    def _instance(self, **values):

        key = tuple(sorted(values.items()))

        if key not in self._cache:

            params_cls = self.model_cls.PARAMS_CLASS

            full = dict(params_cls.defaults())
            full.update(values)

            self._cache.clear()
            self._cache[key] = self.model_cls(
                params_cls(**{n: full[n] for n in params_cls.names()})
            )

        return self._cache[key]

    def solve(self, ctx, **p):

        model = self._instance(
            H0=ctx.H0, Omega_m=ctx.Omega_cb, Omega_k=ctx.Omega_k, **p,
        )

        def E2(z):
            return model.E(z) ** 2

        return E2

    def jumps(self, **p):
        return ()

    def __repr__(self) -> str:
        return f"LegacyExpansion({self.model_cls.__name__})"
