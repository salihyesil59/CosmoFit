"""
Derived-quantity posteriors must be computed at each sample's own
background.

``derived.q_of_z`` wrote each posterior sample into the cosmology's
parameters and read ``q(z)`` straight back, on the stated grounds that
``E(z)`` and ``dE/dz`` are analytic in the parameters. For HDE and ADE
they are not: both solve an ODE for the dark-energy density and
interpolate it, and that solution is rebuilt only by ``refresh()``. So
every sample was evaluated on the background of whichever point the
cosmology had last been refreshed at, and ``z_t`` and ``q0`` came back
with the spread of the analytic part only.
"""

from __future__ import annotations

import numpy as np

from CosmoFit import HDE, LCDM
from CosmoFit.stats import derived


class _Chain:
    """
    The two things ``derived`` reads from a fitter's sampler.
    """

    def __init__(self, samples):

        # (nsteps, nwalkers=1, ndim)
        self._chain = np.asarray(samples, dtype=float)[:, None, :]

    def get_chain(self, discard=0):

        return self._chain[discard:]


class _Fit:

    def __init__(self, model, free_params, samples, **fixed):

        params_cls = model.PARAMS_CLASS
        values = dict(params_cls.defaults())
        values.update(H0=70.0, Omega_m=0.3, **fixed)

        self.cosmology = model(params_cls(**values))
        self.free_params = list(free_params)
        self.sampler = _Chain(samples)
        self.burnin = 0


def _fresh_q(model, z, **values):

    params_cls = model.PARAMS_CLASS
    full = dict(params_cls.defaults())
    full.update(H0=70.0, Omega_m=0.3)
    full.update(values)

    return model(params_cls(**full)).background.q(z)


def test_hde_samples_each_get_their_own_background():
    """
    ``c_hde`` enters only through the ODE, so with a stale solution
    every sample returned the same q(z).
    """

    z = np.array([0.0, 0.5, 1.0])
    c_values = [0.6, 0.8, 1.2]

    fit = _Fit(HDE, ["c_hde"], [[c] for c in c_values])

    q = derived.q_of_z(fit, z, max_samples=None)

    for row, c in zip(q, c_values):

        np.testing.assert_allclose(row, _fresh_q(HDE, z, c_hde=c),
                                   rtol=1e-8)

    # And the samples really differ: this is not a trivial pass.
    assert np.ptp(q[:, 0]) > 0.05


def test_cosmology_is_left_where_it_was():

    fit = _Fit(HDE, ["c_hde"], [[0.6], [1.2]], c_hde=0.8)

    before = fit.cosmology.background.q(np.array([0.3]))

    derived.q_of_z(fit, [0.3], max_samples=None)

    after = fit.cosmology.background.q(np.array([0.3]))

    np.testing.assert_allclose(after, before, rtol=1e-12)


def test_analytic_models_are_unchanged():

    z = np.array([0.0, 1.0])

    fit = _Fit(LCDM, ["Omega_m"], [[0.25], [0.35]])

    q = derived.q_of_z(fit, z, max_samples=None)

    np.testing.assert_allclose(q[0], _fresh_q(LCDM, z, Omega_m=0.25))
    np.testing.assert_allclose(q[1], _fresh_q(LCDM, z, Omega_m=0.35))
