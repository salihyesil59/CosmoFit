"""
The interpolator a Boltzmann theory returns for ``Pk_interpolator``,
on spectra whose values are known everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmofit.theories.power_spectrum import PowerSpectrumInterpolator


def spectrum(z, k):
    """A smooth, positive stand-in: a turnover, damped with redshift."""

    z, k = np.meshgrid(z, k, indexing="ij")

    return 2.0e4 * k / (1.0 + (k / 0.02) ** 2.6) / (1.0 + z) ** 2


Z = np.linspace(0.0, 3.0, 31)
K = np.logspace(-4, 1, 300)


def test_interpolates_a_known_spectrum():

    P = PowerSpectrumInterpolator(Z, K, spectrum(Z, K))

    assert P.islog

    z = np.array([0.05, 0.77, 1.31, 2.9])
    k = np.array([3e-4, 0.013, 0.2, 7.0])

    np.testing.assert_allclose(P.P(z, k), np.diag(spectrum(z, k)), rtol=1e-4)
    np.testing.assert_allclose(P.P(z, k, grid=True), spectrum(z, k), rtol=1e-4)
    np.testing.assert_allclose(P.logP(z, k), np.log(P(z, k)))

    # The grid's own nodes come back as they went in.
    np.testing.assert_allclose(P.P(Z[3], K), spectrum(Z, K)[3], rtol=1e-12)

    assert (P.zmin, P.zmax) == (0.0, 3.0)
    assert P.kmin == K[0] and P.kmax == K[-1]


def test_broadcasts_like_numpy():

    P = PowerSpectrumInterpolator(Z, K, spectrum(Z, K))

    assert np.ndim(P.P(0.5, 0.1)) == 0
    assert P.P(0.5, K).shape == K.shape
    assert P.P(Z[:, None], K[None, :]).shape == (Z.size, K.size)


def test_one_redshift():

    P = PowerSpectrumInterpolator([0.0], K, spectrum([0.0], K))

    np.testing.assert_allclose(P.P(0.0, [0.01, 0.3]), spectrum([0.0], [0.01, 0.3])[0], rtol=1e-4)

    with pytest.raises(ValueError, match="'z'"):
        P.P(0.1, 0.1)


def test_two_redshifts_are_interpolated_linearly():

    z = [0.0, 1.0]
    P = PowerSpectrumInterpolator(z, K, spectrum(z, K))

    # Linear in ln P between the two: the geometric mean half-way.
    middle = np.sqrt(spectrum([0.0], [0.1]) * spectrum([1.0], [0.1]))[0, 0]

    assert P.P(0.5, 0.1) == pytest.approx(middle, rel=1e-6)


def test_a_cross_spectrum_that_changes_sign():

    values = spectrum(Z, K) * np.cos(np.log(K))[None, :]

    P = PowerSpectrumInterpolator(Z, K, values)

    assert not P.islog

    np.testing.assert_allclose(P.P(Z[5], K), values[5], rtol=1e-12, atol=1e-12)

    with pytest.raises(ValueError, match="changes sign"):
        P.logP(0.5, 0.1)


def test_no_extrapolation():

    P = PowerSpectrumInterpolator(Z, K, spectrum(Z, K))

    # The ends themselves are inside.
    P.P(Z[0], K[0])
    P.P(Z[-1], K[-1])

    with pytest.raises(ValueError, match="'z'"):
        P.P(3.1, 0.1)

    with pytest.raises(ValueError, match="k_max"):
        P.P(1.0, 11.0)

    with pytest.raises(ValueError, match="k_max"):
        P.P(1.0, 1e-5)


@pytest.mark.parametrize("z, k, values", [
    ([0.0, 1.0], K, np.ones((3, K.size))),
    ([1.0, 0.0], K, np.ones((2, K.size))),
    ([0.0, 1.0], -K, np.ones((2, K.size))),
])
def test_malformed_grids_are_refused(z, k, values):

    with pytest.raises(ValueError):
        PowerSpectrumInterpolator(z, k, values)
