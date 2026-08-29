"""
The three covariance representations.

Every likelihood in the library ends at one of these -- a dense
matrix, a precision matrix a dataset shipped directly (DES-SN5YR), or
a diagonal of independent errors -- and each is solved against at
every single MCMC step, millions of times over a run.

Which is why `DenseCovariance` has two paths. For a large matrix it
builds an explicit inverse and does a mat-vec, because a triangular
solve is an inherently sequential recurrence that BLAS cannot thread.
That is a real trade of accuracy for speed, and it is *validated at
construction* on random probe vectors rather than trusted: if the
inverse does not reproduce the identity well enough, `solve()` falls
back to the Cholesky factor. Nothing tested either half of that.

The invariant that ties the three together: for the same covariance
expressed three ways, `chi2` must agree. It is the only thing
standing between a dataset shipping a precision matrix and a dataset
shipping the matrix it inverts.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit.data.covariance import (
    DenseCovariance,
    DiagonalCovariance,
    PrecisionCovariance,
    make_covariance,
)


def spd(n, seed=0, scale=1.0):
    """A random symmetric positive-definite matrix."""

    rng = np.random.default_rng(seed)

    a = rng.standard_normal((n, n))

    return scale * (a @ a.T + n * np.eye(n))


# ============================================================
# The three agree
# ============================================================


def test_the_same_covariance_three_ways_gives_the_same_chi_squared():
    """
    The invariant everything else rests on. A dataset that ships a
    precision matrix and one that ships the matrix it inverts must
    produce the same number, or two supernova compilations would not
    be comparable.
    """

    matrix = spd(12, seed=1)

    residual = np.linspace(-1.0, 1.0, 12)

    dense = DenseCovariance(matrix)

    precision = PrecisionCovariance(np.linalg.inv(matrix))

    assert precision.chi2(residual) == pytest.approx(
        dense.chi2(residual), rel=1e-10
    )

    # And for a diagonal one, all three.
    sigma = np.linspace(0.5, 2.0, 12)

    diagonal = DiagonalCovariance(sigma)

    as_dense = DenseCovariance(np.diag(sigma ** 2))

    assert diagonal.chi2(residual) == pytest.approx(
        as_dense.chi2(residual), rel=1e-12
    )

    assert diagonal.solve(residual) == pytest.approx(
        as_dense.solve(residual), rel=1e-12
    )


@pytest.mark.parametrize("n", [3, 40])
def test_solve_really_solves(n):
    """
    `solve(v)` returns `C^-1 v`, whichever path it took to get
    there -- so multiplying back by `C` returns `v`.
    """

    matrix = spd(n, seed=2)

    cov = DenseCovariance(matrix)

    v = np.arange(1.0, n + 1.0)

    np.testing.assert_allclose(matrix @ cov.solve(v), v, rtol=1e-9)


# ============================================================
# The two paths through DenseCovariance.solve
# ============================================================


def test_both_paths_agree_and_the_choice_is_by_size():
    """
    Small matrices use the Cholesky factor, large ones a precomputed
    inverse. The threshold is a performance choice, so the two must
    be numerically interchangeable -- asserted by forcing each.
    """

    matrix = spd(30, seed=3)

    residual = np.linspace(-2.0, 3.0, 30)

    factored = DenseCovariance(matrix, use_inverse=False)

    inverted = DenseCovariance(matrix, use_inverse=True)

    assert factored.uses_inverse is False

    assert inverted.uses_inverse is True

    assert inverted.chi2(residual) == pytest.approx(
        factored.chi2(residual), rel=1e-9
    )

    np.testing.assert_allclose(
        inverted.solve(residual), factored.solve(residual), rtol=1e-9,
    )


def test_a_small_matrix_defaults_to_the_factor():

    assert DenseCovariance(spd(4, seed=4)).uses_inverse is False


def test_an_inverse_that_fails_its_own_accuracy_check_is_not_used(monkeypatch):
    """
    The fallback, which is the point of validating at all: an inverse
    that does not reproduce the identity on probe vectors is
    discarded and `solve()` keeps the Cholesky factor, which is
    always correct and merely slower.

    Forced here by making the built inverse wrong, since a matrix
    ill-conditioned enough to trigger it naturally is not something
    to put in a test.
    """

    from CosmoFit.data import covariance as module

    original = module.cho_solve

    def wrong(*args, **kwargs):
        return original(*args, **kwargs) * 1.5

    monkeypatch.setattr(module, "cho_solve", wrong)

    cov = DenseCovariance(spd(30, seed=5), use_inverse=True)

    assert cov.uses_inverse is False

    monkeypatch.undo()

    # And it still gets the right answer without it.
    matrix = spd(30, seed=5)

    v = np.ones(30)

    np.testing.assert_allclose(matrix @ cov.solve(v), v, rtol=1e-9)


# ============================================================
# Refusing a matrix that is not a covariance
# ============================================================


def test_a_matrix_that_is_not_positive_definite_is_refused():
    """
    A covariance that is not positive definite has no Cholesky
    factor and no meaningful chi-squared. Caught at construction, so
    a bad data file fails at load rather than at the first MCMC step.

    The Cholesky factorization gets there first and raises
    `LinAlgError`; the explicit `ValueError` in `__init__` guards the
    case where the factorization succeeds and the determinant is
    still non-positive.
    """

    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        DenseCovariance(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_positive_definiteness_is_reported_rather_than_assumed():

    assert DenseCovariance(spd(6, seed=6)).is_positive_definite

    assert DiagonalCovariance(np.array([1.0, 2.0])).is_positive_definite

    assert PrecisionCovariance(np.linalg.inv(spd(6, seed=6))).is_positive_definite


# ============================================================
# The derived quantities
# ============================================================


def test_the_log_determinant_matches_a_direct_computation():
    """
    `logdet` enters any likelihood comparison where the covariance
    itself changes, and it is computed by a different route in each
    class -- `slogdet` here, the negated one of the precision matrix
    there, a sum of logs for the diagonal.
    """

    matrix = spd(8, seed=7)

    expected = np.linalg.slogdet(matrix)[1]

    assert DenseCovariance(matrix).logdet == pytest.approx(expected)

    assert PrecisionCovariance(np.linalg.inv(matrix)).logdet == pytest.approx(
        expected, rel=1e-9,
    )

    sigma = np.array([1.0, 2.0, 3.0])

    assert DiagonalCovariance(sigma).logdet == pytest.approx(
        np.sum(np.log(sigma ** 2))
    )


def test_sigma_is_the_marginal_error_bar():
    """
    The square root of the diagonal -- the number a Hubble diagram
    draws as an error bar, even for a dataset whose likelihood uses
    the full matrix.
    """

    matrix = spd(5, seed=8)

    np.testing.assert_allclose(
        DenseCovariance(matrix).sigma, np.sqrt(np.diag(matrix)),
    )

    np.testing.assert_allclose(
        PrecisionCovariance(np.linalg.inv(matrix)).sigma,
        np.sqrt(np.diag(matrix)),
        rtol=1e-9,
    )


def test_the_correlation_matrix_is_unit_diagonal_and_bounded():

    correlation = DenseCovariance(spd(7, seed=9)).correlation

    np.testing.assert_allclose(np.diag(correlation), 1.0)

    assert np.all(np.abs(correlation) <= 1.0 + 1e-12)

    np.testing.assert_allclose(correlation, correlation.T)


def test_the_condition_number_is_reported():

    matrix = spd(6, seed=10)

    assert DenseCovariance(matrix).condition_number == pytest.approx(
        np.linalg.cond(matrix), rel=1e-9,
    )

    sigma = np.array([1.0, 10.0])

    assert DiagonalCovariance(sigma).condition_number == pytest.approx(100.0)


def test_the_shape_and_size_agree():

    for cov in (
        DenseCovariance(spd(5, seed=11)),
        PrecisionCovariance(np.linalg.inv(spd(5, seed=11))),
        DiagonalCovariance(np.ones(5)),
    ):
        assert cov.size == 5
        assert cov.shape == (5, 5)
        assert "5" in str(cov)
        assert "5" in repr(cov)


def test_a_precision_matrix_can_be_turned_back_into_a_covariance():

    matrix = spd(6, seed=12)

    np.testing.assert_allclose(
        PrecisionCovariance(np.linalg.inv(matrix)).matrix, matrix, rtol=1e-9,
    )


# ============================================================
# make_covariance
# ============================================================


def test_the_factory_picks_the_right_class():

    matrix = spd(4, seed=13)

    assert isinstance(make_covariance(cov=matrix), DenseCovariance)

    assert isinstance(
        make_covariance(sigma=np.ones(4)), DiagonalCovariance
    )

    assert isinstance(
        make_covariance(precision=np.linalg.inv(matrix)), PrecisionCovariance
    )


def test_the_factory_insists_on_exactly_one():
    """
    Two of them would be a data file that contradicts itself, and
    none would be a likelihood with no uncertainties at all. Both are
    mistakes worth failing on rather than resolving by precedence.
    """

    with pytest.raises(ValueError, match="exactly one"):
        make_covariance()

    with pytest.raises(ValueError, match="exactly one"):
        make_covariance(cov=spd(3, seed=14), sigma=np.ones(3))
