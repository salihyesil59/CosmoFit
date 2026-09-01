"""
Shared test fixtures.

One thing lives here, and it exists because of a failure that was
very hard to read.

Three of the CMB test modules -- ``test_planck_lensing``,
``test_planck_lowe`` and ``test_act_lensing`` -- build their
likelihood in a **module-scoped** fixture, so every test in the file
shares one mutable cosmology. Several of those tests move a
parameter, check what it does, and move it back. That is fine while
they pass. When one of them fails, the restore line never runs, and
the cosmology stays moved for every test after it.

The result is that a single real failure turns into a cascade of
invented ones. Observed in one full-suite run: CAMB returned NaN for
the baseline spectrum, the first two tests failed on that, the third
failed *before* undoing its ``ln1e10As += 0.10``, the fourth failed
before undoing its own ``+= 0.20`` and ``n_s -= 0.03``, and by the
time the suite reached the sigma8 check the shared cosmology was
0.30 high in ``ln1e10As``, reporting ``sigma8 = 0.932`` against an
expected 0.811. Thirteen failures, one cause. Consecutive runs of
the same suite produced 0, 1, 11 and 14 failures depending on how
far the cascade got, which is what made the underlying problem look
like random flakiness for so long.

The fixture below restores the shared cosmology after every test
that uses one, whether the test passed or failed. It does not fix
whatever makes CAMB return NaN -- that is still open -- but it stops
one failure from manufacturing a dozen more, so the next person to
look at a failing run sees the cause instead of the wreckage.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def restore_shared_cosmology(request):
    """
    Undo any parameter a test left moved on a shared likelihood.

    A no-op for tests that don't ask for a ``likelihood`` fixture,
    which is most of the suite.
    """

    if "likelihood" not in request.fixturenames:

        yield

        return

    try:

        likelihood = request.getfixturevalue("likelihood")

    except pytest.skip.Exception:

        # The module's own fixture skipped, e.g. CAMB missing.
        yield

        return

    cosmology = getattr(likelihood, "cosmology", None)

    if cosmology is None or not hasattr(cosmology, "params"):

        yield

        return

    params = cosmology.params

    before = {name: getattr(params, name) for name in params.names()}

    derive_sigma8 = getattr(cosmology, "derive_sigma8", None)

    try:

        yield

    finally:

        moved = False

        for name, value in before.items():

            if getattr(params, name) != value:

                setattr(params, name, value)

                moved = True

        if (
            derive_sigma8 is not None
            and getattr(cosmology, "derive_sigma8", None) != derive_sigma8
        ):

            cosmology.derive_sigma8 = derive_sigma8

            moved = True

        # Only pay for the invalidation when something actually
        # changed -- refresh() drops the Boltzmann cache, and most
        # tests here leave the cosmology exactly as they found it.
        if moved and hasattr(cosmology, "refresh"):

            cosmology.refresh()
