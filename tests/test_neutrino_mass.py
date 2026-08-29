"""
The massive-neutrino parameter, ``Sum m_nu``, end to end.

This capability worked when it was built and nothing pinned it,
which is the state a parameter is in just before it silently stops
working. It has two consumers that never talk to each other:

* :mod:`CosmoFit.cosmology.calculators.sound_horizon`, where the
  neutrinos are relativistic at the drag epoch and raise ``r_d``;
* :mod:`CosmoFit.cosmology.boltzmann`, where they free-stream out of
  the small-scale power and lower ``sigma8``.

Both of them read the *same* ``Omega_m``, and both have to agree on
what it contains -- CosmoFit counts massive neutrinos as matter, so
each has to subtract them back out before using a cold-matter
density. Getting that wrong in one place and not the other is the
failure these tests exist for: it would not raise, and it would
shift ``r_d`` and ``sigma8`` in opposite directions by a fraction of
a percent, which is comfortably inside the range where a fit still
looks plausible and is wrong.

The precedent is real. ``set_w_a_table`` was being assigned after
``pars.DarkEnergy``, CAMB silently kept the default, and every
``w0-wa`` spectrum came back as pure LCDM until a test compared two
cosmologies that had to differ.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from CosmoFit import LCDM, Fitter
from CosmoFit.cosmology.calculators.sound_horizon import SoundHorizon




def model(m_nu=0.06, **kwargs):
    """An LCDM at Planck's best fit, with the neutrino mass varied."""

    return LCDM(
        LCDM.PARAMS_CLASS(
            H0=67.36, Omega_m=0.3153, Omega_b=0.04930,
            ln1e10As=3.045, n_s=0.9649, tau_reio=0.0544,
            m_nu=m_nu, **kwargs,
        ),
    )


# ============================================================
# The convention: Omega_m contains the neutrinos
# ============================================================

def test_cold_matter_is_omega_m_minus_neutrinos():
    """
    ``omega_cb`` is the *cold* density, so the neutrinos that
    ``Omega_m`` includes have to come back out.
    """

    cosmo = model(m_nu=0.3)
    sh = SoundHorizon(cosmo)

    assert sh.omega_nu > 0.0

    np.testing.assert_allclose(
        sh.omega_cb + sh.omega_nu,
        cosmo.Omega_m * cosmo.h ** 2,
        rtol=1e-12,
    )


def test_the_two_backends_subtract_the_same_neutrino_density():
    """
    The sound horizon derives ``omega_nu`` from the Fermi-Dirac
    integral; the CAMB backend uses the ``Sum m_nu / 93.14``
    shorthand. Those are the same quantity by two routes and must
    agree -- the tolerance here is the 93.0378-vs-93.14 gap between
    the derived constant and the round number, not slack.
    """

    from CosmoFit.cosmology.boltzmann import NEUTRINO_MASS_DENOM

    for m_nu in (0.06, 0.3, 0.6):

        derived = SoundHorizon(model(m_nu=m_nu)).omega_nu
        shorthand = m_nu / NEUTRINO_MASS_DENOM

        assert derived == pytest.approx(shorthand, rel=2e-3), m_nu


def test_zero_mass_removes_the_neutrino_density_entirely():
    """
    Not merely small: exactly zero, and every species massless.
    Treating one species as "massive with zero mass" drops
    ``3.044/3`` effective species and costs 6% in ``r_d``.
    """

    sh = SoundHorizon(model(m_nu=0.0))

    assert sh.n_massive == 0
    assert sh.omega_nu == 0.0

    np.testing.assert_allclose(
        sh.omega_cb, model(m_nu=0.0).Omega_m * sh.h ** 2, rtol=1e-12
    )


# ============================================================
# The sound-horizon side
# ============================================================

def test_rd_rises_with_neutrino_mass():
    """
    At fixed ``Omega_m``, mass moved into neutrinos is mass that was
    relativistic at the drag epoch: less cold matter, later
    matter-radiation equality, a larger sound horizon.
    """

    rd = [SoundHorizon(model(m_nu=m)).rd_computed() for m in (0.0, 0.06, 0.3)]

    assert rd[0] < rd[1] < rd[2]

    # A real effect, not rounding: ~0.5 Mpc from 0 to 0.3 eV.
    assert 0.3 < rd[2] - rd[0] < 1.0


def test_rd_responds_to_mass_at_the_sub_permille_level():
    """
    The minimal-mass shift is small but has to be *there* -- an
    ``m_nu`` that reached nothing at all would leave this at zero.
    """

    baseline = SoundHorizon(model(m_nu=0.0)).rd_computed()
    minimal = SoundHorizon(model(m_nu=0.06)).rd_computed()

    assert minimal - baseline == pytest.approx(0.162, abs=0.03)


# ============================================================
# The Boltzmann side
# ============================================================

def test_neutrino_mass_suppresses_sigma8():
    """
    Free-streaming erases small-scale power. This is the test that
    would have caught ``set_w_a_table``: two cosmologies that must
    differ, compared rather than inspected.
    """

    pytest.importorskip("camb", reason="CAMB not installed (optional 'cmb' extra)")

    from CosmoFit.cosmology.boltzmann import CAMBBackend

    sigma8 = []
    for m_nu in (0.0, 0.06, 0.3):
        cosmo = model(m_nu=m_nu)
        sigma8.append(CAMBBackend.shared(cosmo).sigma8())

    assert sigma8[0] > sigma8[1] > sigma8[2], sigma8

    # Roughly the textbook 1 - 8 f_nu; ~9% at 0.3 eV.
    assert (sigma8[0] - sigma8[2]) / sigma8[0] == pytest.approx(0.09, abs=0.03)


def test_the_free_sigma8_ignores_the_neutrino_mass():
    """
    The trap `derive_sigma8` exists to close, stated as a test.

    ``cosmology.sigma8`` is the *free parameter* unless
    ``derive_sigma8`` is set -- so a fit that samples ``sigma8``
    directly gets the same amplitude at every neutrino mass, and the
    free-streaming suppression above simply does not happen. That is
    not a bug, it is a different question being asked, but it is the
    kind of thing that has to be deliberate rather than discovered.
    """

    pytest.importorskip("camb", reason="CAMB not installed (optional 'cmb' extra)")

    from CosmoFit.cosmology.boltzmann import CAMBBackend

    light, heavy = model(m_nu=0.0), model(m_nu=0.3)

    for cosmo in (light, heavy):
        CAMBBackend.shared(cosmo)

    # Free parameter: identical, and blind to the physics.
    assert light.sigma8 == heavy.sigma8

    # Derived: the suppression reappears.
    light.derive_sigma8 = heavy.derive_sigma8 = True

    assert light.sigma8 > heavy.sigma8


def test_neutrino_mass_is_part_of_the_cache_key():
    """
    The backend caches on a parameter tuple. If ``m_nu`` were
    missing from it, changing the mass on a cosmology that had
    already been run would return the stale spectrum -- silently,
    and only for the second call.
    """

    pytest.importorskip("camb", reason="CAMB not installed (optional 'cmb' extra)")

    from CosmoFit.cosmology.boltzmann import CAMBBackend

    cosmo = model(m_nu=0.06)
    backend = CAMBBackend.shared(cosmo)

    first = backend.sigma8()

    cosmo.params.m_nu = 0.4
    second = backend.sigma8()

    assert second != first
    assert second < first


def test_a_neutrino_mass_heavier_than_the_matter_budget_is_refused():
    """
    Past some mass there is no cold dark matter left. CAMB would
    fail somewhere inside Fortran; the backend has to catch it
    first and say which parameter is at fault.
    """

    pytest.importorskip("camb", reason="CAMB not installed (optional 'cmb' extra)")

    from CosmoFit.cosmology.boltzmann import BoltzmannError, CAMBBackend

    # Omega_m h^2 = 0.143; at ~13 eV the neutrinos exhaust it.
    cosmo = model(m_nu=15.0)

    with pytest.raises(BoltzmannError, match="cold dark matter"):
        CAMBBackend.shared(cosmo).sigma8()


# ============================================================
# As a fitted parameter
# ============================================================

def test_neutrino_mass_can_be_varied_by_the_fitter():
    """
    ``m_nu`` has to be a first-class free parameter: bounded,
    sampled, and actually reaching ``r_d`` rather than sitting in
    the parameter object unread.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["cc", "desi"],
        free_params=["H0", "Omega_m", "m_nu"],
        initial={"H0": 67.4, "Omega_m": 0.315, "Omega_b": 0.0493, "m_nu": 0.06},
        compute_rd=True,
    )

    assert "m_nu" in fit.free_params

    lower, upper = fit.prior.lower[-1], fit.prior.upper[-1]
    assert lower == 0.0 and upper > 0.0

    # The likelihood must respond to it -- through r_d, which is the
    # only route by which a neutrino mass touches BAO distances.
    theta = fit.theta0.copy()

    baseline = fit.logpost.chi2(theta)

    theta[-1] = 0.5
    shifted = fit.logpost.chi2(theta)

    assert shifted != baseline


def test_the_prior_bound_keeps_the_mass_non_negative():
    """
    A negative neutrino mass is not a physical region the sampler
    should be allowed to wander into, and ``omega_nu`` would flip
    sign there rather than raise.
    """

    fit = Fitter(
        model=LCDM,
        datasets=["cc"],
        free_params=["H0", "m_nu"],
        initial={"H0": 67.4, "Omega_m": 0.315, "m_nu": 0.06},
    )

    theta = fit.theta0.copy()
    theta[-1] = -0.1

    assert not np.isfinite(fit.logpost(theta))


# ============================================================
# The combination that cannot measure it
# ============================================================

def test_the_compressed_priors_are_blind_to_the_neutrino_mass():
    """
    The fact behind the warning, asserted rather than described.

    CHW19's ``z_star`` is a fitting formula in
    ``(omega_b, omega_cb)`` alone, calibrated against CAMB at the
    Planck fiducial ``Sum m_nu = 0.06 eV``. Hand it 0.8 eV and it
    returns the 0.06 eV answer -- not approximately, identically.
    ``Omega_r`` is the same story, since CHW19 leave massive
    neutrinos inside ``Omega_m``.

    So the compressed CMB carries no information about the
    neutrino mass at all. That is a property of the priors, not a
    bug here, and it is exactly why a fit using them must not
    report an ``m_nu`` posterior.
    """

    values = []

    for mass in (0.0, 0.06, 0.3, 0.8):

        recombination = model(m_nu=mass).recombination

        values.append(
            (
                recombination.z_star(),
                recombination.Omega_r,
                recombination.sound_horizon(),
            )
        )

    for other in values[1:]:

        assert other == values[0]


def test_a_free_neutrino_mass_without_a_boltzmann_cmb_warns():
    """
    With `compute_rd`, `m_nu` reaches exactly one thing: the sound
    horizon, where it shifts `r_d` with nothing to push back. The
    best fit runs to 0.82 eV on CC+DESI+SN+compressed Planck+BBN,
    against a published bound below 0.1 eV from the same data with
    the full CMB.
    """

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=LCDM,
            datasets=["cc", "desi", "planck", "omega_b"],
            free_params=["H0", "Omega_m", "Omega_b", "m_nu"],
            initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493,
                     "m_nu": 0.06},
            compute_rd=True,
        )

    messages = [str(w.message) for w in caught]

    assert any("neutrino mass" in m for m in messages), messages

    assert any("planck_lite" in m for m in messages), messages


def test_no_warning_when_the_cmb_can_see_it():
    """
    With a Boltzmann-computed CMB the mass changes the spectra --
    lensing smoothing and the early ISW -- so there is nothing to
    warn about.
    """

    pytest.importorskip("camb", reason="CAMB not installed (optional 'cmb' extra)")

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=LCDM,
            datasets=["planck_lite", "desi"],
            free_params=["H0", "Omega_m", "Omega_b", "ln1e10As",
                         "n_s", "tau_reio", "m_nu"],
            initial={"H0": 67.4, "Omega_m": 0.315, "Omega_b": 0.0493,
                     "ln1e10As": 3.045, "n_s": 0.9649,
                     "tau_reio": 0.0544, "m_nu": 0.06},
            compute_rd=True,
        )

    assert not any(
        "neutrino mass" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]


def test_no_warning_when_the_mass_is_fixed():
    """
    The combination is perfectly fine with `m_nu` held fixed --
    which is what every other fit in this library does.
    """

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        Fitter(
            model=LCDM,
            datasets=["cc", "desi", "planck", "omega_b"],
            free_params=["H0", "Omega_m", "Omega_b"],
            initial={"H0": 68.0, "Omega_m": 0.315, "Omega_b": 0.0493},
            compute_rd=True,
        )

    assert not any(
        "neutrino mass" in str(w.message) for w in caught
    )
