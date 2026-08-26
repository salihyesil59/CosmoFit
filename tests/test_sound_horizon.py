"""
Validation of the from-scratch BAO sound horizon.

``r_d`` is a number the whole BAO sector is divided by, so an error
in it is an error in every BAO distance at once -- and one that a fit
will happily absorb into ``H0`` without complaining. These tests are
the ones that would have caught that.

They do not need CAMB. The reference values were computed with it
once and are bundled (``tests/data/camb_drag_reference.npz``), so the
comparison runs anywhere and cannot silently drift with a CAMB
upgrade -- if it ever does, that is itself worth seeing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CosmoFit import LCDM, CPL, CosmologyParameters
from CosmoFit.cosmology.calculators.sound_horizon import (
    NU_ENERGY_FACTOR,
    neutrino_density_ratio,
)


REFERENCE = Path(__file__).parent / "data" / "camb_drag_reference.npz"


#: CAMB 2.0.4 at the Planck 2018 best fit
#: (ombh2 = 0.02237, omch2 = 0.1200, mnu = 0.06, nnu = 3.044).
PLANCK_Z_DRAG = 1059.9322086479028
PLANCK_R_DRAG = 147.1026996805658

PLANCK_OMEGA_B = 0.02237
PLANCK_OMEGA_CB = 0.14237


def build(omega_b, omega_cb, N_eff=3.044, m_nu=0.06, h=0.6736, model=LCDM,
          **extra):
    """
    A cosmology with the given *physical* densities.

    CosmoFit's ``Omega_m`` counts massive neutrinos as matter, so the
    neutrino density is added back in here -- the inverse of what
    :attr:`SoundHorizon.omega_cb` does. Getting this wrong in the test
    rather than in the code is the easiest way to "validate" a bug, so
    it is written out once and reused.
    """

    omega_nu = m_nu / 93.0378 if m_nu > 0 else 0.0

    return model(

        model.PARAMS_CLASS(

            H0=100.0 * h,

            Omega_m=(omega_cb + omega_nu) / h ** 2,

            Omega_b=omega_b / h ** 2,

            N_eff=N_eff,

            m_nu=m_nu,

            **extra,

        ),

    )


@pytest.fixture(scope="module")
def reference():

    with np.load(REFERENCE) as data:

        return {key: data[key] for key in data.files}


# ============================================================
# Neutrino thermodynamics
# ============================================================

def test_fermi_dirac_asymptotes():
    """
    The tabulated ``f(y) = rho_nu / rho_nu,massless`` must hit both
    analytic limits: 1 when relativistic, ``0.3173 y`` when not.
    """

    assert neutrino_density_ratio(1e-6)[0] == pytest.approx(1.0, abs=1e-8)

    y = 1.0e5

    expected = 1.8030853547 / (7.0 * np.pi ** 4 / 120.0) * y

    assert neutrino_density_ratio(y)[0] == pytest.approx(expected, rel=1e-5)

    # Monotonic, and never below the massless value.
    y_grid = np.logspace(-4, 4, 200)

    f = neutrino_density_ratio(y_grid)

    assert np.all(np.diff(f) > 0)
    assert np.all(f >= 1.0 - 1e-12)


def test_neutrino_mass_to_density_relation_is_derived_not_assumed():
    """
    The familiar ``omega_nu h^2 = Sum m_nu / 93.14 eV`` is *not* a
    constant in this module -- it falls out of the Fermi-Dirac
    integral and the neutrino temperature.

    Checking it is checking the whole thermodynamic setup at once:
    the decoupling temperature, the ``(N_eff/3)^{1/4}`` correction,
    and the non-relativistic limit of the density integral. CAMB
    gets 93.04 for the same physics.
    """

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB, m_nu=0.06)

    denominator = 0.06 / model.sound_horizon.omega_nu

    assert denominator == pytest.approx(93.04, abs=0.05)

    # And it must be linear in the mass -- a fixed conversion, not a
    # coincidence at one value.
    heavy = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB, m_nu=0.6)

    assert 0.6 / heavy.sound_horizon.omega_nu == pytest.approx(

        denominator,

        rel=2e-3,

    )


def test_omega_nu_does_not_move_with_N_eff():
    """
    Extra effective relativistic species must go into the *massless*
    component, leaving the massive neutrinos' present-day density
    alone -- CAMB's convention, verified against it.

    If ``EFF_PER_MASSIVE`` were written as ``N_eff/3`` instead of the
    fixed ``3.044/3``, varying ``N_eff`` would silently rescale the
    neutrino masses.
    """

    densities = [

        build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB, N_eff=n).sound_horizon.omega_nu

        for n in (2.0, 3.044, 5.0)

    ]

    assert densities[0] == pytest.approx(densities[1], rel=1e-12)
    assert densities[2] == pytest.approx(densities[1], rel=1e-12)


def test_massless_case_counts_every_species():
    """
    At ``m_nu = 0`` all ``N_eff`` species are massless.

    Leaving one of them in the "massive" bucket would drop
    ``3.044/3`` effective species from the radiation -- a ~14%
    error in the neutrino density and ~6% in ``r_d``, which is 25
    times DESI's best BAO error bar.
    """

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB, m_nu=0.0)

    horizon = model.sound_horizon

    assert horizon.n_massive == 0
    assert horizon.omega_nu == 0.0

    a = 1.0e-4

    expected = (

        horizon.omega_gamma

        * NU_ENERGY_FACTOR

        * model.N_eff

        * a ** -4

    )

    assert horizon.omega_nu_of_a(a) == pytest.approx(expected, rel=1e-12)


# ============================================================
# Against CAMB
# ============================================================

def test_matches_camb_at_the_planck_best_fit():

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB)

    horizon = model.sound_horizon

    assert horizon.omega_b == pytest.approx(PLANCK_OMEGA_B, rel=1e-10)
    assert horizon.omega_cb == pytest.approx(PLANCK_OMEGA_CB, rel=1e-3)

    assert horizon.z_drag() == pytest.approx(PLANCK_Z_DRAG, rel=1e-4)

    assert horizon.rd_computed() == pytest.approx(PLANCK_R_DRAG, rel=1e-4)


def test_matches_camb_across_the_calibration_grid(reference):
    """
    End to end -- this library's ``z_drag`` fit plus its own integral
    -- against CAMB's ``rdrag`` over the full grid the fit was
    calibrated on: ``omega_b`` in [0.018, 0.026], ``omega_cb`` in
    [0.09, 0.20], ``N_eff`` in [2.0, 5.0], ``Sum m_nu`` in [0, 0.6]
    eV.

    The bound is 1e-4 relative. For scale, DESI DR2's best single
    BAO measurement is 0.24%, so this is ~25 times tighter than the
    best data can tell -- and the *measured* worst case is 5e-5, so
    the bound has room to catch a regression before it matters.
    """

    computed = np.empty_like(reference["r_drag"])
    z_drag = np.empty_like(reference["z_drag"])

    for i in range(len(computed)):

        model = build(

            reference["omega_b"][i],

            reference["omega_cb"][i],

            N_eff=reference["N_eff"][i],

            m_nu=reference["m_nu"][i],

        )

        z_drag[i] = model.sound_horizon.z_drag()
        computed[i] = model.sound_horizon.rd_computed()

    z_error = np.abs(z_drag / reference["z_drag"] - 1.0)
    r_error = np.abs(computed / reference["r_drag"] - 1.0)

    assert z_error.max() < 1.0e-4, (

        f"z_drag: max relative error {z_error.max():.2e}"

    )

    assert r_error.max() < 1.0e-4, (

        f"r_d: max relative error {r_error.max():.2e} at "

        f"omega_b={reference['omega_b'][r_error.argmax()]:.4f}, "

        f"omega_cb={reference['omega_cb'][r_error.argmax()]:.3f}, "

        f"N_eff={reference['N_eff'][r_error.argmax()]}, "

        f"m_nu={reference['m_nu'][r_error.argmax()]}"

    )

    # Typical accuracy, not just worst case -- a regression that
    # degraded the median while keeping the maximum would pass the
    # bound above.
    assert np.median(r_error) < 1.0e-5


# ============================================================
# What r_d does and does not depend on
# ============================================================

@pytest.mark.parametrize("H0", [60.0, 67.36, 75.0])
def test_rd_is_independent_of_H0_at_fixed_physical_densities(H0):
    """
    ``r_d`` is fixed by the physical densities. Changing ``H0`` at
    fixed ``omega_b``/``omega_cb`` must not move it -- CAMB agrees to
    1e-7.

    This is not a triviality of the implementation: ``Omega_m`` and
    ``Omega_b`` are the parameters, and ``h^2`` appears in converting
    them, so an ``h`` left in the wrong place would show up here and
    almost nowhere else.
    """

    h = H0 / 100.0

    reference_value = build(

        PLANCK_OMEGA_B, PLANCK_OMEGA_CB, h=0.6736,

    ).sound_horizon.rd_computed()

    value = build(

        PLANCK_OMEGA_B, PLANCK_OMEGA_CB, h=h,

    ).sound_horizon.rd_computed()

    assert value == pytest.approx(reference_value, rel=1e-10)


def test_rd_is_independent_of_curvature_and_dark_energy():
    """
    The integral runs from the drag epoch upward, where dark energy
    and curvature are utterly negligible -- so ``r_d`` must be the
    same for LCDM, an open universe, and a strongly evolving CPL.

    That is what makes this usable for *every* model in the library,
    including ones no Boltzmann code could be given.
    """

    baseline = build(

        PLANCK_OMEGA_B, PLANCK_OMEGA_CB,

    ).sound_horizon.rd_computed()

    curved = build(

        PLANCK_OMEGA_B, PLANCK_OMEGA_CB, Omega_k=0.05,

    ).sound_horizon.rd_computed()

    evolving = build(

        PLANCK_OMEGA_B, PLANCK_OMEGA_CB,

        model=CPL, w0=-0.8, wa=-0.8,

    ).sound_horizon.rd_computed()

    assert curved == pytest.approx(baseline, rel=1e-10)
    assert evolving == pytest.approx(baseline, rel=1e-10)


def test_integral_is_converged():
    """
    The quoted accuracy is meaningless if the quadrature itself is
    not converged. Doubling the grid and extending the lower limit
    must both change nothing at the level being claimed.
    """

    horizon = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB).sound_horizon

    z_drag = horizon.z_drag()

    default = horizon.sound_horizon(z_drag)

    finer = horizon.sound_horizon(z_drag, n_grid=8000)
    deeper = horizon.sound_horizon(z_drag, decades=11.0)

    assert finer == pytest.approx(default, rel=1e-8)
    assert deeper == pytest.approx(default, rel=1e-6)


def test_eh98_is_available_and_quantifiably_worse():
    """
    The Eisenstein & Hu (1998) ``z_drag`` formula is kept for
    comparison only, and the module docstring quotes exactly how
    much worse it is. Pin those numbers, so the claim cannot go
    stale without a test noticing.

    EH98 gives ``z_d = 1020.7`` against CAMB's 1059.9 -- 3.7% low --
    which puts ``r_d`` 2.5% *high*, ten times DESI DR2's best BAO
    error bar. Their ``z_d`` was calibrated jointly with their own
    closed-form ``r_s`` and is not meant to be used with an
    independent integral; this is what happens if you do.
    """

    horizon = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB).sound_horizon

    eh98 = horizon.z_drag_eh98()
    calibrated = horizon.z_drag()

    assert eh98 / calibrated - 1.0 == pytest.approx(-0.037, abs=0.003)

    r_eh98 = horizon.sound_horizon(eh98)
    r_calibrated = horizon.sound_horizon(calibrated)

    assert r_eh98 / r_calibrated - 1.0 == pytest.approx(0.025, abs=0.003)


# ============================================================
# Wiring into a fit
# ============================================================

def test_compute_rd_is_off_by_default():
    """
    Turning this on changes every BAO prediction, so it must never
    happen by accident.
    """

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB)

    model.params.rd = 140.0

    assert model.compute_rd is False
    assert model.rd == 140.0

    model.compute_rd = True

    assert model.rd == pytest.approx(model.sound_horizon.rd_computed())
    assert abs(model.rd - 140.0) > 1.0


def test_fitter_refuses_rd_as_a_free_parameter_when_computing_it():

    from CosmoFit import Fitter

    with pytest.raises(ValueError, match="cannot also be a free parameter"):

        Fitter(

            model=LCDM,

            datasets=["desi"],

            free_params=["H0", "Omega_m", "rd"],

            initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1},

            compute_rd=True,

        )


def test_compute_rd_reaches_the_bao_likelihood():
    """
    The switch has to change the *prediction*, not merely a property
    nothing reads. Checked through the likelihood's own model vector.
    """

    from CosmoFit.likelihoods.desi import DESILikelihood

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB)

    model.params.rd = 120.0

    likelihood = DESILikelihood(model, version="desi2025")

    fitted = likelihood.model().copy()

    model.compute_rd = True

    computed = likelihood.model()

    # r_d appears as a divisor (or a multiplier for rs_over_DV), so
    # every entry must move, and by the same ratio.
    ratio = computed / fitted

    assert np.all(np.abs(ratio - 1.0) > 0.1)

    assert np.allclose(ratio, ratio[0], rtol=1e-12)


def test_compute_rd_is_part_of_the_chain_signature(tmp_path):
    """
    Two fits differing only in ``compute_rd`` sample different
    posteriors, so one's chain must not be resumable as the other's.
    """

    from CosmoFit import Fitter

    common = dict(

        model=LCDM,

        datasets=["desi"],

        dataset_kwargs={"desi": {"version": "desi2025"}},

        free_params=["H0", "Omega_m"],

        initial={"H0": 68.0, "Omega_m": 0.31, "rd": 147.1,
                 "Omega_b": 0.0493},

    )

    fitted = Fitter(**common)
    computed = Fitter(**common, compute_rd=True)

    assert fitted.chain_id() != computed.chain_id()

    from CosmoFit.stats.chains import compare_signatures

    differences = compare_signatures(

        fitted._chain_signature(),

        computed._chain_signature(),

    )

    assert any("compute_rd" in d for d in differences), differences


def test_cached_value_tracks_the_densities():
    """
    ``rd_computed`` caches, because the BAO likelihoods ask for it
    once per data point. The cache must invalidate on anything that
    moves ``r_d`` -- and, equally, must *not* be recomputed for a
    parameter that cannot.
    """

    model = build(PLANCK_OMEGA_B, PLANCK_OMEGA_CB)

    first = model.sound_horizon.rd_computed()

    model.params.w0 = -0.7

    assert model.sound_horizon.rd_computed() == first

    model.params.Omega_b *= 1.05

    assert abs(model.sound_horizon.rd_computed() - first) > 0.1
