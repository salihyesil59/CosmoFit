"""
The parameter containers.

`Parameter`, `ParameterSet` and `CosmologyParameters` sit under every
model, every prior, every plot label and every chain signature in the
library. 67% of the module had ever run -- the parts every fit happens
to touch -- and the rest is the dict/sequence protocol, the validation,
and the class factory that backs a custom model's `EXTRA_PARAMS`.

The mapping interface is the part worth pinning: `params["H0"]`,
`params[0]`, `params.update(theta)` and `dict(params.items())` are all
documented, all reachable from a notebook, and none of them was
exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from CosmoFit import CosmologyParameters
from CosmoFit.cosmology.core.parameters import (
    Parameter,
    ParameterSet,
    build_params_class,
)


# ============================================================
# Parameter
# ============================================================


def test_the_label_falls_back_to_the_name():
    """
    A parameter with no LaTeX label still has to produce something
    an axis can be labelled with, or a custom model's corner plot
    would come out with empty axes.
    """

    assert Parameter("H0", label=r"$H_0$").latex == r"$H_0$"

    assert Parameter("beta").latex == "beta"


def test_bounds_split_into_lower_and_upper():

    bounded = Parameter("H0", bounds=(60.0, 80.0))

    assert bounded.has_bounds
    assert bounded.lower == 60.0
    assert bounded.upper == 80.0

    free = Parameter("H0")

    assert not free.has_bounds
    assert free.lower is None
    assert free.upper is None


def test_parameter_repr_shows_the_bounds():

    assert "bounds=(60.0, 80.0)" in repr(Parameter("H0", bounds=(60.0, 80.0)))


# ============================================================
# ParameterSet
# ============================================================


@pytest.fixture
def pset():

    return ParameterSet(
        Parameter("H0", label=r"$H_0$", unit="km/s/Mpc", bounds=(50.0, 90.0)),
        Parameter("Omega_m", label=r"$\Omega_m$", bounds=(0.05, 0.7)),
        Parameter("w0"),
    )


def test_it_can_be_built_from_an_iterable_as_well_as_from_arguments():

    parameters = [Parameter("H0"), Parameter("Omega_m")]

    assert ParameterSet(*parameters).names == ParameterSet(parameters).names


def test_an_empty_set_is_refused():

    with pytest.raises(ValueError, match="empty"):
        ParameterSet()


def test_duplicate_names_are_refused():
    """
    Two entries with one name would make `index()` and `__getitem__`
    silently disagree with iteration order.
    """

    with pytest.raises(ValueError, match="Duplicate"):
        ParameterSet(Parameter("H0"), Parameter("H0"))


def test_the_sequence_and_mapping_protocols(pset):

    assert len(pset) == 3
    assert pset.ndim == 3

    assert "H0" in pset
    assert "nonsense" not in pset

    assert pset["H0"] is pset[0]

    assert pset.index("Omega_m") == 1

    assert [p.name for p in pset] == ["H0", "Omega_m", "w0"]

    assert "H0" in repr(pset)


def test_the_column_properties(pset):

    assert pset.names == ["H0", "Omega_m", "w0"]

    assert pset.labels == [r"$H_0$", r"$\Omega_m$", "w0"]

    assert pset.units == ["km/s/Mpc", None, None]

    assert pset.bounds == [(50.0, 90.0), (0.05, 0.7), None]

    assert pset.lower_bounds == [50.0, 0.05, None]

    assert pset.upper_bounds == [90.0, 0.7, None]


def test_a_vector_round_trips_through_the_names(pset):

    theta = [70.0, 0.3, -1.0]

    as_dict = pset.to_dict(theta)

    assert as_dict == {"H0": 70.0, "Omega_m": 0.3, "w0": -1.0}

    assert pset.from_dict(as_dict) == theta


def test_a_vector_of_the_wrong_length_is_refused(pset):

    with pytest.raises(ValueError, match="Expected 3"):
        pset.to_dict([70.0, 0.3])


# ============================================================
# ParameterSet.validate
# ============================================================


def test_validate_accepts_a_well_formed_vector(pset):

    assert pset.validate([70.0, 0.3, -1.0]) is None


def test_validate_names_the_offending_parameter(pset):
    """
    Structural validation only -- bounds are a prior's business.
    What matters is that the message says *which* entry is wrong,
    since the caller passed an unlabelled array.
    """

    with pytest.raises(ValueError, match="Expected 3"):
        pset.validate([70.0, 0.3])

    with pytest.raises(TypeError, match="Omega_m"):
        pset.validate([70.0, "0.3", -1.0])

    with pytest.raises(ValueError, match="w0.*finite"):
        pset.validate([70.0, 0.3, float("nan")])

    with pytest.raises(ValueError, match="H0.*finite"):
        pset.validate([float("inf"), 0.3, -1.0])


# ============================================================
# CosmologyParameters
# ============================================================


def test_names_defaults_and_ndim_agree():

    names = CosmologyParameters.names()

    assert "H0" in names and "Omega_m" in names

    assert CosmologyParameters.ndim() == len(names)

    defaults = CosmologyParameters.defaults()

    assert set(defaults) <= set(names)

    # `H0` and `Omega_m` are deliberately *not* among them: they have
    # no default, so a container cannot be built without saying what
    # cosmology it is. Everything else does.
    assert "H0" not in defaults
    assert "Omega_m" not in defaults
    assert defaults["Omega_k"] == 0.0


def test_the_dict_round_trip():

    original = CosmologyParameters(H0=70.0, Omega_m=0.3)

    restored = CosmologyParameters.from_dict(original.as_dict())

    assert restored.as_dict() == original.as_dict()


def test_the_numpy_round_trip():

    original = CosmologyParameters(H0=70.0, Omega_m=0.3)

    theta = original.to_numpy()

    assert theta.shape == (CosmologyParameters.ndim(),)

    assert CosmologyParameters.from_numpy(theta).as_dict() == original.as_dict()


def test_from_numpy_refuses_the_wrong_shape():

    with pytest.raises(ValueError, match="one-dimensional"):
        CosmologyParameters.from_numpy(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Expected"):
        CosmologyParameters.from_numpy(np.zeros(2))


def test_update_from_a_vector_and_by_name():

    params = CosmologyParameters(H0=70.0, Omega_m=0.3)

    params.update(H0=72.0)

    assert params.H0 == 72.0

    theta = params.to_numpy()
    theta[0] = 65.0

    params.update(theta)

    assert params.H0 == 65.0


def test_update_refuses_an_unknown_name():
    """
    A typo in a keyword would otherwise set an attribute nothing
    reads, and the fit would silently run at the old value.
    """

    params = CosmologyParameters(H0=70.0, Omega_m=0.3)

    with pytest.raises(KeyError, match="Omega_M"):
        params.update(Omega_M=0.3)


def test_update_refuses_a_malformed_vector():

    params = CosmologyParameters(H0=70.0, Omega_m=0.3)

    with pytest.raises(ValueError, match="one-dimensional"):
        params.update(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="Expected"):
        params.update(np.zeros(3))


def test_the_sequence_and_mapping_interface():

    params = CosmologyParameters(H0=70.0, Omega_m=0.3)

    assert len(params) == CosmologyParameters.ndim()

    assert params["H0"] == 70.0

    assert params[0] == list(params)[0]

    params["H0"] = 71.0
    assert params.H0 == 71.0

    params[0] = 72.0
    assert params[CosmologyParameters.names()[0]] == 72.0

    assert dict(params.items()) == params.as_dict()

    assert params.values() == tuple(params)

    assert "H0=" in repr(params)


def test_copy_is_deep():

    params = CosmologyParameters(H0=70.0, Omega_m=0.3)

    clone = params.copy()

    clone.H0 = 60.0

    assert params.H0 == 70.0


def test_parameter_set_carries_labels_and_bounds():

    pset = CosmologyParameters.parameter_set()

    assert pset["H0"].label == r"$H_0$"

    assert pset["H0"].bounds is not None

    overridden = CosmologyParameters.parameter_set(bounds={"H0": (10.0, 20.0)})

    assert overridden["H0"].bounds == (10.0, 20.0)

    # The override must not leak into the next call.
    assert CosmologyParameters.parameter_set()["H0"].bounds != (10.0, 20.0)


# ============================================================
# The class factory behind a custom model's EXTRA_PARAMS
# ============================================================


def test_an_extra_parameter_becomes_a_field_with_its_bounds_and_label():

    cls = build_params_class(
        "Toy",
        {"beta": {"default": 0.5, "bounds": (0.0, 2.0), "label": r"$\beta$"}},
    )

    assert cls.__name__ == "ToyParameters"

    assert "beta" in cls.names()

    assert cls(H0=70.0, Omega_m=0.3).beta == 0.5

    pset = cls.parameter_set()

    assert pset["beta"].bounds == (0.0, 2.0)

    assert pset["beta"].label == r"$\beta$"

    # Everything the base had is still there.
    assert set(CosmologyParameters.names()) <= set(cls.names())


def test_an_extra_parameter_defaults_to_zero_with_no_bounds():

    cls = build_params_class("Toy", {"gamma": {}})

    assert cls(H0=70.0, Omega_m=0.3).gamma == 0.0

    assert cls.parameter_set()["gamma"].bounds is None

    assert cls.parameter_set()["gamma"].label == "gamma"


def test_a_name_that_collides_with_an_existing_parameter_is_refused():
    """
    Silently shadowing `Omega_m` would give a model two parameters
    meaning different things under one name -- and the fitter would
    sample whichever one the dataclass resolved to.
    """

    with pytest.raises(ValueError, match="collide"):
        build_params_class("Toy", {"Omega_m": {"default": 0.3}})


def test_the_factory_can_be_stacked():

    first = build_params_class("First", {"beta": {"bounds": (0.0, 1.0)}})

    second = build_params_class("Second", {"gamma": {}}, base=first)

    assert {"beta", "gamma"} <= set(second.names())

    # The first layer's bounds survive the second.
    assert second.parameter_set()["beta"].bounds == (0.0, 1.0)
