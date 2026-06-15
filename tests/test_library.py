"""M2: the candidate-function library X -> Θ.

Tests are written against hand-computed values and the combinatorial feature count, so
the library's columns, ordering, and names are all pinned exactly. The final test is the
drift guard promised in M1: the polynomial library must reproduce the *exact* feature
strings each dataset declares its truth in.
"""

import math

import numpy as np
import pytest

from esindy import datasets
from esindy.library import (
    ConcatLibrary,
    CustomLibrary,
    PolynomialLibrary,
)

# --- polynomial: hand-computed -----------------------------------------------------


def test_polynomial_hand_computed_values_and_names():
    X = np.array([[2.0, 3.0]])
    lib = PolynomialLibrary(degree=2, include_bias=True).fit(X)
    assert lib.get_feature_names() == ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
    Theta = lib.transform(X)
    np.testing.assert_array_equal(Theta, [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])


def test_polynomial_no_bias_drops_constant():
    X = np.array([[2.0, 3.0]])
    lib = PolynomialLibrary(degree=2, include_bias=False).fit(X)
    assert lib.get_feature_names() == ["x0", "x1", "x0^2", "x0 x1", "x1^2"]
    np.testing.assert_array_equal(lib.transform(X), [[2.0, 3.0, 4.0, 6.0, 9.0]])


@pytest.mark.parametrize("n,d", [(1, 1), (2, 2), (2, 3), (3, 3), (4, 2)])
def test_polynomial_feature_count_matches_formula(n, d):
    X = np.zeros((1, n))
    with_bias = PolynomialLibrary(degree=d, include_bias=True).fit(X)
    # number of monomials of total degree <= d in n variables == C(n + d, d)
    assert with_bias.n_output_features == math.comb(n + d, d)
    without_bias = PolynomialLibrary(degree=d, include_bias=False).fit(X)
    assert without_bias.n_output_features == math.comb(n + d, d) - 1


def test_polynomial_custom_input_names():
    X = np.zeros((1, 3))
    lib = PolynomialLibrary(degree=2).fit(X, input_names=("x", "y", "z"))
    names = lib.get_feature_names()
    for expected in ["x", "y", "z", "x^2", "x y", "x z", "y^2", "y z", "z^2"]:
        assert expected in names
    # space-joined in state order, never reversed
    assert "z x" not in names


def test_polynomial_interaction_only():
    X = np.array([[2.0, 3.0]])
    lib = PolynomialLibrary(degree=2, interaction_only=True).fit(X)
    assert lib.get_feature_names() == ["1", "x0", "x1", "x0 x1"]
    np.testing.assert_array_equal(lib.transform(X), [[1.0, 2.0, 3.0, 6.0]])


def test_polynomial_transform_shape_and_columns():
    X = np.arange(12.0).reshape(6, 2)
    lib = PolynomialLibrary(degree=3).fit(X)
    Theta = lib.transform(X)
    assert Theta.shape == (6, lib.n_output_features)
    names = lib.get_feature_names()
    # spot-check that a named column equals its hand-computed value
    np.testing.assert_array_equal(Theta[:, names.index("x0^2")], X[:, 0] ** 2)
    np.testing.assert_array_equal(Theta[:, names.index("x0 x1")], X[:, 0] * X[:, 1])


def test_fit_transform_matches_fit_then_transform():
    X = np.random.default_rng(0).random((5, 2))
    lib = PolynomialLibrary(degree=2)
    a = lib.fit_transform(X)
    b = PolynomialLibrary(degree=2).fit(X).transform(X)
    np.testing.assert_array_equal(a, b)


def test_transform_rejects_wrong_width():
    lib = PolynomialLibrary(degree=2).fit(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        lib.transform(np.zeros((3, 3)))


# --- custom / trig -----------------------------------------------------------------


def test_custom_trig_library():
    X = np.array([[0.0], [np.pi / 2]])
    lib = CustomLibrary(
        functions=[np.sin, np.cos],
        names=[lambda s: f"sin({s})", lambda s: f"cos({s})"],
    ).fit(X)
    assert lib.get_feature_names() == ["sin(x0)", "cos(x0)"]
    np.testing.assert_allclose(lib.transform(X), [[0.0, 1.0], [1.0, 0.0]], atol=1e-12)


def test_concat_library_concatenates_columns_and_names():
    X = np.array([[2.0, 3.0]])
    poly = PolynomialLibrary(degree=1, include_bias=True)
    trig = CustomLibrary(functions=[np.sin], names=[lambda s: f"sin({s})"])
    lib = ConcatLibrary([poly, trig]).fit(X)
    assert lib.get_feature_names() == ["1", "x0", "x1", "sin(x0)", "sin(x1)"]
    expected = [[1.0, 2.0, 3.0, np.sin(2.0), np.sin(3.0)]]
    np.testing.assert_allclose(lib.transform(X), expected)


# --- drift guard: library names == dataset's declared truth -------------------------


@pytest.mark.parametrize("name", datasets.available_systems())
def test_polynomial_library_reproduces_dataset_feature_names(name):
    """Every term a system declares its Ξ in must be a column the library produces."""
    system = datasets.get_system(name)
    X = np.zeros((1, system.n_states))
    lib = PolynomialLibrary(degree=2, include_bias=True).fit(X, input_names=system.state_names)
    produced = set(lib.get_feature_names())
    assert set(system.feature_names()) <= produced
