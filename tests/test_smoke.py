"""M0 smoke tests: the package imports, the scientific stack is present, and the
reproducibility primitives behave deterministically. CI goes green on these before any
real milestone work begins."""

import numpy as np

import esindy
from esindy import _seed


def test_package_imports_and_has_version():
    assert isinstance(esindy.__version__, str)
    assert esindy.__version__.count(".") == 2


def test_scientific_stack_available():
    import scipy  # noqa: F401  (import is the assertion)

    assert np.add(2, 3) == 5


def test_as_generator_is_reproducible():
    a = _seed.as_generator(42).random(5)
    b = _seed.as_generator(42).random(5)
    np.testing.assert_array_equal(a, b)


def test_as_generator_passes_through_existing_generator():
    rng = np.random.default_rng(0)
    assert _seed.as_generator(rng) is rng


def test_child_seeds_are_index_stable_and_independent():
    # The property M8 relies on: child b is the same whether we request 3 or 30,
    # so parallel execution order can never change a bootstrap's result.
    few = _seed.child_generators(123, 3)
    many = _seed.child_generators(123, 30)
    for b in range(3):
        np.testing.assert_array_equal(few[b].random(4), many[b].random(4))

    # Different children produce different streams.
    g0, g1 = _seed.child_generators(123, 2)
    assert not np.array_equal(g0.random(4), g1.random(4))
