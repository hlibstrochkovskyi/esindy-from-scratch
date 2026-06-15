"""Risk #3 made explicit: if the true dynamics aren't in the candidate basis, *no*
method recovers them. This test is *designed* to fail recovery, so the limitation is
documented and understood rather than discovered as a surprise.

The system is a large-amplitude pendulum, θ' = ω, ω' = -sin(θ). A polynomial library
cannot express sin(θ), so it is forced into a wrong-form approximation with spurious
terms. Adding sin to the library recovers the exact equation — proving the failure was
the basis, not the algorithm.
"""

import numpy as np
from scipy.integrate import solve_ivp

from esindy.library import ConcatLibrary, CustomLibrary, PolynomialLibrary
from esindy.metrics import active_set
from esindy.model import SINDy
from esindy.optimizers import STLSQ

STATE_NAMES = ("theta", "omega")


def _pendulum(theta0=3.0):
    """Large-amplitude pendulum with exact (non-polynomial) derivatives."""

    def rhs(_t, u):
        theta, omega = u
        return [omega, -np.sin(theta)]

    t = np.arange(0.0, 20.0, 0.01)
    sol = solve_ivp(rhs, (t[0], t[-1]), [theta0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    X = sol.y.T
    x_dot = np.column_stack([X[:, 1], -np.sin(X[:, 0])])
    return sol.t, X, x_dot


def _omega_dot_error(model, X):
    return float(np.sqrt(np.mean((model.predict(X)[:, 1] + np.sin(X[:, 0])) ** 2)))


def test_polynomial_basis_misses_the_sin_term():
    t, X, x_dot = _pendulum()
    model = SINDy(
        library=PolynomialLibrary(degree=3),
        optimizer=STLSQ(threshold=0.05),
        input_names=STATE_NAMES,
    ).fit(X, t=t, x_dot=x_dot)

    omega_dot_terms = active_set(model.coefficients_[:, 1])
    # theta' = omega is polynomial-expressible and recovered cleanly...
    assert active_set(model.coefficients_[:, 0]).sum() == 1
    # ...but omega' = -sin(theta) is NOT: the basis forces a multi-term approximation.
    assert omega_dot_terms.sum() >= 2
    assert _omega_dot_error(model, X) > 1e-4


def test_adding_sin_to_the_library_fixes_recovery():
    t, X, x_dot = _pendulum()
    library = ConcatLibrary(
        [PolynomialLibrary(degree=1), CustomLibrary([np.sin], [lambda s: f"sin({s})"])]
    )
    model = SINDy(
        library=library, optimizer=STLSQ(threshold=0.05), input_names=STATE_NAMES
    ).fit(X, t=t, x_dot=x_dot)

    names = model.feature_names_
    # omega' is now exactly -sin(theta): a single term, coefficient -1.
    omega_dot = model.coefficients_[:, 1]
    assert active_set(omega_dot).sum() == 1
    np.testing.assert_allclose(omega_dot[names.index("sin(theta)")], -1.0, atol=1e-3)
    assert _omega_dot_error(model, X) < 1e-6
