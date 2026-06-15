"""M9: equation output (text + LaTeX) and the round-trip guarantee.

The headline is the round trip: format a fitted model's coefficients, parse the strings
back, and recover the same matrix to the formatter's precision. If that holds, the
printed equations are a faithful representation, not a lossy summary.
"""

import numpy as np

from esindy import datasets
from esindy.equations import (
    coefficients_from_equations,
    format_equations,
    format_equations_latex,
    parse_equation,
)
from esindy.model import SINDy
from esindy.optimizers import STLSQ


def test_format_equations_basic():
    names = ["1", "x0", "x1", "x0 x1"]
    Xi = np.array([[0.0, 0.0], [-0.1, -2.0], [2.0, -0.1], [0.0, 0.0]])
    eqs = format_equations(names, Xi, ["x0", "x1"], precision=3)
    assert eqs[0] == "x0' = -0.100 x0 + 2.000 x1"
    assert eqs[1] == "x1' = -2.000 x0 - 0.100 x1"


def test_format_equations_constant_term_and_zero_row():
    names = ["1", "x0"]
    Xi = np.array([[1.5, 0.0], [0.0, 0.0]])
    eqs = format_equations(names, Xi, ["x", "y"])
    assert eqs[0] == "x' = 1.500"  # bare constant, no feature
    assert eqs[1] == "y' = 0"  # empty row


def test_format_latex_matches_expected():
    names = ["x0", "x1", "x0^2", "x0 x1"]
    Xi = np.array([[-0.1], [2.0], [0.0], [3.0]])
    latex = format_equations_latex(names, Xi, ["x0"], precision=2)
    assert latex[0] == r"\dot{x_{0}} = -0.10\,x_{0} + 2.00\,x_{1} + 3.00\,x_{0}\,x_{1}"


def test_latex_handles_symbolic_state_names_and_powers():
    names = ["x z", "z^2"]
    Xi = np.array([[1.0], [-2.0]])
    latex = format_equations_latex(names, Xi, ["y"], precision=1)
    assert latex[0] == r"\dot{y} = 1.0\,x\,z - 2.0\,z^{2}"


def test_parse_equation_roundtrip_simple():
    state, terms = parse_equation("x0' = -0.100 x0 + 2.000 x1")
    assert state == "x0"
    assert terms == {"x0": -0.1, "x1": 2.0}


def test_parse_handles_constant_and_products():
    _, terms = parse_equation("z' = 1.500 + 2.000 x0 x1 - 3.000 x0^2")
    assert terms == {"1": 1.5, "x0 x1": 2.0, "x0^2": -3.0}


def test_full_roundtrip_recovers_coefficient_matrix():
    system = datasets.get_system("lorenz")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    eqs = model.equations(precision=6)
    recovered = coefficients_from_equations(eqs, model.feature_names_, model.input_names_)
    np.testing.assert_allclose(recovered, model.coefficients_, atol=1e-6)
