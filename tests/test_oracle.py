"""Oracle tests: validate our implementation against PySINDy.

These convert "the library already exists" into a rigorous external check. They are
behind the ``oracle`` marker and require the oracle extra (``uv sync --extra oracle``),
so the core suite never depends on PySINDy.

Conventions are matched explicitly (see optimizers.py): ridge ``alpha`` and
``normalize_columns`` are set identically on both sides, since PySINDy's STLSQ defaults
(alpha=0.05) differ from ours (alpha=0.0). Our polynomial library also produces the same
column ordering and names as PySINDy's, so coefficients compare position-for-position.
"""

import numpy as np
import pytest

ps = pytest.importorskip("pysindy")

from esindy import datasets  # noqa: E402
from esindy.library import PolynomialLibrary  # noqa: E402
from esindy.model import SINDy  # noqa: E402
from esindy.optimizers import STLSQ  # noqa: E402

pytestmark = pytest.mark.oracle


def test_stlsq_matches_pysindy_plain():
    rng = np.random.default_rng(0)
    Theta = rng.standard_normal((200, 8))
    xi = np.zeros((8, 1))
    xi[[1, 4, 6], 0] = [3.0, -2.0, 1.5]
    y = Theta @ xi

    ours = STLSQ(threshold=0.5, alpha=0.0).fit(Theta, y).coef_
    ref = ps.STLSQ(threshold=0.5, alpha=0.0).fit(Theta, y)
    np.testing.assert_allclose(ours.T, ref.coef_, atol=1e-8)


def test_stlsq_support_matches_pysindy_with_ridge():
    """With ridge on, the *support* matches exactly; the shrinkage magnitude differs
    slightly because PySINDy's ridge solver and our augmented-lstsq are not bit-identical
    formulations — hence the looser tolerance."""
    rng = np.random.default_rng(1)
    Theta = rng.standard_normal((300, 6))
    xi = np.zeros((6, 2))
    xi[[0, 3], 0] = [4.0, -1.0]
    xi[[2, 5], 1] = [2.0, 3.0]
    Y = Theta @ xi

    ours = STLSQ(threshold=0.3, alpha=0.05).fit(Theta, Y).coef_
    ref = ps.STLSQ(threshold=0.3, alpha=0.05).fit(Theta, Y)
    # identical support
    np.testing.assert_array_equal(ours.T != 0, ref.coef_ != 0)
    # coefficients in the same ballpark
    np.testing.assert_allclose(ours.T, ref.coef_, atol=5e-3)


@pytest.mark.parametrize("name", ["lotka_volterra", "lorenz"])
def test_full_sindy_matches_pysindy(name):
    system = datasets.get_system(name)
    traj = datasets.simulate(system)

    ours = SINDy(
        library=PolynomialLibrary(degree=2),
        optimizer=STLSQ(threshold=0.5, alpha=0.0),
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    ref = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.5, alpha=0.0),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    ref.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    # our library and PySINDy's share the same column ordering and names
    assert ours.feature_names_ == ref.get_feature_names()
    np.testing.assert_allclose(ours.coefficients_.T, ref.coefficients(), atol=1e-8)
