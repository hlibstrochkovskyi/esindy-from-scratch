"""esindy — a from-scratch, test-first implementation of SINDy and E-SINDy."""

from esindy._seed import as_generator, child_seeds
from esindy.differentiation import FiniteDifference, SavitzkyGolay, SplineDifferentiation
from esindy.ensemble import ESINDy
from esindy.library import ConcatLibrary, CustomLibrary, PolynomialLibrary
from esindy.model import SINDy
from esindy.optimizers import STLSQ, LeastSquares
from esindy.weak import WeakSINDy

__version__ = "0.0.1"

__all__ = [
    "ESINDy",
    "STLSQ",
    "ConcatLibrary",
    "CustomLibrary",
    "FiniteDifference",
    "LeastSquares",
    "PolynomialLibrary",
    "SINDy",
    "SavitzkyGolay",
    "SplineDifferentiation",
    "WeakSINDy",
    "__version__",
    "as_generator",
    "child_seeds",
]
