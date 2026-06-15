"""Candidate-function libraries: X (m×n) -> Θ (m×p).

Each library turns a state matrix into a matrix of candidate terms and exposes
human-readable names for every column. Names follow the convention shared with
``datasets`` (M1): constant ``"1"``, power ``"x0^2"``, product ``"x0 x1"`` joined in
state order. The :func:`test_polynomial_library_reproduces_dataset_feature_names` guard
keeps the two from ever drifting apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from itertools import combinations, combinations_with_replacement

import numpy as np


def default_input_names(n: int) -> tuple[str, ...]:
    return tuple(f"x{i}" for i in range(n))


class BaseLibrary(ABC):
    """Common fit/transform plumbing. Subclasses define columns and names."""

    n_features_in_: int
    input_names_: tuple[str, ...]

    def fit(self, X: np.ndarray, input_names: Sequence[str] | None = None) -> BaseLibrary:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (m, n); got shape {X.shape}")
        self.n_features_in_ = X.shape[1]
        if input_names is None:
            self.input_names_ = default_input_names(self.n_features_in_)
        else:
            if len(input_names) != self.n_features_in_:
                raise ValueError(
                    f"got {len(input_names)} input names for {self.n_features_in_} columns"
                )
            self.input_names_ = tuple(input_names)
        self._fit()
        return self

    def _fit(self) -> None:  # noqa: B027  (optional hook; default is a no-op)
        """Hook for subclasses to precompute their column structure."""

    def _check_fitted(self) -> None:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} must be fit before use")

    def _check_width(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected X with {self.n_features_in_} columns; got shape {X.shape}")
        return X

    def fit_transform(self, X: np.ndarray, input_names: Sequence[str] | None = None) -> np.ndarray:
        return self.fit(X, input_names).transform(X)

    @property
    def n_output_features(self) -> int:
        return len(self.get_feature_names())

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_feature_names(self) -> list[str]: ...


class PolynomialLibrary(BaseLibrary):
    """All monomials up to a total degree, optionally with a constant term.

    With ``include_bias`` the column count is ``C(n + d, d)``. ``interaction_only``
    keeps only products of *distinct* variables (no powers).
    """

    def __init__(
        self, degree: int = 2, *, include_bias: bool = True, interaction_only: bool = False
    ) -> None:
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def _fit(self) -> None:
        n = self.n_features_in_
        # Each term is a tuple of column indices (a multiset); () is the constant.
        terms: list[tuple[int, ...]] = []
        if self.include_bias:
            terms.append(())
        combiner = combinations if self.interaction_only else combinations_with_replacement
        for d in range(1, self.degree + 1):
            terms.extend(combiner(range(n), d))
        self._terms = terms

    def _term_name(self, term: tuple[int, ...]) -> str:
        if not term:
            return "1"
        factors = []
        for idx in sorted(set(term)):
            power = term.count(idx)
            name = self.input_names_[idx]
            factors.append(name if power == 1 else f"{name}^{power}")
        return " ".join(factors)

    def get_feature_names(self) -> list[str]:
        self._check_fitted()
        return [self._term_name(term) for term in self._terms]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check_width(X)
        columns = []
        for term in self._terms:
            if not term:
                columns.append(np.ones(X.shape[0]))
            else:
                columns.append(np.prod(X[:, list(term)], axis=1))
        return np.column_stack(columns)


class CustomLibrary(BaseLibrary):
    """Apply each scalar function to each input variable.

    ``functions[i]`` is applied elementwise to every column; ``names[i](var)`` produces
    that column's label. Output order is function-major: f0(x0), f0(x1), …, f1(x0), …
    """

    def __init__(
        self,
        functions: Sequence[Callable[[np.ndarray], np.ndarray]],
        names: Sequence[Callable[[str], str]],
    ) -> None:
        if len(functions) != len(names):
            raise ValueError("functions and names must have the same length")
        self.functions = list(functions)
        self.names = list(names)

    def get_feature_names(self) -> list[str]:
        self._check_fitted()
        return [name(var) for name in self.names for var in self.input_names_]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check_width(X)
        columns = [
            np.asarray(func(X[:, j]), dtype=float)
            for func in self.functions
            for j in range(self.n_features_in_)
        ]
        return np.column_stack(columns)


class ConcatLibrary(BaseLibrary):
    """Horizontally concatenate several libraries into one feature matrix."""

    def __init__(self, libraries: Sequence[BaseLibrary]) -> None:
        if not libraries:
            raise ValueError("ConcatLibrary needs at least one library")
        self.libraries = list(libraries)

    def _fit(self) -> None:
        for lib in self.libraries:
            lib.fit(np.zeros((1, self.n_features_in_)), input_names=self.input_names_)

    def get_feature_names(self) -> list[str]:
        self._check_fitted()
        names: list[str] = []
        for lib in self.libraries:
            names.extend(lib.get_feature_names())
        return names

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._check_width(X)
        return np.column_stack([lib.transform(X) for lib in self.libraries])
