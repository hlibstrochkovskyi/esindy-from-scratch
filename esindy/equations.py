"""Render and parse discovered dynamics.

A coefficient matrix Ξ plus feature names is turned into readable text or LaTeX, and
parsed back again. The parse step is what makes the round-trip test possible: print a
model, read it back, and recover the same coefficients — proof the formatter is lossless
to its stated precision.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_VAR = re.compile(r"^([A-Za-z]+)(\d+)$")


def _latex_var(var: str) -> str:
    """``x0`` -> ``x_{0}``; leave already-symbolic names (``x``, ``y``) untouched."""
    m = _VAR.match(var)
    return f"{m.group(1)}_{{{m.group(2)}}}" if m else var


def _latex_feature(name: str) -> str:
    if name == "1":
        return ""
    factors = []
    for factor in name.split():
        if "^" in factor:
            base, exponent = factor.split("^")
            factors.append(f"{_latex_var(base)}^{{{exponent}}}")
        else:
            factors.append(_latex_var(factor))
    return r"\,".join(factors)


def _terms(feature_names, coefficients_col, precision, *, latex):
    parts = []
    for name, coef in zip(feature_names, coefficients_col, strict=True):
        if coef == 0.0:
            continue
        value = f"{abs(coef):.{precision}f}"
        feature = _latex_feature(name) if latex else ("" if name == "1" else name)
        if not feature:
            body = value
        elif latex:
            body = rf"{value}\,{feature}"
        else:
            body = f"{value} {feature}"
        if not parts:
            # first term: sign attached, no leading operator
            parts.append(f"-{body}" if coef < 0 else body)
        else:
            parts.append(f"{' - ' if coef < 0 else ' + '}{body}")
    return "".join(parts) if parts else "0"


def format_equations(
    feature_names: Sequence[str],
    coefficients: np.ndarray,
    state_names: Sequence[str],
    precision: int = 3,
) -> list[str]:
    """One ``x0' = ...`` string per state variable."""
    return [
        f"{state}' = {_terms(feature_names, coefficients[:, k], precision, latex=False)}"
        for k, state in enumerate(state_names)
    ]


def format_equations_latex(
    feature_names: Sequence[str],
    coefficients: np.ndarray,
    state_names: Sequence[str],
    precision: int = 3,
) -> list[str]:
    r"""One ``\dot{x}_{0} = ...`` LaTeX string per state variable."""
    return [
        rf"\dot{{{_latex_var(state)}}} = "
        f"{_terms(feature_names, coefficients[:, k], precision, latex=True)}"
        for k, state in enumerate(state_names)
    ]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_equation(equation: str) -> tuple[str, dict[str, float]]:
    """Parse a pretty-printed ``state' = ...`` line back into (state, {feature: coef})."""
    lhs, rhs = equation.split("=", 1)
    state = lhs.strip().rstrip("'").strip()

    terms: dict[str, float] = {}
    tokens = rhs.split()
    i = 0
    while i < len(tokens):
        # a term's coefficient is either an attached-sign number ("-0.1") or a standalone
        # operator token ("+"/"-") followed by a magnitude.
        if tokens[i] in ("+", "-"):
            sign = -1.0 if tokens[i] == "-" else 1.0
            coef = sign * float(tokens[i + 1])
            i += 2
        else:
            coef = float(tokens[i])
            i += 1
        factors = []
        while i < len(tokens) and tokens[i] not in ("+", "-") and not _is_number(tokens[i]):
            factors.append(tokens[i])
            i += 1
        terms[" ".join(factors) if factors else "1"] = coef
    return state, terms


def coefficients_from_equations(
    equations: Sequence[str],
    feature_names: Sequence[str],
    state_names: Sequence[str],
) -> np.ndarray:
    """Reconstruct Ξ from printed equations (the inverse of :func:`format_equations`)."""
    feature_names = list(feature_names)
    state_names = list(state_names)
    Xi = np.zeros((len(feature_names), len(state_names)))
    for equation in equations:
        state, terms = parse_equation(equation)
        k = state_names.index(state)
        for name, coef in terms.items():
            Xi[feature_names.index(name), k] = coef
    return Xi
