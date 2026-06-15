"""Reproducibility primitives.

One seed must thread through *everything* — the ODE integrator's noise, the library,
and every bootstrap — or the statistical tests flake and the golden tests drift
(plan §5.6). The key requirement for E-SINDy is that bootstrap ``b`` gets a seed
derived from its *index*, not from execution order, so that parallel runs (M8) are
bit-identical to serial runs (M7). ``numpy``'s ``SeedSequence.spawn`` gives us exactly
that: deterministic, independent child streams.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Anything we are willing to turn into a Generator.
SeedLike = None | int | np.random.SeedSequence | np.random.Generator


def as_seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    """Coerce a seed-like value into a ``SeedSequence`` (the spawnable primitive)."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, np.random.Generator):
        # Pull entropy out of the generator's bit stream so spawning stays deterministic
        # for a given generator state.
        return np.random.SeedSequence(int(seed.integers(0, 2**63 - 1)))
    if seed is None or isinstance(seed, int):
        return np.random.SeedSequence(seed)
    raise TypeError(f"cannot interpret {seed!r} as a seed")


def as_generator(seed: SeedLike) -> np.random.Generator:
    """Coerce a seed-like value into a ``numpy`` ``Generator``.

    Passing an existing ``Generator`` returns it unchanged so callers can share a
    stream; everything else is funnelled through ``SeedSequence`` for a fresh stream.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(as_seed_sequence(seed))


def child_seeds(seed: SeedLike, n: int) -> list[np.random.SeedSequence]:
    """Deterministically derive ``n`` independent child seeds from a parent.

    Use this to assign one seed per bootstrap by index. ``child_seeds(s, n)[b]`` is
    stable regardless of how many of the children are actually consumed or in what
    order — the property M8 relies on for "parallel == serial".
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return list(as_seed_sequence(seed).spawn(n))


def child_generators(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """Like :func:`child_seeds` but returns ready-to-use ``Generator`` objects."""
    return [np.random.default_rng(s) for s in child_seeds(seed, n)]


def is_sequence(obj: object) -> bool:
    """Small helper: a non-string sequence (used by config-coercion code paths)."""
    return isinstance(obj, Sequence) and not isinstance(obj, str | bytes)
