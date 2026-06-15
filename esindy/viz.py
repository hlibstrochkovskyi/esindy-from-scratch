"""Optional plotting helpers (requires the ``viz`` extra: ``uv sync --extra viz``).

Kept out of the core import path — nothing in ``esindy`` imports this — so the library
has no hard matplotlib dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(t, X, state_names: Sequence[str] | None = None, ax=None):
    """Plot each state variable against time. Returns the matplotlib Figure."""
    X = np.asarray(X)
    if ax is None:
        _, ax = plt.subplots()
    names = list(state_names) if state_names is not None else [f"x{i}" for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        ax.plot(t, X[:, i], label=names[i])
    ax.set_xlabel("t")
    ax.legend()
    return ax.figure


def plot_phase_portrait(X, ax=None):
    """Phase portrait: 2-D scatter-line, or a 3-D projection for three states."""
    X = np.asarray(X)
    if X.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
        return fig
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], lw=0.7)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    return ax.figure


def plot_inclusion_probabilities(model, threshold: float | None = None):
    """Bar chart of E-SINDy inclusion probabilities per state. Returns the Figure."""
    probs = model.inclusion_probabilities_
    names = model.feature_names_
    n_states = probs.shape[1]
    fig, axes = plt.subplots(1, n_states, figsize=(4 * n_states, 3), squeeze=False)
    for k in range(n_states):
        ax = axes[0, k]
        ax.bar(range(len(names)), probs[:, k])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model.input_names_[k]}'")
        if threshold is not None:
            ax.axhline(threshold, color="red", ls="--", lw=1)
    fig.tight_layout()
    return fig
