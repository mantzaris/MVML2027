"""Leakage-safe state construction and policy targets; no trained model yet."""
import numpy as np


def observed_state(initial, selected, tool_values, width):
    if set(selected) - {0, 1, 2, 3}:
        raise ValueError("Unknown action")
    numeric, observed = [], []
    for action in range(4):
        if action in selected:
            value = np.asarray(tool_values[action], dtype=float)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError("Invalid observed tool fields")
            numeric.extend(value)
            observed.append(1.)
        else:
            # Do not even read an unselected cached tool result.
            numeric.extend([0.] * width)
            observed.append(0.)
    return np.r_[np.asarray(initial, dtype=float), observed, numeric]


def subset_targets(losses):
    if set(losses) != set(range(16)):
        raise ValueError("All 16 independently scored terminal subsets are required")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in losses.values()):
        raise ValueError("Terminal losses must be finite in [0, 1]")
    rows = []
    for mask in range(16):
        available = [not bool(mask & (1 << a)) for a in range(4)]
        delta = [losses[mask] - losses[mask | (1 << a)] if available[a] else 0. for a in range(4)]
        rows.append({"mask": mask, "risk": losses[mask], "delta": delta, "available": available})
    return rows


def choose_action(means, deviations, selected, budget=2, beta=.5, cost_penalty=.02, force=False):
    if len(selected) >= budget:
        return None
    values, spread = np.asarray(means, dtype=float), np.asarray(deviations, dtype=float)
    if values.shape != (4,) or spread.shape != (4,) or not np.isfinite(values).all() or not np.isfinite(spread).all() or np.any(spread < 0):
        raise ValueError("Invalid ensemble predictions")
    scores = values - beta * spread - cost_penalty
    scores[list(selected)] = -np.inf
    action = int(np.argmax(scores))
    return action if np.isfinite(scores[action]) and (force or scores[action] > 0) else None
