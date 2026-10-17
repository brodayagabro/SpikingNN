"""The same robust reduction of episode scores in every experiment."""
import numpy as np


def fitness(metrics):
    metrics = np.asarray(metrics, dtype=float)
    if metrics.ndim != 2 or not len(metrics) or metrics.shape[1] != 9:
        raise ValueError("Expected a nonempty matrix of nine episode metrics")
    if not np.isfinite(metrics).all():
        raise ValueError("Episode metrics must be finite")
    scores = np.sort(metrics[:, 0])
    tail = max(1, int(np.ceil(len(scores) / 4)))
    return float(.5 * scores.mean() + .3 * scores[:tail].mean() + .2 * scores[0])

