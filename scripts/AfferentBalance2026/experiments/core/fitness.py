"""The same robust reduction of episode scores in every experiment."""
import numpy as np


def fitness(metrics):
    """Reduce a nonempty K-by-9 episode-metric matrix to one robust fitness.

    Only the episode-score column enters the weighted reduction: 0.50 times
    the mean, 0.30 times the mean of the worst ceil(K/4) trials, and 0.20 times
    the minimum. Return a Python float; reject nonfinite or malformed input.
    """
    metrics = np.asarray(metrics, dtype=float)
    if metrics.ndim != 2 or not len(metrics) or metrics.shape[1] != 9:
        raise ValueError("Expected a nonempty matrix of nine episode metrics")
    if not np.isfinite(metrics).all():
        raise ValueError("Episode metrics must be finite")
    scores = np.sort(metrics[:, 0])
    tail = max(1, int(np.ceil(len(scores) / 4)))
    return float(.5 * scores.mean() + .3 * scores[:tail].mean() + .2 * scores[0])

