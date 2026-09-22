"""Uplift-model validation: Qini curves, AUUC-style coefficients, uplift@k,
and bootstrap uncertainty bands.

Everything here evaluates a RANKING against ACTUAL randomized outcomes -- no
model predictions are trusted, only the ordering they induce. That is what
makes these metrics honest: within any top-k% slice, treated vs control rows
are still a mini randomized experiment, so their outcome gap is a valid causal
estimate of the uplift concentrated in that slice.
"""
import numpy as np


def uplift_at_k(scores: np.ndarray, y: np.ndarray, T: np.ndarray, k: float) -> float:
    """Actual uplift (treated mean - control mean) inside the top k% by score."""
    idx = np.argsort(-scores)[: int(k * len(scores))]
    yt, yc = y[idx][T[idx] == 1], y[idx][T[idx] == 0]
    if len(yt) == 0 or len(yc) == 0:
        return np.nan
    return yt.mean() - yc.mean()


def qini_curve(scores: np.ndarray, y: np.ndarray, T: np.ndarray, n_bins: int = 100):
    """Cumulative incremental successes vs fraction of population targeted.

    At each fraction f: (treated successes) - (control successes scaled to the
    treated count) among the top-f% ranked customers. The curve of a useless
    ranking is the straight line to the same endpoint (random targeting).
    """
    order = np.argsort(-scores)
    ys, Ts = y[order], T[order].astype(float)
    N = len(y)
    # Vectorized via cumulative sums (needed for fast bootstrapping).
    cum_yt = np.cumsum(ys * Ts)             # treated successes so far
    cum_yc = np.cumsum(ys * (1 - Ts))       # control successes so far
    cum_nt = np.cumsum(Ts)                  # treated count so far
    cum_nc = np.cumsum(1 - Ts)              # control count so far
    ms = np.maximum((N * np.arange(1, n_bins + 1) / n_bins).astype(int), 1) - 1
    nt = np.maximum(cum_nt[ms], 1.0)
    nc = np.maximum(cum_nc[ms], 1.0)
    qini = cum_yt[ms] - cum_yc[ms] * nt / nc
    fracs = (ms + 1) / N
    return np.concatenate([[0.0], fracs]), np.concatenate([[0.0], qini])


def qini_coefficient(scores: np.ndarray, y: np.ndarray, T: np.ndarray) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal.

    0 = no better than random; higher = more uplift concentrated early.
    """
    f, q = qini_curve(scores, y, T)
    return float(np.trapezoid(q - f * q[-1], f))


def bootstrap_metric(metric_fn, scores: np.ndarray, y: np.ndarray, T: np.ndarray,
                     n_boot: int = 500, seed: int = 42, **kwargs):
    """Percentile bootstrap over CUSTOMERS (predictions held fixed).

    Resampling rows answers: "how much would this metric wobble under a fresh
    sample of customers, given this trained model?" -- evaluation uncertainty,
    the honest error bar for the comparisons we report. (It does not include
    model-refit variance; that is noted as a limitation.)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn(scores[idx], y[idx], T[idx], **kwargs)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi), stats
