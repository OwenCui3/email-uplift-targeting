"""Targeting-policy simulation: turn a CATE ranking into a business decision.

The policy question: with budget to email only some customers, whom do we email
and how many? We sweep the contact fraction, and at each fraction estimate the
campaign's incremental profit EMPIRICALLY -- the actual treated-vs-control spend
gap among the targeted slice (valid because randomization holds within any
slice) -- under explicit economic assumptions.
"""
import numpy as np

# Economic assumptions -- stated, not hidden. Sensitivity is checked in nb 05.
DEFAULT_MARGIN = 0.30      # gross margin: $1 of revenue -> $0.30 of profit
DEFAULT_COST = 0.10        # cost per email sent (creative + send + attention)


def profit_curve(scores: np.ndarray, spend: np.ndarray, T: np.ndarray,
                 margin: float = DEFAULT_MARGIN, cost: float = DEFAULT_COST,
                 fracs: np.ndarray | None = None):
    """Expected incremental profit of emailing the top-f% ranked customers.

    profit(f) = [margin * (incremental spend per targeted customer) - cost] * n_targeted
    """
    if fracs is None:
        fracs = np.arange(0.05, 1.0001, 0.05)
    order = np.argsort(-scores)
    n = len(scores)
    profits = []
    for f in fracs:
        m = int(f * n)
        top = order[:m]
        t_mask = T[top] == 1
        if t_mask.sum() == 0 or (~t_mask).sum() == 0:
            profits.append(np.nan)
            continue
        d_spend = spend[top][t_mask].mean() - spend[top][~t_mask].mean()
        profits.append((margin * d_spend - cost) * m)
    return np.asarray(fracs), np.asarray(profits)


def optimal_fraction(fracs: np.ndarray, profits: np.ndarray):
    """(best fraction, its profit, profit at 100% i.e. blanket emailing)."""
    best = int(np.nanargmax(profits))
    return float(fracs[best]), float(profits[best]), float(profits[-1])
