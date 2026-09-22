"""One-shot preview of the Day 2-5 pipeline to produce real resume numbers.

Comparison: Womens E-Mail vs No E-Mail (as planned). Seed fixed at 42 so the
week's rebuilt notebooks reproduce these numbers exactly.

Methodology:
- T-learner: two HistGB classifiers (visit | treated, visit | control); CATE = diff.
- X-learner: imputed individual effects regressed, blended with e=0.5 (known RCT propensity).
- All CATEs are OUT-OF-FOLD (5-fold): each customer scored by models never trained on them.
- Validation on actual outcomes by ranked bins: uplift@k, Qini/AUUC vs random & vs
  a naive outcome-model ranking; incremental-revenue capture; profit simulation
  (stated assumptions); placebo test with permuted treatment labels.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data

SEED = 42
rng = np.random.default_rng(SEED)

df = load_data()
sub = df[df["segment"].isin(["Womens E-Mail", "No E-Mail"])].copy().reset_index(drop=True)
sub["T"] = (sub["segment"] == "Womens E-Mail").astype(int)

X = pd.get_dummies(
    sub[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
    drop_first=True, dtype=float)
y_visit = sub["visit"].to_numpy()
y_spend = sub["spend"].to_numpy()
T = sub["T"].to_numpy()
n = len(sub)
print(f"n={n}, treated={T.sum()}, control={(1-T).sum()}")

def clf():  return HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                                  learning_rate=0.05, max_iter=300)
def reg():  return HistGradientBoostingRegressor(random_state=SEED, max_depth=3,
                                                 learning_rate=0.05, max_iter=300)

def oof_cates(X, y, T, strat):
    """Out-of-fold T-learner and X-learner CATEs for a binary outcome."""
    tau_t = np.zeros(len(y)); tau_x = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, strat):
        Xtr, Xte, ytr, Ttr = X.iloc[tr], X.iloc[te], y[tr], T[tr]
        m1 = clf().fit(Xtr[Ttr == 1], ytr[Ttr == 1])     # E[Y|X, treated]
        m0 = clf().fit(Xtr[Ttr == 0], ytr[Ttr == 0])     # E[Y|X, control]
        p1, p0 = m1.predict_proba(Xte)[:, 1], m0.predict_proba(Xte)[:, 1]
        tau_t[te] = p1 - p0                              # T-learner
        # X-learner: impute individual effects on each arm, regress, blend (e=0.5)
        d1 = ytr[Ttr == 1] - m0.predict_proba(Xtr[Ttr == 1])[:, 1]
        d0 = m1.predict_proba(Xtr[Ttr == 0])[:, 1] - ytr[Ttr == 0]
        g1 = reg().fit(Xtr[Ttr == 1], d1)
        g0 = reg().fit(Xtr[Ttr == 0], d0)
        tau_x[te] = 0.5 * g1.predict(Xte) + 0.5 * g0.predict(Xte)
    return tau_t, tau_x

strat = T * 2 + y_visit   # stratify folds on arm x outcome
tau_t, tau_x = oof_cates(X, y_visit, T, strat)

# Naive baseline: rank by predicted OUTCOME (not uplift) - what a non-causal DS would do
p_out = np.zeros(n)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, te in skf.split(X, strat):
    m = clf().fit(X.iloc[tr], y_visit[tr])
    p_out[te] = m.predict_proba(X.iloc[te])[:, 1]

# ------------------------- evaluation on actual outcomes -------------------------
def bin_uplift(scores, y, T, k_frac):
    """Actual uplift (treated rate - control rate) inside the top k% by score."""
    idx = np.argsort(-scores)[: int(k_frac * len(scores))]
    yt, yc = y[idx][T[idx] == 1], y[idx][T[idx] == 0]
    return yt.mean() - yc.mean()

def qini_curve(scores, y, T, n_bins=100):
    """Cumulative incremental successes (scaled to control size) vs fraction targeted."""
    order = np.argsort(-scores)
    ys, Ts = y[order], T[order]
    fracs, qini = [0.0], [0.0]
    N = len(y)
    for b in range(1, n_bins + 1):
        m = int(N * b / n_bins)
        yt = ys[:m][Ts[:m] == 1].sum(); nt = max((Ts[:m] == 1).sum(), 1)
        yc = ys[:m][Ts[:m] == 0].sum(); nc = max((Ts[:m] == 0).sum(), 1)
        qini.append(yt - yc * nt / nc)   # incremental successes among targeted
        fracs.append(m / N)
    return np.array(fracs), np.array(qini)

def qini_coefficient(scores, y, T):
    """Area between model Qini curve and the random-targeting diagonal."""
    f, q = qini_curve(scores, y, T)
    random_line = f * q[-1]              # random targeting: straight line to same endpoint
    return np.trapezoid(q - random_line, f)

ate = y_visit[T == 1].mean() - y_visit[T == 0].mean()
print(f"\nOverall visit ATE: {ate*100:.2f}pp")
print("\nuplift@k (actual visit uplift inside top-k% by predicted uplift):")
for k in (0.1, 0.2, 0.3):
    ut = bin_uplift(tau_t, y_visit, T, k)
    ux = bin_uplift(tau_x, y_visit, T, k)
    print(f"  top {int(k*100)}%: T-learner {ut*100:.2f}pp ({ut/ate:.2f}x avg) | "
          f"X-learner {ux*100:.2f}pp ({ux/ate:.2f}x avg)")

qt = qini_coefficient(tau_t, y_visit, T)
qx = qini_coefficient(tau_x, y_visit, T)
qo = qini_coefficient(p_out, y_visit, T)
print(f"\nQini coefficient (visit): T-learner {qt:.1f} | X-learner {qx:.1f} | "
      f"outcome-model baseline {qo:.1f} | random 0 by definition")

# Placebo: permute T, refit ONE fold-scheme T-learner, Qini should be ~ 0
T_perm = rng.permutation(T)
tau_p, _ = oof_cates(X, y_visit, T_perm, T_perm * 2 + y_visit)
qp = qini_coefficient(tau_p, y_visit, T_perm)
print(f"Placebo (permuted labels) Qini: {qp:.1f}  (should be ~0 vs model {qt:.1f})")

# --------------------- incremental revenue capture & profit ---------------------
def revenue_capture(scores, k_frac):
    """Share of total incremental spend captured by emailing only top k%."""
    order = np.argsort(-scores)
    m = int(k_frac * n)
    top = order[:m]
    inc_top = (y_spend[top][T[top] == 1].mean() - y_spend[top][T[top] == 0].mean()) * m
    inc_all = (y_spend[T == 1].mean() - y_spend[T == 0].mean()) * n
    return inc_top / inc_all

print("\nIncremental revenue capture (ranked by T-learner visit uplift):")
for k in (0.2, 0.3, 0.4, 0.5):
    print(f"  emailing top {int(k*100)}% captures {revenue_capture(tau_t, k)*100:.0f}% "
          f"of incremental revenue (random would capture {int(k*100)}%)")

# Profit simulation. ASSUMPTIONS (stated): 30% gross margin on spend; $0.10 per email.
MARGIN, COST = 0.30, 0.10
order = np.argsort(-tau_t)
fracs = np.arange(0.05, 1.0001, 0.05)
profits = []
for f in fracs:
    m = int(f * n); top = order[:m]
    d_spend = y_spend[top][T[top] == 1].mean() - y_spend[top][T[top] == 0].mean()
    profits.append((MARGIN * d_spend - COST) * m)
profits = np.array(profits)
best = profits.argmax()
blanket = profits[-1]
print(f"\nProfit simulation (margin {MARGIN:.0%}, cost ${COST:.2f}/email), n={n:,} customers:")
print(f"  blanket emailing (100%): ${blanket:,.0f}")
print(f"  optimal: email top {fracs[best]*100:.0f}% -> ${profits[best]:,.0f} "
      f"({(profits[best]/blanket-1)*100:+.0f}% vs blanket)")
for f, p in zip(fracs, profits):
    if abs(f - 0.3) < 1e-9 or abs(f - 0.5) < 1e-9:
        print(f"  email top {f*100:.0f}%: ${p:,.0f}")

np.save("/tmp/tau_t.npy", tau_t)  # keep OOF preds for figure use later
print("\nDone. Seed=42 throughout; rebuildable in this week's notebooks.")
