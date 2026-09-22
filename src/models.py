"""CATE (uplift) estimators with out-of-fold prediction.

Meta-learners implemented directly on scikit-learn so every step is inspectable:

- T-learner ("two-model"): fit E[Y|X] separately on treated and control rows;
  CATE(x) = mu1(x) - mu0(x). Simple and unbiased-ish, but each model only sees
  half the data and their errors don't cancel.
- X-learner: impute each unit's individual effect using the OTHER arm's outcome
  model, regress those imputed effects, then blend the two regressions weighted
  by the propensity score. Shines when arms are imbalanced or effects are
  smoother than outcomes. Here the RCT propensity within the two-arm subset is
  known (~0.5), so the blend is an even average.

All predictions are OUT-OF-FOLD: each customer is scored by models that never
saw them in training, so downstream validation (Qini, uplift@k) is honest.
Seed and hyperparameters are fixed so results are exactly reproducible.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_SPLITS = 5

# One shared spec so every learner uses the same base model class.
GB_PARAMS = dict(random_state=SEED, max_depth=3, learning_rate=0.05, max_iter=300)


def make_clf():
    return HistGradientBoostingClassifier(**GB_PARAMS)


def make_reg():
    return HistGradientBoostingRegressor(**GB_PARAMS)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix from pre-treatment covariates only.

    `history_segment` is excluded (it is a binned copy of `history`).
    """
    return pd.get_dummies(
        df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
        drop_first=True, dtype=float)


def oof_meta_learners(X: pd.DataFrame, y: np.ndarray, T: np.ndarray,
                      propensity: float = 0.5, seed: int = SEED):
    """Out-of-fold T-learner and X-learner CATEs for a binary outcome.

    Returns (tau_t, tau_x): arrays of per-customer estimated treatment effects.
    """
    tau_t = np.zeros(len(y), dtype=float)
    tau_x = np.zeros(len(y), dtype=float)
    strat = T * 2 + y  # stratify folds jointly on arm and outcome
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for tr, te in skf.split(X, strat):
        Xtr, Xte, ytr, Ttr = X.iloc[tr], X.iloc[te], y[tr], T[tr]

        # --- T-learner: one outcome model per arm ---
        m1 = make_clf().fit(Xtr[Ttr == 1], ytr[Ttr == 1])
        m0 = make_clf().fit(Xtr[Ttr == 0], ytr[Ttr == 0])
        tau_t[te] = m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]

        # --- X-learner: imputed individual effects, cross-model ---
        d1 = ytr[Ttr == 1] - m0.predict_proba(Xtr[Ttr == 1])[:, 1]   # treated: actual - predicted control outcome
        d0 = m1.predict_proba(Xtr[Ttr == 0])[:, 1] - ytr[Ttr == 0]   # control: predicted treated outcome - actual
        g1 = make_reg().fit(Xtr[Ttr == 1], d1)
        g0 = make_reg().fit(Xtr[Ttr == 0], d0)
        # blend weighted by propensity e: tau = e*g0 + (1-e)*g1 ; e=0.5 here
        tau_x[te] = propensity * g0.predict(Xte) + (1 - propensity) * g1.predict(Xte)

    return tau_t, tau_x


def oof_outcome_model(X: pd.DataFrame, y: np.ndarray, T: np.ndarray,
                      seed: int = SEED) -> np.ndarray:
    """Out-of-fold predicted OUTCOME (not uplift) - the naive targeting baseline.

    Ranking by P(visit) finds customers likely to visit ANYWAY; it is the
    mistake uplift modeling exists to correct, and our comparison baseline.
    """
    p = np.zeros(len(y), dtype=float)
    strat = T * 2 + y
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, strat):
        m = make_clf().fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p
