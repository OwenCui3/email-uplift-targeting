"""Experiment-validation tools: SRM check, covariate balance, difference-in-means ATEs.

These are the checks a data scientist runs on ANY experiment before trusting it:
1. Sample-ratio mismatch (SRM): did randomization deliver the intended split?
2. Covariate balance: are pre-treatment covariates distributed the same across arms?
3. Only then: estimate treatment effects.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Sample-ratio mismatch
# ---------------------------------------------------------------------------
def srm_check(df: pd.DataFrame, arm_col: str = "segment",
              expected_shares: dict | None = None) -> pd.DataFrame:
    """Chi-squared goodness-of-fit test of observed arm counts vs intended shares.

    Hillstrom's design is a 1/3 : 1/3 : 1/3 split. A tiny p-value would mean the
    randomization (or logging) is broken and NOTHING downstream can be trusted.
    """
    counts = df[arm_col].value_counts()
    arms = list(counts.index)
    n = counts.sum()
    if expected_shares is None:
        expected_shares = {a: 1 / len(arms) for a in arms}
    expected = np.array([expected_shares[a] * n for a in arms])
    observed = counts.to_numpy()

    chi2, pval = stats.chisquare(f_obs=observed, f_exp=expected)
    out = pd.DataFrame({
        "arm": arms,
        "observed": observed,
        "expected": expected.round(1),
        "share": (observed / n).round(4),
    })
    print(f"SRM chi-squared test: chi2 = {chi2:.3f}, p-value = {pval:.4f}")
    if pval < 0.001:
        print("!! Possible sample-ratio mismatch -- investigate before proceeding.")
    else:
        print("No evidence of sample-ratio mismatch (arms match intended 1/3 split).")
    return out


# ---------------------------------------------------------------------------
# 2. Covariate balance (standardized mean differences)
# ---------------------------------------------------------------------------
def _smd(x_t: pd.Series, x_c: pd.Series) -> float:
    """Standardized mean difference: (mean_t - mean_c) / pooled SD.

    Scale-free, so we can compare balance across covariates with different units.
    Rule of thumb: |SMD| < 0.1 is well balanced. Note this is an effect size,
    not a p-value -- with n = 64k, p-values would flag trivially small gaps.
    """
    m_t, m_c = x_t.mean(), x_c.mean()
    s_t, s_c = x_t.std(ddof=1), x_c.std(ddof=1)
    pooled = np.sqrt((s_t ** 2 + s_c ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (m_t - m_c) / pooled


def balance_table(df: pd.DataFrame, treat_arm: str, control_arm: str,
                  covariates: list[str], arm_col: str = "segment") -> pd.DataFrame:
    """SMD for every covariate, treatment arm vs control.

    Categorical covariates are one-hot expanded so each level gets its own SMD
    (an SMD of a binary indicator compares the two arms' proportions).
    """
    sub = df[df[arm_col].isin([treat_arm, control_arm])].copy()
    X = pd.get_dummies(sub[covariates], drop_first=False, dtype=float)
    is_treat = (sub[arm_col] == treat_arm).to_numpy()

    rows = []
    for col in X.columns:
        rows.append({
            "covariate": col,
            f"mean_{treat_arm}": X.loc[is_treat, col].mean(),
            f"mean_{control_arm}": X.loc[~is_treat, col].mean(),
            "smd": _smd(X.loc[is_treat, col], X.loc[~is_treat, col]),
        })
    out = pd.DataFrame(rows).set_index("covariate").round(4)
    n_flagged = (out["smd"].abs() >= 0.1).sum()
    print(f"{treat_arm} vs {control_arm}: {len(out)} covariate columns, "
          f"{n_flagged} with |SMD| >= 0.1")
    return out


# ---------------------------------------------------------------------------
# 3. Average treatment effects, difference in means
# ---------------------------------------------------------------------------
def ate_binary(df: pd.DataFrame, outcome: str, treat_arm: str, control_arm: str,
               arm_col: str = "segment", alpha: float = 0.05) -> dict:
    """ATE for a binary outcome via two-proportion z-test.

    In an RCT, difference in means is an unbiased ATE estimator. CI uses the
    unpooled SE; the p-value uses the pooled SE (standard practice, because the
    null hypothesis says the two proportions are equal).
    """
    y_t = df.loc[df[arm_col] == treat_arm, outcome].to_numpy()
    y_c = df.loc[df[arm_col] == control_arm, outcome].to_numpy()
    n_t, n_c = len(y_t), len(y_c)
    p_t, p_c = y_t.mean(), y_c.mean()
    diff = p_t - p_c

    se_unpooled = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    p_pool = (y_t.sum() + y_c.sum()) / (n_t + n_c)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))

    z = diff / se_pooled
    pval = 2 * stats.norm.sf(abs(z))
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return {
        "outcome": outcome, "treat_arm": treat_arm,
        "mean_treat": p_t, "mean_control": p_c, "ate": diff,
        "ci_low": diff - zcrit * se_unpooled, "ci_high": diff + zcrit * se_unpooled,
        "rel_lift_pct": 100 * diff / p_c if p_c > 0 else np.nan,
        "p_value": pval,
    }


def ate_continuous(df: pd.DataFrame, outcome: str, treat_arm: str, control_arm: str,
                   arm_col: str = "segment", alpha: float = 0.05) -> dict:
    """ATE for a continuous outcome via Welch's t-test (unequal variances).

    Caveat for `spend`: ~99% zeros, heavy right tail. With n ~ 21k per arm the
    CLT keeps the t-test serviceable for the MEAN, but we note the zero-inflation
    and could bootstrap as a robustness check.
    """
    y_t = df.loc[df[arm_col] == treat_arm, outcome].to_numpy()
    y_c = df.loc[df[arm_col] == control_arm, outcome].to_numpy()
    m_t, m_c = y_t.mean(), y_c.mean()
    diff = m_t - m_c

    t_stat, pval = stats.ttest_ind(y_t, y_c, equal_var=False)
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    # Welch-Satterthwaite dof
    v_t, v_c = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    dof = (v_t + v_c) ** 2 / (v_t ** 2 / (len(y_t) - 1) + v_c ** 2 / (len(y_c) - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "outcome": outcome, "treat_arm": treat_arm,
        "mean_treat": m_t, "mean_control": m_c, "ate": diff,
        "ci_low": diff - tcrit * se, "ci_high": diff + tcrit * se,
        "rel_lift_pct": 100 * diff / m_c if m_c > 0 else np.nan,
        "p_value": pval,
    }


def ate_table(df: pd.DataFrame, control_arm: str = "No E-Mail",
              arm_col: str = "segment") -> pd.DataFrame:
    """All ATEs: each email arm vs control, for visit, conversion, spend."""
    treat_arms = [a for a in df[arm_col].cat.categories if a != control_arm]
    rows = []
    for arm in treat_arms:
        for outcome in ["visit", "conversion"]:
            rows.append(ate_binary(df, outcome, arm, control_arm, arm_col))
        rows.append(ate_continuous(df, "spend", arm, control_arm, arm_col))
    out = pd.DataFrame(rows).set_index(["treat_arm", "outcome"])
    return out.round(5)
