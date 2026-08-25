# Who should get the email? Uplift modeling on a real randomized experiment

Estimating heterogeneous treatment effects and simulating targeting policies on the
[Hillstrom MineThatData E-Mail Analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
dataset — a true randomized experiment on 64,000 customers with three arms
(men's e-mail campaign, women's e-mail campaign, no e-mail control).

**Status: work in progress** (Day 1 of 6 complete).

## Research questions

1. What is the average treatment effect (ATE) of the e-mail campaigns on visit rate, conversion, and spend?
2. Are effects heterogeneous across customer segments (CATE)?
3. If we could only e-mail X% of customers, whom should we target — and how much better is uplift-based targeting than blanket or random targeting?

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -c "from src.data import download_data; download_data()"
```

## Repo layout

- `notebooks/` — narrative analysis, one notebook per project phase
- `src/` — reusable, importable functions (data loading, validation, estimators)
- `data/` — raw data (gitignored; re-download with `src/data.py`)
- `reports/figures/` — exported figures

## Progress

- [x] Day 1 — EDA, experiment validation (SRM, covariate balance), ATEs with CIs
- [ ] Day 2 — Subgroup ATEs, T-learner CATE
- [ ] Day 3 — X-learner, causal forest, out-of-fold predictions
- [ ] Day 4 — Qini / uplift@k / AUUC validation
- [ ] Day 5 — Targeting policy simulation, placebo & robustness checks
- [ ] Day 6 — Write-up, figures, limitations
