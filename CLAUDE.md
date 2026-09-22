# Project context for AI assistants

Six-day causal inference portfolio project on the Hillstrom e-mail RCT (64k customers,
3 arms). Owner is a Stanford ICME MS student building interview-ready understanding —
prefer small, explained changes over clever rewrites, and never commit anything the
owner can't explain.

Conventions:
- Narrative lives in `notebooks/` (run in order 01→05; 03 writes `data/oof_cates.csv`
  consumed by 04–05). Reusable logic lives in `src/` with docstrings.
- Figures save to `reports/figures/` at dpi=200; matplotlib only (no seaborn);
  fixed arm colors: control #6A6A6A, mens #0072B2, womens #CC7A00.
- Pinned deps in `requirements.txt` (Python 3.12). Seed 42 everywhere; reported numbers
  must reproduce exactly.
- Never commit `data/` contents or the venv. Raw data re-downloads via `src/data.py`.
- Honesty over polish: weak or noisy results get reported with caveats, not hidden.
