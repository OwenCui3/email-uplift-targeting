# Who should get the email? — one-page summary

**Setting.** A retailer e-mailed 64,000 recent customers in a three-arm randomized
experiment (men's campaign / women's campaign / no e-mail), tracking site visits,
conversion, and spend for two weeks. E-mail is nearly free but not costless — send costs,
list fatigue, unsubscribes — so the real question is not *whether* to e-mail but *whom*.

**Is the experiment trustworthy?** Yes. Arm sizes match the intended 1/3 split
(χ² p ≈ 0.95); all 18 covariate columns balance within |SMD| < 0.02; and adjusting the
effect estimates for covariates moves them by less than 0.001 — the signature of clean
randomization.

**Average effects.** The men's e-mail lifts visit rate 10.6% → 18.3% (+7.7pp, CI ±0.7)
and spend by $0.77/customer; the women's e-mail +4.5pp and $0.42. Effects shrink down
the funnel: visits are easy to nudge, purchases hard. Conversion effects are positive
but small in absolute terms (+0.3 to +0.7pp).

**Heterogeneity.** Focusing on the women's-campaign arm (moderate average effect, more
room for targeting to matter): subgroup slices hint that high prior-spend customers
respond more and rural customers less. Model-based CATE estimation (T-learner, X-learner,
causal forest; all scores out-of-fold) sharpens this into a per-customer ranking whose
top decile shows 7.6pp realized uplift — 1.7× the average.

**Does the ranking actually work?** Validated on actual randomized outcomes: Qini
coefficients of 151–163 versus 48 for a naive "predict who will visit" ranking (~3×) and
0 for random, with bootstrap CIs excluding both baselines. A placebo run with permuted
treatment labels returns ≈ 0, confirming the pipeline cannot conjure uplift from noise.
The naive baseline's failure is the core lesson: customers *likely to visit* are largely
not the customers *changed by the e-mail*.

**Policy.** Translating the ranking into money (assumed 30% margin, $0.10/e-mail;
incremental spend measured empirically within each targeted slice): profit peaks when
e-mailing roughly the top 40–60% of customers, beating blanket e-mailing by +73% to +93%
in simulation depending on the ranking model. The optimum moves sensibly with the
economics — cheaper e-mails ⇒ e-mail more; thinner margins ⇒ fewer.

**Honest limits.** Simulated policy under stated assumptions, not a deployment; spend's
zero-inflation makes slice-level revenue noisy (the profit curve carries a wide bootstrap
band, and one planned metric — incremental revenue captured at top-k — was too unstable
to report); CATE magnitudes are validated as an ordering, not calibrated values;
two-week window, one dataset.

**What I'd do next with production access.** A/B test the policy itself (uplift-targeted
vs blanket arms), CUPED-style variance reduction on spend, calibration of CATEs for
budget planning, and multi-armed assignment — *which* campaign per customer, not just
whether to send.
