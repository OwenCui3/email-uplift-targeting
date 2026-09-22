# Ten interview questions this project should survive

**1. Why do you need causal methods at all if the data is already an RCT?**
Randomization gives the *average* effect by subtraction. But targeting is an
individual-level counterfactual question — how much does the e-mail change *this*
customer? — and no individual's effect is ever observed. CATE models estimate it by
borrowing strength across similar customers; the RCT's role shifts from answering the
question to making validation honest (any slice of a ranking is still a randomized
comparison).

**2. T-learner vs S-learner vs X-learner?**
S-learner: one model, treatment as a feature — simplest, but regularization can shrink
the treatment's role and bias effects toward homogeneity. T-learner: separate outcome
models per arm, subtract — transparent, but each model sees half the data and their
independent smoothing errors don't cancel. X-learner: impute each unit's counterfactual
with the *other* arm's model, regress the imputed individual effects, blend by propensity
— regularizes the *effect function* directly, best when τ(x) is smoother than the outcome
functions or arms are imbalanced. In this project the X-learner won validation by a nose.

**3. Explain a Qini curve to a PM.**
Rank customers by how much the model thinks the e-mail changes them. Walk down the list,
e-mailing more and more, and plot the *extra* visits caused so far. E-mailing at random
traces a straight line; a good model bows above it — it front-loads the persuadable
customers. The area between curve and line is a single score for "how much better than
random targeting."

**4. How do you validate a model when ground truth is unobservable per person?**
By slices, not individuals. Because assignment ignored the model score (built from
pre-treatment features only), the treated-vs-control gap *within* the model's top-k% is
an unbiased estimate of realized uplift there (uplift@k); Qini generalizes this over all
k. All scores were out-of-fold and error bars come from bootstrapping customers.

**5. Why is out-of-fold prediction non-negotiable here?**
A flexible model scored on its own training data can "find" heterogeneity it memorized.
In-sample, the top decile looks spectacular; out-of-fold is the honest estimate of what
deploying the model on new customers achieves. This inflation is the single most common
error in uplift analyses.

**6. What would change with observational data instead of an RCT?**
Everything gets harder in the same direction: identification. I'd need
unconfoundedness — all confounders measured — plus overlap, and I'd use
propensity-based methods (IPW, AIPW/doubly-robust, DML) instead of raw differences;
validation metrics like Qini would themselves be biased by confounding, so I'd lean on
negative controls, sensitivity analysis (e.g., E-values), and any natural experiments
available. The RCT is what let this project make strong claims with simple tools.

**7. Your spend outcome is 99% zeros. Why is a t-test defensible?**
The t-test needs approximate normality of the *sample means*, not the data. At n≈21k per
arm the CLT delivers that even for a spike-plus-heavy-tail distribution. I verified
empirically: a 2,000-rep bootstrap CI for the spend ATE, [$0.15, $0.69], matches the
Welch CI almost exactly. At n=50 I would not have run the test.

**8. What's a sample-ratio mismatch and why does it invalidate everything, not just counts?**
Arms sized differently than designed means something *removed* people — crashes, bot
filters, logging bugs — and removal is never random (it eats specific devices, behaviors,
segments). The surviving groups are no longer exchangeable, so every downstream
comparison is contaminated. Wrong counts are the symptom; broken comparability is the
disease. It's a stop-the-analysis alarm, not an adjust-around-it nuisance.

**9. Why report covariate balance with SMDs instead of t-tests?**
With 64k rows the luck window shrinks like 1/√n, so t-tests flag differences far too
small to confound anything — p-values detect existence, not importance. SMD (gap ÷
pooled SD) is a sample-size-free, unit-free effect size with a standard actionability
threshold (0.1). The link: t ≈ SMD·√(n/2) — a p-value is magnitude times a sample-size
amplifier; balance checking wants the magnitude alone.

**10. Your profit simulation says "e-mail the top ~50%." How much would you bet on that number?**
Directionally a lot, precisely little. The ranking's quality has tight bootstrap CIs, but
slice-level *spend* is noisy, so the profit optimum is a region (~40–60%), the curve's
band is wide, and the +73–93% gain depends on stated margin/cost assumptions (sensitivity
grid included). In production I'd A/B the policy itself before believing any single
number — and I'd say exactly that to the stakeholder.
