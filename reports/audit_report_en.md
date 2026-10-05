# Model Audit Report: Recourse, Fairness, Stability, Holdout, Sensitivity

> Educational use only. Not for real financial decisions. The audit itself changed nothing. After review, two config decisions were made (autopay removed from paths, recourse target 495); their effect is reported in sections 1.6 and 3.1. The final model is unchanged.

- Code: `notebooks/audit.py` (run `python notebooks/audit.py`, about 3.5 minutes)
- Numbers: `reports/audit/*.json`. Holdout re-selection tables: `reports/audit/holdout_selection_*.csv`. MLflow experiment: `nextstep-audit`
- Cutoff: score 475 (lowest score of grade C, from `config/scoring.yaml`)

## Summary

| Check | Result | Verdict |
|---|---|---|
| 1. Recourse feasibility (12 months, at most 3 variables) | 99.95% of rejected applicants can reach 475; thin-filers 100% | Feasible, but suspiciously easy (see 1.3) |
| 1b. Recourse, decided setup (no autopay, target 495) | 98.7% can reach 495; thin-filers 99.8% | Feasible |
| 2. Fairness by age | Lowest DI 0.81 (under 30); equalized-odds gap 12.3 points | DI passes 0.8 narrowly; **EO gap fails the charter target (≤ 0.10)** |
| 3. Stability (3 retrained models) | 27.9% of applicants within 30 points of the cutoff flip | Above 10%; 1000 trees at lr 0.02 does not help (29.0%) |
| 3. Path robustness (charter ≥ 80%) | Original minimal paths (target 475) 59.7% under all 3 models; decided minimal paths (target 495) **90.6%** | Fails at 475; passes at 495 |
| 4. Holdout (20%, never used for selection) | Same winner. AUC 0.927, KS 0.703, thin-filer gain +0.055 (95% CI 0.039 to 0.070) | All charter targets met |
| 5. Sensitivity | Thin-filer gain +0.057 to +0.108 across latent r 0.30 to 0.50 and thin-filer definitions | Gain stays above +0.03 in every case |

## 1. Recourse feasibility

**Setup:** rejected = deployed model score below 475 (21,479 applicants, 14.3%; 28.7% of thin-filers). Variables: the 8 with `dice_vary: true`, moved only in their YAML direction, by their YAML step, within their YAML bounds, and at most `horizon / months_per_step` steps. Every one of the 8 has a monotonic constraint, so the best reachable score for a set of variables is at the corner where each is moved as far as allowed. Checking all 92 sets of 1 to 3 variables at that corner is therefore an exact search.

### 1.1 Results

| View | Feasible (all) | Thin-filers | Others | Not feasible |
|---|---|---|---|---|
| **Main: 8 variables, 12 months** | **99.95%** | **100%** | 99.94% | 10 |
| 8 variables, 24 months | 99.95% | 100% | 99.94% | 10 |
| Without TIME_ONLY (tenure, insurance), 12 months | 98.99% | 99.58% | 98.88% | 217 |
| Without autopay, 12 months | 99.27% | 99.97% | 99.15% | 156 |

Variables needed (main view): 1 variable for 15,392 applicants (71.7%), 2 for 5,298 (24.7%), 3 for 779 (3.6%). 24 months gives the same result as 12 because most variables reach their bounds within 12 months.

Most common paths (main view; for each applicant, the smallest set with the largest score gain):

| Path | Applicants |
|---|---|
| Insurance paid months (wait up to 12 months) | 4,754 |
| Autopay share (to 100%, about 2.5 months) | 4,318 |
| Utility bill on-time rate | 3,573 |
| Phone bill on-time rate | 2,744 |
| Utility on-time rate + autopay | 1,940 |

### 1.2 What blocks the rest

- **Main view:** the 10 applicants who cannot reach 475 would need 4 or more variables; none is blocked even with all 8. All 10 have at least one 60+ day late payment and a score near 0.
- **Without TIME_ONLY (217) and without autopay (156):** over 92% of the blocked applicants have a 60+ day late payment, and their median score is about 50, more than 420 points short. Late-payment history is NON_DECREASING, so recourse cannot touch it. That is the real blocker.

### 1.3 Interpretation

Recourse is almost always feasible because the simulated alternative variables are strong (single-variable AUC about 0.73 to 0.76) and can be moved to their limits within months. One variable is enough for 72% of rejected applicants, and only 10 of 21,479 cannot reach the cutoff at all, even though late-payment history cannot be changed. This is a property of the simulation, not evidence that real bill-payment data would work this way.

Two consequences for the presentation and for DiCE:
- **Autopay was the single easiest path for 4,318 applicants.** It takes about 2.5 months and, as the original notes say, does not lower real risk. **Decided after this audit:** autopay is removed from recourse paths (`dice_vary: false`) and kept as a model feature.
- **Insurance months is the most common path, but it is TIME_ONLY:** the applicant can only wait. It must be phrased as a duration, never as an action.

### 1.4 Note for 채민규: income and debt ratio

In reality, raising income also lowers the debt ratio, because the ratio is divided by income. The corner search treats every variable as independent: it can raise income without changing the debt ratio, or lower the debt ratio without changing income. DiCE should model this link. When income rises by a factor k, the debt ratio should become (debt ratio / k) unless debt also changes. Otherwise income-only paths will look weaker than they are, and paths combining both can double count.

### 1.5 Limits of this check

- The corner is the maximum allowed change, so feasibility is an upper bound. DiCE will look for the cheapest change and may not find every path the corner finds.
- Waiting 12 months also ages the applicant. Age is held fixed here (it is IMMUTABLE), and the tenure age cap is not re-applied.
- "Rejected" uses the deployed model, which was trained on all rows, because that is the model DiCE will query.

### 1.6 Decided setup: no autopay, recourse target 495

Rerun with the current YAML (7 variables, autopay excluded) and `recourse_target_score` = 495 from `config/scoring.yaml`, 12 months, at most 3 variables.

| | All | Thin-filers | Others |
|---|---|---|---|
| Can reach 495 | **98.68%** | **99.82%** | 98.47% |
| Not feasible | 284 (1.32%) | | |

- Variables needed: 1 for 13,406 (62.4%), 2 for 6,060 (28.2%), 3 for 1,729 (8.0%). Paths need more variables than before because the target is 20 points higher and autopay is gone.
- Not feasible: 257 would need 4 or more variables, and 27 cannot reach 495 even with all 7. Of the 284, 97.9% have a 60+ day late payment, and their median score is 65 (430 points short).
- Most common paths: insurance months (5,850, waiting only), utility on-time rate (4,025), phone on-time rate (3,512), utility on-time rate + insurance months (2,900).

## 2. Fairness by age group

Out-of-fold decisions of the current setup (no in-sample scores). 95% intervals use the normal approximation.

| Age | Applicants | Default rate | Approval rate (95% CI) | DI ratio | Approval if repaid | Approval if defaulted |
|---|---|---|---|---|---|---|
| Under 30 | 8,820 | 11.7% | 73.7% (72.8 to 74.6) | **0.81** | 81.4% | 16.0% |
| 30 to 49 | 57,560 | 9.1% | 80.4% (80.1 to 80.7) | 0.89 | 86.5% | 19.3% |
| 50+ | 83,619 | 4.5% | 90.7% (90.5 to 90.9) | 1.00 | 93.7% | 28.3% |

- **DI ratio:** the lowest is 0.81 (under 30 vs. 50+), just above the 0.8 rule. Part of the gap reflects a real difference in default rate (11.7% vs. 4.5%).
- **Equalized-odds gap: 12.3 points, which fails the charter target of 0.10 or less, even though DI passes.** Among applicants who actually repaid, those under 30 are approved 81.4% of the time, against 93.7% for 50+. Among those who defaulted, the gap is 12.3 points the other way: older defaulters are approved more often. So the model treats age groups differently even when the actual outcome is the same.
- **Why:** age is a model feature, and utilization and dependents carry age information as well (Step 2).
- **Control check, gender (simulated, independent of default):** DI 0.998, EO gap 0.6 points. This is as expected and confirms the measurement works.

These numbers go to 채민규's fairness work. No mitigation was applied here. **Status: the EO gap (0.123) does not meet the charter target (≤ 0.10); mitigation is open and owned by 채민규.**

## 3. Stability

**Seeds alone do nothing:** the model uses no row or column sampling, so two models with different `random_state` give identical predictions (max PD difference 0.0). Stability was therefore measured by training the final setup on 3 different random 80% subsamples (seeds 11, 22, 33) of the holdout training set and scoring the same 30,000 holdout applicants.

| Variant | Applicants within ±30 points | Flip share near cutoff | Flip share overall | Holdout AUC (3 models) |
|---|---|---|---|---|
| 300 trees, lr 0.05 (current) | 2,345 (7.8%) | **27.9%** | 2.2% | 0.926, 0.926, 0.926 |
| 1000 trees, lr 0.02 | 2,345 (7.8%) | 29.0% | 2.4% | 0.926, 0.926, 0.926 |

- The flip share near the cutoff is above the 10% limit. Slower learning does not reduce it, so the cause is the training data sample, not the training settings. AUC barely moves, so ranking is stable while individual decisions near the line are not.
- The ±30-point band is wide: 30 points is a 1.5x change in odds. Inside it, small score shifts of a few points are enough to cross 475.

### 3.1 Path robustness (charter: at least 80% of paths stay approved)

The recourse paths from check 1 (main view) were applied and scored by the 3 retrained models.

| Path type | Still approved (average per model) | Approved by all 3 | Thin-filers, all 3 |
|---|---|---|---|
| Maximum change (corner) | 95.5% | 92.2% | 91.3% |
| Minimal change (just reaches 475) | 77.3% | **59.7%** | 63.8% |
| **Decided: maximum change, target 495** | 98.7% | 97.1% | 95.9% |
| **Decided: minimal change, target 495** | **96.0%** | **90.6%** | **88.3%** |

"Still approved" always means a score of 475 or more under the retrained model. Original minimal paths fail, because they end just above the cutoff (median deployed score 490 after rounding up to whole steps). Minimal paths aimed at 495 (median deployed score 509) pass with 90.6% under all 3 models, which confirms the estimate below (91.1%). With 1000 trees at lr 0.02 the decided minimal paths give 89.1%, so the result does not depend on the training settings.

Estimate made before the decision, from the original minimal paths:

| Extra margin above the minimal path | Average per model | All 3 models |
|---|---|---|
| +0 | 77.3% | 59.7% |
| +10 | 90.0% | 79.6% |
| **+20** | **96.0%** | **91.1%** |
| +30 | 98.5% | 96.4% |

(Estimate: assumes a path aimed 20 points higher is affected by retraining in the same way.)

**Decided after this audit:** `recourse_target_score: 495` in `config/scoring.yaml`; approval stays at 475. Verified above: 90.6%.

## 4. Holdout

A stratified 20% (30,000 rows, seed 2026) was set aside, and the **whole selection was rerun on the other 80%**: the 27-run grid, the constraint stage and the winner rule. It picked **the same winner** (XGBoost, both feature sets, no resampling, all 11 constraints). That model was evaluated once on the holdout. 95% intervals: 500 bootstrap resamples.

| Metric | Holdout | 95% CI | 5-fold CV (Step 5) |
|---|---|---|---|
| AUC | 0.927 | 0.921 to 0.932 | 0.927 |
| KS | 0.703 | 0.688 to 0.718 | 0.704 |
| Thin-filer AUC | 0.919 | | 0.917 |
| Thin-filer AUC, GMSC only | 0.864 | | 0.847 |
| **Thin-filer AUC gain** | **+0.055** | **+0.039 to +0.070** | +0.071 |
| PSI (80% train scores vs. holdout) | 0.0005 | | 0.0005 |
| Brier | 0.040 | | 0.039 |
| Mean predicted PD / actual default rate | 6.71% / 6.68% | | |
| Approval rate (all / thin-filers) | 85.6% / 73.4% | | 85.8% / 71.5% |

Grades on the holdout: the actual default rate per grade is close to the predicted PD (A 0.5% vs. 0.6%, B 3.1% vs. 3.2%, C 7.7% vs. 7.2%, D 14.4% vs. 14.0%, E 49.3% vs. 49.4%).

The thin-filer gain is lower on the holdout (+0.055) than in CV (+0.071), mainly because the GMSC-only model happens to do better on these thin-filers (0.864 vs. 0.847). The whole interval stays above +0.03. One overlap remains: the simulator calibrated its hidden scores on all rows, including the holdout. The simulation is built from the outcome anyway (see the simulator report), so this does not change the bias picture.

## 5. Sensitivity

Final setup, 5-fold paired CV, gain = thin-filer AUC with alternative data minus GMSC only.

**Latent correlation target (all 5 variables):**

| Latent r | Observed Pearson r | AUC | Thin-filer gain |
|---|---|---|---|
| 0.30 | -0.16 to -0.25 | 0.913 | +0.057 ± 0.008 |
| 0.35 (current) | -0.18 to -0.27 | 0.927 | +0.071 ± 0.008 |
| 0.50 | -0.22 to -0.36 | 0.960 | +0.108 ± 0.011 |

**Thin-filer definition (credit lines, with 0 real estate loans):**

| Definition | Share | Default rate | Thin-filer AUC | Gain |
|---|---|---|---|---|
| ≤ 1 line | 4.1% | 16.6% | 0.897 | +0.091 ± 0.012 |
| ≤ 2 lines (current) | 7.7% | 12.9% | 0.917 | +0.071 ± 0.008 |
| ≤ 3 lines | 11.8% | 11.2% | 0.923 | +0.062 ± 0.006 |

The gain clears +0.03 in every case. It grows with the latent correlation, as expected, and it is largest for the thinnest files, which have the least credit history for the GMSC-only model to use. Within the charter range (0.30 to 0.50) the gain roughly doubles, so the size of the gain is mostly a consequence of the chosen setting.

## 6. Decisions and open items

**Decided after review (applied in a separate config commit):**
1. Autopay removed from recourse paths (`dice_vary: false`), kept as a model feature. Its notes in `actionability.yaml` and the Excel files explain why.
2. `recourse_target_score: 495` added to `config/scoring.yaml`, separate from the approval cutoff of 475.

**Handed to 채민규:**
3. **Age fairness:** the EO gap of 12.3 points fails the charter target (≤ 0.10) even though DI (0.81) passes.
4. **Income and debt ratio link** in DiCE (section 1.4).

**Open team items:**
5. **Grey zone around 475:** decisions within ±30 points flip in 27.9% of cases when the model is retrained, and slower learning does not help. A grey zone for manual review, or averaging several models, was not adopted for now.
6. **Autopay's class (resolved):** reclassified from ACTIONABLE to NOT_RECOMMENDED, so the class rule ("fixed in DiCE") matches `dice_vary: false`. Monotone stays -1 and it remains a model feature, so the final model is unchanged.
