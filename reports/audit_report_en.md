# Model Audit Report: Recourse, Fairness, Stability, Sensitivity

> Educational use only. Not for real financial decisions. The audit changes nothing: configs, the shipped model (`models/xgboost_v1.0.joblib`) and the training MLflow runs stay as they are.

- Code: `notebooks/audit.py` (run `python notebooks/audit.py`, about 3 minutes). Numbers: `reports/audit/*.json`. MLflow: `nextstep-audit`, one run per check.
- Every check uses the official split (`data/processed/split.csv`): decisions and paths on the **test set**, retraining on train + validation. The earlier 80/20 holdout re-selection is dropped; the official split replaces it.
- Approval cutoff 475 (fixed rule, `config/scoring.yaml`); recourse target 495 (`recourse_target_score`).
- The simulator uses the target for all rows, test included (as in the mission example), so recourse and performance figures with alternative data are optimistic.

## Summary

| Check | Result | Verdict |
|---|---|---|
| 1. Recourse coverage (charter goal ≥ 90%): rejected test applicants who can reach the target within 12 months with at most 3 variables | target 495: **87.9%** (thin-filers 87.0%); target 475: 90.0% | **Below 90% at the 495 target over 12 months; open item for 체민규** (24 months: 91.8%) |
| 2. Fairness, gender | DI 0.995; EO gap 0.038 | **Pass** |
| 2. Fairness, age band (20-34 / 35-54 / 55+) | DI 0.832; **EO gap 0.121** | DI passes; **EO fails the charter target (≤ 0.10). Open item for 체민규** |
| 3. Stability: approval flips within ±30 points under 3 retrained models | 26.8% | Above 10%; slower learning does not help (26.4%) |
| 3. Path robustness (charter ≥ 80% of paths still approved) | minimal paths aimed at 495: **90.8%** under all 3 models | **Pass** (paths aimed at 475: 51.9%, would fail) |
| 4. Sensitivity: thin-filer gain across mission r −0.30 to −0.50 and thin-filer definitions | +0.084 to +0.149 | Always ≥ +0.03; r = −0.50 cannot meet the 0.60 cap |

## 1. Recourse feasibility

**Setup:** rejected = test applicants with a shipped-model score below 475 (2,941 of 22,500, 13.1%; 25.3% of thin-filers). Path variables: the 8 model features with `dice_vary: true` (utilization, debt ratio, income, both payment rates, tenure, insurance months, spending consistency), each moved only in its YAML direction, by its step, within its bounds and the time horizon. Autopay (not a model feature) and the two count variables (NOT_RECOMMENDED) are never used. All 8 are monotonic, so checking every set of 1 to 3 variables at its maximum allowed change is an exact search.

| View | Reach target | Thin-filers | Not feasible (3-variable limit / even with all 8) |
|---|---|---|---|
| **12 months, target 495 (main)** | **87.9%** | **87.0%** | 357 (238 / 119) |
| 24 months, target 495 | 91.8% | 91.7% | 240 (229 / 11) |
| 12 months, target 495, without TIME_ONLY (tenure, insurance) | 75.3% | 72.4% | 726 (161 / 565) |
| 12 months, target 475 (reference) | 90.0% | 89.7% | 293 (214 / 79) |

**Coverage vs. the charter's 90% path-coverage goal** (share of rejected test applicants with at least one path of at most 3 variables):

| Horizon | Target 475 (approval line) | Target 495 (recourse target) |
|---|---|---|
| 12 months | 90.0% (thin-filers 89.7%, others 90.1%) | **87.9%** (thin-filers 87.0%, others 88.0%) |
| 24 months | 93.6% (thin-filers 95.1%, others 93.4%) | 91.8% (thin-filers 91.7%, others 91.9%) |

- **At the recourse target 495, 12-month coverage is 87.9%, below the 90% goal.** Over 24 months it is 91.8%. At the 475 line, 12-month coverage is exactly 90.0%, with thin-filers at 89.7%.
- The gap comes mostly from the 3-variable limit (238 of the 357 at 12 months / 495 would need 4 or more variables); 119 cannot reach 495 even with all 8, almost all because of 60+ day late payments.
- **Open item for 체민규:** DiCE path coverage is his metric. Options to discuss: allow 4 variables, a 24-month horizon, or report coverage at 475 with the 495 margin as a robustness target. Nothing was changed here.

- Variables needed (main view): 1 for 1,663 applicants, 2 for 700, 3 for 221.
- Most common paths: `spending_consistency` alone (1,329), insurance months + spending consistency (624), insurance months alone (324).
- **What blocks the rest:** 83% of the 357 have a 60+ day late payment, which cannot decrease; their median score is 121, 374 points short.
- **Note for 체민규:** raising income also lowers the debt ratio in reality, while this search moves them independently; DiCE should model that link. TIME_ONLY paths (insurance months) must be phrased as waiting time, not as an action.

## 2. Fairness (test-set decisions of the shipped model)

TPR = approval rate among applicants who repaid; FPR = approval rate among those who defaulted. Equalized-odds (EO) gap = the larger of the TPR and FPR differences. Rules: DI in [0.8, 1.25]; EO gap ≤ 0.10.

**Gender (simulated, independent of default):**

| Group | n | Default rate | Approval | TPR | FPR |
|---|---|---|---|---|---|
| Female | 11,287 | 6.6% | 86.7% | 0.916 | 0.171 |
| Male | 11,213 | 6.7% | 87.2% | 0.920 | 0.209 |

DI 0.995 (pass); TPR difference 0.003, FPR difference 0.038, EO gap 0.038 (pass).

**Age band (mission bands):**

| Group | n | Default rate | Approval (95% CI) | TPR | FPR |
|---|---|---|---|---|---|
| 20 to 34 | 2,970 | 11.8% | 77.4% (75.9 to 78.9) | 0.857 | 0.149 |
| 35 to 54 | 9,753 | 8.4% | 83.8% (83.0 to 84.5) | 0.898 | 0.175 |
| 55+ | 9,777 | 3.5% | 93.0% (92.5 to 93.5) | 0.954 | 0.269 |

- DI = 0.832 (20 to 34 vs. 55+): passes the 0.8 rule.
- **EO: TPR difference 0.096, FPR difference 0.121, so the EO gap is 0.121. This fails the charter target of 0.10 or less.** Among people who repaid, the youngest are approved 85.7% of the time vs. 95.4% for 55+; among defaulters, older applicants are approved more often (26.9% vs. 14.9%).
- **Status: fail, open item for 체민규's mitigation work** (for example reweighting or group thresholds, tuned on validation and reported on test). Nothing in the data, simulator or model was tuned to pass it.

## 3. Stability

The model uses no row or column sampling, so changing only `random_state` changes nothing (max PD difference 0.0). Stability was measured by retraining on 3 random 80% subsamples of train + validation and scoring the test set.

| Variant | Test rows within ±30 points | Flip share near cutoff | Flip share overall | Test AUC (3 models) |
|---|---|---|---|---|
| 300 trees, lr 0.05 (shipped) | 1,413 (6.3%) | **26.8%** | 1.7% | 0.949, 0.949, 0.949 |
| 1000 trees, lr 0.02 | 1,413 (6.3%) | 26.4% | 1.7% | 0.949, 0.949, 0.948 |

Ranking is stable (AUC unchanged), but decisions close to 475 are not, and slower learning does not fix it. A grey zone around 475 is an open team item.

**Path robustness** (paths from section 1, scored by the 3 retrained models; "still approved" = score ≥ 475):

| Paths | Approved, per model | All 3 models | Thin-filers, all 3 |
|---|---|---|---|
| **Minimal, target 495** | 96.7%, 95.5%, 97.3% | **90.8%** | 85.8% |
| Maximum change, target 495 | 99.0%, 97.1%, 99.4% | 95.9% | 91.8% |
| Minimal, target 475 (reference) | 73.9%, 72.9%, 77.0% | 51.9% | 49.5% |

The 495 target meets the charter's 80% robustness target; paths aimed only at 475 would not.

## 4. Sensitivity (5-fold CV inside train, final setup)

**Mission variables' observed r with default** (all 5 set to the same value):

| Observed r | Within the 0.60 pair cap | Largest pair r | CV AUC | Thin-filer gain |
|---|---|---|---|---|
| −0.30 | yes | 0.591 | 0.936 | +0.086 ± 0.010 |
| **−0.32 (current)** | yes | 0.591 | 0.943 | **+0.094 ± 0.008** |
| −0.40 | yes | 0.594 | 0.971 | +0.124 ± 0.006 |
| −0.50 | **no** | 0.816 (lowest reachable) | 0.995 | +0.149 ± 0.004 |

At −0.50 the variables are so strongly tied to default that even without a shared factor two of them correlate 0.82, so the 0.60 cap cannot hold; the row shows the result with the cap lifted. Across the mission's 0.3 to 0.5 range the gain nearly doubles, so its size is set by this choice.

**Thin-filer definition** (open credit lines, with 0 real estate loans):

| Definition | Share | Default rate | Thin-filer AUC | Gain |
|---|---|---|---|---|
| ≤ 1 line | 4.0% | 16.9% | 0.924 | +0.121 ± 0.013 |
| **≤ 2 lines (current)** | 7.7% | 13.1% | 0.939 | **+0.094 ± 0.008** |
| ≤ 3 lines | 11.7% | 11.2% | 0.943 | +0.084 ± 0.011 |

## 5. Open items

1. **Age-band EO gap 0.121 > 0.10 (fail):** 체민규's mitigation.
1b. **Recourse coverage 87.9% < 90% (12 months, target 495):** 체민규 (DiCE coverage); see section 1.
2. **Grey zone around 475** (26.8% of near-cutoff decisions flip on retraining): team decision.
3. **Income and debt-ratio link in DiCE:** 체민규.
4. **TIME_ONLY paths** are phrased as waiting time; the NOT_RECOMMENDED variables appear only as "reference (not changeable)" in rejection reasons.
