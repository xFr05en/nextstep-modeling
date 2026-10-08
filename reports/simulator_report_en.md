# Alternative-Data Simulator Report (Step 3)

> Educational use only. Not for real financial decisions. All alternative variables and gender are simulated, not observed.

- Code: `src/data/simulator.py` (run `python -m src.data.simulator`), settings in `config/simulator.yaml`
- Trade-off sweep and figures: `notebooks/simulator_report.py` (run `python notebooks/simulator_report.py`)
- Outputs: `data/processed/gmsc_sim.csv` (main, b = 0.30), `data/processed/gmsc_sim_target_only.csv` (comparison, b = 0)
- Numbers: `reports/simulator_summary.json`, `reports/simulator_sweep.json`. Figures 08 to 10 in `reports/figures/` (Korean and English)

**Interpretation of the charter rule:** the 0.3 to 0.5 correlation rule is applied to the copula's latent correlation (hidden variable score vs. hidden default score), and the observed Pearson values with the 0/1 default column are listed alongside.

## 1. How the simulator works

Each alternative variable starts as a hidden score `z` on a standard normal scale:

`z = -(a × T + b × w × C) + noise × e`

| Part | Meaning |
|---|---|
| `T` | Hidden default score. Defaulters get a value from the top 6.68% of a normal curve, non-defaulters from the rest. |
| `C` | Credit-behavior score from real GMSC data: total past-due count plus revolving utilization, each converted to rank-based normal scores. Higher = worse. |
| `e` | Independent random noise per variable. |
| `a` | Calibrated so the latent correlation r(z, T) equals -0.35 for every variable. `z` includes the credit part, so the total is calibrated. |
| `b × w` | Strength of the link to credit behavior. Main dataset b = 0.30. Weight w = 1.0 for the two on-time rates, 0.5 for tenure, insurance months and autopay. |

The minus sign makes higher values mean lower default risk, as `actionability.yaml` requires. The copula step then ranks `z` and maps it to each variable's real distribution (beta, gamma, binomial or 6-step discrete). Everything uses seed 42, so the same seed gives the same data.

**Missing values in the credit part:**
- The 269 special-code rows (96/98) have no past-due counts after Step 1. They default at 54.65%, so they are ranked as the worst past-due behavior.
- The 241 utilization outliers (above 10) default at 7.05%, close to the 6.68% average. There is no sign they are riskier, so they get a neutral utilization score (0, the middle of the scale).

**Tenure cap:** phone carrier tenure cannot exceed (age - 18) × 12 months. The cap applies to 0.69% of rows.

**Gender:** a separate 50/50 random draw. Female share 50.26%, correlation with default 0.0007.

## 2. Results per variable (main dataset, b = 0.30)

| Variable | Latent r | Observed Pearson r | Spearman r | AUC alone |
|---|---|---|---|---|
| Phone bill on-time rate | -0.350 | -0.273 | -0.235 | 0.763 |
| Utility bill on-time rate | -0.350 | -0.273 | -0.234 | 0.762 |
| Phone carrier tenure | -0.350 | -0.176 | -0.209 | 0.742 |
| Insurance paid months | -0.350 | -0.223 | -0.207 | 0.737 |
| Autopay share | -0.350 | -0.202 | -0.202 | 0.731 |

Comparison version without the credit link (target-only, b = 0):

| Variable | Latent r | Observed Pearson r | Spearman r | AUC alone |
|---|---|---|---|---|
| Phone bill on-time rate | -0.350 | -0.188 | -0.172 | 0.693 |
| Utility bill on-time rate | -0.350 | -0.185 | -0.171 | 0.691 |
| Phone carrier tenure | -0.350 | -0.153 | -0.176 | 0.704 |
| Insurance paid months | -0.350 | -0.184 | -0.174 | 0.700 |
| Autopay share | -0.350 | -0.171 | -0.171 | 0.695 |

**Check of the expected values:** with a 6.68% default rate, latent 0.35 was expected to give an observed Pearson r of about 0.18 and an AUC of about 0.70. The target-only version confirms this (Pearson 0.15 to 0.19, AUC 0.69 to 0.70).

**Why the linked version is stronger at the same latent r:** the hidden default score `T` contains random variation within each group that has nothing to do with the actual outcome. The credit score `C` comes from real data, so its link to the actual 0/1 outcome is stronger than its link to `T`. As a result, a variable built partly from `C` carries more information about real default (Pearson up to 0.27, AUC up to 0.76) even though its latent r is still 0.35.

**Link to real credit features (main dataset):** the on-time rates correlate -0.25 with total past-due count and -0.27 with utilization. The other three variables, which use half the link, correlate -0.13 to -0.16 with both.

**Between alternative variables:** Pearson r ranges from 0.12 to 0.19, far below the 0.60 cap. The two on-time rates correlate 0.19, so no extra noise was needed (noise stays 1.0 for all).

## 3. Correlation with age (for the fairness analysis)

| Variable | Pearson r with age |
|---|---|
| Phone bill on-time rate | +0.092 |
| Utility bill on-time rate | +0.091 |
| Phone carrier tenure | +0.065 |
| Insurance paid months | +0.056 |
| Autopay share | +0.052 |

The tenure cap adds almost nothing: tenure is no more related to age than the other variables. The small positive relationship comes indirectly, because older borrowers have lower utilization and default less, and the simulator links the variables to both. In the target-only version it is 0.02 to 0.03. So the alternative variables are a weak stand-in for age, and the link to credit behavior makes it 2 to 4 times larger.

## 4. Thin-filer flag

`thin_filer = 1` when `NumberOfOpenCreditLinesAndLoans <= 2` and `NumberRealEstateLoansOrLines == 0` (rule in `config/simulator.yaml`, function in `src/features.py`). It flags 11,564 rows (7.71%). This completes Step 4 early.

## 5. Trade-off: link to credit behavior vs. AUC gain (Figure 08)

LightGBM with default settings, 5-fold stratified CV (by default status and thin-filer flag), the same folds for both models. The gain is AUC with alternative data minus AUC without it. GMSC-only AUC is 0.865 overall and 0.850 for thin-filers.

| Credit link b | Thin-filer gain | Overall gain | Thin-filer AUC with alt. data | Mean single-variable AUC |
|---|---|---|---|---|
| 0 (target-only) | +0.075 ± 0.010 | +0.068 | 0.925 | 0.697 |
| 0.15 | +0.071 ± 0.009 | +0.064 | 0.922 | 0.723 |
| **0.30 (main)** | **+0.067 ± 0.008** | **+0.061** | **0.918** | **0.747** |
| 0.45 | +0.065 ± 0.007 | +0.060 | 0.915 | 0.768 |

(± is 1 standard deviation across the 5 folds.)

**How to explain it:** linking the variables to credit behavior lowers the thin-filer gain from +0.075 to +0.067, a drop of 0.007 or about 10%. This happens even though each variable becomes stronger on its own (mean AUC 0.70 to 0.75). The reason is overlap: part of what the variables know is already in the past-due counts and utilization, so it adds less new information. In every setting the gain stays above the +0.03 goal. The difference between neighboring settings is smaller than the fold-to-fold spread, so only the overall downward trend should be presented, not single steps.

## 6. Limitations

- **Simulation bias (main limitation).** The alternative variables are generated from the real default column, so every row's values already "know" its outcome, including rows that land in a test fold. This cannot be avoided in a simulation, but it means the absolute gain (+0.067) is optimistic and should not be read as what real telecom or utility data would deliver. The trade-off pattern (direction and relative size) is the more reliable result.
- **The size of the gain depends on our settings.** The latent target (0.35), the link strength b and the weights were chosen by the team, not estimated from real alternative data.
- **The overall AUC is high (0.926 with alternative data).** This follows from five independent sources of default information. Treat it as a best case.
- One seed, default LightGBM settings. Step 5 does the full model comparison.
