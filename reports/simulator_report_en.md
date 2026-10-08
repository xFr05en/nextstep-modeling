# Alternative-Data Simulator Report

> Educational use only. Not for real financial decisions. All alternative variables and gender are simulated, not observed.

- Code: `src/data/simulator.py` (run `python -m src.data.simulator`; mission-style entry point `generate_alternative_data()`), settings in `config/simulator.yaml`
- Scenario checks: `notebooks/simulator_scenarios.py` → `reports/simulator_scenarios.json`. Credit-link sweep and figures 08 to 10: `notebooks/simulator_report.py`
- Outputs: `data/processed/gmsc_sim.csv` (main, b = 0.30), `data/processed/gmsc_sim_target_only.csv` (comparison, b = 0)

**Important:** the simulator uses the real default column for **every** row, including rows that later land in the test set, exactly as the mission's own example simulator does. Every result that uses alternative data is therefore optimistic. **The GMSC-only model is the realistic reference** (test AUC about 0.86).

## 1. The 8 variables

| Variable | Role | Range | Distribution | Correlation rule |
|---|---|---|---|---|
| `telecom_payment_rate` | mission | 0 to 1 (steps of 1/12) | Beta(8, 1.5) | observed Pearson r = −0.32 |
| `utility_payment_rate` | mission | 0 to 1 (steps of 1/12) | Beta(8, 1.5) | observed Pearson r = −0.32 |
| `spending_consistency` | mission | 0 to 100 | Beta(5, 2) × 100 | observed Pearson r = −0.32 |
| `regular_payment_count` | mission | 0 to 20, integer | Binomial(20, 0.75) | observed Pearson r = −0.32 |
| `app_login_frequency` | mission | 0 to 30, integer (days with a login) | Binomial(30, 0.75) | observed Pearson r = −0.32 |
| `telecom_tenure_months` | extra | 0 to 240, capped at (age − 18) × 12 | Gamma(2, 24) | latent r = −0.35 |
| `insurance_paid_months` | extra | 0 to 24 | Binomial(24, 0.75) | latent r = −0.35 |
| `autopay_ratio` | extra, **not a model feature** | 0 to 1 (steps of 0.2) | 6-step discrete | latent r = −0.35 |

The two payment rates were renamed from `telecom_ontime_rate` / `utility_ontime_rate` to the mission's names (same variables). Signs are negative: a higher value means a lower default risk.

## 2. How each value is generated

Each variable starts as a hidden score on a standard normal scale:

`z = -(a × T + b × w × C) + g × F + e`

| Part | Meaning |
|---|---|
| `T` | hidden default score (defaulters drawn from the top 6.68% of a normal curve) |
| `C` | credit-behavior score from past-due counts and revolving utilization only (no age, no gender) |
| `F` | one random factor shared by the 5 mission variables only |
| `e` | independent noise per variable |
| `a` | calibrated per variable: **observed** Pearson r with the 0/1 target = −0.32 ± 0.005 for the mission variables (what `df.corr()` shows, as the checklist verifies); **latent** r(z, T) = −0.35 for the extras |
| `b × w` | link to real credit behavior: b = 0.30; w = 1.0 for the two payment rates, 0.5 for the others |
| `g` | searched so the largest correlation between two mission variables stays at or below 0.60 (achieved 0.591) |

The copula step ranks `z` and maps it to the variable's distribution. Seeded (seed 42): the same seed gives byte-identical data.

**Why a shared factor:** the 5 mission variables overlap in information, which limits how much they add together. Its effect is modest: full-model CV AUC 0.968 (average pair 0.30), 0.959 (0.45), 0.957 (largest pair 0.60, chosen).

**Why Binomial for the counts:** with Poisson counts, defaulters bunch together at the low end, so reaching observed r = −0.32 needed a much stronger hidden link (AUC alone 0.86). Binomial with p = 0.75 has a long left tail like the payment rates (AUC alone 0.83). Raising the Poisson mean (λ 10 / 18) did not help (0.86 / 0.85).

## 3. Results per variable (main dataset)

| Variable | Observed Pearson r | Latent r | Spearman r | AUC alone | r with credit score | r with age |
|---|---|---|---|---|---|---|
| telecom_payment_rate | −0.323 | −0.471 | −0.273 | 0.806 | −0.287 | +0.079 |
| utility_payment_rate | −0.320 | −0.465 | −0.270 | 0.802 | −0.288 | +0.078 |
| spending_consistency | −0.318 | −0.525 | −0.277 | 0.821 | −0.202 | +0.056 |
| regular_payment_count | −0.316 | −0.555 | −0.287 | 0.828 | −0.202 | +0.057 |
| app_login_frequency | −0.323 | −0.559 | −0.293 | 0.835 | −0.199 | +0.056 |
| telecom_tenure_months | −0.176 | −0.350 | −0.209 | 0.742 | −0.176 | +0.065 |
| insurance_paid_months | −0.223 | −0.350 | −0.207 | 0.737 | −0.202 | +0.056 |
| autopay_ratio | −0.203 | −0.350 | −0.203 | 0.731 | −0.189 | +0.052 |

- **The mission rule holds:** observed |r| is 0.316 to 0.323 on the full data and 0.314 to 0.339 on each of the train / validation / test splits (tested).
- **Largest correlation between any two of the 8 variables:** 0.591 (cap 0.60).
- **Gender:** 50/50, correlation with default 0.0007.

**Strength warning:** each mission variable alone reaches AUC 0.80 to 0.84, stronger than any real GMSC feature. This comes from the observed-r rule with a 6.7% default rate, and it is the main reason the full model reaches test AUC 0.95.

## 4. `app_login_frequency` and age

The mission example links app logins to age. **This simulator does not**, because age is a protected attribute. Its credit part uses only past-due counts and utilization; a test confirms that shuffling `age` leaves the credit score unchanged. It still correlates +0.056 with age, indirectly: older borrowers have lower utilization and default less. Mean by mission age band: **22.3 (20 to 34), 22.4 (35 to 54), 22.6 (55+)** days.

## 5. Scenario parameters

| Parameter | Default | Effect |
|---|---|---|
| `thin_filer_ratio` | None (natural 7.71%) | `subsample` mode: reaches the ratio by subsampling the larger group without replacement (0.15 → 77,093 rows; 0.30 → 38,547 rows). `mask` mode (demonstration only): blanks the past-due columns so the mission rule flags the ratio (0.30 → exactly 30.0%). |
| `bias_ratio` | 0.0 | shifts the hidden scores of women and the 20 to 34 age band down by `bias_ratio` SD, in [0, 1] |

The mission example uses 0.3 / 0.1 as defaults. Ours are None / 0.0 because the final model must be trained on the natural composition and on unbiased data.

**Bias scenario (final setup retrained in 5-fold CV, mission age bands):**

| bias_ratio | Gender approval gap / DI / EO gap | Age-band approval gap / DI / EO gap |
|---|---|---|
| 0.0 | 0.001 / 0.999 / 0.011 | 0.152 / 0.836 / 0.127 |
| 0.2 | 0.033 / 0.963 / 0.062 | 0.157 / 0.831 / 0.132 |
| 0.5 | 0.074 / 0.918 / 0.132 | 0.156 / 0.832 / 0.151 |

- **Gender:** a clean measurement-bias effect. Women and men default at the same 6.7%, yet women's approval falls from 86.6% to 82.8%.
- **Age band (finding):** retraining mostly offsets the shift. Age is a model feature, so the model learns that young applicants' alternative data reads low; the approval gap and DI barely move. The EO gap still rises.
- The latent link `a` of the mission variables changes only at 0.5, because calibration stops once r is within ±0.005.

## 6. Credit link vs. AUC gain (Figure 08)

LightGBM default settings, 5-fold CV, mission variables always at observed r −0.32.

| Credit link b | Thin-filer gain | Overall gain |
|---|---|---|
| 0 (default only) | +0.101 ± 0.015 | +0.090 |
| 0.15 | +0.095 ± 0.014 | +0.084 |
| **0.30 (main)** | **+0.088 ± 0.014** | **+0.078** |
| 0.45 | +0.083 ± 0.014 | +0.073 |

Linking the variables to real credit behavior lowers the gain by about 13% at b = 0.30, because part of their information overlaps with the past-due counts and utilization. The gain stays far above +0.03 in every setting.

## 7. Limitations

- **Simulation bias:** the target is used for every row, including the test set, as in the mission example. Treat the gain as an upper bound; the trade-off pattern is the more reliable result.
- **Strength set by the rule:** observed r = −0.32 per variable, five such variables and a shared factor. The settings were chosen by the team, not estimated from real data.
- **The 0.60 cap rules out the top of the mission range:** at observed r = −0.50 even the lowest reachable pair correlation is 0.82 (see the audit report).
