# GMSC EDA Report (Step 2)

> Educational use only. Not for real financial decisions.

- Input: `data/processed/gmsc_clean.csv` (149,999 rows, output of Step 1)
- Code: `notebooks/eda.py` (run `python notebooks/eda.py` from the project root)
- Figures: `reports/figures/*_en.png` and `*_ko.png`. Numbers: `reports/eda_summary.json`

## 1. Key findings

1. **Strong class imbalance.** 6.68% of borrowers default (10,026 of 149,999). This is why Step 5 compares class weights and SMOTE. (Figure 01)
2. **Late-payment history is the strongest signal.** One or more 60 to 89 day or 90+ day late payments pushes the default rate from about 5% to over 30%. Three or more pushes it to about 60%. (Figure 02)
3. **Utilization is the strongest actionable feature.** The default rate rises from about 1.4% at low utilization to 23.4% in the top decile (above 0.98). (Figure 02)
4. **Thin-filers are about twice as risky.** The proxy rule flags 7.71% of rows (11,564). Their default rate is 12.93%, compared with 6.16% for everyone else. (Figure 07)
5. **The 96/98 code group is very risky.** It defaults at 54.6%, about the same as borrowers with 2 or more serious late payments. (Figure 05)

## 2. Direction check against `actionability.yaml`

For every feature, the default rate was computed across ranges (deciles, or counts for count features). "Reversals" counts steps between neighboring ranges that go against the overall trend.

| Feature | YAML `monotone` | Observed trend | Reversals | Matches YAML |
|---|---|---|---|---|
| Revolving utilization | +1 | up | 1 (lowest decile only) | yes |
| 30-59 / 60-89 / 90+ days late | +1 | up | 0 | yes |
| Debt ratio | +1 | up, but U-shaped | 3 | yes, weak |
| Monthly income | -1 | down | 1 (lowest decile only) | yes |
| Age | 0 (protected) | down | 0 | not constrained |
| Open loans and credit lines | 0 | U-shaped | 4 | not constrained |
| Real estate loans | 0 | U-shaped | 3 | not constrained |
| Dependents | 0 | up | 0 | not constrained |

All constrained features move in the direction the YAML says. **Debt ratio is the weak case:** its default rate dips in the middle ranges and rises clearly only above about 0.43, and its Pearson r with default is about 0. Forcing it to be monotonic in Step 5 is the most likely source of AUC loss. If the loss goes over 0.01, it is the first constraint to drop.

## 3. Correlation with default (Figure 06)

| Feature | Pearson r | Spearman |
|---|---|---|
| 90+ days late | +0.315 | +0.335 |
| Revolving utilization | +0.282 | +0.241 |
| 30-59 days late | +0.275 | +0.251 |
| 60-89 days late | +0.268 | +0.268 |
| Age | -0.115 | -0.117 |
| All others | below 0.05 | below 0.07 |

**Important for Step 3:** only one real feature reaches an observed |r| of 0.3. Measured the same way, a 0.3 to 0.5 rule would make each simulated variable as strong as the best real credit feature. Resolved in Step 3: the rule is applied to the copula's latent correlation, which gives an observed Pearson r of about 0.18 to 0.27 (see `simulator_report_en.md`).

Pearson r is distorted by extreme values. Debt ratio, for example, has Pearson -0.002 but Spearman +0.058. Step 3 therefore reports latent, Pearson and Spearman correlations for every simulated variable.

## 4. Relationships between features (Figure 04)

- Pairs with rank correlation of at least 0.4: debt ratio and real estate loans (0.61), open lines and real estate loans (0.47), debt ratio and open lines (0.41).
- The three late-payment columns are only moderately related to each other (0.24 to 0.30), so each adds its own information.
- Age is negatively related to utilization (-0.28) and dependents (-0.23). Age is protected, so this matters for the fairness work: other features can partly stand in for age.

## 5. Distributions (Figure 03)

- Utilization has two peaks: near 0 (mostly non-defaulters) and near 1.0 (where defaulters concentrate).
- Debt ratio has a spike at 0 and a long right tail. Defaulters sit slightly higher.
- Defaulters' income is shifted somewhat lower, but the two groups overlap heavily.

## 6. What this changes for later steps

- **Step 3:** correlation definition decided: latent correlation (see section 3).
- **Step 5:** watch the AUC cost of the debt ratio constraint first. Logistic Regression needs a log transform or cap for debt ratio, income and utilization because of the skew.
- **Fairness (채민규):** age correlates with utilization and dependents, so removing age alone will not remove age effects.
