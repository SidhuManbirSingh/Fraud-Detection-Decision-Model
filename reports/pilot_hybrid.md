# Fraud Detection Evaluation (pilot | hybrid | production_strict)

## Overall Performance

| Metric | Value |
| :--- | ---: |
| Transactions Scored | 500 (100 frauds) |
| Confusion Matrix (TP / FP / TN / FN) | 84 / 50 / 350 / 16 |
| Precision | 0.6269 |
| Recall | 0.8400 |
| F1 Score | 0.7179 |
| False Positive Rate (FPR) | 0.1250 |
| PR-AUC | 0.8509 |
| ROC-AUC | 0.9498 |
| Brier Score (Calibration) | 0.0967 |
| Best-F1 Operating Threshold | 0.80 (F1=0.7707, P=0.7524, R=0.7900) |

## Prevalence-Adjusted Production Projection

- **True Population Prevalence:** 2.21%
- **Projected Production Precision:** 0.1316
- **Projected Production Recall:** 0.8400
- **Projected Alert Rate:** 14.08%

## Decision Tier Breakdown

| Tier | Count | Share | Frauds | Predicted Frauds | Precision | Recall | FPR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dmn_auto_approve` | 253 | 50.6% | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `dmn_auto_decline` | 42 | 8.4% | 42 | 42 | 1.0000 | 1.0000 | 0.0000 |
| `gemini_decision_1` | 205 | 41.0% | 58 | 92 | 0.4565 | 0.7241 | 0.3401 |

## Slice Diagnostics

| Slice | Support | Frauds | Precision | Recall | F1 | FPR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `account_age_le_90d` | 16 | 9 | 0.8000 | 0.8889 | 0.8421 | 0.2857 |
| `account_age_gt_90d` | 484 | 91 | 0.6129 | 0.8352 | 0.7070 | 0.1221 |
| `channel_web` | 282 | 83 | 0.6952 | 0.8795 | 0.7766 | 0.1608 |
| `channel_app` | 218 | 17 | 0.3793 | 0.6471 | 0.4783 | 0.0896 |
| `cross_border_true` | 70 | 40 | 0.8611 | 0.7750 | 0.8158 | 0.1667 |
| `cross_border_false` | 430 | 60 | 0.5408 | 0.8833 | 0.6709 | 0.1216 |
| `auth_failures_0` | 302 | 13 | 1.0000 | 0.6154 | 0.7619 | 0.0000 |
| `auth_failures_1` | 83 | 21 | 0.7857 | 0.5238 | 0.6286 | 0.0484 |
| `auth_failures_2` | 44 | 18 | 0.4146 | 0.9444 | 0.5763 | 0.9231 |
| `auth_failures_3` | 71 | 48 | 0.6761 | 1.0000 | 0.8067 | 1.0000 |

## Operational Telemetry

- **DMN Fast-Path Resolution:** 295/500 (59.0% API calls saved)
- **Model API Calls (`gemini-decision-1`):** 205 (DLQ failures: 0)
- **Total Input Tokens (0 output tokens):** 228,713
- **Estimated Cost (USD):** $0.022871
- **Latency (p50 / p95):** 337.0 ms / 51399.6 ms
