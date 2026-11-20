# Fraud Detection Evaluation (pilot | model_only | production_strict)

## Overall Performance

| Metric | Value |
| :--- | ---: |
| Transactions Scored | 500 (100 frauds) |
| Confusion Matrix (TP / FP / TN / FN) | 82 / 50 / 350 / 18 |
| Precision | 0.6212 |
| Recall | 0.8200 |
| F1 Score | 0.7069 |
| False Positive Rate (FPR) | 0.1250 |
| PR-AUC | 0.7441 |
| ROC-AUC | 0.9323 |
| Brier Score (Calibration) | 0.0993 |
| Best-F1 Operating Threshold | 0.25 (F1=0.7407, P=0.6294, R=0.9000) |

## Prevalence-Adjusted Production Projection

- **True Population Prevalence:** 2.21%
- **Projected Production Precision:** 0.1289
- **Projected Production Recall:** 0.8200
- **Projected Alert Rate:** 14.03%

## Decision Tier Breakdown

| Tier | Count | Share | Frauds | Predicted Frauds | Precision | Recall | FPR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dmn_auto_approve` | 0 | 0.0% | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `dmn_auto_decline` | 0 | 0.0% | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| `gemini_decision_1` | 500 | 100.0% | 100 | 132 | 0.6212 | 0.8200 | 0.1250 |

## Slice Diagnostics

| Slice | Support | Frauds | Precision | Recall | F1 | FPR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `account_age_le_90d` | 16 | 9 | 0.8000 | 0.8889 | 0.8421 | 0.2857 |
| `account_age_gt_90d` | 484 | 91 | 0.6066 | 0.8132 | 0.6948 | 0.1221 |
| `channel_web` | 282 | 83 | 0.6893 | 0.8554 | 0.7634 | 0.1608 |
| `channel_app` | 218 | 17 | 0.3793 | 0.6471 | 0.4783 | 0.0896 |
| `cross_border_true` | 70 | 40 | 0.8571 | 0.7500 | 0.8000 | 0.1667 |
| `cross_border_false` | 430 | 60 | 0.5361 | 0.8667 | 0.6624 | 0.1216 |
| `auth_failures_0` | 302 | 13 | 1.0000 | 0.4615 | 0.6316 | 0.0000 |
| `auth_failures_1` | 83 | 21 | 0.7857 | 0.5238 | 0.6286 | 0.0484 |
| `auth_failures_2` | 44 | 18 | 0.4146 | 0.9444 | 0.5763 | 0.9231 |
| `auth_failures_3` | 71 | 48 | 0.6761 | 1.0000 | 0.8067 | 1.0000 |

## Operational Telemetry

- **DMN Fast-Path Resolution:** 0/500 (0.0% API calls saved)
- **Model API Calls (`gemini-decision-1`):** 500 (DLQ failures: 0)
- **Total Input Tokens (0 output tokens):** 555,172
- **Estimated Cost (USD):** $0.055517
- **Latency (p50 / p95):** 340.1 ms / 41595.5 ms
