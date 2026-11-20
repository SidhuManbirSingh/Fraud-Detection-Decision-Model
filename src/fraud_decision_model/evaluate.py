"""Offline evaluation metrics, slice diagnostics, threshold sweeps, and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraud_decision_model.decision_client import DEFAULT_PRICE_PER_1M_INPUT_TOKENS
from fraud_decision_model.schemas import FinalDecision


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "support": int(len(y_true)),
        "actual_frauds": int(np.sum(y_true == 1)),
        "predicted_frauds": int(np.sum(y_pred == 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "fpr": round(fpr, 4),
    }


def _slice_summary(df: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    sub = df[mask]
    if sub.empty:
        return {"support": 0, "actual_frauds": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0}
    m = _binary_metrics(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int))
    return m


def evaluate_decisions(
    decisions: Sequence[FinalDecision],
    df_enriched: pd.DataFrame,
    y_true: pd.Series,
    population_prevalence: float = 0.022062,
    failed_count: int = 0,
    price_per_1m_input_tokens: float = DEFAULT_PRICE_PER_1M_INPUT_TOKENS,
) -> dict[str, Any]:
    """Compute comprehensive offline classification, calibration, tier, slice, and cost metrics."""
    if not decisions:
        raise ValueError("No decisions provided for evaluation.")

    dec_df = pd.DataFrame([d.model_dump() for d in decisions])
    meta_cols = ["transaction_id", "account_age_days", "channel", "is_cross_border", "auth_failures_count"]
    meta = df_enriched[meta_cols].drop_duplicates(subset=["transaction_id"])

    merged = dec_df.merge(meta, on="transaction_id", how="inner")
    merged["y_true"] = merged["transaction_id"].map(y_true).astype(int)
    merged["y_pred"] = merged["is_fraud"].astype(int)

    y_arr = merged["y_true"].to_numpy(dtype=int)
    p_arr = merged["fraud_probability"].to_numpy(dtype=float)
    pred_arr = merged["y_pred"].to_numpy(dtype=int)

    overall = _binary_metrics(y_arr, pred_arr)
    has_both_classes = len(np.unique(y_arr)) == 2
    pr_auc = float(average_precision_score(y_arr, p_arr)) if has_both_classes else None
    roc_auc = float(roc_auc_score(y_arr, p_arr)) if has_both_classes else None
    brier = float(brier_score_loss(y_arr, p_arr))

    # Per-tier breakdown
    tier_breakdown: dict[str, Any] = {}
    tiers = {
        "dmn_auto_approve": (merged["decision_source"] == "dmn_rule") & (merged["y_pred"] == 0),
        "dmn_auto_decline": (merged["decision_source"] == "dmn_rule") & (merged["y_pred"] == 1),
        "gemini_decision_1": merged["decision_source"] == "gemini_decision_1",
    }
    for name, mask in tiers.items():
        sub = merged[mask]
        tier_breakdown[name] = {
            "count": int(len(sub)),
            "share_pct": round(100.0 * len(sub) / len(merged), 2),
            **_binary_metrics(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int)),
        }

    # Threshold sweep over effective probabilities (DMN auto rules keep 0.01/0.98-0.99, model varies)
    sweep: list[dict[str, Any]] = []
    for i in range(17):
        thr = round(0.10 + 0.05 * i, 2)
        thr_pred = (p_arr >= thr).astype(int)
        m = _binary_metrics(y_arr, thr_pred)
        sweep.append(
            {
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
            }
        )

    best_f1_row = max(sweep, key=lambda r: (r["f1"], r["recall"]))
    high_rec_candidates = [r for r in sweep if r["recall"] >= 0.85]
    high_rec_row = max(high_rec_candidates, key=lambda r: (r["precision"], r["f1"])) if high_rec_candidates else None

    # Slice diagnostics
    slices = {
        "account_age_le_90d": _slice_summary(merged, merged["account_age_days"] <= 90),
        "account_age_gt_90d": _slice_summary(merged, merged["account_age_days"] > 90),
        "channel_web": _slice_summary(merged, merged["channel"] == "web"),
        "channel_app": _slice_summary(merged, merged["channel"] == "app"),
        "cross_border_true": _slice_summary(merged, merged["is_cross_border"] == True),  # noqa: E712
        "cross_border_false": _slice_summary(merged, merged["is_cross_border"] == False),  # noqa: E712
        "auth_failures_0": _slice_summary(merged, merged["auth_failures_count"] == 0),
        "auth_failures_1": _slice_summary(merged, merged["auth_failures_count"] == 1),
        "auth_failures_2": _slice_summary(merged, merged["auth_failures_count"] == 2),
        "auth_failures_3": _slice_summary(merged, merged["auth_failures_count"] == 3),
    }

    # Prevalence-adjusted production projection
    tpr = overall["recall"]
    fpr = overall["fpr"]
    p = population_prevalence
    alert_rate = tpr * p + fpr * (1.0 - p)
    proj_precision = (tpr * p / alert_rate) if alert_rate > 0 else 0.0

    # Operational telemetry
    api_rows = merged[merged["decision_source"] == "gemini_decision_1"]
    dmn_rows = merged[merged["decision_source"] == "dmn_rule"]
    total_tokens = int(merged["input_tokens"].sum())
    est_cost = round((total_tokens / 1_000_000.0) * price_per_1m_input_tokens, 6)
    latencies = api_rows.loc[api_rows["latency_ms"] > 0, "latency_ms"].to_numpy(dtype=float)

    telemetry = {
        "total_scored": int(len(merged)),
        "dmn_resolved": int(len(dmn_rows)),
        "dmn_savings_pct": round(100.0 * len(dmn_rows) / len(merged), 2),
        "api_calls": int(len(api_rows)),
        "failed_dlq_count": int(failed_count),
        "total_input_tokens": total_tokens,
        "estimated_cost_usd": est_cost,
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2) if len(latencies) else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if len(latencies) else 0.0,
    }

    return {
        "overall": {
            **overall,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "brier_score": round(brier, 4),
        },
        "prevalence_adjusted_projection": {
            "population_prevalence": population_prevalence,
            "projected_precision": round(proj_precision, 4),
            "projected_recall": round(tpr, 4),
            "projected_alert_rate": round(alert_rate, 4),
        },
        "tier_breakdown": tier_breakdown,
        "threshold_sweep": sweep,
        "best_f1_operating_point": best_f1_row,
        "high_recall_85_operating_point": high_rec_row,
        "slices": slices,
        "telemetry": telemetry,
    }


def format_markdown_report(metrics: dict[str, Any], title: str = "Fraud Decision Evaluation Report") -> str:
    """Render evaluation metrics into a structured GitHub-flavored Markdown report."""
    ov = metrics["overall"]
    tel = metrics["telemetry"]
    proj = metrics["prevalence_adjusted_projection"]
    best_f1 = metrics["best_f1_operating_point"]

    lines = [
        f"# {title}",
        "",
        "## Overall Performance",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Transactions Scored | {ov['support']} ({ov['actual_frauds']} frauds) |",
        f"| Confusion Matrix (TP / FP / TN / FN) | {ov['tp']} / {ov['fp']} / {ov['tn']} / {ov['fn']} |",
        f"| Precision | {ov['precision']:.4f} |",
        f"| Recall | {ov['recall']:.4f} |",
        f"| F1 Score | {ov['f1']:.4f} |",
        f"| False Positive Rate (FPR) | {ov['fpr']:.4f} |",
        f"| PR-AUC | {ov['pr_auc'] if ov['pr_auc'] is not None else 'N/A'} |",
        f"| ROC-AUC | {ov['roc_auc'] if ov['roc_auc'] is not None else 'N/A'} |",
        f"| Brier Score (Calibration) | {ov['brier_score']:.4f} |",
        f"| Best-F1 Operating Threshold | {best_f1['threshold']:.2f} (F1={best_f1['f1']:.4f}, P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}) |",
        "",
        "## Prevalence-Adjusted Production Projection",
        "",
        f"- **True Population Prevalence:** {proj['population_prevalence']:.2%}",
        f"- **Projected Production Precision:** {proj['projected_precision']:.4f}",
        f"- **Projected Production Recall:** {proj['projected_recall']:.4f}",
        f"- **Projected Alert Rate:** {proj['projected_alert_rate']:.2%}",
        "",
        "## Decision Tier Breakdown",
        "",
        "| Tier | Count | Share | Frauds | Predicted Frauds | Precision | Recall | FPR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier_name, t in metrics["tier_breakdown"].items():
        lines.append(
            f"| `{tier_name}` | {t['count']} | {t['share_pct']:.1f}% | {t['actual_frauds']} | "
            f"{t['predicted_frauds']} | {t['precision']:.4f} | {t['recall']:.4f} | {t['fpr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Slice Diagnostics",
            "",
            "| Slice | Support | Frauds | Precision | Recall | F1 | FPR |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s_name, s in metrics["slices"].items():
        lines.append(
            f"| `{s_name}` | {s['support']} | {s['actual_frauds']} | {s['precision']:.4f} | "
            f"{s['recall']:.4f} | {s['f1']:.4f} | {s['fpr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Operational Telemetry",
            "",
            f"- **DMN Fast-Path Resolution:** {tel['dmn_resolved']}/{tel['total_scored']} ({tel['dmn_savings_pct']:.1f}% API calls saved)",
            f"- **Model API Calls (`gemini-decision-1`):** {tel['api_calls']} (DLQ failures: {tel['failed_dlq_count']})",
            f"- **Total Input Tokens (0 output tokens):** {tel['total_input_tokens']:,}",
            f"- **Estimated Cost (USD):** ${tel['estimated_cost_usd']:.6f}",
            f"- **Latency (p50 / p95):** {tel['p50_latency_ms']:.1f} ms / {tel['p95_latency_ms']:.1f} ms",
            "",
        ]
    )
    return "\n".join(lines)


def save_evaluation_outputs(
    metrics: dict[str, Any],
    report_md: str,
    report_path: Path,
) -> tuple[Path, Path]:
    """Write Markdown report and companion JSON metrics artifact."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return report_path, json_path
