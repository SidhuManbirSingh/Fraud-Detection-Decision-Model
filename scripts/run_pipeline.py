"""CLI entry point to run the hybrid DMN + gemini-decision-1 fraud detection pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fraud_decision_model.decision_client import GeminiDecisionClient
from fraud_decision_model.dmn_engine import DMNEngine, RoutingMode
from fraud_decision_model.evaluate import (
    evaluate_decisions,
    format_markdown_report,
    save_evaluation_outputs,
)
from fraud_decision_model.pipeline import FraudDecisionPipeline
from fraud_decision_model.preprocess import (
    engineer_features,
    load_and_quarantine,
    sample_user_grouped_pilot,
    to_enriched_records,
)
from fraud_decision_model.schemas import FeatureMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hybrid DMN + gemini-decision-1 Fraud Detection Pipeline."
    )
    parser.add_argument("--csv", type=Path, default=Path("transactions.csv"), help="Input CSV path.")
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "full"),
        default="pilot",
        help="Execution scope: 5-row smoke, stratified user-grouped pilot, or full dataset.",
    )
    parser.add_argument("--sample-size", type=int, default=500, help="Pilot sample size.")
    parser.add_argument("--target-fraud-count", type=int, default=100, help="Target frauds in pilot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pilot sampling.")
    parser.add_argument(
        "--feature-mode",
        choices=("production_strict", "dataset_benchmark"),
        default="production_strict",
        help="Feature causality mode.",
    )
    parser.add_argument(
        "--routing",
        choices=("hybrid", "model_only"),
        default="hybrid",
        help="Routing strategy: DMN + gemini-decision-1 vs pure model ablation.",
    )
    parser.add_argument("--fraud-threshold", type=float, default=0.50, help="Probability cutoff for fraud.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent API requests.")
    parser.add_argument("--rpm", type=int, default=300, help="Requests-per-minute rate limit.")
    parser.add_argument("--max-cost-usd", type=float, default=5.0, help="Hard USD budget ceiling.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate DMN routing & cost without API calls.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior checkpoints and start fresh.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for artifacts.")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional Markdown report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    feature_mode: FeatureMode = args.feature_mode
    routing_mode: RoutingMode = args.routing

    features, y_true = load_and_quarantine(args.csv)
    pop_prevalence = float(y_true.mean())
    enriched = engineer_features(features, feature_mode=feature_mode)

    if args.mode == "smoke":
        labeled = enriched.copy()
        labeled["_y"] = labeled["transaction_id"].map(y_true)
        frauds = labeled[labeled["_y"] == 1].iloc[[0, 50]]
        legits = labeled[labeled["_y"] == 0].iloc[[0, 4, 8]]
        eval_df = labeled.loc[list(frauds.index) + list(legits.index)].drop(columns=["_y"]).reset_index(drop=True)
    elif args.mode == "pilot":
        eval_df, _ = sample_user_grouped_pilot(
            enriched,
            y_true,
            n_samples=args.sample_size,
            target_fraud_count=args.target_fraud_count,
            seed=args.seed,
        )
    else:
        eval_df = enriched

    records = to_enriched_records(eval_df, feature_mode=feature_mode)
    dmn_engine = DMNEngine(routing_mode=routing_mode)
    client = (
        None
        if args.dry_run
        else GeminiDecisionClient(concurrency=args.concurrency, rpm=args.rpm)
    )

    pipeline = FraudDecisionPipeline(
        dmn_engine=dmn_engine,
        decision_client=client,
        fraud_threshold=args.fraud_threshold,
        max_cost_usd=args.max_cost_usd,
        concurrency=args.concurrency,
    )
    run_name = f"{args.mode}_{routing_mode}_{feature_mode}"
    try:
        result = pipeline.run(
            records=records,
            output_dir=args.output_dir,
            run_name=run_name,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        return 130

    if args.dry_run:
        print(f"=== DRY RUN ({run_name}) ===", flush=True)
        print(f"Total transactions: {len(records):,}", flush=True)
        print(f"DMN routing counts: {result.dmn_counts}", flush=True)
        print(f"Estimated API input tokens: {result.total_input_tokens:,} (0 output tokens)", flush=True)
        print(f"Estimated API cost (USD): ${result.estimated_cost_usd:.6f}", flush=True)
        return 0

    metrics = evaluate_decisions(
        result.decisions,
        eval_df,
        y_true,
        population_prevalence=pop_prevalence,
        failed_count=result.failed_count,
    )
    title = f"Fraud Detection Evaluation ({args.mode} | {routing_mode} | {feature_mode})"
    report_md = format_markdown_report(metrics, title=title)

    ov = metrics["overall"]
    tel = metrics["telemetry"]
    print(f"=== RUN COMPLETE ({run_name}) ===", flush=True)
    print(f"Predictions CSV: {result.predictions_csv_path}", flush=True)
    print(f"Checkpoint JSONL: {result.checkpoint_path}", flush=True)
    print(
        f"Metrics -> Precision={ov['precision']:.4f} Recall={ov['recall']:.4f} "
        f"F1={ov['f1']:.4f} PR-AUC={ov['pr_auc']} ROC-AUC={ov['roc_auc']} Brier={ov['brier_score']:.4f}",
        flush=True,
    )
    print(
        f"Telemetry -> DMN savings={tel['dmn_savings_pct']:.1f}% | API calls={tel['api_calls']} | "
        f"Tokens={tel['total_input_tokens']:,} | Cost=${tel['estimated_cost_usd']:.6f} | DLQ={result.failed_count}",
        flush=True,
    )

    if args.report_path is not None:
        md_path, json_path = save_evaluation_outputs(metrics, report_md, args.report_path)
        print(f"Saved evaluation report: {md_path} (+ {json_path})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
