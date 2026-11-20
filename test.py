"""End-to-end verification runner: unit tests, DMN funnel check, and 5-row live smoke test."""

from __future__ import annotations

import argparse
import sys
import unittest
from collections import Counter
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fraud_decision_model.decision_client import GeminiDecisionClient
from fraud_decision_model.dmn_engine import DMNEngine
from fraud_decision_model.preprocess import (
    engineer_features,
    load_and_quarantine,
    sample_user_grouped_pilot,
    to_enriched_records,
)


def run_unit_tests() -> bool:
    """Discover and execute all unit tests under tests/."""
    print("=== PART 1: UNIT TEST SUITE ===", flush=True)
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def verify_pipeline_funnel(csv_path: Path) -> tuple:
    """Verify leakage quarantine, causal preprocessing, and DMN gating on a 500-row pilot."""
    print("\n=== PART 2: PREPROCESSING & DMN FUNNEL CHECK ===", flush=True)
    features, y_true = load_and_quarantine(csv_path)
    assert "is_fraud" not in features.columns
    assert "total_transactions_user" not in features.columns

    enriched = engineer_features(features, feature_mode="production_strict")
    pilot_df, pilot_y = sample_user_grouped_pilot(
        enriched, y_true, n_samples=500, target_fraud_count=100, seed=42
    )
    records = to_enriched_records(pilot_df)

    engine = DMNEngine(routing_mode="hybrid")
    action_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    escalated_records = []

    for rec in records:
        dec = engine.evaluate(rec)
        action_counts[dec.action] += 1
        rule_counts[dec.rule_id] += 1
        if dec.action == "ESCALATE_TO_MODEL":
            escalated_records.append(rec)

    cost_est = GeminiDecisionClient.estimate_dry_run_cost(escalated_records)
    print(f"Dataset loaded: {len(features):,} rows | Pilot: {len(records)} rows ({int(pilot_y.sum())} frauds)")
    print(f"DMN action counts: {dict(action_counts)}")
    print(f"DMN rule breakdown: {dict(rule_counts)}")
    print(f"Escalated dry-run cost estimate: {cost_est}", flush=True)
    return enriched, y_true


def run_live_smoke_test(enriched, y_true) -> int:
    """Run a 5-row live verification against gemini-decision-1 (2 frauds + 3 legits)."""
    print("\n=== PART 3: LIVE 5-ROW GEMINI-DECISION-1 SMOKE TEST ===", flush=True)
    labeled = enriched.copy()
    labeled["_y"] = labeled["transaction_id"].map(y_true)

    frauds = labeled[labeled["_y"] == 1].iloc[[0, 50]]
    legits = labeled[labeled["_y"] == 0].iloc[[0, 4, 8]]
    sample_df = labeled.loc[list(frauds.index) + list(legits.index)].drop(columns=["_y"])
    records = to_enriched_records(sample_df)

    client = GeminiDecisionClient(use_vertexai=None, concurrency=2, rpm=60, max_attempts=1)
    try:
        results = [client.evaluate_transaction(records[0])]
        mode_note = f"env GEMINI_USE_VERTEXAI={client.use_vertexai}"
    except Exception as exc:
        print(
            f"Initial client failed ({type(exc).__name__}: {exc}). "
            f"Retrying with vertexai={not client.use_vertexai}...",
            flush=True,
        )
        use_vertex = not client.use_vertexai
        client = GeminiDecisionClient(use_vertexai=use_vertex, concurrency=2, rpm=60, max_attempts=1)
        results = [client.evaluate_transaction(records[0])]
        mode_note = f"fallback vertexai={use_vertex}"

    print(f"Authenticated successfully ({mode_note}). Scoring remaining 4 rows...", flush=True)
    for rec in records[1:]:
        results.append(client.evaluate_transaction(rec))

    for rec, res in zip(records, results):
        final = client.build_final_decision(rec, res, threshold=0.50)
        true_label = int(y_true.loc[rec.transaction_id])
        print(
            f"tx={rec.transaction_id:6d} true={true_label} pred={int(final.is_fraud)} "
            f"P(fraud)={final.fraud_probability:.4f} conf={final.confidence_score:.3f} "
            f"sev={final.risk_severity_score:.2f} driver={final.primary_risk_driver:25s} "
            f"in_tok={final.input_tokens} ms={final.latency_ms:.1f}",
            flush=True,
        )
        print(f"  reasoning: {final.reasoning}", flush=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unit tests, DMN check, and gemini-decision-1 smoke test.")
    parser.add_argument("--csv", type=Path, default=Path("transactions.csv"), help="Path to transactions CSV.")
    parser.add_argument("--offline", action="store_true", help="Skip live API calls in Part 3.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not run_unit_tests():
        return 1

    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    enriched, y_true = verify_pipeline_funnel(args.csv)
    if args.offline:
        print("\nSkipped live API smoke test (--offline).", flush=True)
        return 0

    return run_live_smoke_test(enriched, y_true)


if __name__ == "__main__":
    raise SystemExit(main())
