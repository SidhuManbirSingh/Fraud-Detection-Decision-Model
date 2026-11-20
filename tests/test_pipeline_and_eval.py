"""Unit tests for checkpointed pipeline execution, DLQ handling, and offline evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fraud_decision_model.dmn_engine import DMNEngine
from fraud_decision_model.evaluate import (
    evaluate_decisions,
    format_markdown_report,
    save_evaluation_outputs,
)
from fraud_decision_model.pipeline import FraudDecisionPipeline
from fraud_decision_model.preprocess import engineer_features, to_enriched_records
from fraud_decision_model.schemas import (
    EnrichedTransaction,
    FinalDecision,
    ModelDecisionResult,
)


class MockDecisionClient:
    """Deterministic mock checking payload hygiene and simulating scoring/failures."""

    def __init__(self, fail_tx_id: int | None = None) -> None:
        self.calls: list[int] = []
        self.fail_tx_id = fail_tx_id

    def evaluate_transaction(self, tx: EnrichedTransaction) -> ModelDecisionResult:
        payload = tx.to_model_payload()
        for forbidden in ("is_fraud", "transaction_id", "user_id", "total_transactions_user"):
            if forbidden in payload:
                raise AssertionError(f"Forbidden leakage key '{forbidden}' found in model payload!")

        self.calls.append(tx.transaction_id)
        if self.fail_tx_id is not None and tx.transaction_id == self.fail_tx_id:
            raise RuntimeError("Simulated transient failure")

        is_high_risk = tx.auth_failures_count >= 2 or tx.is_cross_border or tx.account_age_days <= 90
        prob = 0.88 if is_high_risk else 0.08
        driver = "auth_failure_takeover" if tx.auth_failures_count >= 2 else "legitimate_routine"
        return ModelDecisionResult(
            transaction_id=tx.transaction_id,
            fraud_probability=prob,
            fraud_confidence=0.92,
            risk_severity_score=3.4 if is_high_risk else 0.5,
            risk_severity_confidence=0.90,
            risk_severity_probs={"0": 0.1, "3": 0.9} if is_high_risk else {"0": 0.9, "3": 0.1},
            primary_risk_driver=driver,
            driver_confidence=0.91,
            driver_probs={driver: 0.91},
            input_tokens=950,
            latency_ms=120.0,
        )

    @classmethod
    def build_final_decision(
        cls,
        tx: EnrichedTransaction,
        result: ModelDecisionResult,
        threshold: float = 0.50,
        rule_id: str = "DMN_GREY_ZONE_ESCALATE",
    ) -> FinalDecision:
        return FinalDecision(
            transaction_id=tx.transaction_id,
            decision_source="gemini_decision_1",
            rule_id=rule_id,
            is_fraud=result.fraud_probability >= threshold,
            fraud_probability=result.fraud_probability,
            confidence_score=result.fraud_confidence,
            risk_severity_score=result.risk_severity_score,
            primary_risk_driver=result.primary_risk_driver,
            reasoning=f"Mock reasoning driver={result.primary_risk_driver}",
            input_tokens=result.input_tokens,
            latency_ms=result.latency_ms,
            feature_mode=tx.feature_mode,
        )


def _sample_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    rows = [
        # Routine mature domestic -> DMN AUTO_APPROVE (legit)
        (101, 1, 400, 100.0, 95.0, "FR", "FR", "app", "grocery", 0, 1, 1, 1, "2024-01-01T10:00:00Z", 50.0, 0),
        # Card testing micro-amount -> DMN AUTO_DECLINE (fraud)
        (102, 2, 400, 100.0, 2.50, "US", "US", "web", "gaming", 0, 1, 1, 1, "2024-01-02T11:00:00Z", 20.0, 1),
        # Grey-zone cross-border + 2 auth fails -> ESCALATE (fraud)
        (103, 3, 200, 100.0, 180.0, "DE", "GB", "web", "electronics", 1, 0, 0, 1, "2024-01-03T12:00:00Z", 1200.0, 1),
        # Grey-zone single auth fail domestic -> ESCALATE (legit)
        (104, 4, 500, 100.0, 110.0, "FR", "FR", "app", "fashion", 0, 0, 1, 1, "2024-01-04T13:00:00Z", 120.0, 0),
    ]
    cols = [
        "transaction_id",
        "user_id",
        "account_age_days",
        "avg_amount_user",
        "amount",
        "country",
        "bin_country",
        "channel",
        "merchant_category",
        "promo_used",
        "avs_match",
        "cvv_result",
        "three_ds_flag",
        "transaction_time",
        "shipping_distance_km",
        "is_fraud",
    ]
    raw = pd.DataFrame(rows, columns=cols)
    y_true = raw.set_index("transaction_id")["is_fraud"].astype(int)
    feats = raw.drop(columns=["is_fraud"])
    enriched = engineer_features(feats, feature_mode="production_strict")
    return enriched, y_true


class PipelineAndEvalTests(unittest.TestCase):
    def test_hybrid_pipeline_checkpoint_resume_and_eval(self) -> None:
        enriched, y_true = _sample_features_and_labels()
        records = to_enriched_records(enriched)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            mock_client = MockDecisionClient()
            pipeline = FraudDecisionPipeline(
                dmn_engine=DMNEngine(routing_mode="hybrid"),
                decision_client=mock_client,
                fraud_threshold=0.50,
                max_cost_usd=1.0,
            )

            res1 = pipeline.run(records, output_dir=out_dir, run_name="test_run", resume=True)
            self.assertEqual(len(res1.decisions), 4)
            self.assertEqual(res1.failed_count, 0)
            # Only tx 103 & 104 should escalate to the model; tx 101 & 102 handled by DMN
            self.assertEqual(sorted(mock_client.calls), [103, 104])
            self.assertTrue(res1.checkpoint_path.exists())
            self.assertTrue(res1.predictions_csv_path.exists())

            # Second run with resume=True must skip all 4 rows and make 0 new API calls
            mock_client.calls.clear()
            res2 = pipeline.run(records, output_dir=out_dir, run_name="test_run", resume=True)
            self.assertEqual(mock_client.calls, [])
            self.assertEqual(res2.resumed_count, 4)

            metrics = evaluate_decisions(res1.decisions, enriched, y_true)
            self.assertEqual(metrics["overall"]["tp"], 2)
            self.assertEqual(metrics["overall"]["tn"], 2)
            self.assertEqual(metrics["overall"]["f1"], 1.0)
            self.assertEqual(metrics["telemetry"]["dmn_savings_pct"], 50.0)

            report_md = format_markdown_report(metrics, title="Unit Test Report")
            self.assertIn("Overall Performance", report_md)
            md_path, json_path = save_evaluation_outputs(
                metrics, report_md, out_dir / "report.md"
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

    def test_dlq_capture_and_budget_guard(self) -> None:
        enriched, _ = _sample_features_and_labels()
        records = to_enriched_records(enriched)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            failing_client = MockDecisionClient(fail_tx_id=103)
            pipeline = FraudDecisionPipeline(
                dmn_engine=DMNEngine(routing_mode="hybrid"),
                decision_client=failing_client,
                max_cost_usd=1.0,
            )
            res = pipeline.run(records, output_dir=out_dir, run_name="dlq_run", resume=False)
            self.assertEqual(res.failed_count, 1)
            self.assertEqual(len(res.decisions), 3)
            self.assertTrue(res.dlq_path.exists())
            dlq_entry = json.loads(res.dlq_path.read_text(encoding="utf-8").strip())
            self.assertEqual(dlq_entry["transaction_id"], 103)

            # Budget guard must halt before API calls
            tight_pipeline = FraudDecisionPipeline(
                dmn_engine=DMNEngine(routing_mode="hybrid"),
                decision_client=failing_client,
                max_cost_usd=0.0,
            )
            with self.assertRaises(RuntimeError):
                tight_pipeline.run(records, output_dir=out_dir, run_name="over_budget", resume=False)


if __name__ == "__main__":
    unittest.main()
