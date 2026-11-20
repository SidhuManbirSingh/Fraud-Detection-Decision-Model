"""Checkpointed execution pipeline combining DMN gating and gemini-decision-1."""

from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
from pydantic import BaseModel

from fraud_decision_model.decision_client import (
    DEFAULT_PRICE_PER_1M_INPUT_TOKENS,
    GeminiDecisionClient,
)
from fraud_decision_model.dmn_engine import DMNEngine
from fraud_decision_model.schemas import (
    EnrichedTransaction,
    FinalDecision,
    ModelDecisionResult,
)


class DecisionClientProtocol(Protocol):
    """Minimal interface required from a decision model client."""

    def evaluate_transaction(self, tx: EnrichedTransaction) -> ModelDecisionResult: ...

    @classmethod
    def build_final_decision(
        cls,
        tx: EnrichedTransaction,
        result: ModelDecisionResult,
        threshold: float = 0.50,
        rule_id: str = "DMN_GREY_ZONE_ESCALATE",
    ) -> FinalDecision: ...


class PipelineResult(BaseModel):
    """Summary and artifacts returned by a FraudDecisionPipeline run."""

    decisions: list[FinalDecision]
    failed_count: int = 0
    resumed_count: int = 0
    dmn_counts: dict[str, int]
    total_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    checkpoint_path: Path
    dlq_path: Path
    predictions_csv_path: Path
    dry_run: bool = False


def load_checkpoint(checkpoint_path: Path) -> dict[int, FinalDecision]:
    """Load previously persisted FinalDecision rows keyed by transaction_id."""
    completed: dict[int, FinalDecision] = {}
    if not checkpoint_path.exists():
        return completed
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            decision = FinalDecision.model_validate_json(line)
            completed[decision.transaction_id] = decision
    return completed


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


class FraudDecisionPipeline:
    """Orchestrates DMN gating, threaded model scoring, JSONL checkpointing, and DLQ capture."""

    def __init__(
        self,
        dmn_engine: DMNEngine,
        decision_client: DecisionClientProtocol | None = None,
        fraud_threshold: float = 0.50,
        max_cost_usd: float = 5.0,
        concurrency: int = 8,
        price_per_1m_input_tokens: float = DEFAULT_PRICE_PER_1M_INPUT_TOKENS,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive.")
        self.dmn_engine = dmn_engine
        self.decision_client = decision_client
        self.fraud_threshold = fraud_threshold
        self.max_cost_usd = max_cost_usd
        self.concurrency = concurrency
        self.price_per_1m_input_tokens = price_per_1m_input_tokens

    def run(
        self,
        records: Sequence[EnrichedTransaction],
        output_dir: Path | str = Path("outputs"),
        run_name: str = "pilot",
        resume: bool = True,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute hybrid or model-only scoring across `records` with resume & budget safety."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"decisions_{run_name}.jsonl"
        dlq_path = out_dir / f"dlq_{run_name}.jsonl"
        predictions_csv_path = out_dir / f"predictions_{run_name}.csv"

        if not resume and not dry_run:
            checkpoint_path.unlink(missing_ok=True)
            dlq_path.unlink(missing_ok=True)

        completed: dict[int, FinalDecision] = load_checkpoint(checkpoint_path) if resume else {}
        resumed_count = len(completed)

        dmn_counts: Counter[str] = Counter()
        dmn_finalized: list[FinalDecision] = []
        escalated: list[tuple[EnrichedTransaction, str]] = []

        for tx in records:
            if tx.transaction_id in completed:
                continue
            dmn_dec = self.dmn_engine.evaluate(tx)
            dmn_counts[dmn_dec.action] += 1
            if dmn_dec.action == "ESCALATE_TO_MODEL":
                escalated.append((tx, dmn_dec.rule_id))
            else:
                dmn_finalized.append(self.dmn_engine.to_final_decision(tx, dmn_dec))

        cost_est = GeminiDecisionClient.estimate_dry_run_cost(
            [tx for tx, _ in escalated],
            price_per_1m_input_tokens=self.price_per_1m_input_tokens,
        )
        est_cost = float(cost_est["estimated_cost_usd"])
        est_tokens = int(cost_est["estimated_total_input_tokens"])

        if dry_run:
            return PipelineResult(
                decisions=list(completed.values()),
                failed_count=0,
                resumed_count=resumed_count,
                dmn_counts=dict(dmn_counts),
                total_input_tokens=est_tokens,
                estimated_cost_usd=est_cost,
                checkpoint_path=checkpoint_path,
                dlq_path=dlq_path,
                predictions_csv_path=predictions_csv_path,
                dry_run=True,
            )

        if est_cost > self.max_cost_usd:
            raise RuntimeError(
                f"Estimated API cost (${est_cost:.4f}) exceeds --max-cost-usd (${self.max_cost_usd:.4f})."
            )

        for final_dec in dmn_finalized:
            _append_jsonl(checkpoint_path, final_dec.model_dump_json())
            completed[final_dec.transaction_id] = final_dec

        failed_count = 0
        if escalated:
            if self.decision_client is None:
                raise RuntimeError("decision_client is required when transactions escalate to the model.")
            client = self.decision_client
            io_lock = threading.Lock()

            def _score_one(tx: EnrichedTransaction, rule_id: str) -> None:
                nonlocal failed_count
                try:
                    model_res = client.evaluate_transaction(tx)
                    final_dec = client.build_final_decision(
                        tx,
                        model_res,
                        threshold=self.fraud_threshold,
                        rule_id=rule_id,
                    )
                    line = final_dec.model_dump_json()
                    with io_lock:
                        _append_jsonl(checkpoint_path, line)
                        completed[tx.transaction_id] = final_dec
                except Exception as exc:  # noqa: BLE001
                    err_line = json.dumps(
                        {
                            "transaction_id": tx.transaction_id,
                            "rule_id": rule_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    with io_lock:
                        failed_count += 1
                        _append_jsonl(dlq_path, err_line)

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = [executor.submit(_score_one, tx, rule_id) for tx, rule_id in escalated]
                for fut in as_completed(futures):
                    fut.result()

        record_ids = {tx.transaction_id for tx in records}
        ordered_decisions = [completed[tx.transaction_id] for tx in records if tx.transaction_id in completed]
        for tx_id in sorted(completed):
            if tx_id not in record_ids:
                ordered_decisions.append(completed[tx_id])

        if ordered_decisions:
            pd.DataFrame([d.model_dump() for d in ordered_decisions]).to_csv(
                predictions_csv_path, index=False
            )

        total_tokens = sum(d.input_tokens for d in ordered_decisions)
        actual_cost = round((total_tokens / 1_000_000.0) * self.price_per_1m_input_tokens, 6)

        return PipelineResult(
            decisions=ordered_decisions,
            failed_count=failed_count,
            resumed_count=resumed_count,
            dmn_counts=dict(dmn_counts),
            total_input_tokens=total_tokens,
            estimated_cost_usd=actual_cost,
            checkpoint_path=checkpoint_path,
            dlq_path=dlq_path,
            predictions_csv_path=predictions_csv_path,
            dry_run=False,
        )
