"""Declarative DMN First-Hit rule engine for deterministic fast-path gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

from fraud_decision_model.schemas import DMNDecision, EnrichedTransaction, FinalDecision

RoutingMode = Literal["hybrid", "model_only"]

DEFAULT_DMN_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "dmn_rules.yaml"


def _matches_conditions(tx: EnrichedTransaction, cond: dict[str, Any]) -> bool:
    """Evaluate a rule's declarative condition block against an EnrichedTransaction."""
    if not cond:
        return True

    if "max_amount_exclusive" in cond and not (tx.amount < float(cond["max_amount_exclusive"])):
        return False
    if "min_amount_inclusive" in cond and not (tx.amount >= float(cond["min_amount_inclusive"])):
        return False

    if "min_amount_ratio_exclusive" in cond:
        if not (tx.amount_to_baseline_ratio > float(cond["min_amount_ratio_exclusive"])):
            return False
    if "min_amount_ratio_inclusive" in cond:
        if not (tx.amount_to_baseline_ratio >= float(cond["min_amount_ratio_inclusive"])):
            return False
    if "max_amount_ratio_inclusive" in cond:
        if not (tx.amount_to_baseline_ratio <= float(cond["max_amount_ratio_inclusive"])):
            return False

    if "min_prior_tx_strict" in cond and tx.feature_mode == "production_strict":
        if not (tx.prior_tx_count >= int(cond["min_prior_tx_strict"])):
            return False

    if "min_account_age_days_exclusive" in cond:
        if not (tx.account_age_days > int(cond["min_account_age_days_exclusive"])):
            return False

    if "required_auth_failures" in cond:
        if tx.auth_failures_count != int(cond["required_auth_failures"]):
            return False

    if cond.get("require_domestic") is True and tx.is_cross_border:
        return False

    if "max_shipping_distance_km" in cond:
        if not (tx.shipping_distance_km <= float(cond["max_shipping_distance_km"])):
            return False

    return True


class DMNEngine:
    """Loads YAML DMN rules and evaluates transactions under a FIRST_HIT policy."""

    def __init__(
        self,
        rules_path: Path | str = DEFAULT_DMN_RULES_PATH,
        routing_mode: RoutingMode = "hybrid",
    ) -> None:
        if routing_mode not in ("hybrid", "model_only"):
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")
        self.routing_mode: RoutingMode = routing_mode
        self.rules_path = Path(rules_path)
        with self.rules_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.hit_policy = cfg.get("hit_policy", "FIRST_HIT")
        self.rules = sorted(cfg["rules"], key=lambda r: int(r["priority"]))

    def evaluate(self, tx: EnrichedTransaction) -> DMNDecision:
        """Evaluate a transaction and return either a terminal DMN decision or escalation."""
        if self.routing_mode == "model_only":
            return DMNDecision(
                action="ESCALATE_TO_MODEL",
                rule_id="BYPASS_MODEL_ONLY",
                reasoning="Model-only ablation mode: bypassed DMN fast-path gates.",
            )

        for rule in self.rules:
            conditions = rule.get("conditions") or {}
            if _matches_conditions(tx, conditions):
                reasoning = rule["reasoning_template"].format(
                    amount=tx.amount,
                    baseline_avg_amount=tx.baseline_avg_amount,
                    amount_to_baseline_ratio=tx.amount_to_baseline_ratio,
                    account_age_days=tx.account_age_days,
                )
                return DMNDecision(
                    action=rule["action"],
                    rule_id=rule["rule_id"],
                    is_fraud=rule.get("is_fraud"),
                    fraud_probability=rule.get("fraud_probability"),
                    confidence_score=rule.get("confidence_score"),
                    primary_risk_driver=rule.get("primary_risk_driver"),
                    reasoning=reasoning,
                )

        return DMNDecision(
            action="ESCALATE_TO_MODEL",
            rule_id="DMN_FALLBACK_ESCALATE",
            reasoning="No DMN rule matched; escalated to gemini-decision-1.",
        )

    @staticmethod
    def to_final_decision(tx: EnrichedTransaction, dmn: DMNDecision) -> FinalDecision:
        """Convert a terminal DMNDecision (AUTO_APPROVE / AUTO_DECLINE) into FinalDecision."""
        if dmn.action == "ESCALATE_TO_MODEL" or dmn.is_fraud is None:
            raise ValueError(f"Cannot finalize non-terminal DMN action: {dmn.action}")
        return FinalDecision(
            transaction_id=tx.transaction_id,
            decision_source="dmn_rule",
            rule_id=dmn.rule_id,
            is_fraud=bool(dmn.is_fraud),
            fraud_probability=float(dmn.fraud_probability or 0.0),
            confidence_score=float(dmn.confidence_score or 0.0),
            risk_severity_score=4.0 if dmn.is_fraud else 0.0,
            primary_risk_driver=dmn.primary_risk_driver or "dmn_rule",
            reasoning=dmn.reasoning,
            input_tokens=0,
            latency_ms=0.0,
            feature_mode=tx.feature_mode,
        )
