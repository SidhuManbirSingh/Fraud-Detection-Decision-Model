"""Pydantic v2 schemas for leakage-free transaction records and decisions."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FeatureMode = Literal["production_strict", "dataset_benchmark"]
DecisionSource = Literal["dmn_rule", "gemini_decision_1"]
DecisionAction = Literal["AUTO_APPROVE", "AUTO_DECLINE", "ESCALATE_TO_MODEL"]
CheckStatus = Literal["PASS", "FAIL"]


class EnrichedTransaction(BaseModel):
    """Canonical transaction payload with strictly causal / leakage-safe features."""

    transaction_id: int
    user_id: int
    amount: float = Field(ge=0.0)
    baseline_avg_amount: float = Field(gt=0.0)
    amount_to_baseline_ratio: float = Field(ge=0.0)
    prior_tx_count: int = Field(ge=0)
    account_age_days: int = Field(ge=0)
    account_maturity: str
    user_country: str
    card_bin_country: str
    is_cross_border: bool
    shipping_distance_km: float = Field(ge=0.0)
    channel: str
    merchant_category: str
    promo_used: bool
    avs_check: CheckStatus
    cvv_check: CheckStatus
    three_ds_auth: CheckStatus
    auth_failures_count: int = Field(ge=0, le=3)
    failed_auth_checks: list[str]
    tx_hour_utc: int = Field(ge=0, le=23)
    feature_mode: FeatureMode

    def to_model_payload(self) -> dict[str, Any]:
        """Serialize only non-leaky semantic attributes for gemini-decision-1.

        Explicitly strips `transaction_id` and `user_id` (alongside never storing
        `is_fraud` or `total_transactions_user`).
        """
        return {
            "amount": round(self.amount, 2),
            "user_baseline_amount": round(self.baseline_avg_amount, 2),
            "amount_to_baseline_ratio": round(self.amount_to_baseline_ratio, 3),
            "prior_transactions_seen": self.prior_tx_count,
            "account_age_days": self.account_age_days,
            "account_maturity": self.account_maturity,
            "user_country": self.user_country,
            "card_bin_country": self.card_bin_country,
            "is_cross_border": self.is_cross_border,
            "shipping_distance_km": round(self.shipping_distance_km, 2),
            "channel": self.channel,
            "merchant_category": self.merchant_category,
            "promo_used": self.promo_used,
            "auth_checks": {
                "avs": self.avs_check,
                "cvv": self.cvv_check,
                "three_ds": self.three_ds_auth,
            },
            "auth_failure_count": self.auth_failures_count,
            "failed_auth_checks": self.failed_auth_checks,
            "tx_hour_utc": self.tx_hour_utc,
        }


class DMNDecision(BaseModel):
    """Result emitted by the deterministic DMN First-Hit rule engine."""

    action: DecisionAction
    rule_id: str
    is_fraud: Optional[bool] = None
    fraud_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    primary_risk_driver: Optional[str] = None
    reasoning: str


class ModelDecisionResult(BaseModel):
    """Structured probabilities and telemetry parsed from gemini-decision-1."""

    transaction_id: int
    fraud_probability: float = Field(ge=0.0, le=1.0)
    fraud_confidence: float = Field(ge=0.0, le=1.0)
    risk_severity_score: float = Field(ge=0.0)
    risk_severity_confidence: float = Field(ge=0.0, le=1.0)
    risk_severity_probs: dict[str, float]
    primary_risk_driver: str
    driver_confidence: float = Field(ge=0.0, le=1.0)
    driver_probs: dict[str, float]
    input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class FinalDecision(BaseModel):
    """Unified audit record across both DMN rules and gemini-decision-1."""

    transaction_id: int
    decision_source: DecisionSource
    rule_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_severity_score: Optional[float] = None
    primary_risk_driver: str
    reasoning: str
    input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    feature_mode: FeatureMode
