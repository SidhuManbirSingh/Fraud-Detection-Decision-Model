"""Threaded client wrapper for Google's gemini-decision-1 Decisions API (v1beta)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.decisions import (
    ChoiceAnswer,
    CreateDecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from fraud_decision_model.schemas import (
    EnrichedTransaction,
    FinalDecision,
    ModelDecisionResult,
)

DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "config" / "decision_questions.yaml"
DEFAULT_PRICE_PER_1M_INPUT_TOKENS = 0.10


def _is_retryable_error(exc: BaseException) -> bool:
    """Retry only transient rate-limit / server / network exceptions."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        return code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class ThreadRateLimiter:
    """Thread-safe concurrency semaphore combined with smooth RPM spacing."""

    def __init__(self, concurrency: int = 8, rpm: int = 120) -> None:
        if concurrency <= 0 or rpm <= 0:
            raise ValueError("concurrency and rpm must be positive integers.")
        self._semaphore = threading.Semaphore(concurrency)
        self._min_interval = 60.0 / float(rpm)
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def __enter__(self) -> "ThreadRateLimiter":
        self._semaphore.acquire()
        try:
            with self._lock:
                now = time.monotonic()
                wait_s = max(0.0, self._next_slot - now)
                self._next_slot = max(now, self._next_slot) + self._min_interval
            if wait_s > 0.0:
                time.sleep(wait_s)
        except BaseException:
            self._semaphore.release()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._semaphore.release()


def load_question_rubric(path: Path | str = DEFAULT_QUESTIONS_PATH) -> dict[str, Any]:
    """Load the closed-ended noul/score/choice question definitions from YAML."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for required in ("is_fraud", "risk_severity", "primary_risk_driver"):
        if required not in data:
            raise ValueError(f"Missing required question '{required}' in {path}")
    return data


def _extract_input_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    for attr in ("total_input_tokens", "initial_input_tokens", "total_tokens"):
        val = getattr(usage, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    return 0


class GeminiDecisionClient:
    """High-throughput client for calibrated fraud decisions via gemini-decision-1."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_vertexai: bool | None = None,
        questions_path: Path | str = DEFAULT_QUESTIONS_PATH,
        concurrency: int = 8,
        rpm: int = 120,
        max_attempts: int = 4,
    ) -> None:
        load_dotenv(override=False)
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to .env or the environment.")

        self.model = model or os.getenv("GEMINI_DECISION_MODEL", "gemini-decision-1")
        if use_vertexai is None:
            env_flag = os.getenv("GEMINI_USE_VERTEXAI", "false").strip().lower()
            use_vertexai = env_flag in ("1", "true", "yes")
        self.use_vertexai = use_vertexai
        self.concurrency = concurrency

        http_opts = types.HttpOptions(api_version="v1beta", timeout=30_000)
        if self.use_vertexai:
            self._client = genai.Client(
                vertexai=True,
                api_key=resolved_key,
                http_options=http_opts,
            )
        else:
            self._client = genai.Client(
                api_key=resolved_key,
                http_options=http_opts,
            )
        self.questions = load_question_rubric(questions_path)
        self._limiter = ThreadRateLimiter(concurrency=concurrency, rpm=rpm)
        self._max_attempts = max_attempts

    def evaluate_transaction(self, tx: EnrichedTransaction) -> ModelDecisionResult:
        """Score one EnrichedTransaction against all calibrated questions in a single pass."""
        payload = tx.to_model_payload()
        input_text = json.dumps(payload, indent=2)

        response: CreateDecisionResponse | None = None
        latency_ms = 0.0

        for attempt in Retrying(
            retry=retry_if_exception(_is_retryable_error),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            reraise=True,
        ):
            with attempt:
                with self._limiter:
                    t0 = time.perf_counter()
                    response = self._client.decisions.create(
                        model=self.model,
                        input=input_text,
                        questions=self.questions,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0

        if response is None:
            raise RuntimeError(f"No response returned for tx={tx.transaction_id}")

        answers = response.answers or {}
        if not answers:
            feedback = response.prompt_feedback
            raise RuntimeError(f"Empty answers returned for tx={tx.transaction_id}: feedback={feedback}")

        fraud_ans = answers.get("is_fraud")
        sev_ans = answers.get("risk_severity")
        drv_ans = answers.get("primary_risk_driver")

        if not isinstance(fraud_ans, NoulAnswer):
            raise TypeError(f"Expected NoulAnswer for 'is_fraud', got {type(fraud_ans).__name__}")
        if not isinstance(sev_ans, ScoreAnswer):
            raise TypeError(f"Expected ScoreAnswer for 'risk_severity', got {type(sev_ans).__name__}")
        if not isinstance(drv_ans, ChoiceAnswer):
            raise TypeError(f"Expected ChoiceAnswer for 'primary_risk_driver', got {type(drv_ans).__name__}")

        return ModelDecisionResult(
            transaction_id=tx.transaction_id,
            fraud_probability=float(fraud_ans.noul),
            fraud_confidence=float(fraud_ans.confidence),
            risk_severity_score=float(sev_ans.score),
            risk_severity_confidence=float(sev_ans.confidence),
            risk_severity_probs={str(k): float(v) for k, v in (sev_ans.probabilities or {}).items()},
            primary_risk_driver=str(drv_ans.choice),
            driver_confidence=float(drv_ans.confidence),
            driver_probs={str(k): float(v) for k, v in (drv_ans.probabilities or {}).items()},
            input_tokens=_extract_input_tokens(response),
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def synthesize_reasoning(tx: EnrichedTransaction, result: ModelDecisionResult) -> str:
        """Build a concise, auditable explanation from calibrated outputs & transaction flags."""
        flags: list[str] = []
        if tx.failed_auth_checks:
            flags.append(f"failed_auth={','.join(tx.failed_auth_checks)}")
        else:
            flags.append("auth=all_pass")
        if tx.is_cross_border:
            flags.append(f"cross_border={tx.user_country}->{tx.card_bin_country}")
        if tx.account_age_days <= 90:
            flags.append(f"new_acct={tx.account_age_days}d")
        if tx.amount < 5.0:
            flags.append(f"micro_amt=${tx.amount:.2f}")
        elif tx.amount_to_baseline_ratio >= 3.0:
            flags.append(f"spend_ratio={tx.amount_to_baseline_ratio:.2f}x")
        if tx.channel == "web":
            flags.append("channel=web")
        if tx.promo_used:
            flags.append("promo=yes")
        if tx.shipping_distance_km > 1000.0:
            flags.append(f"dist={tx.shipping_distance_km:.0f}km")

        flag_str = "; ".join(flags)
        return (
            f"gemini-decision-1 driver={result.primary_risk_driver} "
            f"(P(fraud)={result.fraud_probability:.3f}, conf={result.fraud_confidence:.3f}, "
            f"severity={result.risk_severity_score:.2f}/4.0). Signals: [{flag_str}]."
        )

    @classmethod
    def build_final_decision(
        cls,
        tx: EnrichedTransaction,
        result: ModelDecisionResult,
        threshold: float = 0.50,
        rule_id: str = "DMN_GREY_ZONE_ESCALATE",
    ) -> FinalDecision:
        """Combine ModelDecisionResult with threshold policy into FinalDecision."""
        is_fraud = result.fraud_probability >= threshold
        return FinalDecision(
            transaction_id=tx.transaction_id,
            decision_source="gemini_decision_1",
            rule_id=rule_id,
            is_fraud=is_fraud,
            fraud_probability=round(result.fraud_probability, 6),
            confidence_score=round(result.fraud_confidence, 6),
            risk_severity_score=round(result.risk_severity_score, 4),
            primary_risk_driver=result.primary_risk_driver,
            reasoning=cls.synthesize_reasoning(tx, result),
            input_tokens=result.input_tokens,
            latency_ms=result.latency_ms,
            feature_mode=tx.feature_mode,
        )

    @staticmethod
    def estimate_dry_run_cost(
        records: Sequence[EnrichedTransaction],
        questions_path: Path | str = DEFAULT_QUESTIONS_PATH,
        price_per_1m_input_tokens: float = DEFAULT_PRICE_PER_1M_INPUT_TOKENS,
    ) -> dict[str, Any]:
        """Estimate input token usage and USD cost before making live API calls."""
        rubric = load_question_rubric(questions_path)
        rubric_chars = len(json.dumps(rubric))
        # Rough conservative ~3.5 chars/token heuristic for structured JSON/YAML text.
        rubric_tokens = max(1, int(round(rubric_chars / 3.5)))

        total_tokens = 0
        for tx in records:
            payload_chars = len(json.dumps(tx.to_model_payload(), indent=2))
            payload_tokens = max(1, int(round(payload_chars / 3.5)))
            total_tokens += rubric_tokens + payload_tokens

        est_cost = (total_tokens / 1_000_000.0) * price_per_1m_input_tokens
        return {
            "escalated_calls": len(records),
            "estimated_tokens_per_call": round(total_tokens / len(records), 1) if records else 0.0,
            "estimated_total_input_tokens": total_tokens,
            "estimated_output_tokens": 0,
            "estimated_cost_usd": round(est_cost, 6),
        }
