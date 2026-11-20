"""Unit tests for the declarative DMN First-Hit rule engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_decision_model.dmn_engine import DMNEngine  # noqa: E402
from fraud_decision_model.schemas import EnrichedTransaction  # noqa: E402


def _tx(**overrides: object) -> EnrichedTransaction:
    base = {
        "transaction_id": 100,
        "user_id": 10,
        "amount": 120.0,
        "baseline_avg_amount": 100.0,
        "amount_to_baseline_ratio": 1.2,
        "prior_tx_count": 5,
        "account_age_days": 365,
        "account_maturity": "established_91_365d",
        "user_country": "FR",
        "card_bin_country": "FR",
        "is_cross_border": False,
        "shipping_distance_km": 200.0,
        "channel": "app",
        "merchant_category": "fashion",
        "promo_used": False,
        "avs_check": "PASS",
        "cvv_check": "PASS",
        "three_ds_auth": "PASS",
        "auth_failures_count": 0,
        "failed_auth_checks": [],
        "tx_hour_utc": 14,
        "feature_mode": "production_strict",
    }
    base.update(overrides)
    return EnrichedTransaction(**base)


class DMNEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DMNEngine()

    def test_card_testing_rule_has_first_hit_priority(self) -> None:
        # Even mature domestic passing all auth checks triggers card-testing decline if amount < 5
        tx = _tx(amount=2.50, amount_to_baseline_ratio=0.025)
        dec = self.engine.evaluate(tx)
        self.assertEqual(dec.action, "AUTO_DECLINE")
        self.assertEqual(dec.rule_id, "DMN_HARD_DECLINE_CARD_TESTING")
        self.assertTrue(dec.is_fraud)
        final = self.engine.to_final_decision(tx, dec)
        self.assertEqual(final.decision_source, "dmn_rule")
        self.assertEqual(final.input_tokens, 0)

    def test_extreme_spike_respects_cold_start_in_production_strict(self) -> None:
        # With prior_tx_count < 3 in production_strict, ratio > 10 escalates instead of auto-declining
        cold_tx = _tx(amount=1500.0, amount_to_baseline_ratio=15.0, prior_tx_count=1, feature_mode="production_strict")
        self.assertEqual(self.engine.evaluate(cold_tx).action, "ESCALATE_TO_MODEL")

        warm_tx = _tx(amount=1500.0, amount_to_baseline_ratio=15.0, prior_tx_count=3, feature_mode="production_strict")
        dec = self.engine.evaluate(warm_tx)
        self.assertEqual(dec.action, "AUTO_DECLINE")
        self.assertEqual(dec.rule_id, "DMN_HARD_DECLINE_EXTREME_SPIKE")

    def test_routine_fast_approve_and_grey_zone_escalations(self) -> None:
        routine = _tx()
        dec = self.engine.evaluate(routine)
        self.assertEqual(dec.action, "AUTO_APPROVE")
        self.assertEqual(dec.rule_id, "DMN_FAST_APPROVE_ROUTINE")
        self.assertFalse(dec.is_fraud)

        # Cross-border escalates
        self.assertEqual(
            self.engine.evaluate(_tx(is_cross_border=True, card_bin_country="US")).action,
            "ESCALATE_TO_MODEL",
        )
        # Failed auth escalates
        self.assertEqual(
            self.engine.evaluate(_tx(avs_check="FAIL", auth_failures_count=1, failed_auth_checks=["AVS"])).action,
            "ESCALATE_TO_MODEL",
        )
        # New account <= 90 days escalates
        self.assertEqual(
            self.engine.evaluate(_tx(account_age_days=90, account_maturity="new_0_90d")).action,
            "ESCALATE_TO_MODEL",
        )
        # High domestic shipping distance (> 500 km) escalates to model instead of using synthetic cutoff
        self.assertEqual(
            self.engine.evaluate(_tx(shipping_distance_km=620.0)).action,
            "ESCALATE_TO_MODEL",
        )

    def test_model_only_routing_bypasses_all_rules(self) -> None:
        model_only = DMNEngine(routing_mode="model_only")
        for tx in (_tx(), _tx(amount=1.0), _tx(amount=5000.0, amount_to_baseline_ratio=50.0)):
            dec = model_only.evaluate(tx)
            self.assertEqual(dec.action, "ESCALATE_TO_MODEL")
            self.assertEqual(dec.rule_id, "BYPASS_MODEL_ONLY")


if __name__ == "__main__":
    unittest.main()
