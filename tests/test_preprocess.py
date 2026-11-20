"""Unit tests for leakage quarantine, causal preprocessing, and pilot sampling."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_decision_model.preprocess import (  # noqa: E402
    engineer_features,
    load_and_quarantine,
    sample_user_grouped_pilot,
    to_enriched_records,
)


def _make_sample_df() -> pd.DataFrame:
    rows = []
    # User 1: 4 transactions over time (1 fraud)
    rows.extend(
        [
            (1, 1, 120, 4, 100.0, 80.0, "FR", "FR", "app", "travel", 0, 1, 1, 1, "2024-01-01T10:00:00Z", 50.0, 0),
            (2, 1, 120, 4, 100.0, 120.0, "FR", "FR", "app", "grocery", 0, 1, 1, 1, "2024-01-05T12:00:00Z", 80.0, 0),
            (3, 1, 120, 4, 100.0, 2000.0, "FR", "DE", "web", "electronics", 1, 0, 0, 0, "2024-01-10T15:00:00Z", 1800.0, 1),
            (4, 1, 120, 4, 100.0, 100.0, "FR", "FR", "web", "fashion", 0, 1, 1, 1, "2024-01-15T18:00:00Z", 120.0, 0),
        ]
    )
    # Additional users for pilot grouping tests
    for uid in range(2, 16):
        is_new = uid <= 5
        age = 20 if is_new else 500
        label = 1 if uid % 3 == 0 else 0
        tx_id_base = uid * 10
        rows.append(
            (
                tx_id_base,
                uid,
                age,
                2,
                50.0,
                3.0 if label == 1 else 50.0,
                "US",
                "GB" if label == 1 else "US",
                "web" if uid % 2 == 0 else "app",
                "gaming",
                0,
                0 if label == 1 else 1,
                1,
                1,
                f"2024-02-{uid:02d}T08:00:00Z",
                1200.0 if label == 1 else 100.0,
                label,
            )
        )
        rows.append(
            (
                tx_id_base + 1,
                uid,
                age,
                2,
                50.0,
                55.0,
                "US",
                "US",
                "app",
                "grocery",
                0,
                1,
                1,
                1,
                f"2024-03-{uid:02d}T09:00:00Z",
                90.0,
                0,
            )
        )

    cols = [
        "transaction_id",
        "user_id",
        "account_age_days",
        "total_transactions_user",
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
    return pd.DataFrame(rows, columns=cols)


class PreprocessTests(unittest.TestCase):
    def test_load_and_quarantine_strips_leakage_columns(self) -> None:
        df = _make_sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "tx.csv"
            df.to_csv(csv_path, index=False)

            features, y_true = load_and_quarantine(csv_path)
            self.assertNotIn("is_fraud", features.columns)
            self.assertNotIn("total_transactions_user", features.columns)
            self.assertEqual(len(y_true), len(df))
            self.assertEqual(int(y_true.loc[3]), 1)

    def test_production_strict_causality_and_payload_hygiene(self) -> None:
        df = _make_sample_df()
        f1 = df.drop(columns=["is_fraud", "total_transactions_user"]).copy()
        f2 = f1.copy()
        # Mutate user 1's LAST transaction (tx_id=4) dramatically
        f2.loc[f2["transaction_id"] == 4, "amount"] = 99999.0

        e1 = engineer_features(f1, feature_mode="production_strict")
        e2 = engineer_features(f2, feature_mode="production_strict")

        u1_e1 = e1[e1["user_id"] == 1].sort_values("transaction_id")
        u1_e2 = e2[e2["user_id"] == 1].sort_values("transaction_id")

        # Transactions 1, 2, 3 must remain bit-for-bit identical in causal baseline & prior count
        pd.testing.assert_series_equal(
            u1_e1.iloc[:3]["baseline_avg_amount"].reset_index(drop=True),
            u1_e2.iloc[:3]["baseline_avg_amount"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            u1_e1.iloc[:3]["prior_tx_count"].reset_index(drop=True),
            u1_e2.iloc[:3]["prior_tx_count"].reset_index(drop=True),
        )
        # User 1's 3rd tx saw prior amounts [80.0, 120.0] -> exact mean 100.0
        self.assertAlmostEqual(float(u1_e1.iloc[2]["baseline_avg_amount"]), 100.0, places=4)
        self.assertEqual(int(u1_e1.iloc[2]["prior_tx_count"]), 2)

        records = to_enriched_records(u1_e1)
        payload = records[2].to_model_payload()
        self.assertNotIn("transaction_id", payload)
        self.assertNotIn("user_id", payload)
        self.assertNotIn("is_fraud", payload)
        self.assertNotIn("total_transactions_user", payload)
        self.assertEqual(payload["auth_checks"], {"avs": "FAIL", "cvv": "FAIL", "three_ds": "FAIL"})
        self.assertEqual(payload["failed_auth_checks"], ["AVS", "CVV", "3DS"])
        self.assertTrue(payload["is_cross_border"])

    def test_user_grouped_pilot_sampling(self) -> None:
        df = _make_sample_df()
        y_true = df.set_index("transaction_id")["is_fraud"].astype(int)
        features = df.drop(columns=["is_fraud", "total_transactions_user"])
        enriched = engineer_features(features, feature_mode="production_strict")

        pilot_df, pilot_y = sample_user_grouped_pilot(
            enriched, y_true, n_samples=12, target_fraud_count=4, seed=7
        )
        self.assertEqual(len(pilot_df), 12)
        self.assertEqual(int(pilot_y.sum()), 4)
        fraud_users = set(pilot_df.loc[pilot_y.values == 1, "user_id"])
        legit_users = set(pilot_df.loc[pilot_y.values == 0, "user_id"])
        self.assertEqual(len(fraud_users & legit_users), 0)


if __name__ == "__main__":
    unittest.main()
