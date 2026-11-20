"""Leakage-safe ingestion, feature engineering, and user-grouped pilot sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fraud_decision_model.schemas import EnrichedTransaction, FeatureMode

REQUIRED_COLUMNS: tuple[str, ...] = (
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
)

# Median order baseline used only on the very first chronological transaction in production_strict mode.
DEFAULT_COLD_START_AMOUNT = 90.0


def load_and_quarantine(csv_path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    """Load transactions CSV, validate integrity, and quarantine target/leakage columns.

    Drops `is_fraud` (target) and `total_transactions_user` (100% future look-ahead
    count across the full horizon) before returning feature data.
    """
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {missing_cols}")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
        bad = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Unexpected missing values in CSV columns: {bad}")

    unique_labels = set(df["is_fraud"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid values in is_fraud column: {unique_labels}")

    y_true = df.set_index("transaction_id")["is_fraud"].astype(int)
    df_features = df.drop(columns=["is_fraud", "total_transactions_user"]).copy()
    return df_features, y_true


def _classify_account_maturity(account_age_days: pd.Series) -> pd.Series:
    conditions = [
        account_age_days <= 90,
        (account_age_days > 90) & (account_age_days <= 365),
        account_age_days > 365,
    ]
    choices = ["new_0_90d", "established_91_365d", "mature_over_365d"]
    return pd.Series(
        np.select(conditions, choices, default="mature_over_365d"),
        index=account_age_days.index,
    )


def _compute_causal_global_fallback(work: pd.DataFrame) -> pd.Series:
    """Compute expanding prior global median ordered strictly by timestamp."""
    chrono = work.sort_values(["_ts", "transaction_id"], kind="mergesort")
    causal_med = chrono["amount"].expanding().median().shift(1).fillna(DEFAULT_COLD_START_AMOUNT)
    return causal_med.reindex(work.index)


def engineer_features(
    df_features: pd.DataFrame,
    feature_mode: FeatureMode = "production_strict",
) -> pd.DataFrame:
    """Create strictly causal, semantically clear features for rules and model calls."""
    if feature_mode not in ("production_strict", "dataset_benchmark"):
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    if "is_fraud" in df_features.columns or "total_transactions_user" in df_features.columns:
        raise ValueError("Leakage columns (`is_fraud` or `total_transactions_user`) must be quarantined first.")

    work = df_features.copy()
    work["_ts"] = pd.to_datetime(work["transaction_time"], utc=True)
    work = work.sort_values(["user_id", "_ts", "transaction_id"], kind="mergesort").reset_index(drop=True)

    work["prior_tx_count"] = work.groupby("user_id", sort=False).cumcount().astype(int)

    if feature_mode == "production_strict":
        prior_sum = work.groupby("user_id", sort=False)["amount"].cumsum() - work["amount"]
        has_prior = work["prior_tx_count"] > 0
        user_prior_avg = pd.Series(np.nan, index=work.index, dtype="float64")
        user_prior_avg.loc[has_prior] = prior_sum.loc[has_prior] / work.loc[has_prior, "prior_tx_count"]

        global_fallback = _compute_causal_global_fallback(work)
        baseline = user_prior_avg.fillna(global_fallback).clip(lower=0.01)
        if "avg_amount_user" in work.columns:
            work = work.drop(columns=["avg_amount_user"])
    else:
        baseline = work["avg_amount_user"].astype(float).clip(lower=0.01)

    work["baseline_avg_amount"] = baseline.round(4)
    work["amount_to_baseline_ratio"] = (work["amount"].astype(float) / work["baseline_avg_amount"]).round(3)
    work["is_cross_border"] = (work["country"] != work["bin_country"]).astype(bool)

    work["avs_check"] = np.where(work["avs_match"] == 1, "PASS", "FAIL")
    work["cvv_check"] = np.where(work["cvv_result"] == 1, "PASS", "FAIL")
    work["three_ds_auth"] = np.where(work["three_ds_flag"] == 1, "PASS", "FAIL")
    work["auth_failures_count"] = (
        (1 - work["avs_match"].astype(int))
        + (1 - work["cvv_result"].astype(int))
        + (1 - work["three_ds_flag"].astype(int))
    ).astype(int)

    work["failed_auth_checks"] = [
        [
            label
            for flag, label in ((avs, "AVS"), (cvv, "CVV"), (tds, "3DS"))
            if flag == 0
        ]
        for avs, cvv, tds in zip(work["avs_match"], work["cvv_result"], work["three_ds_flag"])
    ]

    work["account_maturity"] = _classify_account_maturity(work["account_age_days"])
    work["tx_hour_utc"] = work["_ts"].dt.hour.astype(int)
    work["promo_used"] = work["promo_used"].astype(bool)
    work["feature_mode"] = feature_mode

    return work.drop(columns=["_ts"])


def _sample_distinct_users_stratified(
    pool: pd.DataFrame,
    target_n: int,
    strata_cols: list[str],
    rng: np.random.Generator,
) -> list[int]:
    """Sample up to `target_n` transaction_ids prioritizing distinct user_id coverage."""
    if target_n <= 0 or pool.empty:
        return []

    # Shuffle deterministically first, then deduplicate to max 1 tx per user within each pool.
    shuffled_idx = rng.permutation(pool.index.to_numpy())
    shuffled = pool.loc[shuffled_idx]
    one_per_user = shuffled.drop_duplicates(subset=["user_id"], keep="first")

    source = one_per_user if len(one_per_user) >= target_n else shuffled

    groups = list(source.groupby(strata_cols, sort=True))
    total_avail = len(source)
    if total_avail <= target_n:
        return source["transaction_id"].astype(int).tolist()

    selected: list[int] = []
    # Proportional allocation across strata with remainder fill.
    quotas: list[tuple[pd.DataFrame, int]] = []
    allocated = 0
    for _, grp in groups:
        q = int(np.floor(target_n * len(grp) / total_avail))
        q = min(q, len(grp))
        quotas.append((grp, q))
        allocated += q

    used_ids: set[int] = set()
    for grp, q in quotas:
        if q > 0:
            chosen = grp["transaction_id"].iloc[:q].astype(int).tolist()
            selected.extend(chosen)
            used_ids.update(chosen)

    remainder = target_n - len(selected)
    if remainder > 0:
        leftover = source[~source["transaction_id"].isin(used_ids)]
        extra = leftover["transaction_id"].iloc[:remainder].astype(int).tolist()
        selected.extend(extra)

    return selected[:target_n]


def sample_user_grouped_pilot(
    df_enriched: pd.DataFrame,
    y_true: pd.Series,
    n_samples: int = 500,
    target_fraud_count: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw a reproducible, user-grouped stratified pilot sample.

    Ensures fraud and legitimate samples come from disjoint `user_id` pools when
    sufficient users exist, preserving representation across account maturity,
    cross-border flag, and channel without single-user clustering bias.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.default_rng(seed)
    work = df_enriched.copy()
    work["_label"] = work["transaction_id"].map(y_true).astype(int)
    work["_is_new_acct"] = work["account_age_days"] <= 90

    frauds = work[work["_label"] == 1]
    legits = work[work["_label"] == 0]

    n_fraud = min(target_fraud_count, len(frauds), n_samples)
    n_legit = min(n_samples - n_fraud, len(legits))

    fraud_ids = _sample_distinct_users_stratified(
        frauds,
        n_fraud,
        strata_cols=["_is_new_acct", "is_cross_border", "channel"],
        rng=rng,
    )
    fraud_users = set(work.loc[work["transaction_id"].isin(fraud_ids), "user_id"].tolist())

    disjoint_legits = legits[~legits["user_id"].isin(fraud_users)]
    legit_pool = disjoint_legits if len(disjoint_legits) >= n_legit else legits
    legit_ids = _sample_distinct_users_stratified(
        legit_pool,
        n_legit,
        strata_cols=["_is_new_acct", "is_cross_border", "channel"],
        rng=rng,
    )

    chosen_ids = fraud_ids + legit_ids
    rng.shuffle(chosen_ids)

    pilot_df = df_enriched[df_enriched["transaction_id"].isin(chosen_ids)].copy()
    order_map = {tx_id: pos for pos, tx_id in enumerate(chosen_ids)}
    pilot_df["_order"] = pilot_df["transaction_id"].map(order_map)
    pilot_df = pilot_df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    pilot_y = y_true.loc[pilot_df["transaction_id"].tolist()].copy()
    return pilot_df, pilot_y


def to_enriched_records(
    df_enriched: pd.DataFrame,
    feature_mode: FeatureMode | None = None,
) -> list[EnrichedTransaction]:
    """Convert an engineered DataFrame into validated EnrichedTransaction objects."""
    records: list[EnrichedTransaction] = []
    for row in df_enriched.itertuples(index=False):
        row_dict = row._asdict()
        mode = feature_mode or row_dict["feature_mode"]
        records.append(
            EnrichedTransaction(
                transaction_id=int(row_dict["transaction_id"]),
                user_id=int(row_dict["user_id"]),
                amount=float(row_dict["amount"]),
                baseline_avg_amount=float(row_dict["baseline_avg_amount"]),
                amount_to_baseline_ratio=float(row_dict["amount_to_baseline_ratio"]),
                prior_tx_count=int(row_dict["prior_tx_count"]),
                account_age_days=int(row_dict["account_age_days"]),
                account_maturity=str(row_dict["account_maturity"]),
                user_country=str(row_dict["country"]),
                card_bin_country=str(row_dict["bin_country"]),
                is_cross_border=bool(row_dict["is_cross_border"]),
                shipping_distance_km=float(row_dict["shipping_distance_km"]),
                channel=str(row_dict["channel"]),
                merchant_category=str(row_dict["merchant_category"]),
                promo_used=bool(row_dict["promo_used"]),
                avs_check=str(row_dict["avs_check"]),
                cvv_check=str(row_dict["cvv_check"]),
                three_ds_auth=str(row_dict["three_ds_auth"]),
                auth_failures_count=int(row_dict["auth_failures_count"]),
                failed_auth_checks=list(row_dict["failed_auth_checks"]),
                tx_hour_utc=int(row_dict["tx_hour_utc"]),
                feature_mode=mode,
            )
        )
    return records
