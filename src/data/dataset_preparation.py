from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def enforce_unique_hourly_index(
    df: pd.DataFrame,
    *,
    time_column: str = "ts",
    label: str = "frame",
    expected_freq: pd.Timedelta | str = pd.Timedelta(hours=1),
    raise_on_gap: bool = True,
    normalize_to_hour: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """Ensure a dataframe has unique, hourly-spaced timestamps.

    Returns a tuple of (clean dataframe, duplicates_removed, gap_count).
    Duplicates are resolved by keeping the latest row for each timestamp.
    Any non-hourly intervals trigger a log; optionally raise when detected.
    """

    if time_column not in df.columns:
        raise ValueError(f"{label} is missing required column: {time_column}")

    clean = df.copy()
    clean[time_column] = pd.to_datetime(clean[time_column], utc=True, errors="coerce")
    if clean[time_column].isna().any():
        raise ValueError(f"{label} contains unparsable timestamps in {time_column}")

    clean = clean.sort_values(time_column).reset_index(drop=True)

    if normalize_to_hour:
        clean[time_column] = clean[time_column].dt.floor("h")
        clean = clean.sort_values(time_column).reset_index(drop=True)

    before = len(clean)
    clean = clean.drop_duplicates(subset=time_column, keep="last").reset_index(drop=True)
    duplicates_removed = before - len(clean)
    if duplicates_removed:
        print(f"[{label}] Dropped {duplicates_removed} duplicate rows; kept latest entries.")

    expected_delta = pd.Timedelta(expected_freq)
    deltas = clean[time_column].diff().dropna()
    anomaly_mask = deltas != expected_delta
    gap_count = int(anomaly_mask.sum())
    if gap_count:
        anomaly_ts = clean.loc[anomaly_mask.index, time_column]
        first_issue = anomaly_ts.iloc[0].isoformat()
        last_issue = anomaly_ts.iloc[-1].isoformat()
        print(
            f"[{label}] Found {gap_count} non-hourly intervals between timestamps. "
            f"First anomaly at {first_issue}, last at {last_issue}."
        )
        if raise_on_gap:
            raise AssertionError(f"{label} violates expected {expected_delta} spacing; see log above.")

    return clean, duplicates_removed, gap_count


def repair_hourly_continuity(
    df: pd.DataFrame,
    *,
    time_column: str = "ts",
    label: str = "frame",
    expected_freq: pd.Timedelta | str = pd.Timedelta(hours=1),
    forward_fill: bool = True,
    backward_fill: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Reindex a dataframe to continuous hourly timestamps and fill gaps.

    Parameters
    ----------
    df: Dataframe containing a timestamp column.
    time_column: Name of the timestamp column (default: ``ts``).
    label: Descriptive label used in log messages.
    expected_freq: Expected spacing between rows (default: 1 hour).
    forward_fill: Whether to forward-fill numeric columns.
    backward_fill: Whether to backward-fill after forward fill to cover leading gaps.

    Returns
    -------
    (repaired_df, missing_count)
        The repaired dataframe (sorted, deduplicated, and reindexed) and the
        number of missing hourly slots that were introduced.
    """

    if time_column not in df.columns:
        raise ValueError(f"{label} is missing required column: {time_column}")

    freq = pd.Timedelta(expected_freq)

    normalized = df.copy()
    normalized[time_column] = pd.to_datetime(normalized[time_column], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=[time_column]).sort_values(time_column)
    normalized = normalized.drop_duplicates(subset=time_column, keep="last")

    if normalized.empty:
        return normalized.reset_index(drop=True), 0

    start = normalized[time_column].iloc[0]
    end = normalized[time_column].iloc[-1]
    full_index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")

    existing_index = normalized.set_index(time_column).index
    missing_index = full_index.difference(existing_index)

    reindexed = normalized.set_index(time_column).reindex(full_index)

    numeric_cols = reindexed.select_dtypes(include=[np.number]).columns
    if forward_fill and len(numeric_cols) > 0:
        reindexed[numeric_cols] = reindexed[numeric_cols].ffill()
    if backward_fill and len(numeric_cols) > 0:
        reindexed[numeric_cols] = reindexed[numeric_cols].bfill()

    non_numeric_cols = reindexed.select_dtypes(exclude=[np.number]).columns
    for column in non_numeric_cols:
        if forward_fill:
            reindexed[column] = reindexed[column].ffill()
        if backward_fill:
            reindexed[column] = reindexed[column].bfill()

    if len(missing_index) > 0:
        start_gap = missing_index[0].isoformat()
        end_gap = missing_index[-1].isoformat()
        print(
            f"[{label}] Backfilled {len(missing_index)} hourly gaps between {start_gap} and {end_gap}.",
        )

    reindexed = reindexed.reset_index().rename(columns={"index": time_column})
    reindexed[time_column] = pd.to_datetime(reindexed[time_column], utc=True)
    return reindexed, len(missing_index)


def make_features_and_target(
    df: pd.DataFrame,
    target_column: str = "ret_1h",
    dropna: bool = True,
    allowed_features: Optional[Sequence[str]] = None,
    return_ts: bool = False,
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create feature matrix X and target y from the curated features table.

    - Sorts by ts.
    - Drops rows with NA in the target (optional).
    - Uses all remaining numeric columns except ts and the target as features.
    """
    if "ts" not in df.columns:
        raise ValueError("Expected a 'ts' column in the dataframe.")

    df = df.sort_values("ts").reset_index(drop=True)

    if dropna:
        df = df.dropna(subset=[target_column])

    y = df[target_column].copy()

    non_feature_cols = {"ts", target_column, "ret_fwd_3h", "ret_4h"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    if allowed_features is not None:
        missing = [col for col in allowed_features if col not in feature_cols]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        feature_cols = list(allowed_features)

    X = df[feature_cols].copy()
    if return_ts:
        ts_series = df["ts"].copy()
        return X, y, ts_series
    return X, y


def time_series_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> DatasetSplits:
    """Split X, y into time-ordered train/val/test sets and scale features.

    The split respects time ordering: earliest rows go to train, then val, then test.
    Standard scaling is fit on train only and applied to val/test to avoid leakage.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset; cannot create splits.")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]

    X_test = X.iloc[n_train + n_val:]
    y_test = y.iloc[n_train + n_val:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DatasetSplits(
        X_train=X_train_scaled,
        y_train=y_train.to_numpy(),
        X_val=X_val_scaled,
        y_val=y_val.to_numpy(),
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        feature_names=list(X.columns),
    )
