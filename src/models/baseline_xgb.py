"""
baseline_xgb.py
================================================================================
Phase 3 · Step 1 — Baseline ETA Prediction Engine (XGBoost)
Delhivery Network Intelligence Project

PURPOSE
-------
This script establishes the performance floor that every subsequent model
(Node2Vec Hybrid, GraphSAGE) must beat. A baseline that is too weak makes
the advanced models look trivially impressive; a baseline that is well-tuned
proves that graph intelligence adds *genuine* value over best-in-class tabular ML.

COLUMN MAPPING (raw Delhivery schema → internal names used here)
----------------------------------------------------------------
    actual_time                    → actual_eta          [TARGET, minutes]
    osrm_time                      → osrm_predicted_eta  [anchor feature]
    actual_distance_to_destination → distance_km         [route length]
    source_center                  → source_hub_id       [high-cardinality ID]
    destination_center             → dest_hub_id         [high-cardinality ID]
    route_type                     → route_type          [FTL | Carting]
    trip_creation_time             → trip_creation_time  [datetime, sort key]
    trip_uuid                      → trip_id             [row identifier]

OUTPUT
------
    baseline_predictions.csv — (trip_id, actual_eta, predicted_eta)
                                Required by the multi-model benchmarking script.

USAGE
-----
    python baseline_xgb.py --data_dir ../data --output_dir ../data
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import glob
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

# Gradient boosting
from xgboost import XGBRegressor

# Sklearn utilities
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Target encoding for high-cardinality categoricals
# Install: pip install category_encoders
import category_encoders as ce

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# SECTION 1 · COLUMN CONSTANTS
# ==============================================================================

# Raw column names as they appear in the Delhivery CSV shards
RAW_COL_MAP = {
    "actual_time":                     "actual_eta",
    "osrm_time":                       "osrm_predicted_eta",
    "osrm_distance":                   "osrm_distance_km",
    "actual_distance_to_destination":  "distance_km",
    "source_center":                   "source_hub_id",
    "destination_center":              "dest_hub_id",
    "route_type":                      "route_type",
    "trip_creation_time":              "trip_creation_time",
    "trip_uuid":                       "trip_id",
    "factor":                          "delay_factor",
    "is_cutoff":                       "is_cutoff",
    "start_scan_to_end_scan":          "start_scan_to_end_scan",
}

# Features fed into XGBoost (after encoding)
# These are the *only* columns the model sees at inference time.
NUMERIC_FEATURES = [
    "osrm_predicted_eta",   # OSRM's static estimate — strongest single predictor
    "distance_km",          # physical corridor length
    "osrm_distance_km",     # OSRM's own distance estimate (may differ from actual)
    "hour_of_day",          # extracted from trip_creation_time
    "day_of_week",          # 0 = Monday … 6 = Sunday
    "is_weekend",           # binary flag: weekend trips often behave differently
    "is_cutoff",            # hub held shipment past slot → structural delay signal
    "start_scan_to_end_scan",  # total dwell at the source hub (proxy for congestion)
]

# High-cardinality IDs that need target encoding (not one-hot encoding)
HIGH_CARD_CAT_FEATURES = ["source_hub_id", "dest_hub_id"]

# Low-cardinality categoricals that are one-hot encoded inside XGBoost natively
LOW_CARD_CAT_FEATURES = ["route_type"]   # only 2 values: FTL | Carting

TARGET_COL = "actual_eta"
ID_COL     = "trip_id"


# ==============================================================================
# SECTION 2 · DATA LOADING
# ==============================================================================

def load_raw_shards(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all three Delhivery CSV shards into one DataFrame,
    applying the standard dtype map for memory efficiency.

    Parameters
    ----------
    data_dir : Path
        Directory containing data_part_1.csv, data_part_2.csv, data_part_3.csv.
        The function also accepts a pre-cleaned trips_clean.parquet in the same
        directory (preferred — produced by 01_traditional_eda.ipynb).

    Returns
    -------
    pd.DataFrame
        Raw concatenated trip segments — NOT yet cleaned or feature-engineered.
    """
    # Prefer the pre-cleaned parquet produced by the EDA notebook.
    # This avoids repeating outlier removal logic and guarantees consistency.
    parquet_path = data_dir / "trips_clean.parquet"
    if parquet_path.exists():
        print(f"📂 Loading pre-cleaned parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        # Rename to internal schema
        df = df.rename(columns={k: v for k, v in RAW_COL_MAP.items() if k in df.columns})
        print(f"   ✅ {len(df):,} rows × {df.shape[1]} columns loaded")
        return df

    # Fallback: read raw CSV shards
    shard_paths = sorted(glob.glob(str(data_dir / "data_part_*.csv")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No data found in {data_dir}.\n"
            "Expected: trips_clean.parquet OR data_part_*.csv files."
        )

    DTYPE_MAP = {
        "data":                          "category",
        "route_type":                    "category",
        "trip_uuid":                     "string",
        "source_center":                 "string",
        "destination_center":            "string",
        "start_scan_to_end_scan":        "float32",
        "is_cutoff":                     "bool",
        "actual_distance_to_destination":"float32",
        "actual_time":                   "float32",
        "osrm_time":                     "float32",
        "osrm_distance":                 "float32",
        "factor":                        "float32",
        "segment_actual_time":           "float32",
        "segment_osrm_time":             "float32",
        "segment_osrm_distance":         "float32",
        "segment_factor":                "float32",
    }
    DATE_COLS = ["trip_creation_time", "od_start_time", "od_end_time", "cutoff_timestamp"]

    print(f"📂 Loading {len(shard_paths)} raw CSV shard(s) from {data_dir} …")
    parts = [
        pd.read_csv(p, dtype=DTYPE_MAP, parse_dates=DATE_COLS, low_memory=False)
        for p in shard_paths
    ]
    df = pd.concat(parts, ignore_index=True)
    print(f"   ✅ Raw: {len(df):,} rows")

    # Rename to internal schema
    df = df.rename(columns={k: v for k, v in RAW_COL_MAP.items() if k in df.columns})
    return df


# ==============================================================================
# SECTION 3 · PREPROCESSING
# ==============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame and engineer all features needed for the baseline
    XGBoost model.

    Steps performed:
    1. Drop rows with null values in columns that are structurally required.
    2. Apply domain-driven outlier filters (mirrors EDA notebook logic).
    3. Parse and extract temporal features from `trip_creation_time`.
    4. Cast the `route_type` categorical into an integer for XGBoost.
    5. Drop columns that would cause data leakage (segment_factor, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip DataFrame (post-rename, pre-cleaning).

    Returns
    -------
    pd.DataFrame
        Feature-complete, cleaned DataFrame sorted chronologically.
    """
    print("\n── Preprocessing ──────────────────────────────────────────────")
    n_raw = len(df)

    # ── 3.1  Drop critical nulls ──────────────────────────────────────────────
    # Rows missing hub IDs would produce phantom nodes in the graph in later
    # phases; rows missing ETA fields cannot be used for training or evaluation.
    critical_cols = [
        "source_hub_id", "dest_hub_id",
        TARGET_COL, "osrm_predicted_eta", "trip_creation_time"
    ]
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])

    # ── 3.2  Outlier removal (domain-driven, not statistical) ─────────────────
    # These thresholds match 01_traditional_eda.ipynb exactly.
    # We do NOT use IQR / z-score because the delay distribution is heavily
    # right-skewed; those cuts would remove valid long-haul trips.
    LOWER_FACTOR = 0.2   # arriving 5× faster than OSRM: physically impossible
    UPPER_FACTOR = 10.0  # 10× overrun: GPS dropout / multi-day parking / error
    MIN_TIME_MIN = 1.0   # zero-duration trip is a data artefact
    MIN_DIST_KM  = 0.1   # zero-distance leg is a data artefact

    if "delay_factor" in df.columns:
        df = df[
            (df["delay_factor"] >= LOWER_FACTOR) &
            (df["delay_factor"] <= UPPER_FACTOR)
        ]
    df = df[df[TARGET_COL]              >= MIN_TIME_MIN]
    df = df[df["osrm_predicted_eta"]    >= MIN_TIME_MIN]
    if "distance_km" in df.columns:
        df = df[df["distance_km"]       >= MIN_DIST_KM]

    n_after_clean = len(df)
    print(f"   Rows after outlier removal : {n_after_clean:,}  "
          f"({n_raw - n_after_clean:,} dropped, "
          f"{(n_raw - n_after_clean)/n_raw*100:.1f}%)")

    # ── 3.3  Temporal feature extraction ──────────────────────────────────────
    # Why extract hour + day_of_week?
    # OSRM uses static speed profiles and ignores congestion timing entirely.
    # Morning rush (6–10 h) and evening rush (16–20 h) empirically push the
    # delay factor up significantly (shown in EDA). Encoding hour + day gives
    # the model a direct handle on these patterns without requiring a graph.
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"])
    df["hour_of_day"]  = df["trip_creation_time"].dt.hour.astype("int8")
    df["day_of_week"]  = df["trip_creation_time"].dt.dayofweek.astype("int8")
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype("int8")

    # ── 3.4  Route type → integer ─────────────────────────────────────────────
    # XGBoost handles integer columns natively (no one-hot needed for binary).
    route_map = {"FTL": 0, "Carting": 1}
    if "route_type" in df.columns:
        df["route_type"] = (
            df["route_type"].astype(str).map(route_map).fillna(-1).astype("int8")
        )

    # ── 3.5  Boolean → int ────────────────────────────────────────────────────
    if "is_cutoff" in df.columns:
        df["is_cutoff"] = df["is_cutoff"].astype("int8")

    # ── 3.6  Ensure trip_id column exists ─────────────────────────────────────
    if ID_COL not in df.columns:
        df[ID_COL] = df.index.astype(str)

    # ── 3.7  Sort chronologically ─────────────────────────────────────────────
    # MUST be done before the chronological split (Section 4).
    # Sorting here also guarantees reproducibility across different shard
    # load orders, which vary depending on OS glob behaviour.
    df = df.sort_values("trip_creation_time").reset_index(drop=True)

    print(f"   Final row count            : {len(df):,}")
    print(f"   Date range                 : "
          f"{df['trip_creation_time'].min().date()} → "
          f"{df['trip_creation_time'].max().date()}")
    print(f"   Target (actual_eta) median : {df[TARGET_COL].median():.1f} min")

    return df


# ==============================================================================
# SECTION 4 · CHRONOLOGICAL TRAIN / TEST SPLIT
# ==============================================================================

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and test sets by time order, NOT randomly.

    WHY CHRONOLOGICAL SPLIT?
    ────────────────────────
    Logistics ETA prediction is fundamentally a forecasting problem: we always
    predict future trips using models trained on past behaviour.

    A random train_test_split violates this by allowing the model to "see"
    future trips during training (data leakage). The inflated accuracy this
    produces is illusory — at deployment the model only ever sees trips whose
    creation time is AFTER its training window.

    By sorting chronologically and cutting at the 80th percentile timestamp,
    we simulate real deployment conditions precisely.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted trip DataFrame (output of preprocess_data).
    train_frac : float, default 0.80
        Fraction of rows assigned to the training set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) — both retain all original columns.
    """
    # The split point is the row index at the train_frac quantile.
    # Because df is already sorted by trip_creation_time, row index == time order.
    split_idx = int(len(df) * train_frac)
    split_date = df.iloc[split_idx]["trip_creation_time"]

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    print(f"\n── Chronological Split (80 / 20) ───────────────────────────────")
    print(f"   Split timestamp : {split_date}")
    print(f"   Train rows      : {len(train_df):,}  "
          f"({train_df['trip_creation_time'].min().date()} → "
          f"{train_df['trip_creation_time'].max().date()})")
    print(f"   Test  rows      : {len(test_df):,}  "
          f"({test_df['trip_creation_time'].min().date()} → "
          f"{test_df['trip_creation_time'].max().date()})")

    return train_df, test_df


# ==============================================================================
# SECTION 5 · FEATURE MATRIX ASSEMBLY + TARGET ENCODING
# ==============================================================================

def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.Series, pd.Series, ce.TargetEncoder]:
    """
    Assemble the feature matrices X_train / X_test and apply target encoding
    to high-cardinality hub-ID columns.

    WHY TARGET ENCODING (NOT ONE-HOT)?
    ───────────────────────────────────
    The Delhivery network contains hundreds of unique source and destination
    hub IDs. One-hot encoding would create hundreds of sparse binary columns,
    most nearly zero, bloating memory and degrading tree-model performance.

    Target encoding replaces each category level with the *mean of the target*
    (actual_eta) computed within the TRAINING SET ONLY.  This condenses rich
    hub-level congestion information into a single dense numeric per column.

    LEAKAGE PREVENTION (critical):
    • The TargetEncoder is FIT on X_train + y_train exclusively.
    • It is then used to TRANSFORM both X_train and X_test.
    • The test set hub IDs are never visible during encoder fitting.
    • Unknown test-set hub IDs (OOV) fall back to the global target mean,
      which is also computed from training data only.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Split DataFrames from chronological_split.

    Returns
    -------
    X_train, y_train, X_test, y_test : pd.DataFrame / pd.Series
        Feature-encoded matrices and target vectors.
    train_ids, test_ids : pd.Series
        trip_id columns (preserved for the output CSV).
    encoder : ce.TargetEncoder
        Fitted encoder (serialised separately for production serving).
    """
    # ── Determine available feature columns ───────────────────────────────────
    # We check availability at runtime because some parquet exports may not
    # include every raw auxiliary column.
    numeric_cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    cat_cols     = [c for c in HIGH_CARD_CAT_FEATURES if c in train_df.columns]
    low_cat_cols = [c for c in LOW_CARD_CAT_FEATURES if c in train_df.columns]

    all_feature_cols = numeric_cols + cat_cols + low_cat_cols

    print(f"\n── Feature Matrix Assembly ─────────────────────────────────────")
    print(f"   Numeric features   : {numeric_cols}")
    print(f"   High-card categoricals (target-encoded) : {cat_cols}")
    print(f"   Low-card categoricals (int-encoded)     : {low_cat_cols}")

    X_train = train_df[all_feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test  = test_df[all_feature_cols].copy()
    y_test  = test_df[TARGET_COL].copy()

    train_ids = train_df[ID_COL].reset_index(drop=True)
    test_ids  = test_df[ID_COL].reset_index(drop=True)

    # ── Target encoding ───────────────────────────────────────────────────────
    # smoothing=1.0: Bayesian smoothing that shrinks low-count hub estimates
    # toward the global mean. This prevents overfitting to hubs with 1–2 trips.
    if cat_cols:
        encoder = ce.TargetEncoder(
            cols=cat_cols,
            smoothing=1.0,         # shrinkage toward global mean for rare hubs
            handle_unknown="value", # OOV → global training mean
            handle_missing="value", # NaN  → global training mean
        )
        # FIT only on training data (y_train drives the encoding)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        # TRANSFORM test: encoder uses the frozen training statistics
        X_test[cat_cols]  = encoder.transform(X_test[cat_cols])
        print(f"   ✅ TargetEncoder fitted on {len(X_train):,} training rows")
    else:
        encoder = None
        print("   ⚠️  No high-cardinality columns found; skipping target encoding")

    # Fill any residual NaNs with column median (only for numeric cols)
    for col in numeric_cols + low_cat_cols:
        if X_train[col].isnull().any():
            fill_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col]  = X_test[col].fillna(fill_val)

    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test  shape : {X_test.shape}")

    return X_train, y_train, X_test, y_test, train_ids, test_ids, encoder


# ==============================================================================
# SECTION 6 · MODEL TRAINING
# ==============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_cv_splits: int = 5,
    n_iter: int = 20,
) -> XGBRegressor:
    """
    Train a tuned XGBRegressor using TimeSeriesSplit cross-validation inside
    a RandomizedSearchCV.

    WHY TimeSeriesSplit (NOT KFold)?
    ─────────────────────────────────
    Standard k-fold cross-validation shuffles rows randomly. In a time-ordered
    dataset this means a validation fold can precede its training fold — again,
    leakage. TimeSeriesSplit guarantees that each validation fold is always
    strictly AFTER its corresponding training fold, preserving temporal order
    within the cross-validation loop.

    WHY RandomizedSearchCV (NOT GridSearch)?
    ─────────────────────────────────────────
    GridSearchCV evaluates every combination in the hyperparameter grid.
    With 5 parameters × 3–5 values each, that's 375+ XGBoost fits before
    even reaching final training. RandomizedSearchCV samples `n_iter` random
    combinations, finding near-optimal parameters at a fraction of the cost.
    For a baseline model, this is the correct trade-off.

    Parameters
    ----------
    X_train : pd.DataFrame
        Encoded training feature matrix.
    y_train : pd.Series
        Training target vector (actual_eta, in minutes).
    n_cv_splits : int
        Number of TimeSeriesSplit folds. 5 is standard for this dataset size.
    n_iter : int
        Number of random hyperparameter combinations to evaluate.

    Returns
    -------
    XGBRegressor
        Best estimator after RandomizedSearchCV, fitted on the full training set.
    """
    print(f"\n── Hyperparameter Tuning (TimeSeriesSplit n={n_cv_splits}, "
          f"n_iter={n_iter}) ──")

    # ── Hyperparameter search space ───────────────────────────────────────────
    # These ranges are informed by XGBoost best practices for tabular regression:
    # • learning_rate: low values (0.01–0.1) require more trees but generalise better
    # • max_depth: 3–8 balances expressiveness vs. overfitting on tabular data
    # • subsample / colsample_bytree: stochastic sampling reduces variance
    # • min_child_weight: regularises splits, especially for sparse hub encodings
    param_dist = {
        "n_estimators":       randint(300, 1000),      # number of boosting rounds
        "learning_rate":      uniform(0.01, 0.19),     # step size shrinkage [0.01, 0.20]
        "max_depth":          randint(3, 9),            # tree depth [3, 8]
        "subsample":          uniform(0.6, 0.4),        # row sampling per tree [0.6, 1.0]
        "colsample_bytree":   uniform(0.5, 0.5),        # feature sampling per tree [0.5, 1.0]
        "min_child_weight":   randint(1, 10),           # min sum of instance weight in leaf
        "reg_lambda":         uniform(0.5, 2.5),        # L2 regularisation
        "reg_alpha":          uniform(0.0, 1.0),        # L1 regularisation
    }

    # Base estimator
    base_xgb = XGBRegressor(
        objective="reg:squarederror",  # MSE loss; appropriate for continuous ETA
        random_state=RANDOM_SEED,
        n_jobs=-1,                     # use all CPU cores
        tree_method="hist",            # histogram-based splits: fast on large data
        eval_metric="mae",
    )

    # TimeSeriesSplit: each fold's train window grows; val window is always future
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",   # maximise −MAE ↔ minimise MAE
        cv=tscv,
        refit=True,          # refit best params on FULL X_train after search
        random_state=RANDOM_SEED,
        verbose=1,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print(f"\n   Best CV MAE     : {-search.best_score_:.4f} min")
    print(f"   Best parameters :")
    for k, v in sorted(search.best_params_.items()):
        print(f"     {k:<22}: {v}")

    return search.best_estimator_


# ==============================================================================
# SECTION 7 · EVALUATION
# ==============================================================================

def evaluate(
    model:     XGBRegressor,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    test_ids:  pd.Series,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Evaluate the trained model on the held-out test set, print a metrics report,
    and save the prediction CSV required by the multi-model benchmarking script.

    Metrics computed
    ────────────────
    • MAE  — Mean Absolute Error (minutes). Primary business metric.
    • RMSE — Root Mean Squared Error (minutes). Penalises large outlier errors.
    • SLA  — % of predictions within ±15% of actual_eta. Business KPI.
             The SLA threshold is Delhivery's contractual accuracy standard.
    • MAPE — Mean Absolute Percentage Error (%). Scale-independent reference.

    Parameters
    ----------
    model      : XGBRegressor — fitted best estimator.
    X_test     : pd.DataFrame — test feature matrix.
    y_test     : pd.Series    — true actual_eta values.
    test_ids   : pd.Series    — trip_id for each test row.
    output_dir : Path         — where to write baseline_predictions.csv.

    Returns
    -------
    Dict[str, float]
        Dictionary of all computed metrics (also written to metrics JSON).
    """
    print("\n── Evaluation ──────────────────────────────────────────────────")

    y_pred = model.predict(X_test)

    # ── Core regression metrics ───────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── SLA metric: % predictions within ±15% of actual ETA ──────────────────
    # Formula: |predicted − actual| / actual ≤ 0.15
    # This is the primary business KPI. GraphSAGE must exceed this number.
    pct_error  = np.abs(y_pred - y_test.values) / y_test.values
    sla_within_15pct = (pct_error <= 0.15).mean() * 100

    # ── MAPE (scale-free reference) ───────────────────────────────────────────
    mape = pct_error.mean() * 100

    # OSRM baseline for comparison (raw OSRM error on same test set)
    # The model should significantly outperform this.
    osrm_col = "osrm_predicted_eta"
    if osrm_col in X_test.columns:
        osrm_mae  = mean_absolute_error(y_test, X_test[osrm_col])
        osrm_rmse = np.sqrt(mean_squared_error(y_test, X_test[osrm_col]))
        osrm_sla  = (
            (np.abs(X_test[osrm_col].values - y_test.values) / y_test.values <= 0.15).mean() * 100
        )
    else:
        osrm_mae = osrm_rmse = osrm_sla = float("nan")

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  BASELINE MODEL EVALUATION REPORT")
    print(f"{'='*55}")
    print(f"  {'Metric':<32} {'XGBoost':>10}  {'OSRM Raw':>10}")
    print(f"  {'-'*52}")
    print(f"  {'MAE (min)':<32} {mae:>10.4f}  {osrm_mae:>10.4f}")
    print(f"  {'RMSE (min)':<32} {rmse:>10.4f}  {osrm_rmse:>10.4f}")
    print(f"  {'MAPE (%)':<32} {mape:>10.2f}  {'':>10}")
    print(f"  {'SLA ≤15% accuracy (%)':<32} {sla_within_15pct:>10.2f}  {osrm_sla:>10.2f}")
    print(f"{'='*55}")
    print(f"\n  📌 Improvement over OSRM baseline:")
    print(f"     MAE  reduction : {(osrm_mae - mae) / osrm_mae * 100:+.1f}%")
    print(f"     SLA  gain      : {sla_within_15pct - osrm_sla:+.2f} pp")

    metrics = {
        "model":              "XGBoost_Baseline",
        "generated_at":       datetime.utcnow().isoformat(),
        "n_test_rows":        int(len(y_test)),
        "mae_min":            float(round(mae,  4)),
        "rmse_min":           float(round(rmse, 4)),
        "mape_pct":           float(round(mape, 4)),
        "sla_within_15pct":   float(round(sla_within_15pct, 4)),
        "osrm_baseline_mae":  float(round(osrm_mae,  4)),
        "osrm_baseline_rmse": float(round(osrm_rmse, 4)),
        "osrm_baseline_sla":  float(round(osrm_sla,  4)),
    }

    # ── Save predictions CSV ──────────────────────────────────────────────────
    # REQUIRED by the multi-model benchmarking script (03_model_benchmarking.ipynb).
    # Column schema is fixed; do not rename without updating the benchmarker.
    pred_df = pd.DataFrame({
        "trip_id":       test_ids.values,
        "actual_eta":    y_test.values,
        "predicted_eta": y_pred,
        "abs_pct_error": pct_error,
    })
    pred_path = output_dir / "baseline_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n   ✅ Predictions saved → {pred_path}  ({len(pred_df):,} rows)")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved    → {metrics_path}")

    return metrics


# ==============================================================================
# SECTION 8 · MAIN PIPELINE ORCHESTRATOR
# ==============================================================================

def run_baseline_pipeline(
    data_dir:   Path,
    output_dir: Path,
    train_frac: float = 0.80,
    n_cv_splits: int  = 5,
    n_iter: int       = 20,
) -> Dict[str, float]:
    """
    End-to-end orchestrator for the XGBoost baseline pipeline.

    Calls each stage in strict order:
        load → preprocess → split → encode → train → evaluate

    Parameters
    ----------
    data_dir   : Path — directory containing raw CSVs or trips_clean.parquet.
    output_dir : Path — where artefacts (CSV, JSON) are written.
    train_frac : float — chronological train fraction (default 0.80).
    n_cv_splits: int   — TimeSeriesSplit folds during tuning.
    n_iter     : int   — RandomizedSearchCV iterations.

    Returns
    -------
    Dict[str, float]
        Final evaluation metrics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 3 · STEP 1: BASELINE XGBoost ETA PIPELINE")
    print("=" * 60)

    # Stage 1 — Load
    df_raw = load_raw_shards(data_dir)

    # Stage 2 — Clean + Feature engineer
    df_clean = preprocess_data(df_raw)

    # Stage 3 — Chronological split
    train_df, test_df = chronological_split(df_clean, train_frac)

    # Stage 4 — Feature matrices + target encoding
    (X_train, y_train,
     X_test,  y_test,
     train_ids, test_ids,
     encoder) = build_feature_matrices(train_df, test_df)

    # Stage 5 — Train (with cross-validated hyperparameter search)
    model = train_model(X_train, y_train, n_cv_splits, n_iter)

    # Stage 6 — Evaluate + save outputs
    metrics = evaluate(model, X_test, y_test, test_ids, output_dir)

    print("\n" + "=" * 60)
    print("  BASELINE PIPELINE COMPLETE")
    print("  → baseline_predictions.csv ready for benchmarking script")
    print("  → baseline_metrics.json ready for Phase 3 comparison report")
    print("=" * 60)

    return metrics


# ==============================================================================
# SECTION 9 · CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 · Baseline XGBoost ETA Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("../data"),
        help="Directory containing trips_clean.parquet or data_part_*.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../data"),
        help="Directory for output artefacts (baseline_predictions.csv, baseline_metrics.json)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.80,
        help="Fraction of data (chronologically earliest) used for training",
    )
    parser.add_argument(
        "--n_cv_splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for cross-validation",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV iterations (increase for better tuning)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_pipeline(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        train_frac  = args.train_frac,
        n_cv_splits = args.n_cv_splits,
        n_iter      = args.n_iter,
    )
