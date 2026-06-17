"""
node2vec_hybrid.py
================================================================================
Phase 3 · Step 2 — Hybrid ETA Prediction Engine (Node2Vec + XGBoost)
Delhivery Network Intelligence Project

PURPOSE
-------
This script implements the HYBRID model tier. It proves that injecting
structural graph intelligence — even without a full GNN — measurably improves
upon the pure-tabular baseline from Step 1.

CONCEPT
-------
Standard tabular models treat every hub as an isolated category: "source = HUB_A"
and "source = HUB_B" are two unrelated integer codes. They carry zero information
about each hub's structural role in the broader network.

Node2Vec fixes this by performing biased random walks on the logistics graph and
applying Word2Vec-style training. Two hubs that sit in similar topological
positions (e.g., both are high-betweenness sorting centres feeding into the same
downstream spokes) will land near each other in the learned embedding space.

These dense vectors are appended to the tabular feature set, giving XGBoost a
rich, pre-computed understanding of network structure without needing to train a
GNN.

PIPELINE FLOW
-------------
    graph.gpickle  (Phase 2 artefact)
           │
           ▼
    [STAGE 1]  Load graph + trips_clean.parquet
           │
           ▼
    [STAGE 2]  Node2Vec random walks on the MultiDiGraph
               → learn 64-dim embedding per hub
           │
           ▼
    [STAGE 3]  Merge embeddings into trip DataFrame
               (source_hub + dest_hub each contribute 64 dims → 128 new cols)
           │
           ▼
    [STAGE 4]  Chronological split + TargetEncoder (mirrors baseline_xgb.py)
           │
           ▼
    [STAGE 5]  XGBRegressor with TimeSeriesSplit + RandomizedSearchCV
           │
           ▼
    [STAGE 6]  Evaluate → node2vec_predictions.csv + node2vec_metrics.json

OUTPUT FILES
------------
    node2vec_embeddings.parquet    — hub_id → 64-dim float vector
    node2vec_predictions.csv       — (trip_id, actual_eta, predicted_eta)
    node2vec_metrics.json          — MAE, RMSE, SLA ≤15%, MAPE

USAGE
-----
    python node2vec_hybrid.py --data_dir ../data --output_dir ../data
    python node2vec_hybrid.py --data_dir ../data --embed_dim 128 --walk_length 40

DEPENDENCIES
------------
    pip install node2vec xgboost category_encoders scikit-learn networkx
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import glob
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

# Graph
import networkx as nx

# Node2Vec  (install: pip install node2vec)
# Uses gensim's Word2Vec under the hood; performs biased random walks on a graph
# and trains skip-gram embeddings over walk co-occurrences.
from node2vec import Node2Vec

# Gradient boosting
from xgboost import XGBRegressor

# Sklearn utilities
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Target encoding
import category_encoders as ce

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# SECTION 1 · COLUMN CONSTANTS
# ==============================================================================

# Matches the internal naming established in baseline_xgb.py
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

# Base tabular features (identical to baseline for fair comparison)
NUMERIC_FEATURES = [
    "osrm_predicted_eta",
    "distance_km",
    "osrm_distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_cutoff",
    "start_scan_to_end_scan",
]

# These are also present in the baseline, but in the hybrid we use the
# Node2Vec embedding vectors INSTEAD of raw target-encoded IDs.
# We still include them as fallback target-encoded scalars alongside the
# embeddings to give XGBoost maximum signal with minimal engineering burden.
HIGH_CARD_CAT_FEATURES = ["source_hub_id", "dest_hub_id"]
LOW_CARD_CAT_FEATURES  = ["route_type"]

TARGET_COL = "actual_eta"
ID_COL     = "trip_id"

# Node2Vec embedding dimension column prefix
SRC_EMB_PREFIX = "src_emb_"
DST_EMB_PREFIX = "dst_emb_"


# ==============================================================================
# SECTION 2 · DATA LOADING
# ==============================================================================

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, nx.MultiDiGraph]:
    from pathlib import Path
    data_dir = Path(data_dir)
    """
    Load the cleaned trip DataFrame and the pre-built logistics graph.

    The graph (graph.gpickle) is produced by 02_data_pipeline.ipynb.
    If it does not exist, we raise a clear error rather than silently
    constructing a degraded graph — the embeddings depend critically on the
    full topology computed in Phase 2.

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    Tuple[pd.DataFrame, nx.MultiDiGraph]
    """
    # ── Trips ─────────────────────────────────────────────────────────────────
    parquet_path = data_dir / "trips_clean.parquet"
    if parquet_path.exists():
        print(f"📂 Loading trips from : {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df = df.rename(columns={k: v for k, v in RAW_COL_MAP.items() if k in df.columns})
    else:
        shard_paths = sorted(glob.glob(str(data_dir / "data_part_*.csv")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No trip data found in {data_dir}.\n"
                "Run 01_traditional_eda.ipynb first to produce trips_clean.parquet."
            )
        print(f"📂 Loading {len(shard_paths)} raw CSV shard(s) …")
        DATE_COLS = ["trip_creation_time", "od_start_time", "od_end_time", "cutoff_timestamp"]
        parts = [pd.read_csv(p, parse_dates=DATE_COLS, low_memory=False) for p in shard_paths]
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={k: v for k, v in RAW_COL_MAP.items() if k in df.columns})

    print(f"   ✅ {len(df):,} trip rows loaded")

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph_path = data_dir / "graph.gpickle"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph artefact not found: {graph_path}\n"
            "Run 02_data_pipeline.ipynb first to produce graph.gpickle."
        )
    print(f"📂 Loading graph from : {graph_path}")
    with open(graph_path, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)
    print(f"   ✅ Graph loaded — {G.number_of_nodes():,} nodes | "
          f"{G.number_of_edges():,} edges | type={type(G).__name__}")

    return df, G


# ==============================================================================
# SECTION 3 · NODE2VEC EMBEDDING TRAINING
# ==============================================================================

def train_node2vec_embeddings(
    G:           nx.MultiDiGraph,
    embed_dim:   int = 64,
    walk_length: int = 30,
    num_walks:   int = 200,
    p:           float = 1.0,
    q:           float = 0.5,
    window:      int = 10,
    min_count:   int = 1,
    workers:     int = 4,
    epochs:      int = 5,
) -> pd.DataFrame:
    """
    Train Node2Vec embeddings on the logistics network graph.

    ALGORITHM OVERVIEW
    ──────────────────
    Node2Vec performs biased second-order random walks on the graph.  Two
    hyperparameters govern the walk strategy:

    • p (return parameter): probability of returning to the previously
      visited node. High p → depth-first walk → captures structural
      equivalence (hubs in the same structural role get similar embeddings).

    • q (in-out parameter): probability of exploring distant nodes vs.
      staying close. Low q → breadth-first walk → captures community
      membership (hubs in the same geographic cluster get similar embeddings).

    For logistics networks, q < 1 is preferred: we want hubs that serve the
    same city or corridor to share embedding space, so the downstream XGBoost
    can recognise "this source hub is in the same community as this destination".

    The walks are treated as sentences; hub IDs are words. Word2Vec (skip-gram)
    learns to predict context hubs from an anchor hub, producing dense floats.

    MULTIGRAPH HANDLING
    ───────────────────
    Node2Vec operates on simple weighted graphs, not multigraphs. We collapse
    the MultiDiGraph to a DiGraph by retaining the MINIMUM edge weight between
    any (u, v) pair. Minimum weight = lowest delay ratio = fastest typical path,
    which is the most structurally informative edge for embedding purposes.
    Using minimum weight rather than mean avoids the collapsed edge being
    dominated by high-congestion TOD slots.

    Parameters
    ----------
    G          : nx.MultiDiGraph — full logistics network from Phase 2.
    embed_dim  : int   — dimensionality of each hub's embedding vector.
    walk_length: int   — length of each random walk.
    num_walks  : int   — number of walks starting from each node.
    p          : float — return parameter.
    q          : float — in-out parameter (< 1 = community-biased).
    window     : int   — Word2Vec context window size.
    min_count  : int   — minimum occurrences to include a node in vocab.
    workers    : int   — parallel Word2Vec training threads.
    epochs     : int   — Word2Vec training epochs.

    Returns
    -------
    pd.DataFrame
        Index = hub_id.  Columns = [emb_0, emb_1, …, emb_{embed_dim-1}].
    """
    print(f"\n── Node2Vec Embedding Training ─────────────────────────────────")
    print(f"   embed_dim   = {embed_dim}")
    print(f"   walk_length = {walk_length}  |  num_walks = {num_walks}")
    print(f"   p={p}, q={q}  (q<1 → community-biased walks)")

    # ── Collapse MultiDiGraph → DiGraph ──────────────────────────────────────
    # Node2Vec's Python library expects a simple graph. We convert by choosing
    # the minimum edge_weight across all (route_type, TOD) strata for each
    # (u, v) pair. The collapsed edge_weight is stored as the "weight" attribute
    # that Node2Vec uses for transition probabilities during walks.
    print("\n   Collapsing MultiDiGraph → DiGraph (min edge_weight per corridor) …")
    G_simple = nx.DiGraph()
    G_simple.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        w = data.get("edge_weight", 1.0)
        if G_simple.has_edge(u, v):
            # Keep the minimum weight (fastest / least delayed corridor)
            if w < G_simple[u][v]["weight"]:
                G_simple[u][v]["weight"] = w
        else:
            G_simple.add_edge(u, v, weight=float(w))

    print(f"   Collapsed graph: {G_simple.number_of_nodes():,} nodes | "
          f"{G_simple.number_of_edges():,} edges")

    # ── Node2Vec walk + Word2Vec training ────────────────────────────────────
    # weight_key="weight" tells Node2Vec to use the collapsed edge_weight when
    # computing transition probabilities during random walks.
    node2vec_model = Node2Vec(
        graph=G_simple,
        dimensions=embed_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key="weight",
        workers=workers,
        seed=RANDOM_SEED,
        quiet=False,
    )

    print("\n   Training Word2Vec on random walks …")
    wv_model = node2vec_model.fit(
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )

    # ── Extract embedding matrix ──────────────────────────────────────────────
    hub_ids    = list(wv_model.wv.index_to_key)
    embeddings = np.array([wv_model.wv[hub] for hub in hub_ids])

    emb_cols = [f"emb_{i}" for i in range(embed_dim)]
    emb_df   = pd.DataFrame(embeddings, index=hub_ids, columns=emb_cols)
    emb_df.index.name = "hub_id"

    print(f"\n   ✅ Embeddings trained — {len(emb_df):,} hubs × {embed_dim} dims")
    print(f"   Embedding matrix memory : "
          f"{emb_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # ── Embedding quality: variance across dims (sanity check) ───────────────
    # Near-zero variance in any dimension → that dimension is not learning;
    # typically indicates too-short walks or too few walk epochs.
    dim_var = embeddings.var(axis=0)
    print(f"   Embedding dim variance — "
          f"min={dim_var.min():.4f}  mean={dim_var.mean():.4f}  "
          f"max={dim_var.max():.4f}")
    dead_dims = (dim_var < 1e-6).sum()
    if dead_dims > 0:
        print(f"   ⚠️  {dead_dims} near-zero-variance dimensions detected. "
              "Consider increasing num_walks or epochs.")

    return emb_df


# ==============================================================================
# SECTION 4 · MERGE EMBEDDINGS INTO TRIP DATAFRAME
# ==============================================================================

def merge_embeddings(
    df:     pd.DataFrame,
    emb_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach Node2Vec embedding vectors to each trip row.

    For every trip, we look up two embedding vectors:
      (a) the source hub's embedding  → prefixed `src_emb_0 … src_emb_{d-1}`
      (b) the destination hub's embedding → prefixed `dst_emb_0 … dst_emb_{d-1}`

    These 2×d columns are appended to the tabular features. XGBoost then learns
    which combinations of source and destination embeddings correlate with
    specific delay patterns — effectively learning structural route patterns
    without ever seeing the raw graph topology.

    HUBS NOT IN THE EMBEDDING VOCABULARY
    ──────────────────────────────────────
    Some hubs may appear in the trip data but not in the graph (e.g., hubs that
    only appeared in the test split, or hubs with no valid graph edges). These
    receive zero-vectors, which is a conservative fallback: XGBoost will treat
    them like "unknown structural role" rather than interpolating incorrectly.

    Parameters
    ----------
    df     : pd.DataFrame — cleaned trip DataFrame.
    emb_df : pd.DataFrame — hub_id index, emb_0 … emb_{d-1} columns.

    Returns
    -------
    pd.DataFrame
        Original df with 2×embed_dim new columns appended.
    """
    embed_dim = len(emb_df.columns)
    print(f"\n── Merging Embeddings into Trip DataFrame ──────────────────────")
    print(f"   embed_dim     = {embed_dim}")
    print(f"   trip rows     = {len(df):,}")
    print(f"   embedded hubs = {len(emb_df):,}")

    # Rename columns with prefix before merging to avoid collision
    src_emb = emb_df.copy().add_prefix(SRC_EMB_PREFIX)
    src_emb.index.name = "source_hub_id"

    dst_emb = emb_df.copy().add_prefix(DST_EMB_PREFIX)
    dst_emb.index.name = "dest_hub_id"

    # Left-join: trips with unknown hubs get NaN → filled with 0 below
    df = df.merge(src_emb.reset_index(), on="source_hub_id", how="left")
    df = df.merge(dst_emb.reset_index(), on="dest_hub_id",   how="left")

    # Fill unknown hub embeddings with zeros
    src_cols = [f"{SRC_EMB_PREFIX}emb_{i}" for i in range(embed_dim)]
    dst_cols = [f"{DST_EMB_PREFIX}emb_{i}" for i in range(embed_dim)]

    n_unknown_src = df[src_cols[0]].isnull().sum()
    n_unknown_dst = df[dst_cols[0]].isnull().sum()

    df[src_cols] = df[src_cols].fillna(0.0)
    df[dst_cols] = df[dst_cols].fillna(0.0)

    pct_src = n_unknown_src / len(df) * 100
    pct_dst = n_unknown_dst / len(df) * 100
    print(f"   Unknown src hubs (zero-filled) : {n_unknown_src:,}  ({pct_src:.1f}%)")
    print(f"   Unknown dst hubs (zero-filled) : {n_unknown_dst:,}  ({pct_dst:.1f}%)")

    if pct_src > 10 or pct_dst > 10:
        print("   ⚠️  >10% zero-filled embeddings — check graph.gpickle coverage.")

    return df


# ==============================================================================
# SECTION 5 · PREPROCESSING (mirrors baseline_xgb.py)
# ==============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning and temporal feature extraction as baseline_xgb.py.

    Kept identical to ensure that any MAE improvement in the hybrid model is
    attributable solely to the Node2Vec embeddings, not to different preprocessing.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame — cleaned, sorted, feature-engineered.
    """
    print("\n── Preprocessing ──────────────────────────────────────────────")
    n_raw = len(df)

    # ── Critical null removal ─────────────────────────────────────────────────
    critical_cols = [
        "source_hub_id", "dest_hub_id",
        TARGET_COL, "osrm_predicted_eta", "trip_creation_time"
    ]
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])

    # ── Domain-driven outlier removal (identical to EDA notebook) ─────────────
    LOWER_FACTOR = 0.2
    UPPER_FACTOR = 10.0
    MIN_TIME_MIN = 1.0
    MIN_DIST_KM  = 0.1

    if "delay_factor" in df.columns:
        df = df[(df["delay_factor"] >= LOWER_FACTOR) & (df["delay_factor"] <= UPPER_FACTOR)]
    df = df[df[TARGET_COL]           >= MIN_TIME_MIN]
    df = df[df["osrm_predicted_eta"] >= MIN_TIME_MIN]
    if "distance_km" in df.columns:
        df = df[df["distance_km"]    >= MIN_DIST_KM]

    print(f"   Rows after outlier removal : {len(df):,}  "
          f"({n_raw - len(df):,} dropped, "
          f"{(n_raw - len(df))/n_raw*100:.1f}%)")

    # ── Temporal features ─────────────────────────────────────────────────────
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"])
    df["hour_of_day"]  = df["trip_creation_time"].dt.hour.astype("int8")
    df["day_of_week"]  = df["trip_creation_time"].dt.dayofweek.astype("int8")
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype("int8")

    # ── Route type → integer ──────────────────────────────────────────────────
    route_map = {"FTL": 0, "Carting": 1}
    if "route_type" in df.columns:
        df["route_type"] = (
            df["route_type"].astype(str).map(route_map).fillna(-1).astype("int8")
        )

    if "is_cutoff" in df.columns:
        df["is_cutoff"] = df["is_cutoff"].astype("int8")

    if ID_COL not in df.columns:
        df[ID_COL] = df.index.astype(str)

    # ── Chronological sort ────────────────────────────────────────────────────
    df = df.sort_values("trip_creation_time").reset_index(drop=True)
    print(f"   Final rows : {len(df):,}")

    return df


# ==============================================================================
# SECTION 6 · CHRONOLOGICAL SPLIT
# ==============================================================================

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 chronological split. Identical logic to baseline_xgb.py to ensure
    the train and test windows match exactly for a fair model comparison.
    """
    split_idx  = int(len(df) * train_frac)
    split_date = df.iloc[split_idx]["trip_creation_time"]
    train_df   = df.iloc[:split_idx].copy()
    test_df    = df.iloc[split_idx:].copy()

    print(f"\n── Chronological Split (80 / 20) ───────────────────────────────")
    print(f"   Split timestamp : {split_date}")
    print(f"   Train rows      : {len(train_df):,}")
    print(f"   Test  rows      : {len(test_df):,}")
    return train_df, test_df


# ==============================================================================
# SECTION 7 · FEATURE MATRIX ASSEMBLY
# ==============================================================================

def build_feature_matrices(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    embed_dim: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.Series, pd.Series, ce.TargetEncoder]:
    """
    Build X_train / X_test for the hybrid model.

    Feature columns = tabular features (same as baseline)
                    + target-encoded source/dest IDs (scalar, 2 cols)
                    + Node2Vec source embedding  (embed_dim cols)
                    + Node2Vec destination embedding (embed_dim cols)

    The target encoder is still applied on top of the raw IDs as a scalar
    fallback — this gives XGBoost a direct hub-level ETA signal that
    complements the more holistic structural signal from the embedding.

    Leakage prevention (same constraint as baseline):
    • TargetEncoder fitted ONLY on training rows.
    • Zero-filled unknown embeddings are frozen from the Node2Vec training
      which itself ran on the training-split graph.
    """
    # ── Collect embedding column names ────────────────────────────────────────
    src_emb_cols = [f"{SRC_EMB_PREFIX}emb_{i}" for i in range(embed_dim)]
    dst_emb_cols = [f"{DST_EMB_PREFIX}emb_{i}" for i in range(embed_dim)]

    numeric_cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    cat_cols     = [c for c in HIGH_CARD_CAT_FEATURES if c in train_df.columns]
    low_cat_cols = [c for c in LOW_CARD_CAT_FEATURES if c in train_df.columns]
    emb_cols_present = [
        c for c in src_emb_cols + dst_emb_cols if c in train_df.columns
    ]

    all_feature_cols = numeric_cols + cat_cols + low_cat_cols + emb_cols_present

    print(f"\n── Feature Matrix Assembly ─────────────────────────────────────")
    print(f"   Tabular numeric      : {len(numeric_cols)} cols")
    print(f"   Target-encoded IDs   : {len(cat_cols)} cols")
    print(f"   Low-card categoricals: {len(low_cat_cols)} cols")
    print(f"   Embedding columns    : {len(emb_cols_present)} cols "
          f"({embed_dim} src + {embed_dim} dst)")
    print(f"   Total feature cols   : {len(all_feature_cols)}")

    X_train = train_df[all_feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test  = test_df[all_feature_cols].copy()
    y_test  = test_df[TARGET_COL].copy()

    train_ids = train_df[ID_COL].reset_index(drop=True)
    test_ids  = test_df[ID_COL].reset_index(drop=True)

    # ── Target encoding (fitted on training only) ─────────────────────────────
    if cat_cols:
        encoder = ce.TargetEncoder(
            cols=cat_cols,
            smoothing=1.0,
            handle_unknown="value",
            handle_missing="value",
        )
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        X_test[cat_cols]  = encoder.transform(X_test[cat_cols])
        print(f"   ✅ TargetEncoder fitted on {len(X_train):,} training rows")
    else:
        encoder = None

    # ── Fill residual NaNs ────────────────────────────────────────────────────
    fill_cols = numeric_cols + low_cat_cols
    for col in fill_cols:
        if X_train[col].isnull().any():
            fill_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col]  = X_test[col].fillna(fill_val)

    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test  shape : {X_test.shape}")

    return X_train, y_train, X_test, y_test, train_ids, test_ids, encoder


# ==============================================================================
# SECTION 8 · MODEL TRAINING
# ==============================================================================

def train_model(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    n_cv_splits: int = 5,
    n_iter:      int = 20,
) -> XGBRegressor:
    """
    Train a tuned XGBRegressor on the embedding-augmented feature set.

    The hyperparameter grid is identical to baseline_xgb.py. This is
    intentional: any MAE improvement comes from richer features, not
    from a luckier hyperparameter draw.

    We increase `colsample_bytree` range slightly to account for the
    much larger feature space (baseline ~10 cols vs. hybrid ~130+ cols).
    Subsampling features per tree is more important at this scale to
    prevent individual embedding dimensions from dominating splits.
    """
    print(f"\n── Hyperparameter Tuning (TimeSeriesSplit n={n_cv_splits}, "
          f"n_iter={n_iter}) ──")

    param_dist = {
        "n_estimators":     randint(300, 1000),
        "learning_rate":    uniform(0.01, 0.19),
        "max_depth":        randint(3, 9),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.3, 0.5),  # wider range for high-dim features
        "min_child_weight": randint(1, 10),
        "reg_lambda":       uniform(0.5, 2.5),
        "reg_alpha":        uniform(0.0, 1.0),
    }

    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae",
    )

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        refit=True,
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
# SECTION 9 · EVALUATION
# ==============================================================================

def evaluate(
    model:      XGBRegressor,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    test_ids:   pd.Series,
    output_dir: Path,
    embed_dim:  int,
) -> Dict[str, float]:
    """
    Evaluate the hybrid model, compare against OSRM baseline, and save outputs.

    Outputs
    -------
    node2vec_predictions.csv  — required by the multi-model benchmarking script.
    node2vec_metrics.json     — structured metric snapshot.
    """
    print("\n── Evaluation ──────────────────────────────────────────────────")

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    pct_error        = np.abs(y_pred - y_test.values) / y_test.values
    sla_within_15pct = (pct_error <= 0.15).mean() * 100
    mape             = pct_error.mean() * 100

    # OSRM reference on the same test rows
    osrm_col = "osrm_predicted_eta"
    if osrm_col in X_test.columns:
        osrm_mae  = mean_absolute_error(y_test, X_test[osrm_col])
        osrm_rmse = np.sqrt(mean_squared_error(y_test, X_test[osrm_col]))
        osrm_sla  = (
            (np.abs(X_test[osrm_col].values - y_test.values) / y_test.values <= 0.15).mean() * 100
        )
    else:
        osrm_mae = osrm_rmse = osrm_sla = float("nan")

    # ── Feature importance — top embedding dimensions ─────────────────────────
    feat_imp = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)

    emb_importance = feat_imp[
        feat_imp.index.str.startswith(SRC_EMB_PREFIX) |
        feat_imp.index.str.startswith(DST_EMB_PREFIX)
    ].sum()
    total_importance = feat_imp.sum()
    emb_share_pct = emb_importance / total_importance * 100

    print(f"\n{'='*60}")
    print(f"  NODE2VEC HYBRID MODEL EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  {'Metric':<32} {'Node2Vec XGB':>12}  {'OSRM Raw':>10}")
    print(f"  {'-'*56}")
    print(f"  {'MAE (min)':<32} {mae:>12.4f}  {osrm_mae:>10.4f}")
    print(f"  {'RMSE (min)':<32} {rmse:>12.4f}  {osrm_rmse:>10.4f}")
    print(f"  {'MAPE (%)':<32} {mape:>12.2f}  {'':>10}")
    print(f"  {'SLA ≤15% accuracy (%)':<32} {sla_within_15pct:>12.2f}  {osrm_sla:>10.2f}")
    print(f"{'='*60}")
    print(f"\n  📌 Embedding contribution to feature importance:")
    print(f"     {embed_dim*2} embedding dims account for {emb_share_pct:.1f}% of total split gain")
    print(f"\n  📌 Improvement over OSRM baseline:")
    print(f"     MAE  reduction : {(osrm_mae - mae) / osrm_mae * 100:+.1f}%")
    print(f"     SLA  gain      : {sla_within_15pct - osrm_sla:+.2f} pp")

    print(f"\n  Top 15 most important features:")
    for feat, imp in feat_imp.head(15).items():
        bar = "█" * int(imp / feat_imp.iloc[0] * 20)
        print(f"    {feat:<38} {imp:.5f}  {bar}")

    metrics = {
        "model":              "Node2Vec_XGBoost_Hybrid",
        "generated_at":       datetime.utcnow().isoformat(),
        "embed_dim":          embed_dim,
        "n_test_rows":        int(len(y_test)),
        "mae_min":            float(round(mae,  4)),
        "rmse_min":           float(round(rmse, 4)),
        "mape_pct":           float(round(mape, 4)),
        "sla_within_15pct":   float(round(sla_within_15pct, 4)),
        "embedding_feature_importance_pct": float(round(emb_share_pct, 4)),
        "osrm_baseline_mae":  float(round(osrm_mae,  4)),
        "osrm_baseline_rmse": float(round(osrm_rmse, 4)),
        "osrm_baseline_sla":  float(round(osrm_sla,  4)),
    }

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "trip_id":       test_ids.values,
        "actual_eta":    y_test.values,
        "predicted_eta": y_pred,
        "abs_pct_error": pct_error,
    })
    pred_path = output_dir / "node2vec_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n   ✅ Predictions saved → {pred_path}  ({len(pred_df):,} rows)")

    metrics_path = output_dir / "node2vec_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved    → {metrics_path}")

    return metrics


# ==============================================================================
# SECTION 10 · MAIN PIPELINE ORCHESTRATOR
# ==============================================================================

def run_node2vec_pipeline(
    data_dir:    Path,
    output_dir:  Path,
    embed_dim:   int   = 64,
    walk_length: int   = 30,
    num_walks:   int   = 200,
    p:           float = 1.0,
    q:           float = 0.5,
    window:      int   = 10,
    train_frac:  float = 0.80,
    n_cv_splits: int   = 5,
    n_iter:      int   = 20,
    workers:     int   = 4,
    epochs:      int   = 5,
) -> Dict[str, float]:
    """
    End-to-end orchestrator for the Node2Vec Hybrid pipeline.

    Stage order:
        load → preprocess → node2vec embeddings → merge embeddings
        → split → encode → train → evaluate
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 3 · STEP 2: NODE2VEC HYBRID ETA PIPELINE")
    print("=" * 60)

    # Stage 1 — Load trips + graph
    df_raw, G = load_data(data_dir)

    # Stage 2 — Clean + temporal features
    df_clean = preprocess_data(df_raw)

    # Stage 3 — Train Node2Vec embeddings on the logistics graph
    emb_df = train_node2vec_embeddings(
        G=G,
        embed_dim=embed_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        window=window,
        workers=workers,
        epochs=epochs,
    )

    # Save embeddings artefact (useful for Phase 4 route optimisation)
    emb_path = output_dir / "node2vec_embeddings.parquet"
    emb_df.reset_index().to_parquet(emb_path, index=False)
    print(f"\n   ✅ Embeddings saved → {emb_path}  "
          f"({len(emb_df):,} hubs × {embed_dim} dims)")

    # Stage 4 — Merge embeddings into trip rows
    df_merged = merge_embeddings(df_clean, emb_df)

    # Stage 5 — Chronological split
    train_df, test_df = chronological_split(df_merged, train_frac)

    # Stage 6 — Feature matrices + target encoding
    (X_train, y_train,
     X_test,  y_test,
     train_ids, test_ids,
     encoder) = build_feature_matrices(train_df, test_df, embed_dim)

    # Stage 7 — Train
    model = train_model(X_train, y_train, n_cv_splits, n_iter)

    # Stage 8 — Evaluate + save outputs
    metrics = evaluate(model, X_test, y_test, test_ids, output_dir, embed_dim)

    print("\n" + "=" * 60)
    print("  NODE2VEC HYBRID PIPELINE COMPLETE")
    print("  → node2vec_predictions.csv  ready for benchmarking script")
    print("  → node2vec_metrics.json     ready for Phase 3 comparison report")
    print("  → node2vec_embeddings.parquet  available for Phase 4")
    print("=" * 60)

    return metrics


# ==============================================================================
# SECTION 11 · CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 · Step 2 — Node2Vec Hybrid ETA Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir",  type=Path, default=Path("../data"))
    parser.add_argument("--embed_dim",   type=int,   default=64,
                        help="Dimensionality of Node2Vec embedding vectors")
    parser.add_argument("--walk_length", type=int,   default=30,
                        help="Length of each random walk")
    parser.add_argument("--num_walks",   type=int,   default=200,
                        help="Number of walks per node")
    parser.add_argument("--p",           type=float, default=1.0,
                        help="Node2Vec return parameter")
    parser.add_argument("--q",           type=float, default=0.5,
                        help="Node2Vec in-out parameter (< 1 = community bias)")
    parser.add_argument("--window",      type=int,   default=10,
                        help="Word2Vec context window size")
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--train_frac",  type=float, default=0.80)
    parser.add_argument("--n_cv_splits", type=int,   default=5)
    parser.add_argument("--n_iter",      type=int,   default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_node2vec_pipeline(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        embed_dim   = args.embed_dim,
        walk_length = args.walk_length,
        num_walks   = args.num_walks,
        p           = args.p,
        q           = args.q,
        window      = args.window,
        workers     = args.workers,
        epochs      = args.epochs,
        train_frac  = args.train_frac,
        n_cv_splits = args.n_cv_splits,
        n_iter      = args.n_iter,
    )
