"""
gnn_graphsage.py
================
Phase 3 · Model 3: GraphSAGE Link-Level ETA Regression Engine
Delhivery Network Intelligence — Deep Learning Engine

─────────────────────────────────────────────────────────────────────────────
WHY THIS IS LINK REGRESSION, NOT NODE CLASSIFICATION
─────────────────────────────────────────────────────────────────────────────
Standard GNN tutorials predict a SINGLE label per node (e.g., "is this hub
congested?").  Our problem is fundamentally different: we are predicting a
continuous value (ETA in minutes) for a TRIP that travels along a DIRECTED
EDGE from a source hub to a destination hub.

This requires a "link-level" architecture with three distinct stages:

  Stage A — Node Encoder (SAGEConv layers)
    Each hub aggregates information from its neighbours and produces a
    rich, context-aware embedding vector that captures:
      • Its own historical congestion and dwell time
      • The operational state of its 1-hop and 2-hop neighbours
    After Stage A every node has learned "where it sits in the network".

  Stage B — Edge Representation Assembly
    For the specific edge we want to predict (src → dst), we:
      1. Extract the source-hub embedding from Stage A output
      2. Extract the destination-hub embedding from Stage A output
      3. Concatenate them with the raw edge features (osrm_time, distance,
         time_of_day, etc.)
    This gives a single vector that encodes BOTH endpoints AND the raw
    corridor context — none of which a standard node classifier has.

  Stage C — MLP Regression Head
    The concatenated vector passes through a small MLP that outputs one
    continuous number: the predicted actual_eta in minutes.

This three-stage design is the architectural core that makes GraphSAGE
superior to the baseline XGBoost model:  while XGBoost treated hubs as
independent categories, GraphSAGE learns that a hub adjacent to a known
chokepoint carries higher delay risk than an identical hub in a quiet region.

─────────────────────────────────────────────────────────────────────────────
INPUTS (PyTorch Geometric Data object)
─────────────────────────────────────────────────────────────────────────────
  data.x           : FloatTensor [N, node_in_dim]
                     Node feature matrix.  Columns (from Phase 1/2):
                       avg_congestion_factor, cutoff_rate,
                       avg_dwell_time_min, log_dwell_time,
                       unique_destinations, unique_sources,
                       outbound_trips, ftl_ratio, p75_delay_ratio,
                       weekend_congestion_delta
  data.edge_index  : LongTensor  [2, E]
                     Directed edge connectivity (row 0 = src, row 1 = dst).
  data.edge_attr   : FloatTensor [E, edge_in_dim]
                     Per-edge features (columns from Phase 1):
                       osrm_time, log_osrm_time, log_distance,
                       log_dwell, hour_of_day, day_of_week,
                       time_of_day_code, route_type_enc, is_cutoff_int
  data.y           : FloatTensor [E]
                     Target: actual_time (minutes) per edge/trip.
  data.train_mask  : BoolTensor  [E]   edges used for training loss
  data.val_mask    : BoolTensor  [E]   edges used for early stopping
  data.test_mask   : BoolTensor  [E]   held-out edges for final benchmark

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────
  graphsage_predictions.csv   actual_eta | predicted_eta | abs_error |
                              pct_error  | within_15pct
  training_curve.csv          epoch-by-epoch train_loss / val_loss
  Printed benchmark table     MAE, RMSE, SLA-15%

─────────────────────────────────────────────────────────────────────────────
DEPENDENCIES
─────────────────────────────────────────────────────────────────────────────
  pip install torch torch-geometric pandas numpy scikit-learn
  (For CUDA: follow https://pytorch.org/get-started/locally/ for your CUDA version)

AUTHOR : Lead Deep Learning Engineer
PROJECT: Delhivery ETA Optimisation — Phase 3, Deliverable 3
"""

# ============================================================
# SECTION 0 — IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import os
import math
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# ---------- PyTorch core ---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------- PyTorch Geometric ----------------------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError(
        "torch_geometric not found.  Install with:\n"
        "  pip install torch-geometric\n"
        "See https://pytorch-geometric.readthedocs.io for CUDA-specific builds."
    ) from exc

# ---------- Scikit-learn (metrics only) ------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Logging --------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------- Reproducibility ------------------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------- Device detection (GPU → CPU fallback) --------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Compute device: %s", DEVICE)

# ---------- Output directory -----------------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 1 — HYPERPARAMETER DATACLASS
# ============================================================

@dataclass
class GraphSAGEConfig:
    """
    Central hyperparameter store for the GraphSAGE ETA Predictor.

    Keeping all hyperparameters in one dataclass makes ablation studies
    trivial — swap one field and the entire pipeline adapts automatically.

    Architecture
    ------------
    node_in_dim    : Dimensionality of input node features (data.x width).
                     Default matches Phase-1 node_features.parquet columns.
    edge_in_dim    : Dimensionality of input edge features (data.edge_attr).
                     Default matches Phase-2 edge_weights.parquet attributes.
    hidden_dim     : Width of SAGEConv hidden layers and MLP hidden layers.
                     Larger values = richer representations, more parameters.
    n_sage_layers  : Number of SAGEConv message-passing rounds.
                     Each round expands the receptive field by 1 hop.
                     2 layers → model sees each node's 2-hop neighbourhood.
    mlp_hidden_dim : Width of the regression MLP hidden layers.
    dropout_rate   : Dropout applied after each SAGEConv and MLP layer.
                     Regularises against overfitting on dense hub clusters.

    Training
    --------
    lr             : Adam learning rate.
    weight_decay   : L2 regularisation coefficient in Adam.
    max_epochs     : Hard epoch ceiling (early stopping will fire sooner).
    patience       : Early stopping patience in epochs without val improvement.
    min_delta      : Minimum absolute improvement to reset patience counter.
    batch_size     : Mini-batch size for edge-level sampling (set to 0 for
                     full-batch training, which works when the graph fits RAM).
    """
    # ── Architecture ─────────────────────────────────────────────
    node_in_dim:    int   = 10     # Phase-1 hub features: 10 columns
    edge_in_dim:    int   = 9      # Phase-2 edge attrs:   9 columns
    hidden_dim:     int   = 128    # SAGEConv hidden width
    n_sage_layers:  int   = 2      # number of message-passing hops
    mlp_hidden_dim: int   = 64     # MLP regression head hidden width
    dropout_rate:   float = 0.3    # dropout probability

    # ── Training ─────────────────────────────────────────────────
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    max_epochs:     int   = 200
    patience:       int   = 15     # early stopping patience (epochs)
    min_delta:      float = 1e-4   # minimum val loss improvement

    # ── Misc ─────────────────────────────────────────────────────
    random_seed:    int   = RANDOM_SEED
    device:         str   = field(default_factory=lambda: str(DEVICE))


# ============================================================
# SECTION 2 — GRAPHSAGE ETA PREDICTOR ARCHITECTURE
# ============================================================

class GraphSAGE_ETA_Predictor(nn.Module):
    """
    Link-Level Regression GNN for logistics ETA prediction.

    ARCHITECTURE OVERVIEW
    ─────────────────────
                        ┌─────────────────────────────────────┐
                        │     INPUT: data.x  [N, node_in_dim] │
                        └──────────────┬──────────────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │    SAGEConv Layer 1         │
                         │    ReLU + Dropout           │
                         │    [N, node_in_dim]         │
                         │         → [N, hidden_dim]   │
                         └─────────────┬──────────────┘
                                       │ (optional 3rd layer here)
                         ┌─────────────▼──────────────┐
                         │    SAGEConv Layer 2         │
                         │    ReLU + Dropout           │
                         │    [N, hidden_dim]          │
                         │         → [N, hidden_dim]   │
                         └─────────────┬──────────────┘
                                       │
                    Node Embeddings H  [N, hidden_dim]
                                       │
               ┌───────────────────────┤
               │                       │
      H[src_nodes]              H[dst_nodes]      edge_attr
      [E, hidden_dim]         [E, hidden_dim]    [E, edge_in_dim]
               │                       │               │
               └───────────────────────┴───────────────┘
                                       │ CONCATENATE
                         ┌─────────────▼──────────────────────────┐
                         │  Edge vector [E, 2*hidden_dim+edge_dim] │
                         └─────────────┬──────────────────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │    MLP Layer 1              │
                         │    ReLU + Dropout           │
                         │         → [E, mlp_hidden]   │
                         └─────────────┬──────────────┘
                         ┌─────────────▼──────────────┐
                         │    MLP Layer 2              │
                         │    ReLU                     │
                         │         → [E, mlp_hidden//2]│
                         └─────────────┬──────────────┘
                         ┌─────────────▼──────────────┐
                         │    Output Layer             │
                         │    Linear → [E, 1]          │
                         │    → squeeze → [E]          │
                         └─────────────┬──────────────┘
                                       │
                         ETA predictions (minutes)  [E]

    Parameters
    ----------
    cfg : GraphSAGEConfig   Hyperparameter dataclass (see Section 1).
    """

    def __init__(self, cfg: GraphSAGEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Stage A: Node Encoder (SAGEConv layers) ──────────────
        # SAGEConv(in_channels, out_channels) implements the GraphSAGE
        # "SAmple and agGrEgate" operator:
        #
        #   h_v^(k) = W_1 · h_v^(k-1)
        #           + W_2 · MEAN( h_u^(k-1) for u in N(v) )
        #
        # where N(v) is the neighbour set of node v.
        #
        # After k layers every node embedding encodes information from
        # its k-hop neighbourhood.  With k=2, a hub's embedding reflects
        # not just its own features but also the state of its direct
        # corridor partners and THEIR partners — capturing delay
        # propagation across up to 2 hops.
        self.sage_layers = nn.ModuleList()
        in_ch = cfg.node_in_dim

        for layer_idx in range(cfg.n_sage_layers):
            out_ch = cfg.hidden_dim
            self.sage_layers.append(
                SAGEConv(in_channels=in_ch, out_channels=out_ch)
            )
            in_ch = out_ch   # output of layer k is input to layer k+1

        # ── Batch normalisation after each SAGEConv ──────────────
        # BatchNorm stabilises training by normalising activations per
        # feature across the node batch.  This is especially useful for
        # logistics graphs where hub feature scales vary enormously
        # (a mega-hub may have 10,000× more outbound trips than a
        # small last-mile depot).
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(cfg.hidden_dim)
            for _ in range(cfg.n_sage_layers)
        ])

        self.dropout = nn.Dropout(p=cfg.dropout_rate)

        # ── Stage C: MLP Regression Head ─────────────────────────
        # Input dimension to MLP:
        #   2 * hidden_dim   (source embedding + destination embedding)
        # + edge_in_dim      (raw edge features: osrm_time, distance, ...)
        mlp_in_dim = 2 * cfg.hidden_dim + cfg.edge_in_dim

        self.mlp = nn.Sequential(
            # ── MLP Layer 1 ──────────────────────────────────────
            nn.Linear(mlp_in_dim, cfg.mlp_hidden_dim),
            nn.BatchNorm1d(cfg.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate),

            # ── MLP Layer 2 ──────────────────────────────────────
            nn.Linear(cfg.mlp_hidden_dim, cfg.mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout_rate / 2),   # lighter dropout near output

            # ── Output Layer (single continuous ETA value) ───────
            nn.Linear(cfg.mlp_hidden_dim // 2, 1),
        )

        # Weight initialisation: Kaiming uniform for ReLU networks.
        # This prevents vanishing/exploding gradients at the start of
        # training, which is critical for stable GNN training.
        self._init_weights()

    # ─────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Apply Kaiming-uniform initialisation to all Linear layers and
        zero-initialise all biases.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ─────────────────────────────────────────────────────────────

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage A: Run all SAGEConv layers to produce node embeddings.

        Each SAGEConv layer performs one round of neighbourhood
        aggregation.  After this function, every node in the graph has
        a rich embedding that encodes its own features AND the aggregate
        state of its multi-hop neighbourhood.

        Parameters
        ----------
        x          : FloatTensor [N, node_in_dim]   Raw node features.
        edge_index : LongTensor  [2, E]             Graph connectivity.

        Returns
        -------
        FloatTensor [N, hidden_dim]
            Context-aware hub embeddings.
        """
        h = x
        for sage_conv, bn in zip(self.sage_layers, self.bn_layers):
            # Message passing: aggregate from neighbours
            h = sage_conv(h, edge_index)   # [N, hidden_dim]
            h = bn(h)                       # normalise activations
            h = F.relu(h)                   # non-linearity
            h = self.dropout(h)             # regularise
        return h   # [N, hidden_dim]

    # ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pred_edge_index: Optional[torch.Tensor] = None,
        pred_edge_attr:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: node encoding → edge assembly → MLP → ETA.

        ─────────────────────────────────────────────────────────────
        STAGE A — Node Encoder
        ─────────────────────────────────────────────────────────────
        Run SAGEConv layers on the FULL graph (all nodes, all edges).
        This is the "inductive" part of GraphSAGE: the model learns
        HOW to aggregate, not just WHAT to aggregate for fixed nodes.
        Result: H ∈ ℝ^[N × hidden_dim]

        ─────────────────────────────────────────────────────────────
        STAGE B — Edge Representation Assembly (THE CORE MECHANISM)
        ─────────────────────────────────────────────────────────────
        For each edge we want to predict:

          src_emb = H[ edge[0] ]   — embedding of the SOURCE hub
          dst_emb = H[ edge[1] ]   — embedding of the DESTINATION hub

        We then concatenate:
          edge_repr = [ src_emb ‖ dst_emb ‖ edge_attr ]

        This vector carries THREE complementary information sources:
          1. src_emb  : "What kind of hub am I leaving from?"
                        (congestion level, fan-out, structural role)
          2. dst_emb  : "What kind of hub am I arriving at?"
                        (inbound load, dwell time, capacity)
          3. edge_attr: "What is the raw corridor context?"
                        (OSRM estimate, distance, time of day)

        NO standard node-classification GNN does this — they only use
        individual node embeddings without the corridor context.

        ─────────────────────────────────────────────────────────────
        STAGE C — MLP Regression Head
        ─────────────────────────────────────────────────────────────
        The assembled edge_repr is passed through a small MLP that
        maps the high-dimensional representation to a single continuous
        ETA prediction in minutes.

        Parameters
        ----------
        x              : FloatTensor [N, node_in_dim]
                         Full-graph node feature matrix.
        edge_index     : LongTensor  [2, E_full]
                         Full-graph connectivity (used for message passing).
        edge_attr      : FloatTensor [E_full, edge_in_dim]
                         Full-graph edge features.
        pred_edge_index: LongTensor  [2, E_pred]  (optional)
                         Connectivity of edges TO PREDICT.
                         If None, uses edge_index (predict all edges).
        pred_edge_attr : FloatTensor [E_pred, edge_in_dim]  (optional)
                         Edge features of edges to predict.
                         If None, uses edge_attr.

        Returns
        -------
        FloatTensor [E_pred]
            Predicted actual_eta in minutes, one value per edge.
        """

        # ── STAGE A: Node Encoder ─────────────────────────────────
        # Use the FULL graph topology for message passing.
        # This ensures every node's embedding reflects the entire
        # network context, not just the training subset.
        node_embeddings = self.encode_nodes(x, edge_index)
        # node_embeddings : [N, hidden_dim]

        # ── Resolve which edges to predict ───────────────────────
        # During training/evaluation we may want to predict only a
        # SUBSET of edges (train_mask, val_mask, test_mask).
        # If no subset is specified, predict all edges.
        if pred_edge_index is None:
            pred_edge_index = edge_index   # [2, E_full]
        if pred_edge_attr is None:
            pred_edge_attr = edge_attr     # [E_full, edge_in_dim]

        # ── STAGE B: Edge Representation Assembly ─────────────────
        #
        # pred_edge_index[0] = indices of SOURCE nodes for each edge
        # pred_edge_index[1] = indices of DESTINATION nodes for each edge
        #
        # Indexing node_embeddings with these index tensors performs
        # a "lookup" that selects exactly the embedding row corresponding
        # to the source (or destination) hub for every edge.
        #
        # Example:
        #   edge 42: src=hub_7, dst=hub_23
        #   src_embeddings[42] = node_embeddings[7]  → hub_7's embedding
        #   dst_embeddings[42] = node_embeddings[23] → hub_23's embedding
        #
        src_node_idx = pred_edge_index[0]   # [E_pred] — source hub indices
        dst_node_idx = pred_edge_index[1]   # [E_pred] — destination hub indices

        src_embeddings = node_embeddings[src_node_idx]   # [E_pred, hidden_dim]
        dst_embeddings = node_embeddings[dst_node_idx]   # [E_pred, hidden_dim]

        # Concatenate along the feature axis (dim=1):
        #   [ src_emb | dst_emb | edge_attr ]
        #   [hidden_dim | hidden_dim | edge_in_dim]
        #   = [2*hidden_dim + edge_in_dim] per edge
        #
        # IMPORTANT: order matters for reproducibility — we always put
        # src before dst.  The MLP learns to interpret the position.
        edge_repr = torch.cat(
            [src_embeddings, dst_embeddings, pred_edge_attr],
            dim=1,
        )
        # edge_repr : [E_pred, 2*hidden_dim + edge_in_dim]

        # ── STAGE C: MLP Regression Head ─────────────────────────
        # The MLP maps the assembled edge representation to a single
        # continuous ETA prediction.
        #
        # We apply F.softplus to the output rather than raw linear to
        # ensure predictions are strictly positive (ETA cannot be ≤ 0).
        #
        #   softplus(x) = log(1 + exp(x))  ≈ ReLU but smooth at zero.
        eta_logit = self.mlp(edge_repr)          # [E_pred, 1]
        eta_pred  = F.softplus(eta_logit).squeeze(-1)  # [E_pred]
        # squeeze(-1): removes the trailing dimension-1, giving [E_pred]

        return eta_pred


# ============================================================
# SECTION 3 — EARLY STOPPING UTILITY
# ============================================================

class EarlyStopping:
    """
    Monitor validation loss and stop training when it stops improving.

    Logic
    -----
    After each epoch the validator calls `.step(val_loss, model)`.
    If val_loss improves by at least *min_delta*, the patience counter
    resets and the best model weights are saved.
    If patience epochs pass without improvement, `.stop` is set True.

    The saved best weights are used for final evaluation, not the
    weights from the last epoch (which may be overfitted).

    Parameters
    ----------
    patience  : int    Epochs without improvement before stopping.
    min_delta : float  Minimum absolute decrease to count as improvement.
    ckpt_path : Path   Where to save the best model weights (.pt file).
    """

    def __init__(
        self,
        patience:  int   = 15,
        min_delta: float = 1e-4,
        ckpt_path: Path  = OUTPUT_DIR / "best_model.pt",
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.ckpt_path  = ckpt_path
        self.best_loss  = math.inf
        self.counter    = 0
        self.stop       = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """
        Evaluate current epoch validation loss and update state.

        Parameters
        ----------
        val_loss : float       Current epoch's validation MAE.
        model    : nn.Module   Model whose weights to checkpoint.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Genuine improvement — save weights and reset counter
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.ckpt_path)
            log.debug("  EarlyStopping: improvement → best=%.4f", val_loss)
        else:
            self.counter += 1
            log.debug(
                "  EarlyStopping: no improvement (%d/%d)",
                self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.stop = True
                log.info(
                    "Early stopping triggered after %d epochs without "
                    "improvement.  Best val loss: %.4f",
                    self.patience, self.best_loss
                )


# ============================================================
# SECTION 4 — DATA HELPERS: MASK SUBSETTING
# ============================================================

def subset_edges_by_mask(
    data: Data,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the edge_index, edge_attr, and y tensors for the edges
    indicated by a boolean mask.

    This is the bridge between the full-graph Data object and the
    per-split (train / val / test) subsets we feed to the model.

    Parameters
    ----------
    data : torch_geometric.data.Data   Full graph data object.
    mask : BoolTensor [E]              True for edges in this split.

    Returns
    -------
    Tuple of:
      masked_edge_index : LongTensor  [2, E_mask]
      masked_edge_attr  : FloatTensor [E_mask, edge_in_dim]
      masked_y          : FloatTensor [E_mask]
    """
    masked_edge_index = data.edge_index[:, mask]   # select masked columns
    masked_edge_attr  = data.edge_attr[mask]        # select masked rows
    masked_y          = data.y[mask]                # select masked targets
    return masked_edge_index, masked_edge_attr, masked_y


# ============================================================
# SECTION 5 — TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model:     GraphSAGE_ETA_Predictor,
    data:      Data,
    optimizer: Adam,
    criterion: nn.L1Loss,
) -> float:
    """
    Run a single training epoch over all training edges.

    We use FULL-BATCH training on the training mask, which works
    efficiently when the graph fits in GPU memory (Delhivery's ~700-hub
    graph easily fits on any modern GPU).  For very large graphs,
    replace this with a NeighborLoader mini-batch approach.

    Loss function: L1Loss (Mean Absolute Error)
    ───────────────────────────────────────────
    We use MAE as the training loss (not MSE/L2) because:
      1. It is interpretable in the same units as the target (minutes).
      2. It is more robust to the heavy-tailed distribution of actual
         delivery times — extreme multi-day delays (due to festivals,
         floods) are real events but shouldn't dominate the loss surface.
      3. Our business SLA metric (predictions within 15% of actual) is
         linearly related to MAE, making MAE the natural training signal.

    Parameters
    ----------
    model     : GraphSAGE_ETA_Predictor   The GNN model.
    data      : Data                      Full-graph PyG Data object.
    optimizer : Adam                      Optimiser.
    criterion : nn.L1Loss                 MAE loss.

    Returns
    -------
    float   Mean training loss (MAE in minutes) for this epoch.
    """
    model.train()       # sets BatchNorm and Dropout to training mode
    optimizer.zero_grad()

    # Extract train-split edges
    train_edge_index, train_edge_attr, y_train = subset_edges_by_mask(
        data, data.train_mask
    )

    # Forward pass: compute predictions for training edges only
    # Full node embeddings are computed using the COMPLETE graph topology
    # (data.edge_index), but ETA predictions are made only for train edges.
    y_pred = model(
        x              = data.x,
        edge_index     = data.edge_index,   # full graph for message passing
        edge_attr      = data.edge_attr,
        pred_edge_index= train_edge_index,  # only predict train edges
        pred_edge_attr = train_edge_attr,
    )

    loss = criterion(y_pred, y_train)
    loss.backward()                          # compute gradients
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent explosions
    optimizer.step()

    return float(loss.item())


# ============================================================
# SECTION 6 — VALIDATION / TEST EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(
    model:    GraphSAGE_ETA_Predictor,
    data:     Data,
    mask:     torch.Tensor,
    criterion: nn.L1Loss,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a masked edge subset (validation or test).

    `@torch.no_grad()` disables gradient computation during inference,
    which (a) prevents accidental weight updates and (b) saves ~30%
    memory, allowing larger batches during evaluation.

    Parameters
    ----------
    model     : GraphSAGE_ETA_Predictor
    data      : Data         Full-graph PyG Data object.
    mask      : BoolTensor   Which edges to evaluate.
    criterion : nn.L1Loss    Loss for reporting.

    Returns
    -------
    Tuple of:
      loss      : float       MAE on the masked edges.
      y_true_np : np.ndarray  Ground truth actual_eta values.
      y_pred_np : np.ndarray  Model predictions.
    """
    model.eval()    # disables Dropout; BatchNorm uses running statistics

    eval_edge_index, eval_edge_attr, y_true = subset_edges_by_mask(data, mask)

    y_pred = model(
        x              = data.x,
        edge_index     = data.edge_index,
        edge_attr      = data.edge_attr,
        pred_edge_index= eval_edge_index,
        pred_edge_attr = eval_edge_attr,
    )

    loss = criterion(y_pred, y_true)

    # Detach from computation graph and move to CPU numpy
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    return float(loss.item()), y_true_np, y_pred_np


# ============================================================
# SECTION 7 — FULL TRAINING LOOP WITH EARLY STOPPING
# ============================================================

def train_model(
    model:  GraphSAGE_ETA_Predictor,
    data:   Data,
    cfg:    GraphSAGEConfig,
) -> Tuple[GraphSAGE_ETA_Predictor, List[Dict]]:
    """
    Full training loop with validation, early stopping, and LR scheduling.

    Training protocol
    -----------------
    1. Adam optimiser with weight decay (L2 regularisation).
    2. ReduceLROnPlateau scheduler: halves LR when val loss plateaus for
       5 epochs.  This lets the model initially take large learning steps
       and fine-tune near the optimum.
    3. EarlyStopping: saves the best model checkpoint and halts training
       if val loss does not improve for *patience* epochs.
    4. Best model weights are reloaded at the end — we never return an
       overfitted checkpoint.

    Parameters
    ----------
    model : GraphSAGE_ETA_Predictor   Freshly initialised model.
    data  : Data                      Full-graph PyG Data object.
    cfg   : GraphSAGEConfig           Hyperparameter config.

    Returns
    -------
    Tuple of:
      model          : GraphSAGE_ETA_Predictor  Best-checkpoint model.
      training_log   : List[Dict]               Per-epoch metrics.
    """
    criterion  = nn.L1Loss()     # Mean Absolute Error loss
    optimizer  = Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler  = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,        # halve LR on plateau
        patience=5,        # plateau = 5 epochs without val improvement
        min_lr=1e-6,       # floor to prevent LR collapsing to zero
    )
    ckpt_path  = OUTPUT_DIR / "best_graphsage_model.pt"
    stopper    = EarlyStopping(
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        ckpt_path=ckpt_path,
    )

    training_log: List[Dict] = []

    log.info(
        "Starting training  |  max_epochs=%d  patience=%d  lr=%.4f",
        cfg.max_epochs, cfg.patience, cfg.lr
    )
    log.info(
        "Model parameters: %d total  |  Device: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        cfg.device,
    )

    for epoch in range(1, cfg.max_epochs + 1):

        # ── Train ────────────────────────────────────────────────
        train_loss = train_one_epoch(model, data, optimizer, criterion)

        # ── Validate ─────────────────────────────────────────────
        val_loss, _, _ = evaluate(model, data, data.val_mask, criterion)

        # ── LR scheduling ────────────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ──────────────────────────────────────────────
        epoch_record = {
            "epoch"     : epoch,
            "train_loss": round(train_loss, 4),
            "val_loss"  : round(val_loss,   4),
            "lr"        : current_lr,
        }
        training_log.append(epoch_record)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  |  train_MAE=%.4f  val_MAE=%.4f  lr=%.2e",
                epoch, cfg.max_epochs, train_loss, val_loss, current_lr,
            )

        # ── Early stopping check ─────────────────────────────────
        stopper.step(val_loss, model)
        if stopper.stop:
            log.info("Training halted at epoch %d.", epoch)
            break

    # ── Reload best checkpoint ───────────────────────────────────
    # This is critical: we always evaluate on the BEST-performing
    # checkpoint, not the final weights (which may be slightly overfit).
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        log.info(
            "Best checkpoint loaded  |  best val_MAE=%.4f",
            stopper.best_loss
        )

    return model, training_log


# ============================================================
# SECTION 8 — METRICS COMPUTATION
# ============================================================

def compute_sla_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.15,
) -> float:
    """
    Compute the percentage of predictions within *threshold* relative
    error of the true value.

    Business KPI — SLA-15%:
      "What fraction of ETA predictions are within ±15% of actual?"

      Formula: |y_pred - y_true| / y_true ≤ 0.15

    A prediction landing within 15% of actual is considered "on-time"
    in Delhivery's SLA framework.  Maximising this metric directly
    translates to reducing customer complaints and revenue-at-risk.

    Parameters
    ----------
    y_true    : np.ndarray   Ground truth actual_time (minutes).
    y_pred    : np.ndarray   Predicted actual_time (minutes).
    threshold : float        Relative error tolerance (0.15 = ±15%).

    Returns
    -------
    float   Percentage of predictions within threshold (0–100).
    """
    rel_errors = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-6, None)
    return float((rel_errors <= threshold).mean() * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "GraphSAGE ETA Predictor",
) -> Dict:
    """
    Compute MAE, RMSE, and SLA-15% and log a formatted summary.

    Parameters
    ----------
    y_true     : np.ndarray   Ground truth actual_time (minutes).
    y_pred     : np.ndarray   Model predictions.
    model_name : str          Label for the printed summary.

    Returns
    -------
    Dict with keys: model_name, mae, rmse, sla_15_pct.
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sla  = compute_sla_pct(y_true, y_pred)

    separator = "=" * 60
    log.info(
        "\n%s\n  %s\n%s\n"
        "  %-35s  %8.4f min\n"
        "  %-35s  %8.4f min\n"
        "  %-35s  %8.2f %%\n"
        "%s",
        separator, model_name, separator,
        "Mean Absolute Error (MAE):",       mae,
        "Root Mean Squared Error (RMSE):",  rmse,
        "SLA-15%% (within ±15%% of actual):", sla,
        separator,
    )

    return {
        "model_name" : model_name,
        "mae"        : round(mae,  4),
        "rmse"       : round(rmse, 4),
        "sla_15_pct" : round(sla,  2),
    }


# ============================================================
# SECTION 9 — OUTPUT GENERATION
# ============================================================

def save_predictions(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    """
    Build and save the predictions DataFrame for Phase-3 benchmarking.

    Output columns
    --------------
    actual_eta    : true actual_time (minutes)
    predicted_eta : GraphSAGE prediction (minutes)
    abs_error     : |predicted - actual| (minutes)
    pct_error     : |predicted - actual| / actual × 100
    within_15pct  : 1 if pct_error ≤ 15, else 0

    Parameters
    ----------
    y_true      : np.ndarray   Ground truth targets.
    y_pred      : np.ndarray   Model predictions.
    output_path : Path         CSV output file path.

    Returns
    -------
    pd.DataFrame   The predictions table (also saved to CSV).
    """
    df_out = pd.DataFrame({
        "actual_eta"    : y_true,
        "predicted_eta" : y_pred,
    })
    df_out["abs_error"]    = np.abs(df_out["predicted_eta"] - df_out["actual_eta"])
    df_out["pct_error"]    = (
        df_out["abs_error"] / np.clip(df_out["actual_eta"].abs(), 1e-6, None) * 100
    )
    df_out["within_15pct"] = (df_out["pct_error"] <= 15.0).astype("int8")

    df_out.to_csv(output_path, index=False)
    log.info(
        "Predictions saved → %s  (%d rows, %d within 15%%)",
        output_path, len(df_out), df_out["within_15pct"].sum()
    )
    return df_out


def save_training_curve(
    training_log: List[Dict],
    output_path:  Path,
) -> None:
    """
    Save the per-epoch training and validation loss history to CSV.

    This file is used in the Phase-3 benchmarking notebook to plot
    learning curves and diagnose overfitting.

    Parameters
    ----------
    training_log : List[Dict]   Output of train_model().
    output_path  : Path         CSV output file path.
    """
    pd.DataFrame(training_log).to_csv(output_path, index=False)
    log.info("Training curve saved → %s  (%d epochs)", output_path, len(training_log))


# ============================================================
# SECTION 10 — PyG DATA OBJECT BUILDER (for standalone testing)
# ============================================================

def build_pyg_data_from_parquets(
    node_features_path: Path,
    edge_weights_path:  Path,
    trips_clean_path:   Path,
    val_fraction:       float = 0.10,
    test_fraction:      float = 0.10,
) -> Data:
    """
    Construct a PyTorch Geometric Data object from the Phase-1/2 parquet
    artefacts.

    This function makes the script fully self-contained: it can be run
    directly from Phase-1/2 outputs without any intermediate notebooks.
    If you already have a Data object, pass it directly to run_graphsage_pipeline.

    Node feature matrix (data.x)
    ─────────────────────────────
    Source: node_features.parquet (built in 02_data_pipeline.ipynb).
    Hub IDs are mapped to contiguous integer indices 0…N-1, which
    PyTorch Geometric requires as node identifiers.

    Edge feature matrix (data.edge_attr) and target (data.y)
    ─────────────────────────────────────────────────────────
    We derive per-trip edges from the cleaned trip log (trips_clean.parquet)
    rather than the aggregate edge_weights.parquet, because each trip is
    a separate prediction target.  The aggregate parquet averages away
    the variance we need for training.

    Edge connectivity (data.edge_index)
    ─────────────────────────────────────
    Directed: row 0 = source hub index, row 1 = destination hub index.

    Train/Val/Test split
    ─────────────────────
    We respect Delhivery's upstream 'data' column ('training' / 'test').
    Validation is carved from the training split by time: the most recent
    val_fraction of training trips become validation edges.

    Parameters
    ----------
    node_features_path : Path   node_features.parquet from Phase 1.
    edge_weights_path  : Path   edge_weights.parquet  from Phase 1.
    trips_clean_path   : Path   trips_clean.parquet   from Phase 1 EDA.
    val_fraction       : float  Fraction of training edges → validation.
    test_fraction      : float  Fraction for test (from Delhivery's split).

    Returns
    -------
    torch_geometric.data.Data   Fully attributed PyG Data object.
    """
    log.info("Building PyG Data object from parquet artefacts …")

    # ── Load artefacts ────────────────────────────────────────────
    node_df  = pd.read_parquet(node_features_path)
    trips_df = pd.read_parquet(trips_clean_path)

    # ── Hub → integer index mapping ───────────────────────────────
    all_hubs    = sorted(
        set(trips_df["source_center"]) | set(trips_df["destination_center"])
    )
    hub_to_idx  = {hub: i for i, hub in enumerate(all_hubs)}
    n_nodes     = len(hub_to_idx)
    log.info("Total unique hubs (nodes): %d", n_nodes)

    # ── Node feature matrix (x) ───────────────────────────────────
    NODE_FEAT_COLS = [
        "avg_congestion_factor", "cutoff_rate", "avg_dwell_time_min",
        "log_dwell_time", "unique_destinations", "unique_sources",
        "outbound_trips", "ftl_ratio", "p75_delay_ratio",
        "weekend_congestion_delta",
    ]
    # Align node_df to hub_to_idx ordering; fill unknown hubs with 0
    if "hub_id" not in node_df.columns:
        node_df = node_df.reset_index()     # hub_id may be the index

    x_df = (
        pd.DataFrame({"hub_id": all_hubs})
        .merge(node_df[["hub_id"] + NODE_FEAT_COLS], on="hub_id", how="left")
        .fillna(0.0)
    )
    x_tensor = torch.tensor(
        x_df[NODE_FEAT_COLS].values, dtype=torch.float32
    )   # [N, node_in_dim]

    # ── Edge connectivity and features ────────────────────────────
    # Feature engineering: mirror Phase-1 EDA derived columns
    trips_df = trips_df.copy()
    trips_df["hour_of_day"]    = pd.to_datetime(
        trips_df["trip_creation_time"]
    ).dt.hour.astype("int8")
    trips_df["day_of_week"]    = pd.to_datetime(
        trips_df["trip_creation_time"]
    ).dt.dayofweek.astype("int8")
    tod_bins   = [0, 6, 10, 16, 20, 24]
    tod_codes  = [0, 1, 2, 3, 4]
    trips_df["time_of_day_code"] = pd.cut(
        trips_df["hour_of_day"], bins=tod_bins, labels=tod_codes,
        right=False, include_lowest=True,
    ).astype("int8")
    trips_df["route_type_enc"]  = (trips_df["route_type"] == "FTL").astype("int8")
    trips_df["is_cutoff_int"]   = trips_df["is_cutoff"].astype("int8")
    trips_df["log_osrm_time"]   = np.log1p(trips_df["osrm_time"].clip(lower=0))
    trips_df["log_distance"]    = np.log1p(
        trips_df["actual_distance_to_destination"].clip(lower=0)
    )
    trips_df["log_dwell"]       = np.log1p(
        trips_df["start_scan_to_end_scan"].clip(lower=0)
    )

    EDGE_FEAT_COLS = [
        "osrm_time", "log_osrm_time", "log_distance",
        "log_dwell", "hour_of_day", "day_of_week",
        "time_of_day_code", "route_type_enc", "is_cutoff_int",
    ]

    # Map hub IDs → integer indices for edge_index
    valid_mask = (
        trips_df["source_center"].isin(hub_to_idx) &
        trips_df["destination_center"].isin(hub_to_idx)
    )
    trips_df = trips_df[valid_mask].copy()

    src_indices = trips_df["source_center"].map(hub_to_idx).values
    dst_indices = trips_df["destination_center"].map(hub_to_idx).values

    edge_index_tensor = torch.tensor(
        np.stack([src_indices, dst_indices], axis=0),
        dtype=torch.long,
    )   # [2, E]

    edge_attr_tensor = torch.tensor(
        trips_df[EDGE_FEAT_COLS].values,
        dtype=torch.float32,
    )   # [E, edge_in_dim]

    y_tensor = torch.tensor(
        trips_df["actual_time"].values,
        dtype=torch.float32,
    )   # [E]

    n_edges = len(trips_df)
    log.info("Total edges (trips): %d", n_edges)

    # ── Train / Val / Test masks ──────────────────────────────────
    # Use Delhivery's own 'data' column for train/test designation.
    # Carve validation from the tail of the training split.
    train_orig = (trips_df["data"] == "training").values
    test_mask  = (~train_orig)

    train_indices = np.where(train_orig)[0]
    n_val         = max(1, int(len(train_indices) * val_fraction))
    # Sort by trip_creation_time to make the validation set the most
    # recent trips — this mirrors real-world temporal deployment.
    sorted_train_idx = train_indices[
        trips_df.iloc[train_indices]["trip_creation_time"]
        .argsort().values
    ]
    val_set   = set(sorted_train_idx[-n_val:])
    train_set = set(sorted_train_idx[:-n_val])

    train_mask_arr = np.array([i in train_set for i in range(n_edges)])
    val_mask_arr   = np.array([i in val_set   for i in range(n_edges)])

    train_mask_tensor = torch.tensor(train_mask_arr, dtype=torch.bool)
    val_mask_tensor   = torch.tensor(val_mask_arr,   dtype=torch.bool)
    test_mask_tensor  = torch.tensor(test_mask,      dtype=torch.bool)

    log.info(
        "Split sizes → train: %d  val: %d  test: %d",
        train_mask_tensor.sum().item(),
        val_mask_tensor.sum().item(),
        test_mask_tensor.sum().item(),
    )

    # ── Assemble PyG Data object ──────────────────────────────────
    data = Data(
        x          = x_tensor,
        edge_index = edge_index_tensor,
        edge_attr  = edge_attr_tensor,
        y          = y_tensor,
        train_mask = train_mask_tensor,
        val_mask   = val_mask_tensor,
        test_mask  = test_mask_tensor,
    )
    data.hub_to_idx = hub_to_idx    # store mapping for downstream use

    log.info(
        "PyG Data object ready:\n"
        "  x         : %s\n"
        "  edge_index: %s\n"
        "  edge_attr : %s\n"
        "  y         : %s",
        tuple(data.x.shape),
        tuple(data.edge_index.shape),
        tuple(data.edge_attr.shape),
        tuple(data.y.shape),
    )
    return data


# ============================================================
# SECTION 11 — MAIN PIPELINE ORCHESTRATOR
# ============================================================

def run_graphsage_pipeline(
    data: Data,
    cfg:  Optional[GraphSAGEConfig] = None,
    baseline_results: Optional[Dict] = None,
    hybrid_results:   Optional[Dict] = None,
) -> Dict:
    """
    End-to-end GraphSAGE ETA prediction pipeline.

    Execution order
    ───────────────
    1. Validate that data object has required attributes
    2. Move data to compute device
    3. Auto-detect feature dimensions from data tensors
    4. Instantiate GraphSAGE_ETA_Predictor with correct dims
    5. Train model with early stopping
    6. Evaluate on test set
    7. Save predictions CSV and training curve CSV
    8. Print final benchmark comparison table

    Parameters
    ----------
    data             : torch_geometric.data.Data
                       Fully attributed PyG Data object.  Must contain:
                       x, edge_index, edge_attr, y,
                       train_mask, val_mask, test_mask.
    cfg              : GraphSAGEConfig | None
                       Hyperparameter config.  If None, auto-populated from
                       data tensor shapes.
    baseline_results : Dict | None
                       evaluate_model output from Phase-3 baseline XGBoost.
    hybrid_results   : Dict | None
                       evaluate_model output from Phase-3 hybrid Node2Vec.

    Returns
    -------
    Dict   Final test metrics: mae, rmse, sla_15_pct.
    """

    # ── Step 1: Validate data object ─────────────────────────────
    required_attrs = [
        "x", "edge_index", "edge_attr", "y",
        "train_mask", "val_mask", "test_mask",
    ]
    missing = [a for a in required_attrs if not hasattr(data, a)]
    if missing:
        raise ValueError(
            f"PyG Data object missing required attributes: {missing}\n"
            "Ensure your data pipeline assigns all required tensors."
        )

    # ── Step 2: Move data to compute device ──────────────────────
    data = data.to(DEVICE)
    log.info("Data moved to device: %s", DEVICE)

    # ── Step 3: Auto-detect feature dimensions ───────────────────
    node_in_dim = data.x.shape[1]
    edge_in_dim = data.edge_attr.shape[1]
    log.info(
        "Auto-detected dims →  node_in=%d  edge_in=%d",
        node_in_dim, edge_in_dim
    )

    # ── Step 4: Instantiate model ─────────────────────────────────
    if cfg is None:
        cfg = GraphSAGEConfig(
            node_in_dim = node_in_dim,
            edge_in_dim = edge_in_dim,
        )
    else:
        # Override dims from data regardless of what cfg says
        cfg.node_in_dim = node_in_dim
        cfg.edge_in_dim = edge_in_dim
    cfg.device = str(DEVICE)

    model = GraphSAGE_ETA_Predictor(cfg).to(DEVICE)
    log.info(
        "Model instantiated:\n%s\n"
        "Trainable parameters: %d",
        model,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ── Step 5: Train ─────────────────────────────────────────────
    log.info("=" * 55)
    log.info("TRAINING: GraphSAGE Link-Level ETA Predictor")
    log.info("=" * 55)
    model, training_log = train_model(model, data, cfg)

    # ── Step 6: Test evaluation ───────────────────────────────────
    log.info("=" * 55)
    log.info("EVALUATING on held-out test edges …")
    log.info("=" * 55)
    _, y_true_np, y_pred_np = evaluate(
        model, data, data.test_mask,
        criterion=nn.L1Loss()
    )

    metrics = compute_all_metrics(
        y_true_np, y_pred_np,
        model_name="GraphSAGE ETA Predictor (Link Regression)"
    )

    # ── Step 7: Save outputs ──────────────────────────────────────
    pred_path  = OUTPUT_DIR / "graphsage_predictions.csv"
    curve_path = OUTPUT_DIR / "training_curve.csv"

    save_predictions(y_true_np, y_pred_np, pred_path)
    save_training_curve(training_log, curve_path)

    # ── Step 8: Benchmark comparison table ───────────────────────
    _print_benchmark_table(baseline_results, hybrid_results, metrics)

    return metrics


# ============================================================
# SECTION 12 — BENCHMARK COMPARISON TABLE
# ============================================================

def _print_benchmark_table(
    baseline_results: Optional[Dict],
    hybrid_results:   Optional[Dict],
    graphsage_results: Dict,
) -> None:
    """
    Print a formatted three-way comparison of all Phase-3 models.

    Parameters
    ----------
    baseline_results  : Dict | None   Phase-3 baseline XGBoost metrics.
    hybrid_results    : Dict | None   Phase-3 hybrid Node2Vec metrics.
    graphsage_results : Dict          This model's metrics.
    """
    SEP  = "=" * 80
    SEP2 = "-" * 80

    print(f"\n{SEP}")
    print("  PHASE 3 — FINAL BENCHMARK REPORT: All Three Models")
    print(SEP)
    print(f"  {'Metric':<35} {'Baseline':>12} {'Hybrid N2V':>12} {'GraphSAGE':>12}")
    print(SEP2)

    METRICS = [
        ("MAE (minutes)",              "mae",        True),
        ("RMSE (minutes)",             "rmse",       True),
        ("SLA-15% (% within ±15%)",    "sla_15_pct", False),
    ]

    def _fmt(d: Optional[Dict], key: str) -> str:
        if d is None:
            return "  N/A"
        v = d.get(key, None)
        if v is None:
            return "  N/A"
        return f"{v:>12}"

    for label, key, lower_better in METRICS:
        b = _fmt(baseline_results,  key)
        h = _fmt(hybrid_results,    key)
        g = _fmt(graphsage_results, key)
        print(f"  {label:<35} {b} {h} {g}")

    print(SEP)

    # Verdict — did GraphSAGE beat the baseline?
    if baseline_results:
        b_mae = baseline_results.get("mae", None)
        g_mae = graphsage_results["mae"]
        b_sla = baseline_results.get("sla_15_pct", None)
        g_sla = graphsage_results["sla_15_pct"]

        if b_mae and g_mae < b_mae:
            pct_improve_mae = (b_mae - g_mae) / b_mae * 100
            print(
                f"  ✅ GraphSAGE BEATS baseline by {pct_improve_mae:.1f}%% MAE reduction"
            )
        else:
            print("  ⚠️  GraphSAGE did NOT outperform baseline on MAE")

        if b_sla and g_sla > b_sla:
            print(
                f"  ✅ GraphSAGE SLA-15%% improved by "
                f"{g_sla - b_sla:.2f} percentage points"
            )
        else:
            print("  ⚠️  GraphSAGE did NOT outperform baseline on SLA-15%%")
    print(f"{SEP}\n")


# ============================================================
# ENTRY POINT — run as a standalone script
# ============================================================

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description=(
            "GraphSAGE Link-Level ETA Regression Engine — "
            "Phase 3, Deliverable 3"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data",
        help="Directory containing trips_clean.parquet, "
             "node_features.parquet, edge_weights.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Where to write graphsage_predictions.csv and training_curve.csv.",
    )
    parser.add_argument(
        "--pyg-data-pickle",
        type=Path,
        default=None,
        help="Optional: path to a pre-built PyG Data object (.pt or .pkl). "
             "If provided, skips data construction entirely.",
    )
    parser.add_argument(
        "--hidden-dim",    type=int,   default=128,  help="SAGEConv hidden width."
    )
    parser.add_argument(
        "--n-sage-layers", type=int,   default=2,    help="Number of SAGEConv layers."
    )
    parser.add_argument(
        "--dropout",       type=float, default=0.3,  help="Dropout probability."
    )
    parser.add_argument(
        "--lr",            type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument(
        "--max-epochs",    type=int,   default=200,  help="Max training epochs."
    )
    parser.add_argument(
        "--patience",      type=int,   default=15,   help="Early stopping patience."
    )
    args = parser.parse_args()

    # Override global output dir from args
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or build PyG Data object ────────────────────────────
    if args.pyg_data_pickle and args.pyg_data_pickle.exists():
        log.info("Loading pre-built PyG Data from %s", args.pyg_data_pickle)
        if args.pyg_data_pickle.suffix == ".pt":
            pyg_data = torch.load(args.pyg_data_pickle, map_location=DEVICE)
        else:
            with open(args.pyg_data_pickle, "rb") as fh:
                pyg_data = pickle.load(fh)
    else:
        # Build from Phase-1/2 parquet artefacts
        node_feat_path = args.data_dir / "node_features.parquet"
        edge_wt_path   = args.data_dir / "edge_weights.parquet"
        trips_path     = args.data_dir / "trips_clean.parquet"

        for p in [node_feat_path, trips_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Required artefact not found: {p}\n"
                    "Run notebooks 01_traditional_eda.ipynb and "
                    "02_data_pipeline.ipynb first."
                )

        pyg_data = build_pyg_data_from_parquets(
            node_features_path = node_feat_path,
            edge_weights_path  = edge_wt_path,
            trips_clean_path   = trips_path,
        )

    # ── Build config from CLI args ────────────────────────────────
    cfg = GraphSAGEConfig(
        hidden_dim    = args.hidden_dim,
        n_sage_layers = args.n_sage_layers,
        dropout_rate  = args.dropout,
        lr            = args.lr,
        max_epochs    = args.max_epochs,
        patience      = args.patience,
    )

    # ── Run pipeline ──────────────────────────────────────────────
    # Pass baseline_results and hybrid_results when available
    # (load from CSV or pass dict directly after running the other models)
    final_metrics = run_graphsage_pipeline(
        data             = pyg_data,
        cfg              = cfg,
        baseline_results = None,   # replace with baseline dict when available
        hybrid_results   = None,   # replace with hybrid dict when available
    )

    print("\nGraphSAGE pipeline complete.")
    print(f"  MAE      : {final_metrics['mae']:.4f} min")
    print(f"  RMSE     : {final_metrics['rmse']:.4f} min")
    print(f"  SLA-15%  : {final_metrics['sla_15_pct']:.2f}%")
