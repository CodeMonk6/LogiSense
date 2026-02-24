"""
Signal Fusion Engine
=====================

Fuses four heterogeneous real-time signal sources into a unified risk
state tensor for each node in the supply network.

Fusion architecture
--------------------
1. Each source processor returns (N, D_source) features for the latest
   timestep (or (N, T, D_source) for temporal history).
2. A CrossSourceAttention module learns which signal sources matter for
   each node type (port vs supplier vs DC have different sensitivities).
3. A TemporalEncoder (3-layer Transformer) summarises the T-step history
   into a (N, d_model) context vector.
4. A lightweight head produces a quick pre-risk score per node before
   the full CausalDisruptionEngine runs.

Lead-time differentiation
--------------------------
Signal type         Predictive horizon
Satellite           1–5 days
Weather NWP         1–10 days
Geopolitics         5–30 days
Supplier sentiment  7–45 days   ← earliest warning

Dimensions
----------
D_signal = 16 + 12 + 24 + 32 = 84 (total signal features per node)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from logisense.signals.geopolitics import N_GEO_FEATURES, GeopoliticsProcessor
from logisense.signals.satellite import N_SATELLITE_FEATURES, SatelliteProcessor
from logisense.signals.sentiment import N_SENTIMENT_FEATURES, SentimentProcessor
from logisense.signals.weather import N_WEATHER_FEATURES, WeatherProcessor

logger = logging.getLogger(__name__)

SOURCE_DIMS = [
    N_SATELLITE_FEATURES,
    N_WEATHER_FEATURES,
    N_GEO_FEATURES,
    N_SENTIMENT_FEATURES,
]
SOURCE_NAMES = ["satellite", "weather", "geopolitics", "sentiment"]
D_SIGNAL = sum(SOURCE_DIMS)  # 84
SOURCE_SLICES = []
_start = 0
for _d in SOURCE_DIMS:
    SOURCE_SLICES.append((_start, _start + _d))
    _start += _d


# ─────────────────────────── data containers ──────────────────────────────


@dataclass
class NodeSignal:
    node_id: str
    timestamp: str
    satellite: np.ndarray  # (16,)
    weather: np.ndarray  # (12,)
    geopolitics: np.ndarray  # (24,)
    sentiment: np.ndarray  # (32,)
    fused_risk_score: float = 0.0
    confidence: float = 1.0

    @property
    def feature_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.satellite, self.weather, self.geopolitics, self.sentiment]
        )


@dataclass
class FusedSignalState:
    """
    Complete fused signal state across all supply chain nodes.

    Attributes:
        node_signals:   node_id → NodeSignal
        signal_tensor:  (N, T, D_signal) float tensor for model input
        risk_scores:    (N,) quick pre-risk scores
    """

    node_signals: Dict[str, NodeSignal]
    timestamp: str
    signal_tensor: torch.Tensor  # (N, T, D_signal)
    node_ids: List[str]
    risk_scores: np.ndarray  # (N,)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def top_risk_nodes(self, n: int = 10) -> List[Tuple[str, float]]:
        pairs = sorted(
            zip(self.node_ids, self.risk_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:n]


# ───────────────────────────── neural modules ─────────────────────────────


class CrossSourceAttention(nn.Module):
    """
    Learns per-node-type attention weights across signal sources.

    Ports are dominated by satellite + weather; supplier nodes by
    sentiment + geopolitics.  This module learns those weights end-to-end.
    """

    def __init__(self, n_node_types: int = 5, d_hidden: int = 64):
        super().__init__()
        self.source_projs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden))
                for d in SOURCE_DIMS
            ]
        )
        self.node_embed = nn.Embedding(n_node_types, d_hidden)
        self.attn = nn.MultiheadAttention(d_hidden, num_heads=4, batch_first=True)
        self.out_proj = nn.Linear(d_hidden, D_SIGNAL)

    def forward(
        self,
        signal: torch.Tensor,  # (N, D_signal)
        node_types: torch.Tensor,  # (N,) int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns fused (N, D_signal) and attention weights (N, n_sources)."""
        splits = torch.split(signal, SOURCE_DIMS, dim=-1)
        src_seq = torch.stack(
            [proj(s) for proj, s in zip(self.source_projs, splits)], dim=1
        )  # (N,4,d)
        query = self.node_embed(node_types).unsqueeze(1)  # (N,1,d)
        out, w = self.attn(query, src_seq, src_seq)
        return self.out_proj(out.squeeze(1)), w.squeeze(1)


class TemporalEncoder(nn.Module):
    """3-layer Transformer over the T-step signal history per node."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        T_max: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(D_SIGNAL)
        self.in_proj = nn.Linear(D_SIGNAL, d_model)
        self.pos_embed = nn.Embedding(T_max + 4, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, D_signal)  →  (N, d_model)"""
        T = x.shape[1]
        x = self.in_proj(self.norm(x))
        x = x + self.pos_embed(torch.arange(T, device=x.device))
        return self.out_norm(self.transformer(x)[:, -1, :])


# ─────────────────────────── main engine ──────────────────────────────────


class SignalFusionEngine(nn.Module):
    """
    Full signal fusion pipeline.

    For each node: ingest 4 sources → cross-source attention → temporal
    encoding → fused state tensor + pre-risk scores.

    Args:
        T_lookback: Timesteps of historical signals to encode.
        d_model:    Signal encoding dimension (fed to CausalEngine).
        device:     Compute device string.
    """

    def __init__(self, T_lookback: int = 30, d_model: int = 128, device: str = "cpu"):
        super().__init__()
        self.T_lookback = T_lookback
        self.d_model = d_model
        self._dev = torch.device(device)

        self.cross_attn = CrossSourceAttention()
        self.temporal_enc = TemporalEncoder(d_model=d_model, T_max=T_lookback)
        self.pre_risk_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        self._satellite = SatelliteProcessor()
        self._weather = WeatherProcessor()
        self._geo = GeopoliticsProcessor()
        self._sentiment = SentimentProcessor()

        self.to(self._dev)

    @torch.no_grad()
    def fetch_and_fuse(
        self,
        network_id: str,
        node_ids: Optional[List[str]] = None,
        node_types: Optional[Dict[str, int]] = None,
        mock: bool = True,
    ) -> FusedSignalState:
        """
        Fetch signals for all nodes and produce FusedSignalState.

        Args:
            network_id: Supply network identifier.
            node_ids:   Node IDs to fetch (None = generate 20 sample nodes).
            node_types: node_id → integer type (0–4).
            mock:       Use synthetic data (True for demo / testing).
        """
        from datetime import datetime, timezone

        self.eval()

        node_ids = node_ids or [f"node_{i:03d}" for i in range(20)]
        node_types = node_types or {nid: i % 5 for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(f"[LogiSense] Fetching signals — network={network_id}, nodes={N}")

        # Fetch each source → (N, D_source)
        sat = torch.tensor(self._satellite.fetch(node_ids, mock), device=self._dev)
        wea = torch.tensor(self._weather.fetch(node_ids, mock), device=self._dev)
        geo = torch.tensor(self._geo.fetch(node_ids, mock), device=self._dev)
        sen = torch.tensor(self._sentiment.fetch(node_ids, mock), device=self._dev)

        latest = torch.cat([sat, wea, geo, sen], dim=-1)  # (N, D_signal)

        # Build T-step history by adding mild temporal noise
        rng = torch.Generator(device=self._dev)
        hist = latest.unsqueeze(1).expand(N, self.T_lookback, D_SIGNAL).clone()
        hist = hist + 0.05 * torch.randn(
            N, self.T_lookback, D_SIGNAL, generator=rng, device=self._dev
        )
        hist = torch.clamp(hist, 0.0, 1.0)
        hist[:, -1, :] = latest  # last step = actual signal

        # Encode
        nt_tensor = torch.tensor(
            [node_types.get(nid, 0) for nid in node_ids],
            dtype=torch.long,
            device=self._dev,
        )
        fused, w = self.cross_attn(latest, nt_tensor)
        encoded = self.temporal_enc(hist)
        risk_scores = self.pre_risk_head(encoded).squeeze(-1).cpu().numpy()

        # Build NodeSignal objects
        node_signals = {}
        latest_np = latest.cpu().numpy()
        for i, nid in enumerate(node_ids):
            row = latest_np[i]
            node_signals[nid] = NodeSignal(
                node_id=nid,
                timestamp=timestamp,
                satellite=row[SOURCE_SLICES[0][0] : SOURCE_SLICES[0][1]],
                weather=row[SOURCE_SLICES[1][0] : SOURCE_SLICES[1][1]],
                geopolitics=row[SOURCE_SLICES[2][0] : SOURCE_SLICES[2][1]],
                sentiment=row[SOURCE_SLICES[3][0] : SOURCE_SLICES[3][1]],
                fused_risk_score=float(risk_scores[i]),
                confidence=float(w[i].max().item()),
            )

        logger.info(
            f"[LogiSense] Fusion complete — mean pre-risk {risk_scores.mean():.3f}"
        )

        return FusedSignalState(
            node_signals=node_signals,
            timestamp=timestamp,
            signal_tensor=hist.cpu(),
            node_ids=node_ids,
            risk_scores=risk_scores,
        )

    def save(self, path: str) -> None:
        import json

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "signal_fusion.pt")
        with open(p / "config.json", "w") as f:
            json.dump({"T_lookback": self.T_lookback, "d_model": self.d_model}, f)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "SignalFusionEngine":
        import json

        p = Path(path)
        with open(p / "config.json") as f:
            cfg = json.load(f)
        obj = cls(**cfg, device=device)
        obj.load_state_dict(torch.load(p / "signal_fusion.pt", map_location=device))
        return obj
