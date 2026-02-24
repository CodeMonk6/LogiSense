"""
Satellite / AIS Signal Processor
==================================

Processes Automatic Identification System (AIS) vessel-tracking data and
satellite imagery to produce per-node lane congestion and closure features.

Signal sources
--------------
- AIS streaming feeds (AISstream.io, ExactEarth, Spire Maritime)
- Satellite SAR imagery for vessel detection in AIS-denied areas
- Port authority queue data

Features produced (16 per node)
---------------------------------
vessel_density_norm, speed_zscore, port_queue_anomaly,
lane_utilization_rate, diversion_rate, anchor_count_delta,
avg_waiting_hours, lane_closure_indicator, ais_signal_gap_rate,
cargo_fraction, tanker_fraction, container_fraction,
bulk_fraction, night_activity_index, weather_delay_proxy,
satellite_coverage_quality

Key lanes monitored
--------------------
Suez Canal, Strait of Hormuz, Malacca Strait, Panama Canal,
Taiwan Strait, Danish Straits, Cape of Good Hope, English Channel
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

N_SATELLITE_FEATURES = 16

FEATURE_NAMES = [
    "vessel_density_norm",
    "speed_zscore",
    "port_queue_anomaly",
    "lane_utilization_rate",
    "diversion_rate",
    "anchor_count_delta",
    "avg_waiting_hours",
    "lane_closure_indicator",
    "ais_signal_gap_rate",
    "cargo_fraction",
    "tanker_fraction",
    "container_fraction",
    "bulk_fraction",
    "night_activity_index",
    "weather_delay_proxy",
    "satellite_coverage_quality",
]


@dataclass
class LaneCongestion:
    """Congestion assessment for one shipping lane."""

    lane_id: str
    congestion_index: float  # 1.0 = baseline; >1.5 = elevated
    closure_probability: float
    affected_nodes: List[str]


class SatelliteProcessor:
    """
    Converts AIS / satellite data into per-node feature vectors.

    In demo / test mode (mock=True) realistic synthetic signals are
    generated, including injected disruption spikes.  In production mode
    calls to an AIS streaming API are made using AISSTREAM_API_KEY.

    Args:
        api_key: AIS API key (required for live mode only).
    """

    KEY_LANES = [
        "suez_canal",
        "hormuz",
        "malacca",
        "panama",
        "taiwan_strait",
        "cape_good_hope",
        "english_channel",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch(
        self,
        node_ids: List[str],
        mock: bool = True,
    ) -> np.ndarray:
        """
        Return (N, 16) satellite feature matrix.

        Args:
            node_ids: Supply network node identifiers.
            mock:     Generate synthetic data instead of calling API.
        """
        if mock:
            return self._mock(len(node_ids))
        if not self.api_key:
            raise RuntimeError("Set AISSTREAM_API_KEY for live satellite data.")
        return self._live(node_ids)

    # ------------------------------------------------------------------
    def _mock(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed=7)
        data = rng.uniform(0, 0.25, (n, N_SATELLITE_FEATURES)).astype(np.float32)
        # Inject congestion spike in first 3 nodes
        data[:3, 0] = rng.uniform(0.7, 1.0, 3)  # vessel density
        data[:3, 1] = rng.uniform(0.5, 0.9, 3)  # speed drop
        data[0, 7] = 0.88  # closure indicator
        return data

    def _live(self, node_ids: List[str]) -> np.ndarray:
        raise NotImplementedError("Live AIS feed not implemented in this release.")

    # ------------------------------------------------------------------
    @staticmethod
    def congestion_index(current: float, baseline: float, std: float = 0.1) -> float:
        """Z-score of current density vs baseline, clamped [0, 5]."""
        if std < 1e-9:
            return 0.0
        return float(np.clip((current - baseline) / std, 0.0, 5.0))

    @staticmethod
    def closure_signal(speed_z: float, density_z: float, ais_gap: float) -> float:
        """Fuse three sub-signals into a lane-closure probability."""
        return float(
            np.clip(
                0.4 * np.clip(-speed_z / 3.0, 0, 1)
                + 0.4 * np.clip(density_z / 3.0, 0, 1)
                + 0.2 * np.clip(ais_gap / 0.3, 0, 1),
                0.0,
                1.0,
            )
        )
