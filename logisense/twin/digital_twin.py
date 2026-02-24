"""
Digital Twin
=============

Integrates the SupplyNetwork, Simulator, and StateEncoder into a
single interface consumed by the RL agent.

The digital twin:
    1. Holds the canonical supply network graph.
    2. Receives risk scores from the CausalDisruptionEngine.
    3. Steps the Simulator forward one day per RL timestep.
    4. Encodes the resulting state into the agent's observation vector.
    5. Tracks service-level KPIs for reward computation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from logisense.twin.network_graph import (
    NodeStatus,
    NodeType,
    SupplyLane,
    SupplyNetwork,
    SupplyNode,
)
from logisense.twin.simulator import Simulator
from logisense.twin.state_encoder import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class TwinState:
    """Observation state emitted by the digital twin."""

    obs: np.ndarray  # (obs_dim,) flat RL observation
    snapshot: Dict[str, dict]  # raw node metrics
    day: int
    risk_7d: Dict[str, float]
    risk_14d: Dict[str, float]
    kpis: Dict[str, float] = field(default_factory=dict)


class DigitalTwin:
    """
    Dynamic digital twin of a supply chain network.

    Args:
        network:       SupplyNetwork to simulate.
        demand_sigma:  Demand noise fraction.
    """

    def __init__(self, network: SupplyNetwork, demand_sigma: float = 0.10):
        self.network = network
        self.sim = Simulator(network, demand_sigma)
        self.encoder = StateEncoder(network)
        self._risk_7d: Dict[str, float] = {}
        self._risk_14d: Dict[str, float] = {}
        self._history: List[TwinState] = []

    # ── configuration ────────────────────────────────────────────────────

    def apply_risk_scores(self, forecast) -> None:
        """
        Apply disruption forecasts from CausalDisruptionEngine to the twin.

        Args:
            forecast: DisruptionForecast object.
        """
        risk_map = {}
        self._risk_7d = {}
        self._risk_14d = {}

        for nid, node_risk in forecast.node_risks.items():
            risk_map[nid] = node_risk.peak_score
            self._risk_7d[nid] = node_risk.risk_7d
            self._risk_14d[nid] = node_risk.risk_14d

        self.sim.apply_risk_scores(risk_map)
        logger.info("[Twin] Risk scores applied for %d nodes.", len(risk_map))

    def inject_disruption(
        self, node_id: str, status: NodeStatus = NodeStatus.CLOSED
    ) -> None:
        """Inject a scenario disruption directly."""
        self.sim.inject_disruption(node_id, status)

    # ── stepping ─────────────────────────────────────────────────────────

    def step(self) -> TwinState:
        """Advance one day; return TwinState."""
        snap = self.sim.step()
        obs = self.encoder.encode(snap, self._risk_7d, self._risk_14d)
        kpis = self._compute_kpis(snap)
        state = TwinState(
            obs=obs,
            snapshot=snap,
            day=self.sim.day,
            risk_7d=dict(self._risk_7d),
            risk_14d=dict(self._risk_14d),
            kpis=kpis,
        )
        self._history.append(state)
        return state

    def simulate(self, steps: int = 14) -> TwinState:
        """Run `steps` days and return the final TwinState."""
        state = None
        for _ in range(steps):
            state = self.step()
        return state

    def reset(self) -> TwinState:
        """Reset simulator to day 0."""
        self.sim = Simulator(self.network)
        self._history.clear()
        snap = {
            nid: {
                "inventory": n.inventory,
                "status": 0,
                "fill_rate": 1.0,
                "demand": 0.0,
                "stockout_days": 0,
                "capacity": n.capacity,
                "risk_score": 0.0,
            }
            for nid, n in self.network.nodes.items()
        }
        obs = self.encoder.encode(snap)
        return TwinState(
            obs=obs, snapshot=snap, day=0, risk_7d={}, risk_14d={}, kpis={}
        )

    # ── KPI computation ──────────────────────────────────────────────────

    def _compute_kpis(self, snap: Dict[str, dict]) -> Dict[str, float]:
        fill_rates = [v["fill_rate"] for v in snap.values()]
        stockout_cnt = sum(1 for v in snap.values() if v.get("stockout_days", 0) > 0)
        high_risk_cnt = sum(
            1 for n in self.network.nodes.values() if n.risk_score > 0.6
        )

        return {
            "avg_fill_rate": float(np.mean(fill_rates)),
            "min_fill_rate": float(np.min(fill_rates)),
            "stockout_nodes": stockout_cnt,
            "high_risk_nodes": high_risk_cnt,
            "service_level": float(np.mean([r >= 0.95 for r in fill_rates])),
        }

    # ── observation helpers ──────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return self.encoder.obs_dim

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str) -> "DigitalTwin":
        """Load a DigitalTwin from a YAML network config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        network = SupplyNetwork(cfg.get("network_id", "default"))

        for n_cfg in cfg.get("nodes", []):
            n_cfg = dict(n_cfg)
            n_cfg["node_type"] = NodeType[n_cfg["node_type"].upper()]
            n_cfg["location"] = tuple(n_cfg["location"])
            node = SupplyNode(**n_cfg)
            network.add_node(node)

        for l_cfg in cfg.get("lanes", []):
            lane = SupplyLane(**l_cfg)
            network.add_lane(lane)

        if not network.nodes:
            logger.info("No nodes in config — using sample network.")
            network = SupplyNetwork.sample(n_nodes=20)

        return cls(network)

    @classmethod
    def sample(cls, n_nodes: int = 20) -> "DigitalTwin":
        return cls(SupplyNetwork.sample(n_nodes))
