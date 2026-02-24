"""
Twin State Encoder
===================

Converts the raw digital twin snapshot (node inventories, capacities,
statuses, risk scores) plus causal disruption forecasts into a flat
observation vector suitable for the PPO RL agent.

Observation vector layout (per node × N_nodes):
    inventory_norm      — inventory / safety_stock
    capacity_norm       — effective_capacity / max_capacity
    risk_score          — causal risk score [0, 1]
    status_onehot[4]    — one-hot NodeStatus
    fill_rate           — last-step demand fill rate
    stockout_flag       — 1 if inventory < safety_stock
    risk_7d             — 7-day horizon risk from forecast
    risk_14d            — 14-day horizon risk from forecast

Total: N_nodes × 11 features
"""

import numpy as np
from typing import Dict, List, Optional
from logisense.twin.network_graph import SupplyNetwork, NodeStatus

N_NODE_FEATURES = 11


class StateEncoder:
    """
    Encodes twin snapshot + forecast into a flat numpy observation vector.

    Args:
        network: The supply network (used for normalisation constants).
    """

    def __init__(self, network: SupplyNetwork):
        self.network   = network
        self.node_ids  = sorted(network.nodes.keys())
        self.obs_dim   = len(self.node_ids) * N_NODE_FEATURES

    def encode(
        self,
        snapshot:   Dict[str, dict],
        risk_7d:    Optional[Dict[str, float]] = None,
        risk_14d:   Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Build flat observation vector.

        Args:
            snapshot:  Output of Simulator.step() / _snapshot().
            risk_7d:   node_id → 7-day risk probability (from forecast).
            risk_14d:  node_id → 14-day risk probability (from forecast).

        Returns:
            obs: (obs_dim,) float32 array in [0, 1].
        """
        risk_7d  = risk_7d  or {}
        risk_14d = risk_14d or {}
        vec      = np.zeros(self.obs_dim, dtype=np.float32)

        for i, nid in enumerate(self.node_ids):
            node = self.network.nodes[nid]
            snap = snapshot.get(nid, {})
            base = i * N_NODE_FEATURES

            inv_norm = np.clip(
                node.inventory / max(node.safety_stock, 1.0), 0.0, 3.0
            ) / 3.0
            cap_max  = node.capacity if node.capacity > 0 else 1.0
            cap_norm = np.clip(node.effective_capacity / cap_max, 0.0, 1.0)

            status   = int(snap.get("status", int(node.status)))
            onehot   = np.zeros(4, dtype=np.float32)
            onehot[min(status, 3)] = 1.0

            stockout_flag = float(node.inventory < node.safety_stock)
            fill_rate     = float(snap.get("fill_rate", 1.0))

            vec[base + 0]     = inv_norm
            vec[base + 1]     = cap_norm
            vec[base + 2]     = float(node.risk_score)
            vec[base + 3:base + 7] = onehot
            vec[base + 7]     = fill_rate
            vec[base + 8]     = stockout_flag
            vec[base + 9]     = float(risk_7d.get(nid, 0.0))
            vec[base + 10]    = float(risk_14d.get(nid, 0.0))

        return np.clip(vec, 0.0, 1.0)

    def decode_action(self, action_idx: int) -> Dict[str, object]:
        """Map integer action index to a structured action dict."""
        from logisense.agent.action_space import ACTION_REGISTRY
        return ACTION_REGISTRY.get(action_idx, {"type": "noop"})
