"""
Action Space
=============

Defines the discrete action vocabulary for the PPO mitigation agent.

Action categories
------------------
NOOP           — Do nothing (baseline / wait)
REROUTE        — Redirect shipment to alternate lane / port
REALLOCATE     — Transfer inventory between DCs
PROCURE        — Trigger contingent procurement from alternate supplier
EXPEDITE       — Upgrade transport mode (road → air) on a specific lane
HEDGE          — Signal financial / commodity hedging need

Action encoding
---------------
The agent selects from N_ACTIONS discrete actions per step.
Each action maps to (type, target_node_or_lane, parameter).

In production the action set is auto-generated from the current network
topology, producing one action per at-risk node × valid mitigation type.
For the research implementation we use a fixed-size discrete action space
with the top-K most impactful actions per category.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# ── action type constants ─────────────────────────────────────────────────
NOOP       = 0
REROUTE    = 1
REALLOCATE = 2
PROCURE    = 3
EXPEDITE   = 4
HEDGE      = 5

N_ACTION_TYPES = 6


@dataclass
class ActionSpec:
    """Specification for one discrete action."""
    action_id:    int
    action_type:  int
    target:       str          # node_id or lane_id
    parameter:    float = 1.0  # e.g. order_qty_factor, inventory_fraction
    description:  str  = ""


class ActionSpace:
    """
    Dynamic action space built from the current supply network topology.

    For each at-risk node we generate:
        - REROUTE:    alternate lanes sorted by lead-time + cost
        - REALLOCATE: nearest DCs with excess inventory
        - PROCURE:    alternate qualified suppliers
        - EXPEDITE:   upgrade transport mode on incoming lanes
        - HEDGE:      flag financial risk

    Args:
        network:     SupplyNetwork object.
        max_actions: Maximum total discrete actions (agent output size).
    """

    def __init__(self, network=None, max_actions: int = 64):
        self.max_actions = max_actions
        self._actions:   List[ActionSpec] = [ActionSpec(0, NOOP, "global", 0.0, "No action")]
        if network is not None:
            self._build(network)

    def _build(self, network) -> None:
        aid = 1
        for nid, node in list(network.nodes.items())[:20]:
            if aid >= self.max_actions:
                break

            # REROUTE
            preds = list(network.graph.predecessors(nid))
            for p in preds[:2]:
                self._actions.append(ActionSpec(
                    aid, REROUTE, nid, 1.0,
                    f"Reroute inbound to {nid} via {p}"
                ))
                aid += 1
                if aid >= self.max_actions:
                    return

            # REALLOCATE
            self._actions.append(ActionSpec(
                aid, REALLOCATE, nid, 0.5,
                f"Transfer 50% excess inventory to {nid}"
            ))
            aid += 1
            if aid >= self.max_actions:
                return

            # PROCURE
            self._actions.append(ActionSpec(
                aid, PROCURE, nid, 2.0,
                f"Trigger contingent procurement for {nid} (2× safety stock)"
            ))
            aid += 1

        # Pad remaining slots with NOOP
        while aid < self.max_actions:
            self._actions.append(ActionSpec(aid, NOOP, "global", 0.0, "No action"))
            aid += 1

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, idx: int) -> ActionSpec:
        return self._actions[min(idx, len(self._actions) - 1)]

    def action_mask(self, snapshot: dict) -> List[bool]:
        """True for valid actions given current network state."""
        mask = [True] * len(self._actions)
        # NOOP always valid; others masked if target not at risk
        for i, a in enumerate(self._actions[1:], 1):
            node_snap = snapshot.get(a.target, {})
            # Disable action if target is fine
            if a.action_type in (REROUTE, REALLOCATE, PROCURE):
                risk = node_snap.get("risk_score", 0.0)
                mask[i] = risk > 0.3 or node_snap.get("fill_rate", 1.0) < 0.95
        return mask


# Default registry (used before network topology is known)
ACTION_REGISTRY: Dict[int, dict] = {
    0: {"type": "noop"},
    1: {"type": "reroute",    "target": "node_000"},
    2: {"type": "reallocate", "target": "node_001"},
    3: {"type": "procure",    "target": "node_002"},
    4: {"type": "expedite",   "target": "node_003"},
    5: {"type": "hedge",      "target": "global"},
}
