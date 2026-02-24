"""
Discrete-Event Simulator
=========================

Steps the supply network forward one day at a time, modelling:

    - Demand fulfilment from each DC / customer node
    - Inventory depletion and replenishment triggers
    - Transit pipeline: in-flight shipments arriving after lead time
    - Disruption application: nodes / lanes degrade based on risk scores
    - Service-level tracking: fill rate, stockout days, excess cost

Each step returns the network state as a dict of per-node metrics, which
is consumed by the StateEncoder to produce the RL observation vector.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from logisense.twin.network_graph import SupplyNetwork, NodeType, NodeStatus

logger = logging.getLogger(__name__)

# Disruption probability thresholds
DEGRADED_THRESHOLD  = 0.40
DISRUPTED_THRESHOLD = 0.65
CLOSED_THRESHOLD    = 0.85


class InTransitShipment:
    """Tracks a shipment moving through a lane."""
    __slots__ = ("lane_id", "src", "dst", "quantity", "arrival_day")

    def __init__(self, lane_id, src, dst, quantity, arrival_day):
        self.lane_id     = lane_id
        self.src         = src
        self.dst         = dst
        self.quantity    = quantity
        self.arrival_day = arrival_day


class Simulator:
    """
    Single-day step discrete-event simulator for a SupplyNetwork.

    Args:
        network:      The supply network to simulate.
        demand_sigma: Fractional demand noise (default 10%).
    """

    def __init__(self, network: SupplyNetwork, demand_sigma: float = 0.10):
        self.network      = network
        self.demand_sigma = demand_sigma
        self.day          = 0
        self.rng          = np.random.default_rng(seed=42)

        self._pipeline:   List[InTransitShipment] = []
        self._metrics:    Dict[str, dict]          = defaultdict(dict)
        self._stockout_days: Dict[str, int]        = defaultdict(int)

    # ── public interface ─────────────────────────────────────────────────

    def step(self) -> Dict[str, dict]:
        """
        Advance simulation by one day.

        Returns:
            metrics: node_id → {inventory, demand, fill_rate,
                                 stockout, status, risk_score}
        """
        self.day += 1
        self._apply_disruptions()
        self._receive_shipments()
        self._generate_demand()
        self._trigger_replenishment()
        return self._snapshot()

    def run(self, n_steps: int) -> List[Dict[str, dict]]:
        """Run n_steps days and return all snapshots."""
        return [self.step() for _ in range(n_steps)]

    def apply_risk_scores(self, risk_scores: Dict[str, float]) -> None:
        """Set risk scores on nodes (called after CausalDisruptionEngine)."""
        for nid, score in risk_scores.items():
            if nid in self.network.nodes:
                self.network.nodes[nid].risk_score = score

    def inject_disruption(self, node_id: str, status: NodeStatus) -> None:
        """Manually inject a disruption for scenario testing."""
        if node_id in self.network.nodes:
            self.network.nodes[node_id].status = status
            logger.info("Disruption injected: %s → %s", node_id, status.name)

    # ── private helpers ──────────────────────────────────────────────────

    def _apply_disruptions(self) -> None:
        """Convert risk scores to node status changes."""
        for node in self.network.nodes.values():
            r = node.risk_score
            if r >= CLOSED_THRESHOLD:
                node.status = NodeStatus.CLOSED
            elif r >= DISRUPTED_THRESHOLD:
                node.status = NodeStatus.DISRUPTED
            elif r >= DEGRADED_THRESHOLD:
                node.status = NodeStatus.DEGRADED
            else:
                node.status = NodeStatus.OPERATIONAL

    def _receive_shipments(self) -> None:
        """Deliver in-transit shipments that have arrived."""
        arriving = [s for s in self._pipeline if s.arrival_day <= self.day]
        for s in arriving:
            if s.dst in self.network.nodes:
                self.network.nodes[s.dst].inventory += s.quantity
        self._pipeline = [s for s in self._pipeline if s.arrival_day > self.day]

    def _generate_demand(self) -> None:
        """Consume inventory at customer and DC nodes."""
        for node in self.network.nodes.values():
            if node.node_type not in (NodeType.CUSTOMER, NodeType.DISTRIBUTION_CENTER):
                continue
            base_demand = node.metadata.get("avg_daily_demand", 100.0)
            noise       = self.rng.normal(1.0, self.demand_sigma)
            demand      = max(0.0, base_demand * noise)

            fulfilled   = min(demand, node.inventory)
            node.inventory -= fulfilled

            fill_rate = fulfilled / demand if demand > 0 else 1.0
            self._metrics[node.node_id]["demand"]    = demand
            self._metrics[node.node_id]["fulfilled"] = fulfilled
            self._metrics[node.node_id]["fill_rate"] = fill_rate

            if node.inventory < node.safety_stock:
                self._stockout_days[node.node_id] += 1

    def _trigger_replenishment(self) -> None:
        """Place replenishment orders when inventory drops below reorder point."""
        for node in self.network.nodes.values():
            if node.inventory > node.reorder_point:
                continue
            # Find best upstream supplier lane
            predecessors = list(self.network.graph.predecessors(node.node_id))
            if not predecessors:
                continue
            # Pick least-disrupted predecessor
            best_pred = min(
                predecessors,
                key=lambda p: self.network.nodes[p].risk_score
                              + (1 if self.network.nodes[p].status != NodeStatus.OPERATIONAL else 0)
            )
            lane_data  = self.network.graph.get_edge_data(best_pred, node.node_id, {})
            lane_id    = lane_data.get("lane_id", f"{best_pred}→{node.node_id}")
            lt         = lane_data.get("lead_time", node.lead_time_days)
            order_qty  = max(0, node.reorder_point * 2 - node.inventory)

            self._pipeline.append(InTransitShipment(
                lane_id     = lane_id,
                src         = best_pred,
                dst         = node.node_id,
                quantity    = order_qty,
                arrival_day = self.day + int(np.ceil(lt)),
            ))

    def _snapshot(self) -> Dict[str, dict]:
        snap = {}
        for nid, node in self.network.nodes.items():
            snap[nid] = {
                "inventory":    node.inventory,
                "capacity":     node.effective_capacity,
                "risk_score":   node.risk_score,
                "status":       int(node.status),
                "fill_rate":    self._metrics[nid].get("fill_rate", 1.0),
                "demand":       self._metrics[nid].get("demand", 0.0),
                "stockout_days": self._stockout_days[nid],
            }
        return snap
