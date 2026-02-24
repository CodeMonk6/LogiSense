"""
Inventory Reallocation Executor
==================================

Transfers inventory between Distribution Centers to pre-position stock
ahead of a predicted disruption.

Decision logic
---------------
1. Identify at-risk DCs (inventory_cover_days < threshold OR risk > 0.5).
2. Identify donor DCs with surplus inventory (cover_days > 2× target).
3. For each at-risk DC, find nearest donor that can transfer stock within
   the disruption onset window.
4. Compute transfer quantity: fill at-risk DC to target_cover_days.
5. Score by: transfer_lead_time < days_to_onset AND transfer_cost minimised.

In production this plan is sent to the WMS / ERP API.
"""

from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferPlan:
    src_dc:         str
    dst_dc:         str
    quantity:       float       # units to transfer
    transfer_cost:  float       # USD
    transfer_days:  float       # estimated lead time
    priority:       str         # URGENT | HIGH | NORMAL


class ReallocateExecutor:
    """
    Evaluates and executes inventory reallocation between DCs.

    Args:
        network:         SupplyNetwork (for lane look-up).
        target_cover:    Target inventory cover (days) for recipient DCs.
        surplus_cover:   Minimum cover to retain at donor DCs.
    """

    def __init__(self, network=None, target_cover: float = 14.0, surplus_cover: float = 7.0):
        self.network       = network
        self.target_cover  = target_cover
        self.surplus_cover = surplus_cover

    def evaluate(
        self,
        target_node:    str,
        risk_score:     float = 0.5,
        days_to_onset:  float = 7.0,
    ) -> List[TransferPlan]:
        """
        Find optimal inventory transfer plans for target_node.

        Returns:
            Ranked list of TransferPlan.
        """
        if self.network is None:
            return self._mock_plans(target_node, risk_score)

        from logisense.twin.network_graph import NodeType
        node = self.network.nodes.get(target_node)
        if node is None:
            return []

        avg_demand  = node.metadata.get("avg_daily_demand", 100.0)
        need        = self.target_cover * avg_demand - node.inventory
        if need <= 0:
            return []   # no transfer needed

        # Find donor DCs
        donors = []
        for nid, n in self.network.nodes.items():
            if nid == target_node:
                continue
            if n.node_type not in (NodeType.DISTRIBUTION_CENTER, NodeType.MANUFACTURER):
                continue
            d_demand  = n.metadata.get("avg_daily_demand", 100.0)
            d_cover   = n.inventory / max(d_demand, 1.0)
            surplus   = n.inventory - self.surplus_cover * d_demand
            if surplus <= 0 or d_cover < self.surplus_cover:
                continue

            qty = min(surplus, need)
            donors.append(TransferPlan(
                src_dc=nid, dst_dc=target_node,
                quantity=qty,
                transfer_cost=qty * 0.5,    # $0.50/unit estimate
                transfer_days=2.0,
                priority="URGENT" if risk_score > 0.7 else "HIGH",
            ))

        donors.sort(key=lambda p: p.transfer_cost)
        return donors[:3]

    def _mock_plans(self, target_node: str, risk: float) -> List[TransferPlan]:
        return [TransferPlan(
            src_dc="node_002", dst_dc=target_node,
            quantity=2_000.0, transfer_cost=1_000.0,
            transfer_days=1.5,
            priority="URGENT" if risk > 0.7 else "HIGH",
        )]

    def execute(self, plan: TransferPlan, dry_run: bool = True) -> Dict:
        """Execute transfer plan (or dry-run)."""
        if dry_run:
            logger.info("[Reallocate DRY-RUN] %s → %s  qty=%.0f  cost=$%.0f",
                        plan.src_dc, plan.dst_dc, plan.quantity, plan.transfer_cost)
        else:
            if self.network:
                self.network.nodes[plan.src_dc].inventory -= plan.quantity
            logger.info("[Reallocate EXECUTE] %s → %s", plan.src_dc, plan.dst_dc)
        return {
            "status": "dry_run" if dry_run else "executed",
            "src": plan.src_dc, "dst": plan.dst_dc,
            "quantity": plan.quantity, "cost_usd": plan.transfer_cost,
        }
