"""
Shipment Reroute Executor
==========================

Redirects in-transit or future shipments away from disrupted / congested
lanes to alternate routes.

Decision logic
---------------
1. Identify in-flight shipments on disrupted lanes.
2. For each, find alternate paths via Dijkstra on the supply graph.
3. Score alternates: composite = α × lead_time_increase + β × cost_increase
   + γ × alternate_lane_disruption_prob.
4. Select best alternate (lowest composite score).
5. Return reroute plan with cost and delay estimates.

In production this plan is sent to the TMS API.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReroutePlan:
    original_lane: str
    alternate_lane: str
    src_node: str
    dst_node: str
    extra_cost: float  # $ above baseline
    extra_days: float  # additional lead time days
    disruption_risk: float  # disruption prob on alternate


class RerouteExecutor:
    """
    Evaluates and executes shipment rerouting decisions.

    Args:
        network: SupplyNetwork to query for alternate paths.
        alpha:   Lead-time weight in scoring.
        beta:    Cost weight in scoring.
        gamma:   Alternate-lane risk weight in scoring.
    """

    def __init__(
        self,
        network=None,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate(
        self,
        target_node: str,
        max_alternates: int = 5,
    ) -> List[ReroutePlan]:
        """
        Evaluate rerouting options for shipments arriving at target_node.

        Returns:
            Ranked list of ReroutePlan (best first).
        """
        if self.network is None:
            return self._mock_plans(target_node)

        plans = []
        preds = list(self.network.graph.predecessors(target_node))

        for src in preds:
            current_data = self.network.graph.get_edge_data(src, target_node, {})
            current_lt = current_data.get("lead_time", 10.0)
            current_cost = current_data.get("cost", 1.0)
            current_risk = current_data.get("disruption_prob", 0.0)

            if current_risk < 0.4:
                continue  # lane is healthy — no reroute needed

            # Find alternate paths from src to target
            try:
                import networkx as nx

                paths = list(
                    nx.all_simple_paths(self.network.graph, src, target_node, cutoff=3)
                )
            except Exception:
                paths = []

            for path in paths[:max_alternates]:
                if len(path) < 2:
                    continue
                alt_lt = sum(
                    self.network.graph[path[i]][path[i + 1]].get("lead_time", 10)
                    for i in range(len(path) - 1)
                )
                alt_cost = sum(
                    self.network.graph[path[i]][path[i + 1]].get("cost", 1)
                    for i in range(len(path) - 1)
                )
                alt_risk = max(
                    self.network.graph[path[i]][path[i + 1]].get("disruption_prob", 0)
                    for i in range(len(path) - 1)
                )

                score = (
                    self.alpha * (alt_lt - current_lt) / max(current_lt, 1)
                    + self.beta * (alt_cost - current_cost) / max(current_cost, 1)
                    + self.gamma * alt_risk
                )

                plans.append(
                    (
                        score,
                        ReroutePlan(
                            original_lane=f"{src}→{target_node}",
                            alternate_lane="→".join(path),
                            src_node=src,
                            dst_node=target_node,
                            extra_cost=max(0, alt_cost - current_cost),
                            extra_days=max(0, alt_lt - current_lt),
                            disruption_risk=alt_risk,
                        ),
                    )
                )

        plans.sort(key=lambda x: x[0])
        return [p for _, p in plans[:max_alternates]]

    def _mock_plans(self, target_node: str) -> List[ReroutePlan]:
        return [
            ReroutePlan(
                original_lane=f"lane_primary_{target_node}",
                alternate_lane=f"lane_alt_{target_node}",
                src_node="node_000",
                dst_node=target_node,
                extra_cost=2_500.0,
                extra_days=2.5,
                disruption_risk=0.08,
            )
        ]

    def execute(self, plan: ReroutePlan, dry_run: bool = True) -> Dict:
        """
        Execute reroute plan (or dry-run).

        In production: calls TMS API to update shipment routing.
        """
        if dry_run:
            logger.info(
                "[Reroute DRY-RUN] %s → %s (+$%.0f, +%.1fd)",
                plan.original_lane,
                plan.alternate_lane,
                plan.extra_cost,
                plan.extra_days,
            )
        else:
            logger.info(
                "[Reroute EXECUTE] %s → %s", plan.original_lane, plan.alternate_lane
            )
            # TODO: integrate with TMS API (SAP TM, Oracle TMS, Flexport, etc.)

        return {
            "status": "dry_run" if dry_run else "executed",
            "original_lane": plan.original_lane,
            "alternate_lane": plan.alternate_lane,
            "extra_cost_usd": plan.extra_cost,
            "extra_days": plan.extra_days,
        }
