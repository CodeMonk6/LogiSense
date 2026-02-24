"""
Supply Network Graph
=====================

Directed graph representation of a supply chain network.

Nodes: Supplier | Manufacturer | Distribution Center | Port | Customer
Edges: Transportation lanes (sea, air, road, rail)

Each node tracks inventory, capacity, operational status, and risk score.
Each lane tracks lead time, cost, utilisation, and disruption probability.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class NodeType(IntEnum):
    SUPPLIER             = 0
    MANUFACTURER         = 1
    DISTRIBUTION_CENTER  = 2
    PORT                 = 3
    CUSTOMER             = 4


class NodeStatus(IntEnum):
    OPERATIONAL = 0
    DEGRADED    = 1   # 60% capacity
    DISRUPTED   = 2   # 20% capacity
    CLOSED      = 3   # 0% capacity


_STATUS_FACTOR = {
    NodeStatus.OPERATIONAL: 1.0,
    NodeStatus.DEGRADED:    0.60,
    NodeStatus.DISRUPTED:   0.20,
    NodeStatus.CLOSED:      0.00,
}


@dataclass
class SupplyNode:
    node_id:       str
    name:          str
    node_type:     NodeType
    location:      Tuple[float, float]   # (lat, lon)
    capacity:      float                 # units/day
    inventory:     float                 # on-hand units
    safety_stock:  float
    reorder_point: float
    lead_time_days: float
    status:        NodeStatus = NodeStatus.OPERATIONAL
    risk_score:    float = 0.0
    country:       str = "US"
    critical:      bool = False
    metadata:      dict = field(default_factory=dict)

    @property
    def effective_capacity(self) -> float:
        return self.capacity * _STATUS_FACTOR[self.status]

    @property
    def inventory_cover_days(self) -> float:
        d = self.metadata.get("avg_daily_demand", 100.0)
        return self.inventory / max(d, 1.0)

    @property
    def is_at_risk(self) -> bool:
        return self.risk_score > 0.5 or self.inventory < self.safety_stock

    def __repr__(self) -> str:
        return (f"SupplyNode({self.node_id}, {NodeType(self.node_type).name}, "
                f"inv={self.inventory:.0f}, risk={self.risk_score:.2f})")


@dataclass
class SupplyLane:
    lane_id:          str
    src_node:         str
    dst_node:         str
    transport_mode:   str       # sea | air | road | rail
    lead_time_days:   float
    cost_per_unit:    float
    capacity_units:   float
    utilisation:      float = 0.5
    disruption_prob:  float = 0.0
    congestion_index: float = 1.0
    alternate_lanes:  List[str] = field(default_factory=list)

    @property
    def effective_lead_time(self) -> float:
        return self.lead_time_days * max(1.0, self.congestion_index)

    @property
    def is_congested(self) -> bool:
        return self.congestion_index > 1.5 or self.utilisation > 0.85

    def __repr__(self) -> str:
        return (f"Lane({self.lane_id}: {self.src_node}→{self.dst_node}, "
                f"{self.transport_mode}, LT={self.effective_lead_time:.1f}d, "
                f"p_disrupt={self.disruption_prob:.2f})")


class SupplyNetwork:
    """
    Directed supply chain graph (NetworkX backend).

    Provides:
    - Shortest-path by lead time / cost
    - Bottleneck identification (betweenness centrality)
    - Critical-path risk computation
    """

    def __init__(self, network_id: str):
        self.network_id = network_id
        self.graph      = nx.DiGraph()
        self.nodes:     Dict[str, SupplyNode] = {}
        self.lanes:     Dict[str, SupplyLane] = {}

    # ── mutation ─────────────────────────────────────────────────────────

    def add_node(self, n: SupplyNode) -> None:
        self.nodes[n.node_id] = n
        self.graph.add_node(n.node_id,
                            node_type=int(n.node_type),
                            inventory=n.inventory,
                            risk_score=n.risk_score,
                            status=int(n.status))

    def add_lane(self, l: SupplyLane) -> None:
        self.lanes[l.lane_id] = l
        self.graph.add_edge(l.src_node, l.dst_node,
                            lane_id=l.lane_id,
                            lead_time=l.lead_time_days,
                            cost=l.cost_per_unit,
                            disruption_prob=l.disruption_prob)

    def update_node(self, node_id: str, **kwargs) -> None:
        node = self.nodes[node_id]
        for k, v in kwargs.items():
            setattr(node, k, v)
        nx.set_node_attributes(self.graph, {node_id: kwargs})

    # ── queries ──────────────────────────────────────────────────────────

    def shortest_path(
        self, src: str, dst: str, weight: str = "lead_time"
    ) -> Tuple[List[str], float]:
        try:
            path = nx.shortest_path(self.graph, src, dst, weight=weight)
            cost = sum(self.graph[path[i]][path[i+1]][weight]
                       for i in range(len(path) - 1))
            return path, cost
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float("inf")

    def bottlenecks(self, k: int = 10) -> List[str]:
        """Top-k nodes by betweenness centrality."""
        bc = nx.betweenness_centrality(self.graph)
        return [n for n, _ in sorted(bc.items(), key=lambda x: x[1], reverse=True)[:k]]

    def path_risk(self, src: str, dst: str) -> float:
        """1 - product(1 - edge_disruption_prob) along shortest path."""
        path, _ = self.shortest_path(src, dst)
        if not path:
            return 1.0
        rel = 1.0
        for i in range(len(path) - 1):
            rel *= 1 - self.graph[path[i]][path[i+1]].get("disruption_prob", 0.0)
        return 1.0 - rel

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def sample(cls, n_nodes: int = 20) -> "SupplyNetwork":
        """Build a synthetic global supply network for demo / testing."""
        rng  = np.random.default_rng(seed=0)
        net  = cls("sample_network")
        types    = list(NodeType)
        modes    = ["sea", "air", "road"]
        countries = ["CN", "TW", "US", "DE", "JP", "VN", "SG", "NL", "MX", "IN"]

        for i in range(n_nodes):
            ntype = types[i % len(types)]
            node  = SupplyNode(
                node_id=f"node_{i:03d}",
                name=f"{NodeType(ntype).name.title()} {i}",
                node_type=ntype,
                location=(rng.uniform(-60, 60), rng.uniform(-160, 160)),
                capacity=rng.uniform(500, 5000),
                inventory=rng.uniform(200, 3000),
                safety_stock=500.0,
                reorder_point=800.0,
                lead_time_days=rng.uniform(2, 30),
                country=countries[i % len(countries)],
                risk_score=0.0,
            )
            net.add_node(node)

        node_ids = list(net.nodes)
        for i in range(n_nodes):
            n_connections = rng.integers(1, 4)
            for _ in range(n_connections):
                j = rng.integers(0, n_nodes)
                if i == j:
                    continue
                mode = modes[rng.integers(len(modes))]
                lt   = rng.uniform(1, 45) if mode == "sea" else rng.uniform(0.5, 7)
                lane = SupplyLane(
                    lane_id=f"lane_{i}_{j}",
                    src_node=node_ids[i],
                    dst_node=node_ids[j],
                    transport_mode=mode,
                    lead_time_days=float(lt),
                    cost_per_unit=float(rng.uniform(0.5, 20.0)),
                    capacity_units=float(rng.uniform(100, 2000)),
                    disruption_prob=float(rng.uniform(0, 0.15)),
                )
                net.add_lane(lane)

        return net

    @property
    def n_nodes(self) -> int: return len(self.nodes)
    @property
    def n_lanes(self) -> int: return len(self.lanes)
