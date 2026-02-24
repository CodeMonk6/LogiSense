"""
Reward Function
================

Computes the scalar reward signal for the PPO agent at each timestep.

Reward decomposition
---------------------
r = w_continuity * R_continuity
  + w_cost       * R_cost
  + w_speed      * R_speed
  - w_penalty    * P_stockout

Where:
    R_continuity  = weighted avg fill rate across all demand nodes
    R_cost        = −normalised mitigation cost (reroute / expedite penalty)
    R_speed       = bonus for acting before predicted disruption onset
    P_stockout    = cumulative stockout-node fraction

Default weights
----------------
continuity : 0.50
cost       : 0.20
speed      : 0.20
stockout   : 0.10

The agent is incentivised to maintain service levels at minimum cost and
as early as possible (preemption bonus decays with proximity to disruption).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RewardComponents:
    continuity: float
    cost: float
    speed: float
    stockout: float
    total: float


class RewardFunction:
    """
    Computes step reward for the RL mitigation agent.

    Args:
        w_continuity:  Weight for service level continuity.
        w_cost:        Weight for mitigation cost penalty.
        w_speed:       Weight for preemption speed bonus.
        w_stockout:    Weight for stockout penalty.
        target_fill:   Target fill rate (default 0.95 = 95%).
    """

    def __init__(
        self,
        w_continuity: float = 0.50,
        w_cost: float = 0.20,
        w_speed: float = 0.20,
        w_stockout: float = 0.10,
        target_fill: float = 0.95,
    ):
        self.w_continuity = w_continuity
        self.w_cost = w_cost
        self.w_speed = w_speed
        self.w_stockout = w_stockout
        self.target_fill = target_fill

    def compute(
        self,
        snapshot: Dict[str, dict],
        action_cost: float = 0.0,
        days_to_onset: Optional[float] = None,
        max_action_cost: float = 10_000.0,
    ) -> RewardComponents:
        """
        Compute reward for one timestep.

        Args:
            snapshot:        Simulator snapshot (node_id → metrics).
            action_cost:     Estimated cost of chosen action ($).
            days_to_onset:   Days until predicted disruption (None = unknown).
            max_action_cost: Normalisation constant for cost penalty.

        Returns:
            RewardComponents with individual terms and total.
        """
        fill_rates = [v.get("fill_rate", 1.0) for v in snapshot.values()]
        avg_fill = float(np.mean(fill_rates))

        # Service continuity: reward for being at or above target fill rate
        r_continuity = np.clip(avg_fill / self.target_fill, 0.0, 1.0)

        # Cost: penalise expensive mitigation (normalised)
        r_cost = -np.clip(action_cost / max_action_cost, 0.0, 1.0)

        # Speed bonus: acting early (many days before onset) earns bonus
        if days_to_onset is not None and days_to_onset > 0:
            r_speed = np.clip(days_to_onset / 21.0, 0.0, 1.0)  # max bonus at 21d lead
        else:
            r_speed = 0.0

        # Stockout penalty
        stockout_frac = sum(
            1 for v in snapshot.values() if v.get("fill_rate", 1.0) < 0.80
        ) / max(len(snapshot), 1)
        p_stockout = stockout_frac

        total = (
            self.w_continuity * r_continuity
            + self.w_cost * r_cost
            + self.w_speed * r_speed
            - self.w_stockout * p_stockout
        )

        return RewardComponents(
            continuity=float(r_continuity),
            cost=float(r_cost),
            speed=float(r_speed),
            stockout=float(p_stockout),
            total=float(total),
        )
