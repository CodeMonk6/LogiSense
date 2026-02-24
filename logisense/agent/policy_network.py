"""
Policy Network (Actor-Critic)
==============================

Shared-trunk actor-critic network for PPO.

Architecture
-------------
Observation → LayerNorm → MLP trunk (3 × 256, GELU)
                         ↓              ↓
                  Actor head        Critic head
              (logits → softmax)  (scalar value)

The actor outputs a probability distribution over discrete actions.
The critic estimates the value function V(s) for advantage estimation.

PPO objective
--------------
L = E[ min(r_t(θ) · A_t,  clip(r_t(θ), 1−ε, 1+ε) · A_t) ]
  − c_v · (V_θ(s) − V_target)²
  + c_e · H(π_θ)

where r_t(θ) = π_θ(a|s) / π_old(a|s),  ε = 0.2 (default),
      c_v = value loss coefficient,  c_e = entropy coefficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Shared-trunk actor-critic network.

    Args:
        obs_dim:      Observation dimension (from StateEncoder).
        n_actions:    Number of discrete actions (from ActionSpace).
        hidden_dim:   Hidden layer width.
        n_layers:     Number of MLP layers in the shared trunk.
    """

    def __init__(
        self,
        obs_dim:    int,
        n_actions:  int,
        hidden_dim: int = 256,
        n_layers:   int = 3,
    ):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        # Shared trunk
        layers = [nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.trunk = nn.Sequential(*layers)

        # Actor head
        self.actor_head = nn.Linear(hidden_dim, n_actions)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self,
        obs:         torch.Tensor,               # (B, obs_dim)
        action_mask: torch.Tensor | None = None, # (B, n_actions) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits: (B, n_actions) unnormalised log probabilities
            values: (B,) state value estimates
        """
        h      = self.trunk(obs)
        logits = self.actor_head(h)
        values = self.critic_head(h).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits, values

    def act(
        self,
        obs:         torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or greedily select action.

        Returns:
            actions:   (B,) integer action indices
            log_probs: (B,) log-probabilities of selected actions
            values:    (B,) state value estimates
        """
        logits, values = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()

        return actions, dist.log_prob(actions), values

    def evaluate(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and value for PPO update.

        Returns:
            log_probs: (B,) log-probabilities of taken actions
            values:    (B,) state values
            entropy:   (B,) entropy of action distribution
        """
        logits, values = self.forward(obs)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy
