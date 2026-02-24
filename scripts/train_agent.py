"""
Train RL Mitigation Agent (PPO)
==================================

Trains the PPO agent against the LogiSense digital twin simulator over
a library of disruption scenarios.

Usage:
    python scripts/train_agent.py
    python scripts/train_agent.py --episodes 50000 --n-nodes 20

PPO hyperparameters
--------------------
gamma:       0.99   (discount)
gae_lambda:  0.95   (GAE advantage estimation)
clip_eps:    0.20   (policy ratio clip)
entropy:     0.01   (exploration bonus)
value_coef:  0.50   (value loss weight)
lr:          3e-4
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from logisense.twin  import DigitalTwin
from logisense.agent import MitigationAgent, RewardFunction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalised Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv   = 0.0
    for t in reversed(range(T)):
        next_val   = values[t + 1] if t + 1 < T else 0.0
        delta      = rewards[t] + gamma * next_val - values[t]
        last_adv   = delta + gamma * lam * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, opt, rollout, args):
    """One PPO update epoch."""
    obs      = torch.tensor(rollout["obs"],       dtype=torch.float32)
    actions  = torch.tensor(rollout["actions"],   dtype=torch.long)
    old_lps  = torch.tensor(rollout["log_probs"], dtype=torch.float32)
    advs     = torch.tensor(rollout["advantages"],dtype=torch.float32)
    rets     = torch.tensor(rollout["returns"],   dtype=torch.float32)

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    total_loss = 0.0
    for _ in range(args.n_epochs):
        lps, vals, ent = policy.evaluate(obs, actions)
        ratio   = (lps - old_lps).exp()
        clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        policy_loss = -torch.min(ratio * advs, clipped * advs).mean()
        value_loss  = F.mse_loss(vals, rets)
        entropy_loss = -ent.mean()

        loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        total_loss += loss.item()

    return total_loss / args.n_epochs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    twin    = DigitalTwin.sample(n_nodes=args.n_nodes)
    agent   = MitigationAgent(obs_dim=twin.obs_dim, n_actions=args.n_actions, device=str(device))
    rf      = RewardFunction()
    opt     = torch.optim.Adam(agent.policy.parameters(), lr=args.lr)

    episode_returns = []

    for ep in range(1, args.episodes + 1):
        rollout  = agent.collect_rollout(twin, n_steps=args.rollout_steps)
        advs, rets = compute_gae(rollout["rewards"], rollout["values"],
                                  args.gamma, args.gae_lambda)
        rollout["advantages"] = advs
        rollout["returns"]    = rets

        loss = ppo_update(agent.policy, opt, rollout, args)
        episode_returns.append(rollout["rewards"].sum())

        if ep % 100 == 0:
            avg_ret = np.mean(episode_returns[-100:])
            logger.info("Episode %5d / %5d  avg_return=%.3f  loss=%.4f",
                        ep, args.episodes, avg_ret, loss)

        if ep % 1000 == 0:
            agent.save(f"{args.output_dir}/agent")
            logger.info("  ✓ Checkpoint saved at episode %d", ep)

    agent.save(f"{args.output_dir}/agent")
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--n-nodes",       type=int,   default=20)
    parser.add_argument("--n-actions",     type=int,   default=64)
    parser.add_argument("--rollout-steps", type=int,   default=128)
    parser.add_argument("--n-epochs",      type=int,   default=4)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--gae-lambda",    type=float, default=0.95)
    parser.add_argument("--clip-eps",      type=float, default=0.20)
    parser.add_argument("--entropy-coef",  type=float, default=0.01)
    parser.add_argument("--value-coef",    type=float, default=0.50)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--output-dir",    default="checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
