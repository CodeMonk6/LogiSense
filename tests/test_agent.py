"""Tests for the PPO mitigation agent."""

import numpy as np
import pytest
import torch

from logisense.agent import MitigationAction, MitigationAgent, RewardFunction
from logisense.agent.action_space import ActionSpace
from logisense.agent.policy_network import PolicyNetwork
from logisense.twin import DigitalTwin


@pytest.fixture
def twin():
    return DigitalTwin.sample(n_nodes=10)


@pytest.fixture
def agent(twin):
    return MitigationAgent(obs_dim=twin.obs_dim, n_actions=16)


class TestPolicyNetwork:
    def test_forward_shapes(self):
        net = PolicyNetwork(obs_dim=64, n_actions=16)
        obs = torch.randn(4, 64)
        logits, values = net(obs)
        assert logits.shape == (4, 16)
        assert values.shape == (4,)

    def test_act_shapes(self):
        net = PolicyNetwork(obs_dim=64, n_actions=16)
        obs = torch.randn(2, 64)
        acts, lps, vals = net.act(obs)
        assert acts.shape == (2,)
        assert lps.shape == (2,)
        assert vals.shape == (2,)

    def test_deterministic_act(self):
        net = PolicyNetwork(obs_dim=32, n_actions=8)
        obs = torch.randn(1, 32)
        a1, _, _ = net.act(obs, deterministic=True)
        a2, _, _ = net.act(obs, deterministic=True)
        assert a1.item() == a2.item()

    def test_evaluate_shapes(self):
        net = PolicyNetwork(obs_dim=32, n_actions=8)
        obs = torch.randn(4, 32)
        acts = torch.randint(0, 8, (4,))
        lps, vals, ent = net.evaluate(obs, acts)
        assert lps.shape == (4,)
        assert vals.shape == (4,)
        assert ent.shape == (4,)


class TestRewardFunction:
    def test_reward_range(self):
        rf = RewardFunction()
        snap = {
            f"node_{i}": {"fill_rate": np.random.uniform(0.7, 1.0)} for i in range(5)
        }
        r = rf.compute(snap, action_cost=0.0)
        assert -2.0 <= r.total <= 2.0

    def test_speed_bonus(self):
        rf = RewardFunction()
        snap = {"n0": {"fill_rate": 0.95}}
        r_early = rf.compute(snap, days_to_onset=20.0)
        r_late = rf.compute(snap, days_to_onset=1.0)
        assert r_early.speed > r_late.speed

    def test_high_fill_rate_reward(self):
        rf = RewardFunction()
        good = {"n0": {"fill_rate": 0.99}}
        bad = {"n0": {"fill_rate": 0.60}}
        assert rf.compute(good).continuity > rf.compute(bad).continuity


class TestMitigationAgent:
    def test_act_returns_list(self, agent, twin):
        state = twin.simulate(steps=2)
        acts = agent.act(state, top_k=3)
        assert isinstance(acts, list)
        assert len(acts) <= 3

    def test_action_types(self, agent, twin):
        state = twin.simulate(steps=2)
        for a in agent.act(state):
            assert isinstance(a, MitigationAction)
            assert a.action_type in (
                "reroute",
                "reallocate",
                "procure",
                "expedite",
                "hedge",
                "noop",
            )

    def test_action_priorities(self, agent, twin):
        state = twin.simulate(steps=2)
        for a in agent.act(state):
            assert a.priority in ("HIGH", "MEDIUM", "LOW")

    def test_save_load(self, agent, tmp_path):
        agent.save(str(tmp_path))
        loaded = MitigationAgent.from_pretrained(
            str(tmp_path), obs_dim=agent.policy.obs_dim, n_actions=agent.n_actions
        )
        assert isinstance(loaded, MitigationAgent)
