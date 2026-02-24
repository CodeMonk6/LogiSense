"""Tests for the Digital Twin and supporting components."""

import numpy as np
import pytest

from logisense.twin import DigitalTwin, SupplyNetwork, TwinState
from logisense.twin.network_graph import NodeStatus
from logisense.twin.simulator import Simulator
from logisense.twin.state_encoder import StateEncoder


@pytest.fixture
def twin():
    return DigitalTwin.sample(n_nodes=10)


class TestSupplyNetwork:
    def test_sample_builds_correctly(self):
        net = SupplyNetwork.sample(n_nodes=15)
        assert net.n_nodes == 15
        assert net.n_lanes > 0

    def test_bottlenecks_returns_list(self):
        net = SupplyNetwork.sample(n_nodes=15)
        bt = net.bottlenecks(k=5)
        assert len(bt) <= 5

    def test_shortest_path(self):
        net = SupplyNetwork.sample(n_nodes=15)
        nids = list(net.nodes.keys())
        path, cost = net.shortest_path(nids[0], nids[-1])
        # Either found or no path
        assert cost >= 0 or cost == float("inf")

    def test_path_risk_range(self):
        net = SupplyNetwork.sample(n_nodes=15)
        nids = list(net.nodes.keys())
        risk = net.path_risk(nids[0], nids[5])
        assert 0.0 <= risk <= 1.0


class TestSimulator:
    def test_step_returns_dict(self, twin):
        snap = twin.sim.step()
        assert isinstance(snap, dict)
        assert len(snap) == twin.network.n_nodes

    def test_step_metrics_keys(self, twin):
        snap = twin.sim.step()
        nid = list(snap.keys())[0]
        for key in ("inventory", "fill_rate", "demand", "status"):
            assert key in snap[nid]

    def test_inventory_non_negative(self, twin):
        for _ in range(5):
            snap = twin.sim.step()
        for v in snap.values():
            assert v["inventory"] >= 0.0

    def test_disruption_inject(self, twin):
        nid = list(twin.network.nodes.keys())[0]
        twin.sim.inject_disruption(nid, NodeStatus.CLOSED)
        assert twin.network.nodes[nid].status == NodeStatus.CLOSED


class TestStateEncoder:
    def test_obs_shape(self, twin):
        snap = twin.sim.step()
        obs = twin.encoder.encode(snap)
        assert obs.shape == (twin.obs_dim,)

    def test_obs_range(self, twin):
        snap = twin.sim.step()
        obs = twin.encoder.encode(snap)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_no_nans(self, twin):
        snap = twin.sim.step()
        obs = twin.encoder.encode(snap)
        assert not np.isnan(obs).any()


class TestDigitalTwin:
    def test_simulate_returns_twin_state(self, twin):
        state = twin.simulate(steps=3)
        assert isinstance(state, TwinState)

    def test_kpis_in_state(self, twin):
        state = twin.simulate(steps=2)
        for key in ("avg_fill_rate", "service_level", "stockout_nodes"):
            assert key in state.kpis

    def test_reset(self, twin):
        twin.simulate(steps=5)
        assert twin.sim.day == 5
        twin.reset()
        assert twin.sim.day == 0
