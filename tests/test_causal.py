"""Tests for the causal disruption engine."""

import numpy as np
import pytest
import torch

from logisense.causal import CausalDisruptionEngine, DisruptionForecast, NodeRisk
from logisense.causal.notears import NOTEARSLearner
from logisense.signals import SignalFusionEngine

NODE_IDS = [f"node_{i:03d}" for i in range(8)]


@pytest.fixture
def signal_state():
    return SignalFusionEngine(T_lookback=5).fetch_and_fuse(
        "test", node_ids=NODE_IDS, mock=True
    )


class TestNOTEARS:
    def test_output_shape(self):
        learner = NOTEARSLearner(n_vars=10, max_iter=5)
        X = np.random.randn(50, 10).astype(np.float64)
        W = learner.fit(X)
        assert W.shape == (10, 10)
        assert W.dtype == np.float32

    def test_no_self_loops(self):
        learner = NOTEARSLearner(n_vars=8, max_iter=3)
        X = np.random.randn(30, 8).astype(np.float64)
        W = learner.fit(X)
        assert np.all(np.diag(W) == 0)

    def test_parents_children(self):
        learner = NOTEARSLearner(n_vars=5)
        learner.W_ = np.zeros((5, 5), dtype=np.float64)
        learner.W_[0, 2] = 0.5  # 2 → 0
        parents = learner.parents(0)
        assert 2 in parents


class TestCausalDisruptionEngine:
    def test_forecast_returns_correct_type(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        assert isinstance(forecast, DisruptionForecast)

    def test_all_nodes_present(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        for nid in NODE_IDS:
            assert nid in forecast.node_risks

    def test_risk_probabilities_range(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        for nr in forecast.node_risks.values():
            for p in nr.risk_by_day.values():
                assert 0.0 <= p <= 1.0

    def test_risk_matrix_shape(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        mat = forecast.risk_matrix()
        assert mat.shape == (len(NODE_IDS), 5)

    def test_high_risk_nodes_property(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        hr = forecast.high_risk_nodes
        assert all(r.peak_score > 0.6 for r in hr)

    def test_attribution_sums_to_one(self, signal_state):
        engine = CausalDisruptionEngine()
        forecast = engine.forecast(signal_state)
        for nr in forecast.node_risks.values():
            total = sum(nr.attribution.values())
            assert abs(total - 1.0) < 1e-5

    def test_save_load(self, signal_state, tmp_path):
        engine = CausalDisruptionEngine()
        engine.save(str(tmp_path))
        loaded = CausalDisruptionEngine.from_pretrained(str(tmp_path))
        assert isinstance(loaded, CausalDisruptionEngine)
