"""Tests for the signal fusion pipeline."""

import numpy as np
import torch
import pytest

from logisense.signals import SignalFusionEngine, FusedSignalState
from logisense.signals.satellite   import SatelliteProcessor, N_SATELLITE_FEATURES
from logisense.signals.weather     import WeatherProcessor,   N_WEATHER_FEATURES
from logisense.signals.geopolitics import GeopoliticsProcessor, N_GEO_FEATURES
from logisense.signals.sentiment   import SentimentProcessor,  N_SENTIMENT_FEATURES


NODE_IDS = [f"node_{i:03d}" for i in range(10)]


class TestSignalProcessors:
    def test_satellite_shape(self):
        proc = SatelliteProcessor()
        out  = proc.fetch(NODE_IDS, mock=True)
        assert out.shape == (10, N_SATELLITE_FEATURES)
        assert out.dtype == np.float32
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_weather_shape(self):
        out = WeatherProcessor().fetch(NODE_IDS, mock=True)
        assert out.shape == (10, N_WEATHER_FEATURES)

    def test_geopolitics_shape(self):
        out = GeopoliticsProcessor().fetch(NODE_IDS, mock=True)
        assert out.shape == (10, N_GEO_FEATURES)

    def test_sentiment_shape(self):
        out = SentimentProcessor().fetch(NODE_IDS, mock=True)
        assert out.shape == (10, N_SENTIMENT_FEATURES)

    def test_cyclone_risk_formula(self):
        risk = WeatherProcessor.cyclone_risk(
            node_lat=25.0, node_lon=121.0,
            storm_lat=22.0, storm_lon=118.0,
            category=4, landfall_hours=24.0,
        )
        assert 0.0 <= risk <= 1.0

    def test_cds_zscore(self):
        z = SentimentProcessor.cds_zscore(600, 250, 80)
        assert z > 0
        # Clamp at 5
        z2 = SentimentProcessor.cds_zscore(10000, 250, 80)
        assert z2 == 5.0


class TestSignalFusionEngine:
    @pytest.fixture
    def engine(self):
        return SignalFusionEngine(T_lookback=10)

    def test_output_type(self, engine):
        state = engine.fetch_and_fuse("test_net", node_ids=NODE_IDS, mock=True)
        assert isinstance(state, FusedSignalState)

    def test_output_shape(self, engine):
        state = engine.fetch_and_fuse("test_net", node_ids=NODE_IDS, mock=True)
        assert state.signal_tensor.shape == (10, 10, 84)
        assert len(state.risk_scores) == 10

    def test_risk_scores_range(self, engine):
        state = engine.fetch_and_fuse("test_net", node_ids=NODE_IDS, mock=True)
        assert np.all(state.risk_scores >= 0.0)
        assert np.all(state.risk_scores <= 1.0)

    def test_top_risk_nodes(self, engine):
        state = engine.fetch_and_fuse("test_net", node_ids=NODE_IDS, mock=True)
        top5  = state.top_risk_nodes(5)
        assert len(top5) == 5
        scores = [s for _, s in top5]
        assert scores == sorted(scores, reverse=True)

    def test_node_signal_feature_vector_dim(self, engine):
        state = engine.fetch_and_fuse("test_net", node_ids=NODE_IDS, mock=True)
        ns = state.node_signals[NODE_IDS[0]]
        assert ns.feature_vector.shape == (84,)

    def test_save_load(self, engine, tmp_path):
        engine.save(str(tmp_path))
        loaded = SignalFusionEngine.from_pretrained(str(tmp_path))
        assert isinstance(loaded, SignalFusionEngine)
