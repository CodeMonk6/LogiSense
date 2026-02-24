"""Integration tests — full pipeline end to end."""

import pytest
from logisense import LogiSensePipeline
from logisense.pipeline import PipelineResult


class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        return LogiSensePipeline(mock_signals=True)

    def test_run_returns_result(self, pipeline):
        result = pipeline.run(horizon_days=3)
        assert isinstance(result, PipelineResult)

    def test_forecast_populated(self, pipeline):
        result = pipeline.run(horizon_days=3)
        assert len(result.forecast.node_risks) > 0

    def test_actions_populated(self, pipeline):
        result = pipeline.run(horizon_days=3)
        assert len(result.actions) >= 1

    def test_twin_state_obs_no_nan(self, pipeline):
        import numpy as np
        result = pipeline.run(horizon_days=3)
        assert not np.isnan(result.twin_state.obs).any()

    def test_summary_is_string(self, pipeline):
        result = pipeline.run(horizon_days=3)
        s = result.summary()
        assert isinstance(s, str)
        assert "LogiSense" in s

    def test_from_config(self, tmp_path):
        import yaml
        cfg = {
            "n_nodes": 10,
            "n_actions": 16,
            "mock_signals": True,
        }
        p = tmp_path / "config.yaml"
        with open(p, "w") as f:
            yaml.dump(cfg, f)
        pl = LogiSensePipeline.from_config(str(p))
        result = pl.run(horizon_days=2)
        assert isinstance(result, PipelineResult)
