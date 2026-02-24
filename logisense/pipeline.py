"""
LogiSense Pipeline Orchestrator
=================================

Wires together all four components into a single end-to-end call:

    signals  = SignalFusionEngine.fetch_and_fuse(...)
    forecast = CausalDisruptionEngine.forecast(signals)
    twin.apply_risk_scores(forecast)
    state    = twin.simulate(steps=horizon_days)
    actions  = MitigationAgent.act(state)

Returns a PipelineResult with the forecast, twin state, and
recommended mitigation actions.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from logisense.signals  import SignalFusionEngine
from logisense.causal   import CausalDisruptionEngine, DisruptionForecast
from logisense.twin     import DigitalTwin
from logisense.agent    import MitigationAgent, MitigationAction
from logisense.twin.digital_twin import TwinState

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full output of one LogiSense pipeline run."""
    forecast:   DisruptionForecast
    twin_state: TwinState
    actions:    List[MitigationAction]
    network_id: str
    horizon_days: int
    metadata:   dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  LogiSense Pipeline Result — network={self.network_id}",
            "=" * 60,
            f"  Horizon:        {self.horizon_days} days",
            f"  Nodes assessed: {len(self.forecast.node_risks)}",
            f"  High-risk:      {len(self.forecast.high_risk_nodes)}",
            "",
            "  Top Risks:",
        ]
        for r in self.forecast.top_nodes(n=5):
            bar = "█" * int(r.peak_score * 20)
            lines.append(f"    {r.node_id:<12s}  day {r.peak_risk_day:>2d}  "
                          f"{r.peak_score:.1%}  {bar}")
        lines += ["", "  Recommended Actions:"]
        for a in self.actions:
            lines.append(f"    [{a.priority:<6s}] {a.description}")
            lines.append(f"             Impact: {a.expected_impact}")
            lines.append(f"             Cost:   ${a.estimated_cost:>8,.0f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class LogiSensePipeline:
    """
    End-to-end LogiSense pipeline.

    Args:
        signal_engine:  SignalFusionEngine instance.
        causal_engine:  CausalDisruptionEngine instance.
        twin:           DigitalTwin instance.
        agent:          MitigationAgent instance.
        mock_signals:   Use synthetic signals (True for demo / testing).
    """

    def __init__(
        self,
        signal_engine: Optional[SignalFusionEngine]     = None,
        causal_engine: Optional[CausalDisruptionEngine] = None,
        twin:          Optional[DigitalTwin]             = None,
        agent:         Optional[MitigationAgent]         = None,
        mock_signals:  bool = True,
    ):
        self.signals      = signal_engine or SignalFusionEngine()
        self.causal       = causal_engine or CausalDisruptionEngine()
        self.twin         = twin          or DigitalTwin.sample()
        self.agent        = agent         or MitigationAgent(obs_dim=self.twin.obs_dim)
        self.mock_signals = mock_signals

    # ── run ──────────────────────────────────────────────────────────────

    def run(
        self,
        network_id:    str  = "default",
        horizon_days:  int  = 14,
        top_k_actions: int  = 3,
        update_dag:    bool = False,
    ) -> PipelineResult:
        """
        Execute the full pipeline once.

        Args:
            network_id:    Supply network identifier.
            horizon_days:  Forecast and simulation horizon.
            top_k_actions: Number of mitigation actions to return.
            update_dag:    Re-learn causal DAG (slow; do weekly).

        Returns:
            PipelineResult
        """
        logger.info("=" * 55)
        logger.info("  LogiSense — pipeline start  network=%s", network_id)
        logger.info("=" * 55)

        # 1. Signal fusion
        logger.info("[1/4] Signal fusion...")
        node_ids   = list(self.twin.network.nodes.keys())
        node_types = {nid: int(n.node_type)
                      for nid, n in self.twin.network.nodes.items()}
        sig_state  = self.signals.fetch_and_fuse(
            network_id=network_id,
            node_ids=node_ids,
            node_types=node_types,
            mock=self.mock_signals,
        )

        # 2. Causal disruption forecast
        logger.info("[2/4] Causal disruption forecast...")
        forecast = self.causal.forecast(sig_state, update_dag=update_dag)
        logger.info("  %s", forecast)

        # 3. Digital twin simulation
        logger.info("[3/4] Digital twin simulation (%dd)...", horizon_days)
        self.twin.apply_risk_scores(forecast)
        twin_state = self.twin.simulate(steps=horizon_days)

        # 4. RL agent — select mitigations
        logger.info("[4/4] RL agent selecting mitigations...")
        actions = self.agent.act(twin_state, top_k=top_k_actions)

        result = PipelineResult(
            forecast=forecast,
            twin_state=twin_state,
            actions=actions,
            network_id=network_id,
            horizon_days=horizon_days,
        )
        logger.info(result.summary())
        return result

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str) -> "LogiSensePipeline":
        """Instantiate pipeline from a YAML config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        twin  = DigitalTwin.sample(n_nodes=cfg.get("n_nodes", 20))
        agent = MitigationAgent(
            obs_dim=twin.obs_dim,
            n_actions=cfg.get("n_actions", 64),
        )
        return cls(
            twin=twin,
            agent=agent,
            mock_signals=cfg.get("mock_signals", True),
        )

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> "LogiSensePipeline":
        """Load all pretrained weights from a checkpoint directory."""
        ckpt = Path(checkpoint_dir)
        pipeline = cls(
            signal_engine=SignalFusionEngine.from_pretrained(str(ckpt / "signals")),
            causal_engine=CausalDisruptionEngine.from_pretrained(str(ckpt / "causal")),
        )
        return pipeline
