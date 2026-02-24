"""
LogiSense — Autonomous Preemptive Supply Chain Resilience Platform
==================================================================

Four integrated components:

    SignalFusionEngine      — Fuses satellite, weather, geopolitics, sentiment
    CausalDisruptionEngine  — NOTEARS DAG + Temporal Causal Transformer
    DigitalTwin             — Graph-based supply network simulation
    MitigationAgent         — PPO reinforcement learning over twin state

Usage:
    from logisense import LogiSensePipeline
    pipeline = LogiSensePipeline.from_config("configs/full_pipeline.yaml")
    result   = pipeline.run(network_id="my_network", horizon_days=14)
"""

from logisense.pipeline import LogiSensePipeline
from logisense.signals  import SignalFusionEngine
from logisense.causal   import CausalDisruptionEngine
from logisense.twin     import DigitalTwin
from logisense.agent    import MitigationAgent

__version__ = "0.1.0"
__author__  = "Sourabh Sharma"

__all__ = [
    "LogiSensePipeline",
    "SignalFusionEngine",
    "CausalDisruptionEngine",
    "DigitalTwin",
    "MitigationAgent",
]
