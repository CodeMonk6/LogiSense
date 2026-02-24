"""
Run Disruption Scenario
=========================

Loads a pre-built scenario YAML and runs the full LogiSense pipeline
with injected disruption signals.

Usage:
    python scripts/run_scenario.py --scenario suez_disruption
    python scripts/run_scenario.py --scenario taiwan_strait_tension --horizon 21
"""

import argparse
import sys
import yaml
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logisense import LogiSensePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent.parent / "data" / "scenarios"


def load_scenario(name: str) -> dict:
    path = SCENARIOS_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in SCENARIOS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Scenario '{name}' not found. Available: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def print_scenario_header(scenario: dict) -> None:
    print()
    print("╔" + "═" * 60 + "╗")
    print(f"║  SCENARIO: {scenario['name']:<48s}║")
    print(f"║  Type:     {scenario['type']:<48s}║")
    print("╠" + "═" * 60 + "╣")
    desc = scenario.get("description", "")
    for line in desc.strip().splitlines():
        print(f"║  {line.strip():<58s}║")
    print("╚" + "═" * 60 + "╝")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="suez_disruption",
                        help="Scenario name (without .yaml)")
    parser.add_argument("--horizon",  type=int, default=14)
    parser.add_argument("--n-nodes",  type=int, default=20)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    print_scenario_header(scenario)

    pipeline = LogiSensePipeline(mock_signals=True)
    result   = pipeline.run(
        network_id=scenario["name"],
        horizon_days=args.horizon,
        top_k_actions=3,
    )

    print(result.summary())

    # Report vs expected outcomes
    expected = scenario.get("expected_outcomes", {})
    if expected:
        print("\n  Expected Outcomes (from scenario spec):")
        for k, v in expected.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
