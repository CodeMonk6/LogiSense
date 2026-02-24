"""
LogiSense — Interactive Demo
==============================

Runs the full pipeline and renders a Rich-formatted terminal dashboard.

Usage:
    python demo.py
    python demo.py --scenario suez_disruption
    python demo.py --network configs/sample_network.yaml --horizon 14
"""

import argparse
import sys
import time

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text    import Text
    from rich         import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from logisense import LogiSensePipeline

console = Console() if HAS_RICH else None


def _risk_colour(score: float) -> str:
    if score >= 0.70: return "bold red"
    if score >= 0.50: return "bold yellow"
    if score >= 0.30: return "yellow"
    return "green"


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def render_dashboard(result) -> None:
    if not HAS_RICH:
        print(result.summary())
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]LogiSense[/bold cyan] — Autonomous Preemptive Supply Chain Resilience",
        border_style="cyan",
    ))

    # ── Risk Forecast Table ────────────────────────────────────────────
    rt = Table("Node", "7-Day Risk", "14-Day Risk", "Peak Day", "Peak Risk",
               "Risk Bar", "Attribution",
               title="[bold]Disruption Risk Forecast[/bold]",
               box=box.ROUNDED, show_lines=True)

    for nr in result.forecast.top_nodes(n=12):
        colour   = _risk_colour(nr.peak_score)
        bar      = _bar(nr.peak_score)
        top_attr = max(nr.attribution, key=nr.attribution.get) if nr.attribution else "—"
        rt.add_row(
            nr.node_id,
            f"{nr.risk_7d:.0%}",
            f"{nr.risk_14d:.0%}",
            f"Day {nr.peak_day}",
            Text(f"{nr.peak_score:.0%}", style=colour),
            Text(bar[:10], style=colour),
            top_attr,
        )

    console.print(rt)

    # ── Twin KPIs ─────────────────────────────────────────────────────
    kpis = result.twin_state.kpis
    kt   = Table("KPI", "Value", title="[bold]Digital Twin KPIs[/bold]",
                  box=box.SIMPLE)
    kt.add_row("Service Level",    f"{kpis.get('service_level', 0):.1%}")
    kt.add_row("Avg Fill Rate",    f"{kpis.get('avg_fill_rate', 0):.1%}")
    kt.add_row("Min Fill Rate",    f"{kpis.get('min_fill_rate', 0):.1%}")
    kt.add_row("Stockout Nodes",   str(kpis.get('stockout_nodes', 0)))
    kt.add_row("High-Risk Nodes",  str(kpis.get('high_risk_nodes', 0)))
    kt.add_row("Simulation Day",   str(result.twin_state.day))
    console.print(kt)

    # ── Recommended Actions ───────────────────────────────────────────
    at = Table("Priority", "Action", "Target", "Expected Impact", "Est. Cost",
               title="[bold]RL Agent — Recommended Mitigations[/bold]",
               box=box.ROUNDED, show_lines=True)

    priority_style = {"HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "green"}

    for a in result.actions:
        style = priority_style.get(a.priority, "white")
        at.add_row(
            Text(a.priority, style=style),
            a.action_type.upper(),
            a.target,
            a.expected_impact[:60],
            f"${a.estimated_cost:>8,.0f}",
        )

    console.print(at)

    # ── Signal Attribution ────────────────────────────────────────────
    if result.forecast.node_risks:
        top_node = result.forecast.top_nodes(1)[0]
        sa       = Table("Signal Source", "Attribution", "Bar",
                          title=f"[bold]Signal Attribution — {top_node.node_id}[/bold]",
                          box=box.SIMPLE)
        attrs = sorted(top_node.attribution.items(), key=lambda x: x[1], reverse=True)
        for src, score in attrs:
            bar = "█" * int(score * 30)
            sa.add_row(src, f"{score:.1%}", Text(bar, style=_risk_colour(score)))
        console.print(sa)

    console.print(Panel.fit(
        f"[dim]LogiSense v0.1.0 — For research purposes only[/dim]",
        border_style="dim"
    ))


def main():
    parser = argparse.ArgumentParser(description="LogiSense interactive demo")
    parser.add_argument("--scenario", default=None,
                        help="Pre-built scenario name (e.g. suez_disruption)")
    parser.add_argument("--network",  default="configs/sample_network.yaml",
                        help="Supply network config YAML")
    parser.add_argument("--horizon",  type=int, default=14,
                        help="Forecast horizon in days")
    parser.add_argument("--n-nodes",  type=int, default=20)
    args = parser.parse_args()

    if HAS_RICH:
        with Progress(SpinnerColumn(),
                      TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                      console=console, transient=True) as progress:
            task = progress.add_task("Initialising LogiSense pipeline...", total=None)
            pipeline = LogiSensePipeline(mock_signals=True)
            progress.update(task, description="Running pipeline...")
            network_id = args.scenario or "demo_network"
            result = pipeline.run(
                network_id=network_id,
                horizon_days=args.horizon,
                top_k_actions=3,
            )
    else:
        print("Installing 'rich' gives a nicer dashboard:  pip install rich")
        pipeline = LogiSensePipeline(mock_signals=True)
        result   = pipeline.run(network_id="demo_network", horizon_days=args.horizon)

    render_dashboard(result)


if __name__ == "__main__":
    main()
