"""
Supply Chain & Forecast Evaluation Metrics
============================================

Metrics for:
    - Disruption forecast evaluation (precision, recall, lead time)
    - Mitigation effectiveness (service level, cost efficiency)
    - General ML (RMSE, Pearson, Spearman)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SupplyChainMetrics:
    """Collection of supply chain and forecasting evaluation metrics."""

    # ── forecast metrics ─────────────────────────────────────────────────

    @staticmethod
    def precision_at_k(
        predicted_risks: np.ndarray,   # (N,) predicted risk scores
        true_disrupted:  np.ndarray,   # (N,) binary disruption labels
        k:               int = 5,
    ) -> float:
        """
        Precision@K: fraction of top-K risk-flagged nodes that
        actually experienced a disruption.
        """
        top_k_idx    = np.argsort(predicted_risks)[::-1][:k]
        true_pos     = true_disrupted[top_k_idx].sum()
        return float(true_pos / k)

    @staticmethod
    def forecast_lead_time(
        predicted_risk_series: np.ndarray,   # (T,) risk score over time
        actual_onset_day:      int,
        threshold:             float = 0.5,
    ) -> float:
        """
        Days of lead time: first day forecast > threshold vs actual onset.
        Returns 0 if forecast never exceeds threshold before onset.
        """
        for t in range(actual_onset_day):
            if predicted_risk_series[t] >= threshold:
                return float(actual_onset_day - t)
        return 0.0

    @staticmethod
    def avg_precision(
        predicted: np.ndarray,
        true:      np.ndarray,
    ) -> float:
        """Average Precision (AP) for disruption forecasting."""
        sorted_idx = np.argsort(predicted)[::-1]
        true_sorted = true[sorted_idx]
        n_pos = true.sum()
        if n_pos == 0:
            return 0.0
        precision_vals = np.cumsum(true_sorted) / np.arange(1, len(true_sorted) + 1)
        return float((precision_vals * true_sorted).sum() / n_pos)

    # ── service level metrics ─────────────────────────────────────────────

    @staticmethod
    def service_level(fill_rates: np.ndarray, threshold: float = 0.95) -> float:
        """Fraction of nodes meeting fill-rate threshold."""
        return float((fill_rates >= threshold).mean())

    @staticmethod
    def weighted_service_level(
        fill_rates:   np.ndarray,
        node_weights: np.ndarray,
    ) -> float:
        """Demand-weighted average fill rate."""
        return float(np.average(fill_rates, weights=node_weights))

    @staticmethod
    def inventory_cover_distribution(
        inventories:  np.ndarray,
        daily_demands: np.ndarray,
    ) -> Dict[str, float]:
        """Summary stats for inventory cover (days) across nodes."""
        cover = inventories / np.maximum(daily_demands, 1.0)
        return {
            "mean_days":   float(np.mean(cover)),
            "median_days": float(np.median(cover)),
            "p5_days":     float(np.percentile(cover, 5)),
            "min_days":    float(np.min(cover)),
        }

    # ── ML metrics ───────────────────────────────────────────────────────

    @staticmethod
    def rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def mae(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - true)))

    @staticmethod
    def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
        if pred.std() < 1e-8 or true.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(pred, true)[0, 1])

    @staticmethod
    def spearman_r(pred: np.ndarray, true: np.ndarray) -> float:
        from scipy.stats import spearmanr
        r, _ = spearmanr(pred, true)
        return float(r)

    # ── mitigation cost-effectiveness ────────────────────────────────────

    @staticmethod
    def cost_of_disruption(
        stockout_units:  float,
        avg_margin_usd:  float,
        backorder_cost:  float,
        lost_sale_frac:  float = 0.30,
    ) -> float:
        """
        Estimated cost of a disruption event.

        Args:
            stockout_units:  Units unable to be fulfilled.
            avg_margin_usd:  Revenue margin per unit.
            backorder_cost:  Handling cost per backordered unit.
            lost_sale_frac:  Fraction of stockouts that become lost sales.
        """
        lost     = stockout_units * lost_sale_frac * avg_margin_usd
        backorder = stockout_units * (1 - lost_sale_frac) * backorder_cost
        return float(lost + backorder)

    @staticmethod
    def mitigation_roi(
        disruption_cost_baseline: float,
        disruption_cost_mitigated: float,
        mitigation_cost:          float,
    ) -> float:
        """
        Return on investment for mitigation actions.
        ROI = (savings − mitigation_cost) / mitigation_cost
        """
        savings = disruption_cost_baseline - disruption_cost_mitigated
        if mitigation_cost < 1e-3:
            return float("inf")
        return float((savings - mitigation_cost) / mitigation_cost)
