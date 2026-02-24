"""
Supplier Sentiment Processor  (FinBERT)
========================================

Extracts financial distress and operational risk signals from supplier
earnings calls, SEC filings, credit markets, and news using FinBERT.

Why sentiment has the longest lead time
-----------------------------------------
Financial distress in an earnings call typically precedes actual supply
disruption by 30–90 days.  CDS spread widening can precede bankruptcy by
45–120 days.  Sentiment signals are the earliest warning available in
the LogiSense stack.

Sources
-------
- SEC EDGAR: 8-K / 10-K filings, earnings call transcripts
- CDS spreads: 5-year credit default swap market
- News wire: workforce reduction, factory closure, debt downgrade
- ESG databases: labor / environmental compliance flags

Features produced (32 per node)
---------------------------------
distress_score, earnings_neg_sentiment, cds_spread_z,
going_concern, layoff_mentions, factory_closure,
debt_downgrade, litigation_risk, esg_labor,
esg_env, revenue_miss, margin_compression,
capex_cut, inventory_writedown, supplier_concentration,
sole_source, payment_delay, credit_utilization,
mgmt_turnover, force_majeure, strike_signal,
absenteeism_proxy, regulatory_risk, cert_lapse,
financial_health, operational_risk, geo_exposure,
single_site, tech_obsolescence, quality_incidents,
delivery_decline, overall_risk

Reference
----------
Yang et al. (2020). FinBERT: A Pretrained Language Model for Financial
Communications. arXiv:2006.08097.
"""

from typing import Dict, List, Optional

import numpy as np

N_SENTIMENT_FEATURES = 32

FEATURE_NAMES = [
    "distress_score",
    "earnings_neg_sentiment",
    "cds_spread_z",
    "going_concern",
    "layoff_mentions",
    "factory_closure",
    "debt_downgrade",
    "litigation_risk",
    "esg_labor",
    "esg_env",
    "revenue_miss",
    "margin_compression",
    "capex_cut",
    "inventory_writedown",
    "supplier_concentration",
    "sole_source",
    "payment_delay",
    "credit_utilization",
    "mgmt_turnover",
    "force_majeure",
    "strike_signal",
    "absenteeism_proxy",
    "regulatory_risk",
    "cert_lapse",
    "financial_health",
    "operational_risk",
    "geo_exposure",
    "single_site",
    "tech_obsolescence",
    "quality_incidents",
    "delivery_decline",
    "overall_risk",
]

HIGH_RISK_PHRASES = [
    "force majeure",
    "going concern",
    "material uncertainty",
    "production halt",
    "facility closure",
    "workforce reduction",
    "supplier default",
    "raw material shortage",
    "inventory depletion",
    "quality hold",
    "regulatory hold",
    "shipment delay",
    "capacity constraint",
    "single source",
    "cash flow concerns",
]


class SentimentProcessor:
    """
    Produces (N, 32) supplier sentiment feature matrix.

    Args:
        model_name:  HuggingFace FinBERT model identifier.
        threshold:   Minimum score to flag text as negative.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        threshold: float = 0.6,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    def fetch(self, node_ids: List[str], mock: bool = True) -> np.ndarray:
        """Return (N, 32) sentiment feature matrix."""
        if mock:
            return self._mock(len(node_ids))
        self._load()
        raise NotImplementedError("Live supplier sentiment requires API credentials.")

    def _mock(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed=37)
        data = rng.uniform(0, 0.2, (n, N_SENTIMENT_FEATURES)).astype(np.float32)
        # Distress on node 0
        data[0, 0] = 0.83  # distress
        data[0, 3] = 0.77  # going concern
        data[0, 4] = 0.72  # layoff mentions
        data[0, 31] = 0.81  # overall risk
        # Moderate risk on node 4
        data[4, 24] = 0.61
        data[4, 25] = 0.58
        return data

    def _load(self):
        if self._model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._model = hf_pipeline(
                    "text-classification", model=self.model_name, device=-1
                )
            except Exception:
                self._model = "unavailable"

    def score_transcript(self, text: str) -> Dict[str, float]:
        """
        Score an earnings call transcript for supply risk.

        Returns dict with:
            distress_score, neg_sentiment, phrase_density
        """
        self._load()
        lower = text.lower()
        hits = sum(1 for p in HIGH_RISK_PHRASES if p in lower)
        phrase_density = min(1.0, hits / 5.0)

        if self._model in (None, "unavailable"):
            return {
                "distress_score": phrase_density * 0.8,
                "neg_sentiment": phrase_density * 0.7,
                "phrase_density": phrase_density,
            }

        chunks = [text[i : i + 512] for i in range(0, min(len(text), 4096), 512)]
        results = self._model(chunks)
        neg_scores = [
            r["score"] if r["label"] == "negative" else 1 - r["score"] for r in results
        ]
        avg_neg = float(np.mean(neg_scores))
        return {
            "distress_score": max(avg_neg, phrase_density),
            "neg_sentiment": avg_neg,
            "phrase_density": phrase_density,
        }

    @staticmethod
    def cds_zscore(current_bps: float, mean_bps: float, std_bps: float) -> float:
        """
        Z-score of CDS spread vs historical baseline, clamped [0, 5].
        Spreads > 500 bps signal significant default concern.
        """
        if std_bps < 1e-9:
            return 0.0
        return float(np.clip((current_bps - mean_bps) / std_bps, 0.0, 5.0))
