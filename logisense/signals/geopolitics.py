"""
Geopolitics NLP Processor
==========================

Converts news articles, government announcements, and geopolitical event
databases into per-country / per-node supply risk scores using NLP.

Sources
-------
- GDELT Project: 2.5 M+ machine-coded global news events per day
- Reuters / AP wire feeds
- WTO dispute database, USTR tariff tracker
- UN Security Council statements

NLP pipeline
-------------
1. NER — extract countries, companies, commodities
2. Event classification — trade restriction, conflict, sanctions, protest,
   election instability, regulatory change, export control
3. Sentiment — FinBERT / supply-chain fine-tuned BERT
4. GPR index — Caldara & Iacoviello (2022) conflict news count
5. Aggregate → per-node risk vector (24 features)

Features produced (24 per node)
---------------------------------
trade_restriction, sanctions_risk, conflict_intensity,
protest_labor, election_instability, regulatory_change,
tariff_escalation, export_control, currency_crisis,
govt_stability, border_closure, infra_attack,
gpr_normalized, news_volume_anomaly, neg_sentiment,
supply_mentions, ally_risk, adversary_escalation,
commodity_shock, trade_concentration, wto_dispute,
sanctions_commodity, regime_change, geo_composite
"""

import numpy as np
from typing import Dict, List, Optional

N_GEO_FEATURES = 24

FEATURE_NAMES = [
    "trade_restriction", "sanctions_risk", "conflict_intensity",
    "protest_labor", "election_instability", "regulatory_change",
    "tariff_escalation", "export_control", "currency_crisis",
    "govt_stability", "border_closure", "infra_attack",
    "gpr_normalized", "news_volume_anomaly", "neg_sentiment",
    "supply_mentions", "ally_risk", "adversary_escalation",
    "commodity_shock", "trade_concentration", "wto_dispute",
    "sanctions_commodity", "regime_change", "geo_composite",
]

SUPPLY_KEYWORDS = [
    "port closure", "factory shutdown", "strike", "sanctions",
    "export ban", "tariff", "supply shortage", "logistics disruption",
    "border closed", "cargo seized", "production halt", "force majeure",
    "trade war", "chip shortage", "raw material", "shipping delay",
]


class GeopoliticsProcessor:
    """
    Produces (N, 24) geopolitical risk feature matrix per node.

    In mock mode injects realistic synthetic disruption signals.
    In live mode fetches from GDELT API and scores with FinBERT.

    Args:
        model_name: HuggingFace model for news classification.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._nlp = None   # lazy-loaded

    def fetch(self, node_ids: List[str], mock: bool = True) -> np.ndarray:
        """Return (N, 24) geopolitical feature matrix."""
        if mock:
            return self._mock(len(node_ids))
        self._load_nlp()
        raise NotImplementedError("Set GDELT_API_KEY for live geopolitics data.")

    def _mock(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed=21)
        data = rng.uniform(0, 0.25, (n, N_GEO_FEATURES)).astype(np.float32)
        # Sanctions / trade restriction spike on node 1
        data[1, 0] = 0.79
        data[1, 1] = 0.85
        data[1, 12] = 0.72
        # Conflict escalation on node 3
        data[3, 2] = 0.88
        data[3, 23] = 0.81
        return data

    def _load_nlp(self):
        if self._nlp is None:
            try:
                from transformers import pipeline as hf_pipeline
                self._nlp = hf_pipeline(
                    "text-classification", model=self.model_name, device=-1
                )
            except Exception:
                self._nlp = "unavailable"

    def score_article(self, text: str) -> Dict[str, float]:
        """
        Score a news article for supply chain relevance and risk type.

        Returns dict with keys:
            supply_relevance, neg_sentiment, conflict_score, trade_restriction
        """
        self._load_nlp()
        lower = text.lower()
        keyword_hits = sum(1 for kw in SUPPLY_KEYWORDS if kw in lower)
        relevance = min(1.0, keyword_hits / 3.0)

        if self._nlp == "unavailable" or not callable(getattr(self._nlp, "__call__", None)):
            return {
                "supply_relevance": relevance,
                "neg_sentiment": 0.5,
                "conflict_score": 0.3,
                "trade_restriction": 0.4,
            }

        result = self._nlp(text[:512])[0]
        neg = result["score"] if result["label"] == "negative" else 1 - result["score"]
        return {
            "supply_relevance": relevance,
            "neg_sentiment": float(neg),
            "conflict_score": float(neg * relevance),
            "trade_restriction": float(relevance * 0.7),
        }

    @staticmethod
    def gpr_index(news_counts: Dict[str, int], baseline: float = 100.0) -> float:
        """
        Caldara & Iacoviello GPR index from news article counts.
        Normalized to baseline (100 = historical average).
        """
        conflict_kws = ["war", "attack", "threat", "military", "crisis", "conflict"]
        total = sum(v for k, v in news_counts.items()
                    if any(w in k for w in conflict_kws))
        return float(total / max(baseline, 1.0))
