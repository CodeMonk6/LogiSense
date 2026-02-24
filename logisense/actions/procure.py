"""
Contingent Procurement Executor
=================================

Triggers contingent / emergency procurement from pre-qualified alternate
suppliers when a primary supplier is flagged as high-risk.

Decision logic
---------------
1. Identify primary suppliers with risk_score > threshold.
2. For each, look up pre-qualified alternate suppliers (from vendor master).
3. Score alternates: lead_time, cost_premium, quality_score, risk_score.
4. Select best alternate and issue contingent purchase order.
5. Track in-flight contingent orders to avoid duplicate triggers.

In production this integrates with ERP (SAP Ariba, Oracle Procurement).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContingentPO:
    """A contingent purchase order to an alternate supplier."""

    po_id: str
    primary_supplier: str
    alt_supplier: str
    item_code: str
    quantity: float
    unit_cost: float
    lead_time_days: float
    total_cost: float
    trigger_reason: str
    status: str = "PENDING"  # PENDING | CONFIRMED | CANCELLED


@dataclass
class SupplierProfile:
    """Pre-qualified alternate supplier profile."""

    supplier_id: str
    name: str
    lead_time_days: float
    unit_cost: float
    quality_score: float  # 0-1
    risk_score: float  # current risk
    capacity_units: float  # max order units
    items: List[str] = field(default_factory=list)


# Sample vendor master (in production loaded from ERP)
_SAMPLE_ALTERNATES: Dict[str, List[SupplierProfile]] = {
    "node_000": [
        SupplierProfile(
            "alt_sup_001",
            "Alternate Supplier A",
            12.0,
            1.05,
            0.92,
            0.10,
            10_000,
            ["ITEM_A", "ITEM_B"],
        ),
        SupplierProfile(
            "alt_sup_002",
            "Alternate Supplier B",
            18.0,
            0.98,
            0.88,
            0.15,
            8_000,
            ["ITEM_A"],
        ),
    ],
    "node_001": [
        SupplierProfile(
            "alt_sup_003",
            "Alternate Supplier C",
            10.0,
            1.12,
            0.95,
            0.08,
            12_000,
            ["ITEM_C"],
        ),
    ],
}


class ProcureExecutor:
    """
    Issues contingent procurement to alternate suppliers.

    Args:
        vendor_master: Dict of primary_supplier_id → List[SupplierProfile].
        risk_threshold: Trigger procurement when primary supplier risk > this.
    """

    def __init__(
        self,
        vendor_master: Optional[Dict] = None,
        risk_threshold: float = 0.55,
    ):
        self.vendor_master = vendor_master or _SAMPLE_ALTERNATES
        self.risk_threshold = risk_threshold
        self._po_counter = 0
        self._active_pos: List[ContingentPO] = []

    def evaluate(
        self,
        primary_supplier: str,
        item_code: str = "ITEM_A",
        quantity: float = 5_000.0,
        trigger_risk: float = 0.60,
    ) -> List[ContingentPO]:
        """
        Evaluate contingent PO options for a primary supplier at risk.

        Returns:
            Ranked list of ContingentPO (best alternate first).
        """
        alternates = self.vendor_master.get(primary_supplier, [])
        if not alternates:
            logger.warning("No alternates found for %s — using mock.", primary_supplier)
            alternates = list(_SAMPLE_ALTERNATES.get("node_000", []))

        candidates = []
        for alt in alternates:
            if item_code not in alt.items and item_code != "ANY":
                continue
            if alt.risk_score > 0.5:
                continue  # alternate is also at risk — skip

            qty = min(quantity, alt.capacity_units)
            total_cost = qty * alt.unit_cost

            # Score: lower is better
            score = (
                0.5 * alt.lead_time_days / 30.0
                + 0.3 * (alt.unit_cost - 1.0)
                + 0.2 * alt.risk_score
            )

            candidates.append(
                (
                    score,
                    ContingentPO(
                        po_id=self._next_po_id(),
                        primary_supplier=primary_supplier,
                        alt_supplier=alt.supplier_id,
                        item_code=item_code,
                        quantity=qty,
                        unit_cost=alt.unit_cost,
                        lead_time_days=alt.lead_time_days,
                        total_cost=total_cost,
                        trigger_reason=f"Primary supplier risk={trigger_risk:.0%}",
                    ),
                )
            )

        candidates.sort(key=lambda x: x[0])
        return [po for _, po in candidates[:3]]

    def execute(self, po: ContingentPO, dry_run: bool = True) -> Dict:
        """Issue the contingent PO (or dry-run)."""
        if dry_run:
            logger.info(
                "[Procure DRY-RUN] PO %s: %s units from %s — $%.0f",
                po.po_id,
                po.quantity,
                po.alt_supplier,
                po.total_cost,
            )
        else:
            po.status = "CONFIRMED"
            self._active_pos.append(po)
            logger.info("[Procure EXECUTE] PO %s confirmed.", po.po_id)

        return {
            "status": "dry_run" if dry_run else "confirmed",
            "po_id": po.po_id,
            "alt_supplier": po.alt_supplier,
            "quantity": po.quantity,
            "total_cost_usd": po.total_cost,
            "lead_time_days": po.lead_time_days,
        }

    def _next_po_id(self) -> str:
        self._po_counter += 1
        return f"CPO-{self._po_counter:05d}"
