"""
NOTEARS — Differentiable DAG Structure Learning
=================================================

Learns a Directed Acyclic Graph (DAG) over supply chain variables via
continuous optimisation instead of combinatorial search.

Key idea
---------
Reformulate the acyclicity constraint as a smooth algebraic condition:

    h(W) = tr(e^{W ∘ W}) − d = 0   iff W is acyclic

Optimisation
-------------
    min  ½‖X − XW‖²_F + λ₁‖W‖₁ + ρ/2 · h(W)² + α · h(W)
     W

Solved via augmented Lagrangian with L-BFGS inner steps.

In LogiSense, NOTEARS learns which supply chain signals causally
influence which nodes, producing a structural map of disruption
propagation:

    geopolitical escalation → export restriction → port congestion
    → shipment delay → inventory stockout

Reference
----------
Zheng et al. (2018). DAGs with NO TEARS: Continuous Optimization for
Structure Learning. NeurIPS 2018.
"""

import logging
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class NOTEARSLearner:
    """
    NOTEARS continuous structure learning.

    W[i, j] > 0  ⟹  variable j is a causal parent of variable i.

    Args:
        n_vars:    Number of variables.
        lambda1:   L1 sparsity weight.
        lambda2:   Acyclicity penalty (unused — handled via rho / alpha).
        rho_init:  Initial augmented Lagrangian penalty.
        rho_max:   Maximum rho before early stop.
        h_tol:     Acyclicity tolerance.
        max_iter:  Maximum outer iterations.
        threshold: Post-fit edge pruning threshold.
    """

    def __init__(
        self,
        n_vars: int = 84,
        lambda1: float = 0.01,
        lambda2: float = 0.01,
        rho_init: float = 1.0,
        rho_max: float = 1e16,
        h_tol: float = 1e-8,
        max_iter: int = 100,
        threshold: float = 0.3,
    ):
        self.n_vars = n_vars
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.h_tol = h_tol
        self.max_iter = max_iter
        self.threshold = threshold
        self.W_: Optional[np.ndarray] = None

    # ── constraint and its gradient ──────────────────────────────────────

    def _h(self, W: np.ndarray) -> float:
        return float(np.trace(expm(W * W)) - W.shape[0])

    def _h_grad(self, W: np.ndarray) -> np.ndarray:
        return 2.0 * W * expm(W * W).T

    # ── L-BFGS objective ─────────────────────────────────────────────────

    def _objective(
        self,
        W_flat: np.ndarray,
        X: np.ndarray,
        rho: float,
        alpha: float,
    ):
        d = self.n_vars
        W = W_flat.reshape(d, d)
        n = X.shape[0]

        res = X - X @ W
        loss_rec = 0.5 / n * (res**2).sum()
        grad_rec = -1.0 / n * X.T @ res

        loss_l1 = self.lambda1 * np.abs(W).sum()
        grad_l1 = self.lambda1 * np.sign(W)

        h = self._h(W)
        loss_h = 0.5 * rho * h**2 + alpha * h
        grad_h = (rho * h + alpha) * self._h_grad(W)

        return (
            float(loss_rec + loss_l1 + loss_h),
            (grad_rec + grad_l1 + grad_h).flatten(),
        )

    # ── public interface ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Learn causal DAG from observations X.

        Args:
            X: (n_samples, n_vars) data matrix.

        Returns:
            W: (n_vars, n_vars) thresholded adjacency matrix.
        """
        n, d = X.shape
        if d != self.n_vars:
            logger.warning("Adapting n_vars from %d to %d", self.n_vars, d)
            self.n_vars = d

        X = (X - X.mean(0)).astype(np.float64)
        W = np.zeros((d, d), dtype=np.float64)
        rho, alpha, h_prev = self.rho_init, 0.0, np.inf

        logger.info("NOTEARS fitting on (%d × %d) ...", n, d)

        for it in range(self.max_iter):
            result = minimize(
                self._objective,
                W.flatten(),
                args=(X, rho, alpha),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 100, "ftol": 1e-12},
            )
            W = result.x.reshape(d, d)
            np.fill_diagonal(W, 0)

            h = self._h(W)
            logger.debug("iter %3d  h=%.2e  rho=%.1e", it, h, rho)

            if h <= self.h_tol:
                logger.info("NOTEARS converged at iter %d (h=%.2e)", it, h)
                break

            alpha += rho * h
            if h > 0.25 * h_prev:
                rho = min(rho * 10, self.rho_max)
            h_prev = h

            if rho >= self.rho_max:
                logger.warning("NOTEARS hit rho_max — partial convergence.")
                break

        W[np.abs(W) < self.threshold] = 0.0
        np.fill_diagonal(W, 0)
        self.W_ = W

        n_edges = int((W != 0).sum())
        logger.info("NOTEARS complete — %d causal edges learned.", n_edges)
        return W.astype(np.float32)

    def parents(self, var_idx: int) -> np.ndarray:
        """Indices of causal parents of variable var_idx."""
        if self.W_ is None:
            return np.array([], dtype=int)
        return np.where(self.W_[var_idx] != 0)[0]

    def children(self, var_idx: int) -> np.ndarray:
        """Indices of causal children of variable var_idx."""
        if self.W_ is None:
            return np.array([], dtype=int)
        return np.where(self.W_[:, var_idx] != 0)[0]
