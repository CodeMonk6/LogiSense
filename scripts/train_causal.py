"""
Train Causal Disruption Engine
================================

Trains the TemporalCausalNet on historical supply chain disruption data.

Usage:
    python scripts/train_causal.py --config configs/full_pipeline.yaml
    python scripts/train_causal.py --epochs 50 --lr 1e-4

Data format:
    Training: (N_batches, N_nodes, T, D_signal) signal tensors
    Labels:   (N_batches, N_nodes, n_horizons) binary disruption flags
"""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from logisense.causal import CausalDisruptionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_batches: int = 200, n_nodes: int = 20,
                              T: int = 30, d_signal: int = 128):
    """Generate synthetic training data for the causal engine."""
    rng = np.random.default_rng(42)
    X   = rng.uniform(0, 1, (n_batches, n_nodes, T, d_signal)).astype(np.float32)
    y   = rng.binomial(1, 0.2, (n_batches, n_nodes, 5)).astype(np.float32)
    # Inject signal correlation: high signal values → more disruptions
    for b in range(n_batches):
        for i in range(n_nodes):
            if X[b, i, -1, 0] > 0.7:
                y[b, i, 2] = 1.0   # 7-day horizon disruption
    return torch.tensor(X), torch.tensor(y)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = CausalDisruptionEngine(device=str(device))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCELoss()

    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_data(args.n_batches, args.n_nodes, 30, 128)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        idx = torch.randperm(len(X))
        X, y = X[idx], y[idx]

        for b in range(0, len(X), args.batch_size):
            xb = X[b: b + args.batch_size].to(device)
            yb = y[b: b + args.batch_size].to(device)

            # Forward pass through TemporalCausalNet
            xb_flat = xb.view(-1, xb.shape[2], xb.shape[3])  # (B*N, T, D)
            risk, _ = model.net(xb_flat, adj=None)
            risk     = risk.view(xb.shape[0], xb.shape[1], -1)

            loss = criterion(risk, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / (len(X) / args.batch_size)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d / %3d  loss=%.4f  lr=%.2e",
                        epoch, args.epochs, avg_loss,
                        scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(args.output_dir + "/causal")
            logger.info("  ✓ Checkpoint saved (loss=%.4f)", best_loss)

    logger.info("Training complete. Best loss: %.4f", best_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=None)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--n-batches",  type=int,   default=200)
    parser.add_argument("--n-nodes",    type=int,   default=20)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        args.epochs     = cfg.get("epochs", args.epochs)
        args.lr         = float(cfg.get("causal", {}).get("lr", args.lr))

    train(args)


if __name__ == "__main__":
    main()
