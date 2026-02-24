"""
Temporal Causal Network
========================

A Transformer encoder that integrates the NOTEARS-learned causal DAG as
a structural prior over the attention mechanism, then forecasts disruption
probabilities across multiple horizons.

Design
-------
Standard self-attention lets every token attend every other token.
Here we split heads into two groups:

    Causal heads (n_causal_heads):
        Attention is masked to the causal parents identified by NOTEARS.
        Information flows only along learned causal paths.

    Free heads (n_heads − n_causal_heads):
        Unconstrained — can discover new causal relationships from data
        not yet captured by the DAG.

Temporal dimension: for each node we first run a causal-masked temporal
self-attention over the T-step history, then cross-node graph attention.

Output
-------
    risk_probs:  (N, n_horizons) disruption probability per node
    node_repr:   (N, d_model)    node embeddings for attribution
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGraphAttn(nn.Module):
    """Multi-head attention with optional causal-graph masking."""

    def __init__(self, d_model: int, n_heads: int, n_causal: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.n_causal  = n_causal
        self.head_dim  = d_model // n_heads
        self.scale     = math.sqrt(self.head_dim)

        self.norm  = nn.LayerNorm(d_model)
        self.qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out   = nn.Linear(d_model, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x:   torch.Tensor,                     # (B, N, d_model)
        adj: Optional[torch.Tensor] = None,    # (N, N) causal adjacency
    ) -> torch.Tensor:
        B, N, _ = x.shape
        res = x
        x   = self.norm(x)

        QKV = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        Q, K, V = QKV.unbind(2)
        Q = Q.permute(0, 2, 1, 3)   # (B, H, N, head_dim)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) / self.scale   # (B, H, N, N)

        if adj is not None and self.n_causal > 0:
            no_edge = (adj.T < 0.1).unsqueeze(0).unsqueeze(0)          # (1,1,N,N)
            mask    = no_edge.expand(B, self.n_causal, N, N)
            attn[:, :self.n_causal] = attn[:, :self.n_causal].masked_fill(mask, -1e9)

        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.out(out) + res


class TemporalSelfAttn(nn.Module):
    """Causal-masked temporal self-attention over T timesteps per node."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BN, T, d_model)"""
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        xn     = self.norm(x)
        out, _ = self.attn(xn, xn, xn, attn_mask=causal)
        return out + x


class CausalBlock(nn.Module):
    """Temporal attention → graph attention → FFN."""

    def __init__(self, d_model: int, n_heads: int, n_causal: int, dropout: float = 0.1):
        super().__init__()
        self.temp_attn  = TemporalSelfAttn(d_model, 4, dropout)
        self.graph_attn = CausalGraphAttn(d_model, n_heads, n_causal, dropout)
        self.ffn        = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (1, N, T, d_model)  adj: (N, N)"""
        B, N, T, D = x.shape

        # temporal attention over each node's history
        xt       = x.reshape(B * N, T, D)
        xt       = self.temp_attn(xt)
        x        = xt.reshape(B, N, T, D)

        # graph attention on latest timestep
        x_last   = x[:, :, -1, :]
        x_last   = self.graph_attn(x_last, adj)
        x        = x.clone()
        x[:, :, -1, :] = x_last

        return x + self.ffn(x)


class TemporalCausalNet(nn.Module):
    """
    Full Temporal Causal Network for multi-horizon disruption forecasting.

    Args:
        d_signal:    Input signal dimension per node per timestep.
        d_model:     Transformer hidden dimension.
        n_layers:    Number of CausalBlock layers.
        n_horizons:  Number of forecast horizons.
        dropout:     Dropout rate.
    """

    HORIZONS = [1, 3, 7, 14, 21]   # days

    def __init__(
        self,
        d_signal:   int   = 128,
        d_model:    int   = 256,
        n_layers:   int   = 6,
        n_horizons: int   = 5,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.in_proj   = nn.Linear(d_signal, d_model)
        self.pos_embed = nn.Embedding(256, d_model)
        self.blocks    = nn.ModuleList([
            CausalBlock(d_model, n_heads=8, n_causal=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm  = nn.LayerNorm(d_model)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_horizons), nn.Sigmoid(),
        )

    def forward(
        self,
        x:   torch.Tensor,                     # (N, T, d_signal)
        adj: Optional[torch.Tensor] = None,    # (N, N) causal adjacency
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            risk_probs: (N, n_horizons)
            node_repr:  (N, d_model)
        """
        N, T, _ = x.shape
        x = self.in_proj(x).unsqueeze(0)   # (1, N, T, d_model)
        x = x + self.pos_embed(torch.arange(T, device=x.device))

        # Build node-level adj from signal-level adj (simplified)
        node_adj = adj[:N, :N] if (adj is not None and adj.shape[0] >= N) else None

        for blk in self.blocks:
            x = blk(x, node_adj)

        x         = x.squeeze(0)                       # (N, T, d_model)
        node_repr = self.out_norm(x[:, -1, :])         # (N, d_model)
        risk_probs = self.risk_head(node_repr)          # (N, n_horizons)
        return risk_probs, node_repr
