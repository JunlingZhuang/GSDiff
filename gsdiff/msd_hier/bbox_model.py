"""BBox smoke models for MSD hierarchical GSDiff.

These models are deliberately small and parameterized. They are not the final
diffusion generator; they validate that the newly constructed coarse/local
hierarchical graphs contain learnable geometry targets before we invest in a
full corner/polygon diffusion model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeAwareBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_edge_types: int, dropout: float = 0.0):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)
        self.edge_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, h: torch.Tensor, edge_types: torch.Tensor, edge_present: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_embedding(edge_types)
        degree = edge_present.sum(dim=2, keepdim=True).clamp_min(1.0)
        edge_context = (edge_emb * edge_present.unsqueeze(-1)).sum(dim=2) / degree
        h = h + self.edge_fusion(edge_context) * node_mask

        key_padding_mask = (node_mask.squeeze(-1) < 0.5).to(torch.bool)
        h_norm = self.attn_norm(h)
        attn_out, _ = self.attn(
            h_norm,
            h_norm,
            h_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = (h + attn_out) * node_mask
        h = (h + self.ffn(self.ffn_norm(h))) * node_mask
        return h


class HierBBoxModel(nn.Module):
    """Predict normalized node bboxes from a coarse or local hierarchy graph."""

    def __init__(
        self,
        input_dim: int,
        num_edge_types: int,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [EdgeAwareBlock(d_model, n_heads, num_edge_types, dropout=dropout) for _ in range(n_layers)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4),
            nn.Tanh(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_types: torch.Tensor,
        edge_present: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(node_features) * node_mask
        for block in self.blocks:
            h = block(h, edge_types=edge_types, edge_present=edge_present, node_mask=node_mask)
        return self.out(h) * node_mask


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * node_mask
    return loss.sum() / node_mask.sum().clamp_min(1.0) / pred.shape[-1]
