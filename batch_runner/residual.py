# residual_injection.py

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class ResidualInjectionConfig:
    enabled: bool = False

    topk: int = 16
    alpha: float = 0.05

    layer_start: int = 16
    layer_end: int = -1

    use_gate: bool = True
    proj: bool = True

class ResidualInjector(nn.Module):
    def __init__(self, hidden_size, topk=16, use_gate=True, proj=True):
        super().__init__()

        self.topk = topk
        self.use_gate = use_gate

        if proj:
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.proj = nn.Identity()

        if use_gate:
            self.gate = nn.Linear(2, 1)  # energy + margin

    def compute_soft_embedding(self, logits, embed_weight):
        vals, idx = torch.topk(logits, k=self.topk, dim=-1)
        probs = torch.softmax(vals, dim=-1)

        emb = embed_weight[idx]                # [B, K, H]
        soft_emb = (probs.unsqueeze(-1) * emb).sum(dim=1)

        return soft_emb

    def compute_gate(self, logits):
        # energy
        energy = torch.logsumexp(logits, dim=-1, keepdim=True)

        # margin
        top2 = torch.topk(logits, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)

        feat = torch.cat([energy, margin], dim=-1)

        gate = torch.sigmoid(self.gate(feat))   # [B,1]
        return gate

    def forward(self, logits, embed_weight):
        v = self.compute_soft_embedding(logits, embed_weight)
        # v = self.proj(v)

        if self.use_gate:
            g = self.compute_gate(logits)
            v = v * g

        return v