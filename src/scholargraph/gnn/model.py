"""A directed GraphSAGE-style network written in plain PyTorch (no PyTorch Geometric needed, so it installs everywhere).

Message passing for paper *i* (one layer)::

    h_i' = ReLU( W_self h_i  +  W_ref * mean_{j cited by i} h_j  +  W_cit * mean_{j citing i} h_j )

Two separate neighbour sets (the papers *i cites* and the papers *that cite i*) keep the direction of a citation, which matters:
the first are the paper's intellectual ancestors, the second its impact.

A pair (u, v) is scored by an MLP over ``[z_u, z_v, z_u * z_v, text cosine, year gap]``, giving the logit of "u cites v".

Apple Silicon note: sparse matrix products are not fully supported on the ``mps`` device, so the aggregation step runs on the CPU and
the dense layers on the GPU (the transfers are differentiable). On CUDA and CPU everything runs on the device.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .data import CitationData


class Adjacency:
    """Row-normalised (mean) neighbour operators for one message graph, plus degree statistics."""

    def __init__(self, edges: np.ndarray, n: int, device: str = "cpu"):
        self.n, self.device = n, device
        ones = np.ones(len(edges), dtype=np.float32)
        refs = sp.csr_matrix((ones, (edges[:, 0], edges[:, 1])), shape=(n, n))      # row i: papers that i cites
        cits = refs.T.tocsr()                                                        # row i: papers that cite i
        self.out_deg = np.asarray(refs.sum(axis=1)).ravel()
        self.in_deg = np.asarray(cits.sum(axis=1)).ravel()
        self._sparse_device = "cpu" if device == "mps" else device
        self.refs = self._to_torch(self._row_normalise(refs))
        self.cits = self._to_torch(self._row_normalise(cits))

    @staticmethod
    def _row_normalise(m: sp.csr_matrix) -> sp.coo_matrix:
        deg = np.asarray(m.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float32), where=deg > 0)
        return sp.diags(inv).dot(m).tocoo()

    def _to_torch(self, m: sp.coo_matrix) -> torch.Tensor:
        idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
        return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)), m.shape).coalesce().to(self._sparse_device)

    def mean_refs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.refs, h.to(self._sparse_device)).to(h.device)

    def mean_cits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.cits, h.to(self._sparse_device)).to(h.device)


def node_features(data: CitationData, adj: Adjacency, device: str) -> torch.Tensor:
    """[text embedding | scaled year | log in-degree | log out-degree]; degrees come from the *message graph* only."""
    y = (data.year - data.year.min()) / max(1, data.year.max() - data.year.min())
    deg = np.stack([y, np.log1p(adj.in_deg), np.log1p(adj.out_deg)], axis=1).astype(np.float32)
    return torch.from_numpy(np.concatenate([data.text, deg], axis=1)).to(device)


class SAGEBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.self_lin, self.ref_lin, self.cit_lin = nn.Linear(dim, dim), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: Adjacency) -> torch.Tensor:
        out = self.self_lin(h) + self.ref_lin(adj.mean_refs(h)) + self.cit_lin(adj.mean_cits(h))
        return h + self.drop(torch.relu(self.norm(out)))                          # residual connection


class CitationGNN(nn.Module):
    def __init__(self, in_dim: int, text_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.text_dim = text_dim
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([SAGEBlock(hidden, dropout) for _ in range(layers)])
        self.pair = nn.Sequential(nn.Linear(3 * hidden + 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, x: torch.Tensor, adj: Adjacency) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for b in self.blocks:
            h = b(h, adj)
        return h

    def score(self, z: torch.Tensor, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cos = (x[u, :self.text_dim] * x[v, :self.text_dim]).sum(-1, keepdim=True)
        gap = (x[u, self.text_dim:self.text_dim + 1] - x[v, self.text_dim:self.text_dim + 1])     # scaled year difference
        return self.pair(torch.cat([z[u], z[v], z[u] * z[v], cos, gap], dim=-1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
