"""
https://github.com/lucidrains/graph-transformer-pytorch

MIT License

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import TypeVar

import torch
from einops import rearrange, repeat
from torch import einsum, nn

from .rotary_embedding import RotaryEmbedding, apply_rotary_emb

# helpers

_T = TypeVar("_T")


def default(val: _T | None, d: _T) -> _T:
    return val if val is not None else d


# normalizations


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# gated residual


class Residual(nn.Module):
    def forward(self, x, res):
        return x + res


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, 1, bias=False), nn.Sigmoid())

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):
    def __init__(self, dim, pos_emb=None, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask=None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim=-1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(lambda t: rearrange(t, "b ... (h d) -> (b h) ... d", h=h), (q, k, v, e_kv))

        if self.pos_emb is not None:
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device=nodes.device))
            freqs = rearrange(freqs, "n d -> () n d")
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, "b j d -> b () j d "), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum("b i d, b i j d -> b i j", q, k) * self.scale

        if mask is not None:
            # Note: different from ludidrain's implementation, where False is dropped.
            #       Here we drop True which is consistent with PyTorch's implementation.
            # Change: originally this was `&` but now it should be `|`
            mask = rearrange(mask, "b i -> b i ()") | rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# optional feedforward


def FeedForward(dim, ff_mult=4):
    return nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))


# classes


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        gated_residual=True,
        with_feedforwards=False,
        norm_edges=False,
        rel_pos_emb=False,
        accept_adjacency_matrix=False,
        adj_types: int = 2,
        output_norm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        self.adj_emb = nn.Embedding(adj_types, edge_dim) if accept_adjacency_matrix else None

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                PreNorm(
                                    dim,
                                    Attention(dim, pos_emb=pos_emb, edge_dim=edge_dim, dim_head=dim_head, heads=heads),
                                ),
                                GatedResidual(dim),
                            ]
                        ),
                        (
                            nn.ModuleList(
                                [
                                    PreNorm(dim, FeedForward(dim)),
                                    GatedResidual(dim),
                                ]
                            )
                            if with_feedforwards
                            else nn.ModuleList([])
                        ),
                    ]
                )
            )

        self.out_norm_nodes = nn.LayerNorm(dim) if output_norm else nn.Identity()
        self.out_norm_edges = nn.LayerNorm(edge_dim) if output_norm else nn.Identity()

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor | None = None,
        adj_mat: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """
        Args:
            nodes: size (batch, seq, dim)
            edges: size (batch, seq, seq, edge_dim)
            adj_mat: size (batch, seq, seq)
            mask: node mask, size (batch, seq)
        """
        batch, seq, _ = nodes.shape

        if edges is not None:
            edges = self.norm_edges(edges)

        if adj_mat is not None:
            assert adj_mat.shape == (batch, seq, seq)
            assert self.adj_emb is not None, "accept_adjacency_matrix must be set to True"
            adj_mat = self.adj_emb(adj_mat.long())

        all_edges: torch.Tensor | int = 0
        if edges is not None:
            all_edges = edges + all_edges
        if adj_mat is not None:
            all_edges = adj_mat + all_edges

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, all_edges, mask=mask), nodes)

            if len(ff_block) > 0:
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        nodes = self.out_norm_nodes(nodes)
        if edges is not None:
            edges = self.out_norm_edges(edges)

        return nodes, edges
