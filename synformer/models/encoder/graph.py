from torch import nn

from synformer.data.common import ProjectionBatch
from synformer.models.transformer.graph_transformer import GraphTransformer

from .base import BaseEncoder, EncoderOutput


class GraphEncoder(BaseEncoder):
    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        dim: int,
        depth: int,
        dim_head: int,
        edge_dim: int,
        heads: int,
        rel_pos_emb: bool,
        output_norm: bool,
    ):
        super().__init__()
        self._dim = dim
        self.atom_emb = nn.Embedding(num_atom_classes + 1, dim, padding_idx=0)
        self.bond_emb = nn.Embedding(num_bond_classes + 1, edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            edge_dim=edge_dim,
            heads=heads,
            rel_pos_emb=rel_pos_emb,
            output_norm=output_norm,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch):
        if "atoms" not in batch or "bonds" not in batch or "atom_padding_mask" not in batch:
            raise ValueError("atoms, bonds and atom_padding_mask must be in batch")
        atoms = batch["atoms"]
        bonds = batch["bonds"]
        atom_padding_mask = batch["atom_padding_mask"]

        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=atom_padding_mask)
        return EncoderOutput(node, atom_padding_mask)
