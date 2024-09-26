from torch import nn

from synformer.data.common import ProjectionBatch
from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import BaseEncoder, EncoderOutput


class SMILESEncoder(BaseEncoder):
    def __init__(
        self,
        num_token_types: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        pe_max_len: int,
    ):
        super().__init__()
        self._dim = d_model
        self.smiles_emb = nn.Embedding(num_token_types, d_model, padding_idx=0)
        self.pe_enc = PositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch):
        if "smiles" not in batch:
            raise ValueError("smiles must be in batch")
        smiles = batch["smiles"]
        h = self.pe_enc(self.smiles_emb(smiles))
        padding_mask = smiles == 0  # the positions with the value of True will be ignored
        out = self.enc(h, src_key_padding_mask=padding_mask)
        return EncoderOutput(out, padding_mask)
