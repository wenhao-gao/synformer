import torch
from torch import nn

from synformer.data.common import ProjectionBatch
from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import BaseEncoder, EncoderOutput
from .smiles import SMILESEncoder


class FixedSizeEncoder(BaseEncoder):
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        smiles_encoder_cfg: dict,
        latent_dec_nhead: int,
        latent_dec_dim_feedforward: int,
        latent_dec_num_layers: int,
        latent_dec_norm: bool,
        temperature_train: float = 1.0,
        temperature_eval: float = 0.1,
    ) -> None:
        super().__init__()
        self._n_tokens = n_tokens
        self._dim = d_model
        self._temperature_train = temperature_train
        self._temperature_eval = temperature_eval

        self.smiles_enc = SMILESEncoder(d_model=d_model, **smiles_encoder_cfg)
        self.pe_dec = PositionalEncoding(d_model=d_model, max_len=n_tokens)
        self.latent_dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=latent_dec_nhead,
                dim_feedforward=latent_dec_dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=latent_dec_num_layers,
            norm=nn.LayerNorm(d_model) if latent_dec_norm else None,
        )

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, batch: ProjectionBatch):
        smiles_code, smiles_padding_mask = self.smiles_enc(batch)
        batch_size = smiles_code.size(0)

        pe = self.pe_dec(torch.zeros([batch_size, self._n_tokens, self.dim], device=smiles_code.device))
        code = self.latent_dec(
            tgt=pe,
            memory=smiles_code,
            memory_key_padding_mask=smiles_padding_mask,
        )  # (bsz, n_tokens, vocab_size)
        code_padding_mask = torch.zeros([code.size(0), code.size(1)], dtype=torch.bool, device=code.device)
        return EncoderOutput(code, code_padding_mask)
