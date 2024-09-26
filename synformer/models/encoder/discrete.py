import torch
from torch import nn

from synformer.data.common import ProjectionBatch
from synformer.models.quantizer.lfq import LFQ
from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import BaseEncoder, EncoderOutput
from .smiles import SMILESEncoder


class DiscreteEncoder(BaseEncoder):
    def __init__(
        self,
        n_dims: int,
        vocab_size: int,
        d_model: int,
        smiles_encoder_cfg: dict,
        latent_dec_nhead: int,
        latent_dec_dim_feedforward: int,
        latent_dec_num_layers: int,
        latent_dec_norm: bool,
        temperature_train: float,
        temperature_eval: float,
    ) -> None:
        super().__init__()
        self._n_dims = n_dims
        self._vocab_size = vocab_size
        self._dim = d_model
        self._temperature_train = temperature_train
        self._temperature_eval = temperature_eval

        self.smiles_enc = SMILESEncoder(d_model=d_model, **smiles_encoder_cfg)
        self.pe_dec = PositionalEncoding(d_model=d_model, max_len=n_dims)
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
        self.quantizer = LFQ(
            codebook_size=vocab_size,
            dim=d_model,
        )

        self.latent_embed = nn.Parameter(torch.randn(n_dims, d_model, vocab_size))

    @property
    def dim(self) -> int:
        return self._dim

    def get_discrete_code(self, batch: ProjectionBatch, temperature: float | None = None) -> torch.Tensor:
        smiles_code, smiles_padding_mask, _ = self.smiles_enc(batch)
        batch_size = smiles_code.size(0)

        x = self.pe_dec(torch.zeros([batch_size, self._n_dims, self.dim], device=smiles_code.device))
        code = self.latent_dec(
            tgt=x,
            memory=smiles_code,
            memory_key_padding_mask=smiles_padding_mask,
        )  # (bsz, n_dims, d_model)

        if temperature is None:
            temperature = self._temperature_train if self.training else self._temperature_eval

        return self.quantizer(code, inv_temperature=1.0 / temperature)

    def forward(self, batch: ProjectionBatch):
        code, _, entropy_aux_loss = self.get_discrete_code(batch)  # (bsz, n_dims, vocab_size)
        code_padding_mask = torch.zeros([code.size(0), code.size(1)], dtype=torch.bool, device=code.device)
        return EncoderOutput(code, code_padding_mask, loss_dict={"quantizer": entropy_aux_loss})
