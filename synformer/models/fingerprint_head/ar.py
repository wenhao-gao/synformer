import torch
from torch import nn
from torch.nn import functional as F

from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import AuxDict, BaseFingerprintHead, LossDict


class AutoRegressiveFingerprintHead(BaseFingerprintHead):
    def __init__(
        self,
        d_model: int,
        fingerprint_dim: int,
        n_bits_per_segment: int,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_layers: int = 3,
        norm: bool = True,
    ) -> None:
        super().__init__(fingerprint_dim=fingerprint_dim)
        self.d_model = d_model
        self.n_bits_per_segment = n_bits_per_segment
        self.num_segments = fingerprint_dim // n_bits_per_segment
        vocab_size = 2**n_bits_per_segment
        self.exponential_part = nn.Parameter(
            torch.bitwise_left_shift(
                torch.ones([n_bits_per_segment], dtype=torch.long),
                torch.arange(n_bits_per_segment, dtype=torch.long),
            ),
            requires_grad=False,
        )

        self.start_embed = nn.Parameter(torch.randn([d_model]), requires_grad=True)
        self.embed = nn.Sequential(
            nn.Linear(n_bits_per_segment, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pe = PositionalEncoding(
            d_model=d_model,
            max_len=self.num_segments + 1,
        )
        self.dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm else None,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def fingerprint_to_sequence(self, fingerprint: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = fingerprint.shape
        segments = fingerprint.view(bsz, seqlen, self.num_segments, self.n_bits_per_segment).long().clamp(0, 1)
        seq = torch.sum(segments * self.exponential_part, dim=-1)
        return seq

    def sequence_to_fingerprint(self, seq: torch.Tensor, segmented: bool = False) -> torch.Tensor:
        seq = torch.bitwise_and(seq[..., None], self.exponential_part).clamp(0, 1)  # (bsz, seqlen, n_segs, n_bits)
        if segmented:
            return seq.float()
        else:
            return seq.flatten(-2, -1).float()

    def embed_fp_segs_input(self, h: torch.Tensor, fp_seg: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, n_segs, n_bits = fp_seg.shape
        embed = self.start_embed.view(1, 1, 1, -1).expand([bsz, seqlen, 1, self.d_model])
        if n_segs > 0:
            embed = torch.cat([embed, self.embed(fp_seg)], dim=-2)

        embed = embed.flatten(0, 1)  # (bsz*seqlen, n_segs, d_model)
        x = self.pe(embed)

        h = h[:, :, None, :].repeat([1, 1, n_segs + 1, 1]).flatten(0, 1)  # (bsz*seqlen, n_segs, d_model)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=n_segs + 1, dtype=fp_seg.dtype, device=fp_seg.device
        )

        y = self.dec(tgt=x, tgt_mask=causal_mask, memory=h)  # (bsz*seqlen, n_segs, d_model)
        y = y.view(bsz, seqlen, n_segs + 1, self.d_model)
        return y

    def predict(self, h: torch.Tensor, *, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        is_2d = h.dim() == 2
        if is_2d:
            h = h[:, None, :]
        bsz, seqlen, _ = h.shape
        index_seq: list[torch.Tensor] = []
        fp_seg = torch.empty([bsz, seqlen, 0, self.n_bits_per_segment], device=h.device)

        for i in range(self.num_segments):
            y = self.embed_fp_segs_input(h, fp_seg)
            logits = self.out(y[..., -1:, :])  # (bsz, seqlen, 1, vocab_size)
            index = torch.multinomial(
                F.softmax(logits.reshape(-1, logits.size(-1)) / temperature, dim=-1), num_samples=1
            ).squeeze(-1)
            index_seq.append(index.reshape(bsz, seqlen, 1))
            fp_seg = self.sequence_to_fingerprint(torch.cat(index_seq, dim=-1), segmented=True)

        out = fp_seg.flatten(-2, -1)
        if is_2d:
            out = out.squeeze(1)
        return out

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        limit_batch: bool = True,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        """
        Args:
            h:  (bsz, seqlen, d_model)
            fp_target:  (bsz, seqlen, fingerprint_dim)
            fp_mask:  (bsz, seqlen)
        """
        if limit_batch:
            fp_index = torch.multinomial(fp_mask.float(), num_samples=1).squeeze(-1)  # (bsz, )
            bsz = h.size(0)
            h = h[range(bsz), fp_index].unsqueeze(1)
            fp_target = fp_target[range(bsz), fp_index].unsqueeze(1)
            fp_mask = fp_mask[range(bsz), fp_index].unsqueeze(1)

        bsz, seqlen, _ = h.shape
        fp_seg = fp_target.view(bsz, seqlen, self.num_segments, self.n_bits_per_segment)
        y = self.embed_fp_segs_input(h, fp_seg[:, :, :-1])  # (bsz, seqlen, n_segs, d_model)
        logits = self.out(y)  # (bsz, seqlen, n_segs, vocab_size)

        fp_seq = self.fingerprint_to_sequence(fp_target)  # (bsz, seqlen, n_segs)
        fp_mask = fp_mask[..., None].expand_as(fp_seq)  # (bsz, seqlen, n_segs)
        loss_all = (
            F.cross_entropy(
                input=logits.permute(0, 3, 1, 2),
                target=fp_seq,
                reduction="none",
            )
            * fp_mask.float()
        )
        loss_avg = loss_all.sum() / (fp_mask.sum() + 1e-6)
        return {"fingerprint": loss_avg}, {}

    def get_log_likelihood(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seqlen, _ = h.shape
        fp_seq = self.fingerprint_to_sequence(fp_target)  # (bsz, seqlen, n_segs)

        fp_seg = fp_target.view(bsz, seqlen, self.num_segments, self.n_bits_per_segment)
        y = self.embed_fp_segs_input(h, fp_seg[:, :, :-1])  # (bsz, seqlen, n_segs, d_model)
        logits = self.out(y)  # (bsz, seqlen, n_segs, vocab_size)
        ll_all = F.log_softmax(logits, dim=-1)  # (bsz, seqlen, n_segs, vocab_size)
        ll = torch.gather(ll_all, dim=-1, index=fp_seq[..., None]).squeeze(-1)  # (bsz, seqlen, n_segs)

        ll = ll.sum(dim=-1) * fp_mask
        return ll


if __name__ == "__main__":
    model = AutoRegressiveFingerprintHead(
        d_model=512,
        fingerprint_dim=256,
        n_bits_per_segment=16,
    )

    fp = (torch.randn([4, 24, 256]) > 0).float()
    seq = model.fingerprint_to_sequence(fp)
    fp_ = model.sequence_to_fingerprint(seq)
    print(seq)
    print((fp != fp_).sum())
    assert torch.allclose(fp, fp_)
