import torch
from torch import nn
from torch.nn import functional as F

from .base import AuxDict, BaseFingerprintHead, LossDict


class MultiFingerprintHead(BaseFingerprintHead):
    def __init__(self, d_model: int, num_out_fingerprints: int, fingerprint_dim: int):
        super().__init__(fingerprint_dim=fingerprint_dim)
        self.d_model = d_model
        self.num_out_fingerprints = num_out_fingerprints
        d_out = fingerprint_dim * num_out_fingerprints
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        y_fingerprint = torch.sigmoid(self.mlp(h))
        out_shape = h.shape[:-1] + (self.num_out_fingerprints, self.fingerprint_dim)
        return y_fingerprint.view(out_shape)

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        warmup: bool = False,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        bsz, seqlen, _ = h.shape
        y_fingerprint = self.mlp(h)  # (bsz, seqlen, n_fps * fp_dim)
        fp_shape = [bsz, seqlen, self.num_out_fingerprints, self.fingerprint_dim]
        y_fingerprint = y_fingerprint.view(fp_shape)
        fp_target = fp_target[:, :, None, :].expand(fp_shape)
        loss_fingerprint_all = F.binary_cross_entropy_with_logits(
            y_fingerprint,
            fp_target,
            reduction="none",
        ).sum(dim=-1)
        loss_fingerprint_min, fp_select = loss_fingerprint_all.min(dim=-1)
        if self.training and warmup:
            loss_fingerprint_avg = loss_fingerprint_all.mean(dim=-1)
            loss_fingerprint = torch.where(
                torch.rand_like(loss_fingerprint_min) < 0.01,
                loss_fingerprint_avg,
                loss_fingerprint_min,
            )
        else:
            loss_fingerprint = loss_fingerprint_min
        loss_fingerprint = (loss_fingerprint * fp_mask).sum() / (fp_mask.sum() + 1e-6)

        return {"fingerprint": loss_fingerprint}, {"fp_select": fp_select}

    def get_log_likelihood(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError("MultiFingerprintHead does not support get_log_likelihood.")
