from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _SimpleMLP(dim_in: int, dim_out: int, dim_hidden: int) -> Callable[[torch.Tensor], torch.Tensor]:
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_out),
    )


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dim_hidden: int | None = None):
        super().__init__()
        dim_hidden = dim_hidden or d_model * 2
        self.mlp = _SimpleMLP(d_model, num_classes, dim_hidden=dim_hidden)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

    def get_loss(self, h: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        logits = self.predict(h)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1)
            total = mask_flat.sum().to(logits_flat) + 1e-6
            loss = (F.cross_entropy(logits_flat, target_flat, reduction="none") * mask_flat).sum() / total
        else:
            loss = F.cross_entropy(logits_flat, target_flat)
        return loss

    def get_log_likelihood(self, h: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            h:  (bsz, seqlen, d_model)
            target:  (bsz, seqlen)
            mask:  (bsz, seqlen)
        Returns:
            ll:  (bsz, seqlen)
        """
        logits = self.predict(h)
        ll_all = F.log_softmax(logits, dim=-1)
        ll = torch.gather(ll_all, dim=-1, index=target[..., None]).squeeze(-1) * mask
        return ll
