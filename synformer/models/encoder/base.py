import abc
import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from synformer.data.common import ProjectionBatch


@dataclasses.dataclass
class EncoderOutput:
    code: torch.Tensor
    code_padding_mask: torch.Tensor
    loss_dict: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    def __iter__(self):
        yield from (self.code, self.code_padding_mask, self.loss_dict)


class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch: ProjectionBatch) -> EncoderOutput:
        ...

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        ...

    if TYPE_CHECKING:

        def __call__(self, batch: ProjectionBatch) -> EncoderOutput:
            ...


class NoEncoder(BaseEncoder):
    def __init__(self, d_model: int):
        super().__init__()
        self._dim = d_model

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def infer_batch_size(batch: ProjectionBatch) -> int:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1

    def forward(self, batch: ProjectionBatch) -> EncoderOutput:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        bsz = self.infer_batch_size(batch)
        code = torch.zeros([bsz, 0, self.dim], device=device)
        code_padding_mask = torch.zeros([bsz, 0], dtype=torch.bool, device=device)
        return EncoderOutput(code, code_padding_mask)
