import abc
import dataclasses
from typing import TypeAlias

import numpy as np
import torch
from torch import nn

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.mol import Molecule

LossDict: TypeAlias = dict[str, torch.Tensor]
AuxDict: TypeAlias = dict[str, torch.Tensor]


@dataclasses.dataclass
class ReactantRetrievalResult:
    reactants: np.ndarray
    fingerprint_predicted: np.ndarray
    fingerprint_retrieved: np.ndarray
    distance: np.ndarray
    indices: np.ndarray


class BaseFingerprintHead(nn.Module, abc.ABC):
    def __init__(self, fingerprint_dim: int):
        super().__init__()
        self._fingerprint_dim = fingerprint_dim

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    @abc.abstractmethod
    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    def retrieve_reactants(
        self,
        h: torch.Tensor,
        fpindex: FingerprintIndex,
        topk: int = 4,
        **options,
    ) -> ReactantRetrievalResult:
        """
        Args:
            h:  Tensor of shape (*batch, h_dim).
            fpindex:  FingerprintIndex
            topk:  Number of reactants to retrieve per fingerprint.
        Returns:
            - numpy Molecule array of shape (*batch, n_fps, topk).
            - numpy fingerprint array of shape (*batch, n_fps, topk, fp_dim).
        """
        fp = self.predict(h, **options)  # (*batch, n_fps, fp_dim)
        fp_dim = fp.shape[-1]
        out = np.empty(list(fp.shape[:-1]) + [topk], dtype=Molecule)
        out_fp = np.empty(list(fp.shape[:-1]) + [topk, fp_dim], dtype=np.float32)
        out_fp_pred = fp[..., None, :].expand(*out_fp.shape)
        out_dist = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.float32)
        out_idx = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.int64)

        fp_flat = fp.view(-1, fp_dim)
        out_flat = out.reshape(-1, topk)
        out_fp_flat = out_fp.reshape(-1, topk, fp_dim)
        out_dist_flat = out_dist.reshape(-1, topk)
        out_idx_flat = out_idx.reshape(-1, topk)

        query_res = fpindex.query_cuda(q=fp_flat, k=topk)
        for i, q_res_subl in enumerate(query_res):
            for j, q_res in enumerate(q_res_subl):
                out_flat[i, j] = q_res.molecule
                out_fp_flat[i, j] = q_res.fingerprint
                out_dist_flat[i, j] = q_res.distance
                out_idx_flat[i, j] = q_res.index

        return ReactantRetrievalResult(
            reactants=out,
            fingerprint_predicted=out_fp_pred.detach().cpu().numpy(),
            fingerprint_retrieved=out_fp,
            distance=out_dist,
            indices=out_idx,
        )

    @abc.abstractmethod
    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        ...

    @abc.abstractmethod
    def get_log_likelihood(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        ...
