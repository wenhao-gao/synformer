from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


def collate_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, max_size - f.size(-1)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, max_size - f.size(-1), 0, max_size - f.size(-2)], mode="constant", value=0) for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_1d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, 0, 0, max_size - f.size(-2)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)


def collate_2d_features(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [
        F.pad(f, pad=[0, 0, 0, max_size - f.size(-2), 0, max_size - f.size(-3)], mode="constant", value=0)
        for f in features
    ]
    return torch.stack(features_padded, dim=0)


def collate_padding_masks(masks: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    masks_padded = [F.pad(m, pad=[0, max_size - m.size(-1)], mode="constant", value=True) for m in masks]
    return torch.stack(masks_padded, dim=0)


def apply_collate(
    spec: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[dict[str, torch.Tensor]],
    max_size: int,
) -> dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in spec.keys()}
    batch = {k: spec[k](transpose[k], max_size) for k in spec.keys()}
    return batch
