import dataclasses
from collections import OrderedDict

import torch

from synformer.chem.stack import Stack
from synformer.models.synformer import GenerateResult


def get_key_of_stack(stack: Stack) -> str:
    smiles = sorted([m.csmiles for m in stack.get_top()])
    return ";".join(smiles)


@dataclasses.dataclass
class Experience:
    code: torch.Tensor  # (code_len, d_model)
    code_padding_mask: torch.Tensor  # (code_len,)
    token_types: torch.Tensor  # (seq_len, )
    rxn_indices: torch.Tensor  # (seq_len, )
    reactant_fps: torch.Tensor  # (seq_len, n_bits)
    token_padding_mask: torch.Tensor  # (seq_len, )

    stack: Stack
    score: float

    @property
    def key(self) -> str:
        return get_key_of_stack(self.stack)

    @classmethod
    def from_samples(
        cls,
        result: GenerateResult,
        stacks: list[Stack],
        scores: torch.Tensor,
        ignore_invalid_stacks: bool = True,
    ):
        bsz = len(stacks)
        for i in range(bsz):
            stack = stacks[i]
            if ignore_invalid_stacks and stack.get_stack_depth() != 1:
                continue
            yield cls(
                code=result.code[i].detach().cpu(),
                code_padding_mask=result.code_padding_mask[i].detach().cpu(),
                token_types=result.token_types[i].detach().cpu(),
                rxn_indices=result.rxn_indices[i].detach().cpu(),
                reactant_fps=result.reactant_fps[i].detach().cpu(),
                token_padding_mask=result.token_padding_mask[i].detach().cpu(),
                stack=stack,
                score=scores[i].item(),
            )
        yield from []


@dataclasses.dataclass
class ReplayBatch:
    code: torch.Tensor
    code_padding_mask: torch.Tensor
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    scores: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.code.size(0)


class ReplayBuffer:
    def __init__(self, capacity: int):
        super().__init__()
        self._capacity = capacity
        self._buffer: OrderedDict[str, Experience] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buffer)

    def _sort_buffer(self):
        self._buffer = OrderedDict(sorted(self._buffer.items(), key=lambda x: x[1].score, reverse=True))

    def add(self, experience: Experience):
        if len(self._buffer) >= self._capacity and experience.key not in self._buffer:
            self._sort_buffer()
            self._buffer.popitem()
        self._buffer[experience.key] = experience

    def sample(self, count: int, temperature: float = 0.1, device: torch.device | str = torch.device("cpu")):
        keys = list(self._buffer.keys())
        count = min(count, len(keys))
        if count == 0:
            return None
        probs = torch.softmax(torch.tensor([self._buffer[key].score for key in keys]) / temperature, dim=0)
        indices = torch.multinomial(probs, count, replacement=False)
        samples = [self._buffer[keys[i]] for i in indices]

        code = torch.stack([s.code for s in samples], dim=0).to(device)
        code_padding_mask = torch.stack([s.code_padding_mask for s in samples], dim=0).to(device)
        token_types = torch.stack([s.token_types for s in samples], dim=0).to(device)
        rxn_indices = torch.stack([s.rxn_indices for s in samples], dim=0).to(device)
        reactant_fps = torch.stack([s.reactant_fps for s in samples], dim=0).to(device)
        token_padding_mask = torch.stack([s.token_padding_mask for s in samples], dim=0).to(device)

        scores = torch.tensor([s.score for s in samples], device=device, dtype=torch.float)
        return ReplayBatch(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=token_padding_mask,
            scores=scores,
        )
