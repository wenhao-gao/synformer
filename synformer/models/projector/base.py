import abc
from typing import TypedDict, final

import torch
from torch import nn
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.data.common import ProjectionBatch, TokenType


class NetworkOutputDict(TypedDict, total=False):
    token: torch.Tensor
    reaction: torch.Tensor
    fingerprint: torch.Tensor


class LossDict(TypedDict, total=False):
    token: torch.Tensor
    reaction: torch.Tensor
    fingerprint: torch.Tensor

    fingerprint_diffusion: torch.Tensor
    fingerprint_bce: torch.Tensor


class PredictResult(TypedDict):
    y_token: torch.Tensor
    y_reaction: torch.Tensor
    y_fingerprint: torch.Tensor

    token_next: torch.Tensor

    rxn_indices_next: torch.Tensor
    reaction_next: list[list[Reaction | None]]
    rxn_scores_next: torch.Tensor

    reactant_next: list[list[Molecule | None]]
    fingerprint_next: torch.Tensor
    reactant_indices_next: torch.Tensor
    reactant_scores_next: torch.Tensor


class GenerateResult(TypedDict):
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    reactants: list[list[Molecule | None]]
    reactions: list[list[Reaction | None]]


class BaseProjector(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def fingerprint_dim(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def model_dim(self) -> int:
        ...

    def get_empty_code(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code = torch.zeros([batch_size, 0, self.model_dim], dtype=dtype, device=device)
        code_padding_mask = torch.zeros([batch_size, 0], dtype=torch.bool, device=device)
        return code, code_padding_mask

    @abc.abstractmethod
    def encode(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def embed_seq(
        self,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_loss(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ) -> tuple[NetworkOutputDict, LossDict]:
        ...

    @final
    def get_loss_shortcut(self, batch: ProjectionBatch, **options):
        code, code_padding_mask = self.encode(batch)
        return self.get_loss(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            **options,
        )

    @abc.abstractmethod
    def predict(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        topk: int = 4,
        result_device: torch.device | None = None,
        **options,
    ) -> PredictResult:
        ...

    @final
    def generate_without_stack(
        self,
        batch: ProjectionBatch,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_len: int = 24,
        **options,
    ) -> GenerateResult:
        """
        Generate synthesis sequence without using stack. This is for previewing purposes.
        """
        fp_dim = self.fingerprint_dim

        code, code_padding_mask = self.encode(batch)
        bsz = code.size(0)

        token_types = torch.full([bsz, 1], fill_value=TokenType.START, dtype=torch.long, device=code.device)
        rxn_indices = torch.full([bsz, 1], fill_value=0, dtype=torch.long, device=code.device)
        reactant_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=code.device)
        reactants: list[list[Molecule | None]] = [[None] for _ in range(bsz)]
        reactions: list[list[Reaction | None]] = [[None] for _ in range(bsz)]

        for _ in tqdm(range(max_len - 1)):
            pred = self.predict(
                code=code,
                code_padding_mask=code_padding_mask,
                token_types=token_types,
                rxn_indices=rxn_indices,
                reactant_fps=reactant_fps,
                rxn_matrix=rxn_matrix,
                fpindex=fpindex,
                **options,
            )

            token_types = torch.cat([token_types, pred["token_next"]], dim=1)
            rxn_indices = torch.cat([rxn_indices, pred["rxn_indices_next"][:, :1]], dim=1)
            reactant_fps = torch.cat([reactant_fps, pred["fingerprint_next"][:, :1]], dim=1)
            for i, m in enumerate(pred["reactant_next"]):
                reactants[i].append(m[0])
            for i, r in enumerate(pred["reaction_next"]):
                reactions[i].append(r[0])

        return {
            "token_types": token_types,
            "rxn_indices": rxn_indices,
            "reactant_fps": reactant_fps,
            "reactants": reactants,
            "reactions": reactions,
        }


def draw_generation_results(result: GenerateResult):
    from PIL import Image

    from synformer.utils.image import draw_text, make_grid

    bsz, len = result["token_types"].size()
    im_list: list[Image.Image] = []
    for b in range(bsz):
        im: list[Image.Image] = []
        for l in range(len):
            if result["token_types"][b, l] == TokenType.START:
                im.append(draw_text("START"))
            elif result["token_types"][b, l] == TokenType.END:
                im.append(draw_text("END"))
                break
            elif result["token_types"][b, l] == TokenType.REACTION:
                rxn = result["reactions"][b][l]
                if rxn is not None:
                    im.append(rxn.draw())
            elif result["token_types"][b, l] == TokenType.REACTANT:
                reactant = result["reactants"][b][l]
                if reactant is not None:
                    im.append(reactant.draw())

        im_list.append(make_grid(im))
    return im_list
