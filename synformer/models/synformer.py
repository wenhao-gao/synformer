import dataclasses

import torch
from torch import nn
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.chem.stack import Stack
from synformer.data.common import ProjectionBatch, TokenType

from .classifier_head import ClassifierHead
from .decoder import Decoder
from .encoder import get_encoder
from .fingerprint_head import ReactantRetrievalResult, get_fingerprint_head


@dataclasses.dataclass
class _ReactantItem:
    reactant: Molecule
    index: int
    score: float

    def __iter__(self):
        return iter([self.reactant, self.index, self.score])


@dataclasses.dataclass
class _ReactionItem:
    reaction: Reaction
    index: int
    score: float

    def __iter__(self):
        return iter([self.reaction, self.index, self.score])


@dataclasses.dataclass
class PredictResult:
    token_logits: torch.Tensor
    token_sampled: torch.Tensor
    reaction_logits: torch.Tensor
    retrieved_reactants: ReactantRetrievalResult

    def to(self, device: torch.device):
        self.__class__(
            self.token_logits.to(device),
            self.token_sampled.to(device),
            self.reaction_logits.to(device),
            self.retrieved_reactants,
        )
        return self

    def best_token(self) -> list[TokenType]:
        return [TokenType(t) for t in self.token_logits.argmax(dim=-1).detach().cpu().tolist()]

    def top_reactions(self, topk: int, rxn_matrix: ReactantReactionMatrix) -> list[list[_ReactionItem]]:
        topk = min(topk, self.reaction_logits.size(-1))
        logit, index = self.reaction_logits.topk(topk, dim=-1, largest=True)
        bsz = logit.size(0)
        out: list[list[_ReactionItem]] = []
        for i in range(bsz):
            out_i: list[_ReactionItem] = []
            for j in range(topk):
                idx = int(index[i, j].item())
                out_i.append(
                    _ReactionItem(
                        reaction=rxn_matrix.reactions[idx],
                        index=idx,
                        score=float(logit[i, j].item()),
                    )
                )
            out.append(out_i)
        return out

    def top_reactants(self, topk: int) -> list[list[_ReactantItem]]:
        bsz = self.retrieved_reactants.reactants.shape[0]
        score_all = 1.0 / (self.retrieved_reactants.distance.reshape(bsz, -1) + 0.1)
        index_all = self.retrieved_reactants.indices.reshape(bsz, -1)
        mols = self.retrieved_reactants.reactants.reshape(bsz, -1)

        topk = min(topk, mols.shape[-1])
        best_index = (-score_all).argsort(axis=-1)

        out: list[list[_ReactantItem]] = []
        for i in range(bsz):
            out_i: list[_ReactantItem] = []
            for j in range(topk):
                idx = int(best_index[i, j])
                out_i.append(
                    _ReactantItem(
                        reactant=mols[i, idx],
                        index=index_all[i, idx],
                        score=score_all[i, idx],
                    )
                )
            out.append(out_i)
        return out


@dataclasses.dataclass
class GenerateResult:
    code: torch.Tensor
    code_padding_mask: torch.Tensor

    token_types: torch.Tensor
    token_padding_mask: torch.Tensor

    rxn_indices: torch.Tensor

    reactant_fps: torch.Tensor
    predicted_fps: torch.Tensor
    reactant_indices: torch.Tensor

    reactants: list[list[Molecule | None]]
    reactions: list[list[Reaction | None]]

    @property
    def batch_size(self):
        return self.token_types.size(0)

    @property
    def seq_len(self):
        return self.token_types.size(1)

    def to_(self, device: str | torch.device):
        self.code = self.code.to(device)
        self.code_padding_mask = self.code_padding_mask.to(device)
        self.token_types = self.token_types.to(device)
        self.token_padding_mask = self.token_padding_mask.to(device)
        self.rxn_indices = self.rxn_indices.to(device)
        self.reactant_fps = self.reactant_fps.to(device)
        self.predicted_fps = self.predicted_fps.to(device)
        self.reactant_indices = self.reactant_indices.to(device)

    def build(self):
        stacks = [Stack() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                if self.token_types[i, j] == TokenType.START:
                    continue
                if self.token_types[i, j] == TokenType.END:
                    break
                if self.token_types[i, j] == TokenType.REACTION:
                    rxn = self.reactions[i][j]
                    if rxn is None:
                        break
                    success = stacks[i].push_rxn(rxn, int(self.rxn_indices[i, j].item()))
                    if not success:
                        break
                elif self.token_types[i, j] == TokenType.REACTANT:
                    mol = self.reactants[i][j]
                    if mol is None:
                        break
                    stacks[i].push_mol(mol, int(self.reactant_indices[i, j].item()))
        return stacks


class Synformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = get_encoder(cfg.encoder_type, cfg.encoder)
        decoder_kwargs = {}
        if "decoder_only" not in cfg.decoder and cfg.encoder_type == "none":
            decoder_kwargs["decoder_only"] = True
        self.decoder = Decoder(**cfg.decoder, **decoder_kwargs)
        self.d_model: int = self.encoder.dim

        self.token_head = ClassifierHead(self.d_model, max(TokenType) + 1)
        self.reaction_head = ClassifierHead(self.d_model, cfg.decoder.num_reaction_classes)
        self.fingerprint_head = get_fingerprint_head(cfg.fingerprint_head_type, cfg.fingerprint_head)

    def encode(self, batch: ProjectionBatch):
        return self.encoder(batch)

    def get_loss(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ):
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=token_padding_mask,
        )[:, :-1]

        token_types_gt = token_types[:, 1:].contiguous()
        rxn_indices_gt = rxn_indices[:, 1:].contiguous()
        reactant_fps_gt = reactant_fps[:, 1:].contiguous()

        loss_dict: dict[str, torch.Tensor] = {}
        aux_dict: dict[str, torch.Tensor] = {}

        # NOTE: token_padding_mask is True for padding tokens: ~token_padding_mask[:, :-1].contiguous()
        # We set the mask to None so the model perfers producing the `END` token when the embedding makes no sense
        loss_dict["token"] = self.token_head.get_loss(h, token_types_gt, None)
        loss_dict["reaction"] = self.reaction_head.get_loss(h, rxn_indices_gt, token_types_gt == TokenType.REACTION)

        fp_loss, fp_aux = self.fingerprint_head.get_loss(
            h,
            reactant_fps_gt,
            token_types_gt == TokenType.REACTANT,
            **options,
        )
        loss_dict.update(fp_loss)
        aux_dict.update(fp_aux)

        return loss_dict, aux_dict

    def get_loss_shortcut(self, batch: ProjectionBatch, **options):
        code, code_padding_mask, encoder_loss_dict = self.encode(batch)
        loss_dict, aux_dict = self.get_loss(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            **options,
        )
        loss_dict.update(encoder_loss_dict)
        return loss_dict, aux_dict

    def get_log_likelihood(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        **options,
    ) -> dict[str, torch.Tensor]:
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=token_padding_mask,
        )[:, :-1]

        token_types_gt = token_types[:, 1:].contiguous()
        rxn_indices_gt = rxn_indices[:, 1:].contiguous()
        reactant_fps_gt = reactant_fps[:, 1:].contiguous()

        ll_token_types = self.token_head.get_log_likelihood(h, token_types_gt, ~token_padding_mask[:, 1:])
        ll_rxn = self.reaction_head.get_log_likelihood(h, rxn_indices_gt, token_types_gt == TokenType.REACTION)
        ll_bb = self.fingerprint_head.get_log_likelihood(h, reactant_fps_gt, token_types_gt == TokenType.REACTANT)
        ll = ll_token_types + ll_rxn + ll_bb
        return {
            "token": ll_token_types,
            "reaction": ll_rxn,
            "reactant": ll_bb,
            "total": ll,
        }

    def get_log_likelihood_shortcut(self, batch: ProjectionBatch, **options):
        code, code_padding_mask, _ = self.encode(batch)
        return self.get_log_likelihood(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            **options,
        )

    @torch.no_grad()
    def predict(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        topk: int = 4,
        temperature_token: float = 0.1,
        **options,
    ):
        h = self.decoder(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            token_padding_mask=None,
        )
        h_next = h[:, -1]  # (bsz, h_dim)

        token_logits = self.token_head.predict(h_next)
        token_sampled = torch.multinomial(
            torch.nn.functional.softmax(token_logits / temperature_token, dim=-1),
            num_samples=1,
        )
        reaction_logits = self.reaction_head.predict(h_next)[..., : len(rxn_matrix.reactions)]
        retrieved_reactants = self.fingerprint_head.retrieve_reactants(
            h_next,
            fpindex,
            topk,
            mask=token_sampled == TokenType.REACTANT,
            **options,
        )
        return PredictResult(token_logits, token_sampled, reaction_logits, retrieved_reactants)

    @torch.no_grad()
    def generate_without_stack(
        self,
        batch: ProjectionBatch,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_len: int = 24,
        temperature_token: float = 1.0,
        temperature_reaction: float = 1.0,
        temperature_reactant: float = 1.0,
        **options,
    ):
        code, code_padding_mask, _ = self.encode(batch)
        bsz = code.size(0)
        fp_dim = self.fingerprint_head.fingerprint_dim

        token_padding_mask = torch.full([bsz, 1], fill_value=False, dtype=torch.bool, device=code.device)
        token_types = torch.full([bsz, 1], fill_value=TokenType.START, dtype=torch.long, device=code.device)
        rxn_indices = torch.full([bsz, 1], fill_value=0, dtype=torch.long, device=code.device)
        reactant_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=code.device)
        predicted_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=code.device)
        reactant_indices = torch.full([bsz, 1], fill_value=-1, dtype=torch.long, device=code.device)
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
                temperature_token=temperature_token,
                **options,
            )

            token_padding_mask_next = torch.logical_or(
                token_types[:, -1:] == TokenType.END, token_padding_mask[:, -1:]
            )
            token_padding_mask = torch.cat([token_padding_mask, token_padding_mask_next], dim=-1)

            token_next = pred.token_sampled
            token_types = torch.cat([token_types, token_next], dim=-1)

            # Reaction
            rxn_idx_next = torch.multinomial(
                torch.nn.functional.softmax(pred.reaction_logits / temperature_reaction, dim=-1),
                num_samples=1,
            )[..., 0]
            rxn_indices = torch.cat([rxn_indices, rxn_idx_next[..., None]], dim=-1)
            for b, idx in enumerate(rxn_idx_next):
                reactions[b].append(rxn_matrix.reactions[int(idx.item())])

            # Reactant (building block)
            fp_scores = (
                torch.from_numpy(1.0 / (pred.retrieved_reactants.distance + 1e-4)).to(reactant_fps).reshape(bsz, -1)
            )
            fp_idx_next = torch.multinomial(
                torch.nn.functional.softmax(fp_scores / temperature_reactant, dim=-1),
                num_samples=1,
            )[..., 0]

            fp_next = (
                torch.from_numpy(pred.retrieved_reactants.fingerprint_retrieved)
                .to(reactant_fps)
                .reshape(bsz, -1, fp_dim)  # (bsz, n_fps*topk, fp_dim)
            )[range(bsz), fp_idx_next]
            reactant_fps = torch.cat([reactant_fps, fp_next[..., None, :]], dim=-2)

            pfp_next = (
                torch.from_numpy(pred.retrieved_reactants.fingerprint_predicted)
                .to(predicted_fps)
                .reshape(bsz, -1, fp_dim)  # (bsz, n_fps*topk, fp_dim)
            )[range(bsz), fp_idx_next]
            predicted_fps = torch.cat([predicted_fps, pfp_next[..., None, :]], dim=-2)

            ridx_next = (
                torch.from_numpy(pred.retrieved_reactants.indices)
                .to(reactant_indices)
                .reshape(bsz, -1)[range(bsz), fp_idx_next]
            )
            reactant_indices = torch.cat([reactant_indices, ridx_next[..., None]], dim=-1)

            reactant_next = pred.retrieved_reactants.reactants.reshape(bsz, -1)[range(bsz), fp_idx_next.cpu().numpy()]
            for b, m in enumerate(reactant_next):
                reactants[b].append(m)

        return GenerateResult(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=token_types,
            token_padding_mask=token_padding_mask,
            rxn_indices=rxn_indices,
            reactant_fps=reactant_fps,
            predicted_fps=predicted_fps,
            reactant_indices=reactant_indices,
            reactants=reactants,
            reactions=reactions,
        )


def draw_generation_results(result: GenerateResult):
    from PIL import Image

    from synformer.utils.image import draw_text, make_grid

    bsz, len = result.token_types.size()
    im_list: list[Image.Image] = []
    for b in range(bsz):
        im: list[Image.Image] = []
        for l in range(len):
            if result.token_types[b, l] == TokenType.START:
                im.append(draw_text("START"))
            elif result.token_types[b, l] == TokenType.END:
                im.append(draw_text("END"))
                break
            elif result.token_types[b, l] == TokenType.REACTION:
                rxn = result.reactions[b][l]
                if rxn is not None:
                    im.append(rxn.draw())
            elif result.token_types[b, l] == TokenType.REACTANT:
                reactant = result.reactants[b][l]
                if reactant is not None:
                    im.append(reactant.draw())

        im_list.append(make_grid(im))
    return im_list
