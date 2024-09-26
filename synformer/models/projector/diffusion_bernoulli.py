import dataclasses
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.chem.reaction import Reaction
from synformer.data.common import ProjectionBatch, TokenType
from synformer.models.diffusion import BernoulliDiffusion, NoiseConditionedMLP
from synformer.models.transformer.graph_transformer import GraphTransformer
from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import BaseProjector, LossDict, NetworkOutputDict, PredictResult


@dataclasses.dataclass
class EncoderConfig:
    num_atom_classes: int = 100
    num_bond_classes: int = 10

    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    edge_dim: int = 128
    heads: int = 8
    rel_pos_emb: bool = False
    output_norm: bool = False


@dataclasses.dataclass
class DecoderConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    pe_max_len: int = 32
    output_norm: bool = False

    fingerprint_dim: int = 256
    num_reaction_classes: int = 100

    fingerprint_hidden_dim: int = 1024
    diffusion_num_steps: int = 100
    diffusion_s: float = 0.01


@dataclasses.dataclass
class ProjectorConfig:
    enc: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    dec: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)


def SimpleMLP(dim_in, dim_out) -> Callable[[torch.Tensor], torch.Tensor]:
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.ReLU(),
        nn.Linear(dim_out, dim_out),
        nn.ReLU(),
        nn.Linear(dim_out, dim_out),
    )


class BernoulliDiffusionProjector(BaseProjector):
    def __init__(self, cfg: ProjectorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ProjectorConfig()
        self.cfg = cfg

        self.atom_emb = nn.Embedding(cfg.enc.num_atom_classes + 1, cfg.enc.dim, padding_idx=0)
        self.bond_emb = nn.Embedding(cfg.enc.num_bond_classes + 1, cfg.enc.edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=cfg.enc.dim,
            depth=cfg.enc.depth,
            dim_head=cfg.enc.dim_head,
            edge_dim=cfg.enc.edge_dim,
            heads=cfg.enc.heads,
            rel_pos_emb=cfg.enc.rel_pos_emb,
            output_norm=cfg.enc.output_norm,
        )

        self.in_token = nn.Embedding(max(TokenType) + 1, cfg.dec.d_model)
        self.in_reaction = nn.Embedding(cfg.dec.num_reaction_classes, cfg.dec.d_model)
        self.in_fingerprint = SimpleMLP(cfg.dec.fingerprint_dim, cfg.dec.d_model)
        self.pe = PositionalEncoding(
            d_model=cfg.dec.d_model,
            max_len=cfg.dec.pe_max_len,
        )
        self.dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=cfg.dec.d_model,
                nhead=cfg.dec.nhead,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=cfg.dec.num_layers,
            norm=nn.LayerNorm(cfg.dec.d_model) if cfg.dec.output_norm else None,
        )
        self.out_token = SimpleMLP(cfg.dec.d_model, max(TokenType) + 1)
        self.out_reaction = SimpleMLP(cfg.dec.d_model, cfg.dec.num_reaction_classes)
        self.fingerprint_var_sched = BernoulliDiffusion(num_steps=cfg.dec.diffusion_num_steps, s=cfg.dec.diffusion_s)
        self.fingerprint_denoiser = NoiseConditionedMLP(
            dim_in=cfg.dec.d_model + cfg.dec.fingerprint_dim,
            dim_out=cfg.dec.fingerprint_dim,
            dim_hidden=cfg.dec.fingerprint_hidden_dim,
        )

    @property
    def fingerprint_dim(self) -> int:
        return self.cfg.dec.fingerprint_dim

    @property
    def model_dim(self) -> int:
        return self.cfg.dec.d_model

    def encode(self, batch: ProjectionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if "atoms" not in batch or "bonds" not in batch or "atom_padding_mask" not in batch:
            raise ValueError("atoms, bonds and atom_padding_mask must be in batch")
        atoms = batch["atoms"]
        bonds = batch["bonds"]
        atom_padding_mask = batch["atom_padding_mask"]

        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=atom_padding_mask)
        return node, atom_padding_mask

    def embed_seq(
        self,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
    ) -> torch.Tensor:
        emb_token = self.in_token(token_types)
        emb_rxn = self.in_reaction(rxn_indices)
        emb_fingerprint = self.in_fingerprint(reactant_fps)
        token_types_expand = token_types.unsqueeze(-1).expand(
            [token_types.size(0), token_types.size(1), self.cfg.dec.d_model]
        )
        emb_token = torch.where(token_types_expand == TokenType.REACTION, emb_rxn, emb_token)
        emb_token = torch.where(token_types_expand == TokenType.REACTANT, emb_fingerprint, emb_token)
        emb_token = self.pe(emb_token)
        return emb_token

    def predict_fingerprint_logit(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var_sched: BernoulliDiffusion | None = None,
    ) -> torch.Tensor:
        return self.fingerprint_denoiser(
            torch.cat([h, x_t], dim=-1),
            t=t,
            var_sched=var_sched or self.fingerprint_var_sched,
        )

    def get_loss(
        self,
        code: torch.Tensor | None,
        code_padding_mask: torch.Tensor | None,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        *,
        num_timestep_samples: int = 32,
        **options,
    ) -> tuple[NetworkOutputDict, LossDict]:
        bsz = token_types.size(0)
        if code is None:
            code, code_padding_mask = self.get_empty_code(bsz, device=reactant_fps.device, dtype=reactant_fps.dtype)

        emb_token = self.embed_seq(token_types, rxn_indices, reactant_fps)
        x_in = emb_token[:, :-1]
        seqlen = x_in.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seqlen,
            dtype=x_in.dtype,
            device=x_in.device,
        )
        tgt_key_padding_mask = torch.zeros(
            [bsz, seqlen],
            dtype=causal_mask.dtype,
            device=causal_mask.device,
        ).masked_fill_(token_padding_mask[:, :-1], -torch.finfo(causal_mask.dtype).max)
        y: torch.Tensor = self.dec(
            tgt=x_in,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=code_padding_mask,
            tgt_is_causal=True,
        )

        token_types_out = token_types[:, 1:]
        rxn_indices_out = rxn_indices[:, 1:]
        reactant_fps_out = reactant_fps[:, 1:]

        y_token = self.out_token(y)
        loss_token = F.cross_entropy(y_token.transpose(1, 2), token_types_out)

        y_reaction = self.out_reaction(y)
        reaction_flag = token_types_out == TokenType.REACTION
        loss_reaction = (
            F.cross_entropy(y_reaction.transpose(1, 2), rxn_indices_out, reduction="none") * reaction_flag
        ).sum() / (reaction_flag.sum() + 1e-6)

        fingerprint_flag = token_types_out == TokenType.REACTANT  # (bsz, len)
        fp_gt = reactant_fps_out[None].repeat(num_timestep_samples, 1, 1, 1)  # (n_smp, bsz, len, fp_dim)
        timestep = torch.randint(
            low=1,
            high=self.fingerprint_var_sched.num_steps + 1,
            size=fp_gt.shape[:-1],
            device=fp_gt.device,
        )  # (n_smp, bsz, len)
        timestamp_bitwise = timestep[..., None].expand_as(fp_gt)
        noisy_fp = self.fingerprint_var_sched.add_noise(x_0=fp_gt, t=timestamp_bitwise).value
        p_denoised = self.predict_fingerprint_logit(
            h=y[None].repeat(num_timestep_samples, 1, 1, 1),
            x_t=noisy_fp,
            t=timestep,
        ).sigmoid()
        posterior_fp_gt = self.fingerprint_var_sched.posterior(p_t=noisy_fp, p_0=fp_gt, t=timestamp_bitwise)
        posterior_fp_pred = self.fingerprint_var_sched.posterior(p_t=noisy_fp, p_0=p_denoised, t=timestamp_bitwise)
        bce_diffusion = F.binary_cross_entropy(input=posterior_fp_pred, target=posterior_fp_gt, reduction="none")
        loss_fingerprint_diffusion = (bce_diffusion.mean(0).sum(-1) * fingerprint_flag).sum() / (
            fingerprint_flag.sum() + 1e-6
        )

        bce_direct = F.binary_cross_entropy(input=p_denoised, target=fp_gt, reduction="none")
        loss_fingerprint_bce = (bce_direct.mean(0).sum(-1) * fingerprint_flag).sum() / (fingerprint_flag.sum() + 1e-6)

        y_dict: NetworkOutputDict = {
            "token": y_token,
            "reaction": y_reaction,
        }
        loss_dict: LossDict = {
            "token": loss_token,
            "reaction": loss_reaction,
            "fingerprint_diffusion": loss_fingerprint_diffusion,
            "fingerprint_bce": loss_fingerprint_bce,
        }
        return y_dict, loss_dict

    @torch.no_grad()
    def generate_fingerprint(
        self,
        h: torch.Tensor,
        x_T: torch.Tensor | None = None,
        var_sched: BernoulliDiffusion | None = None,
    ) -> torch.Tensor:
        var_sched = var_sched or self.fingerprint_var_sched
        fp_shape = [*h.shape[:-1], self.fingerprint_dim]
        if x_T is None:
            x_T = torch.randint(
                low=0,
                high=2,
                size=fp_shape,
                dtype=h.dtype,
                device=h.device,
            )

        x_t = x_T
        for t_index in range(var_sched.num_steps, 0, -1):
            t_bitwise = torch.full(fp_shape, fill_value=t_index, dtype=torch.long, device=h.device)
            t_fp = t_bitwise[..., 0]
            p_0_pred = self.predict_fingerprint_logit(h=h, x_t=x_t, t=t_fp, var_sched=var_sched).sigmoid()
            x_t = var_sched.denoise(x_t=x_t, p_0_pred=p_0_pred, t=t_bitwise).value
        return x_t

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
        result_device: torch.device | None = None,
        *,
        fingerprint_var_sched: BernoulliDiffusion | None = None,
        num_fingerprint_samples: int = 4,
        **options,
    ) -> PredictResult:
        bsz = token_types.size(0)
        if code is None:
            code, code_padding_mask = self.get_empty_code(bsz, device=reactant_fps.device, dtype=reactant_fps.dtype)
        result_device = result_device or code.device

        x = self.embed_seq(token_types, rxn_indices, reactant_fps)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        y: torch.Tensor = self.dec(
            tgt=x,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=code_padding_mask,
        )  # (bsz, seq_len, d_model)
        y_next = y[:, -1:]  # (bsz, 1, d_model)
        y_token = self.out_token(y_next)
        y_reaction = self.out_reaction(y_next)[..., : len(rxn_matrix.reactions)]
        y_fingerprint = self.generate_fingerprint(
            y_next.repeat(1, num_fingerprint_samples, 1),
            var_sched=fingerprint_var_sched,
        )  # (bsz, n_fps, fp_dim)

        token_next = torch.argmax(y_token, dim=-1)  # (bsz, 1)

        rxn_scores_next, rxn_indices_next = torch.sort(y_reaction, dim=-1, descending=True)
        rxn_scores_next = rxn_scores_next[:, 0, :topk]  # (bsz, 1, n_rxn) -> (bsz, topk)
        rxn_indices_next = rxn_indices_next[:, 0, :topk]
        reaction_next: list[list[Reaction | None]] = []
        for i in range(bsz):
            if token_next[i].item() != TokenType.REACTION:
                reaction_next.append([None] * topk)
            else:
                reaction_next.append([])
                for j in range(topk):
                    ridx = int(rxn_indices_next[i, j].item())
                    reaction_next[i].append(rxn_matrix.reactions[ridx])

        fp_query = y_fingerprint.detach()  # (bsz, n_fps, fp_dim)
        query_res = fpindex.query_cuda(fp_query, k=topk)
        fp_next_list: list[torch.Tensor] = []
        reactant_scores_next_list: list[torch.Tensor] = []
        reactant_next: list[list[Molecule | None]] = []
        reactant_indices_next_list: list[torch.Tensor] = []
        for i, q_res_subl in enumerate(query_res):
            fp_i: list[torch.Tensor] = []
            sc_i: list[float] = []
            mo_i: list[Molecule | None] = []
            mo_idx_i: list[int] = []

            for j, q_res in enumerate(q_res_subl):
                fp_i.append(torch.tensor(q_res.fingerprint, dtype=torch.float))
                sc_i.append(1 / q_res.distance)
                if token_next[i].item() != TokenType.REACTANT:
                    mo_i.append(None)
                    mo_idx_i.append(-1)
                else:
                    mo_i.append(q_res.molecule)
                    mo_idx_i.append(q_res.index)

            fp_next_list.append(torch.stack(fp_i, dim=0))
            reactant_next.append(mo_i)
            reactant_indices_next_list.append(torch.tensor(mo_idx_i))
            reactant_scores_next_list.append(torch.tensor(sc_i))

        fp_next = torch.stack(fp_next_list, dim=0).to(y_fingerprint)  # (bsz, topk, fp_dim)
        reactant_indices_next = torch.stack(reactant_indices_next_list, dim=0).to(y_fingerprint)  # (bsz, topk)
        reactant_scores_next = torch.stack(reactant_scores_next_list, dim=0).to(y_fingerprint)  # (bsz, topk)

        return {
            "y_token": y_token.to(result_device),
            "y_reaction": y_reaction.to(result_device),
            "y_fingerprint": y_fingerprint.to(result_device),
            "token_next": token_next.to(result_device),
            "rxn_indices_next": rxn_indices_next.to(result_device),
            "reaction_next": reaction_next,
            "rxn_scores_next": rxn_scores_next.to(result_device),
            "reactant_next": reactant_next,
            "fingerprint_next": fp_next.to(result_device),
            "reactant_indices_next": reactant_indices_next.to(result_device),
            "reactant_scores_next": reactant_scores_next.to(result_device),
        }
