import pathlib
import pickle

import torch
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.stack import Stack
from synformer.models.synformer import GenerateResult, Synformer

from .replay import ReplayBatch


def load_model(model_path: pathlib.Path):
    ckpt = torch.load(model_path, map_location="cpu")
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = Synformer(config.model)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})

    fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return model, config, fpindex, rxn_matrix


def save_model(model: Synformer, config: OmegaConf, output_path: pathlib.Path):
    ckpt = {
        "hyper_parameters": {"config": OmegaConf.to_container(config)},
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
    }
    torch.save(ckpt, output_path)


def create_dummy_feature(batch_size: int, device: torch.device | str):
    feat = {"_": torch.empty([batch_size, 1], dtype=torch.float, device=device)}
    return feat


def score_stack(sfn, stack: Stack, penalize_no_reaction: bool = True) -> float:
    from ipdb import set_trace as st

    st()
    if stack.get_stack_depth() != 1:
        score = 0.0
    else:
        score = max([sfn(m.smiles) for m in stack.get_top()])

    if penalize_no_reaction and stack.count_reactions() == 0:
        score *= 0.1

    return score


def sample(
    model: Synformer,
    feat: dict[str, torch.Tensor],
    rxn_matrix: ReactantReactionMatrix,
    fpindex: FingerprintIndex,
):
    with torch.no_grad():
        result = model.generate_without_stack(
            feat,
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            temperature_token=1.0,
            temperature_reactant=0.1,
            temperature_reaction=1.0,
        )
        stacks = result.build()
    return result, stacks


def get_loss(
    model_agent: Synformer,
    model_prior: Synformer | None,
    result: GenerateResult | ReplayBatch,
    scores: torch.Tensor,
    sigma: float,
):
    ll_agent_items = model_agent.get_log_likelihood(
        code=result.code,
        code_padding_mask=result.code_padding_mask,
        token_types=result.token_types,
        rxn_indices=result.rxn_indices,
        reactant_fps=result.reactant_fps,
        token_padding_mask=result.token_padding_mask,
    )
    ll_agent = ll_agent_items["total"].sum(-1)

    if model_prior is not None:
        with torch.no_grad():
            ll_prior_items = model_prior.get_log_likelihood(
                code=result.code,
                code_padding_mask=result.code_padding_mask,
                token_types=result.token_types,
                rxn_indices=result.rxn_indices,
                reactant_fps=result.reactant_fps,
                token_padding_mask=result.token_padding_mask,
            )
            ll_prior = ll_prior_items["total"].sum(-1)
    else:
        ll_prior = torch.zeros_like(ll_agent)

    ll_target = ll_prior + sigma * scores
    loss = (ll_agent - ll_target).pow(2).mean()
    return loss
