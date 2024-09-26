import os
import pathlib
import pickle

import click
import numpy as np
import tdc
import torch
import yaml
from omegaconf import OmegaConf
from rdkit import Chem, rdBase

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.stack import Stack
from synformer.models.synformer import Synformer

rdBase.DisableLog("rdApp.error")


def load_model(model_path: pathlib.Path):
    ckpt = torch.load(model_path, map_location="cpu")
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    model = Synformer(config.model)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})

    fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return model, fpindex, rxn_matrix


def create_dummy_feature(batch_size: int, device: torch.device):
    feat = {"_": torch.empty([batch_size, 1], dtype=torch.float, device=device)}
    return feat


def score_stack(sfn, stack: Stack, penalize_no_reaction: bool = True) -> float:
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
    ll = model.get_log_likelihood(
        code=result.code,
        code_padding_mask=result.code_padding_mask,
        token_types=result.token_types,
        rxn_indices=result.rxn_indices,
        reactant_fps=result.reactant_fps,
        token_padding_mask=result.token_padding_mask,
    )
    return result, stacks, ll


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")
        self.last_log = 0
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.output_dir, "results.yaml")
        else:
            output_file_path = os.path.join(self.output_dir, "results_" + suffix + ".yaml")

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):
        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[
                        : self.max_oracle_calls
                    ]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)

        print(
            f"{n_calls}/{self.max_oracle_calls} | "
            f"avg_top1: {avg_top1:.3f} | "
            f"avg_top10: {avg_top10:.3f} | "
            f"avg_top100: {avg_top100:.3f} | "
            f"avg_sa: {avg_sa:.3f} | "
            f"div: {diversity_top100:.3f}"
        )
        if finish:
            print(
                {
                    "avg_top1": avg_top1,
                    "avg_top10": avg_top10,
                    "avg_top100": avg_top100,
                    "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
                    "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
                    "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
                    "avg_sa": avg_sa,
                    "diversity_top100": diversity_top100,
                    "n_oracle": n_calls,
                }
            )

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if isinstance(smiles_lst, list):
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  # a string of SMILES
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


@click.command()
@click.option("--model-path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--device", type=torch.device, default="cuda")
@click.option("--task", type=str, required=True)
# @click.option("--task", type=get_oracle, required=True)
@click.option("--lr", type=float, default=1e-4)
@click.option("--batch-size", type=int, default=256)
@click.option("--sigma", type=float, default=500.0)
@click.option("--save-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(
    model_path: pathlib.Path,
    device: torch.device,
    task: str,
    lr: float,
    batch_size: int,
    sigma: float,
    save_path: pathlib.Path,
):
    oracle = Oracle()
    oracle.assign_evaluator(tdc.Oracle(name=task))
    # oracle.assign_evaluator(task)

    # print("Loading prior model...")
    # model_prior, fpindex, rxn_matrix = load_model(model_path)
    # model_prior = model_prior.to(device)
    # model_prior.eval()

    print("Creating agent model...")
    # model_agent = copy.deepcopy(model_prior)
    model_agent, fpindex, rxn_matrix = load_model(model_path)
    model_agent = model_agent.to(device)
    model_agent.train()

    # for param in model_prior.parameters():
    #     param.requires_grad_(False)

    feat = create_dummy_feature(batch_size, device)

    optimizer = torch.optim.Adam(model_agent.parameters(), lr=lr)

    try:
        while True:
            result, stacks, ll_agent_items = sample(model_agent, feat, rxn_matrix, fpindex)
            ll_agent = ll_agent_items["total"].sum(-1)
            # with torch.no_grad():
            #     ll_prior_items = model_prior.get_log_likelihood(
            #         code=result.code,
            #         code_padding_mask=result.code_padding_mask,
            #         token_types=result.token_types,
            #         rxn_indices=result.rxn_indices,
            #         reactant_fps=result.reactant_fps,
            #         token_padding_mask=result.token_padding_mask,
            #     )
            #     ll_prior = ll_prior_items["total"].sum(-1)

            scores = torch.tensor(
                [score_stack(oracle, stack) for stack in stacks],
                device=ll_agent.device,
                dtype=ll_agent.dtype,
            )
            # print(f"Average score: {scores.mean().item():.4f}")
            ll_augmented = sigma * scores
            # ll_augmented = ll_prior + sigma * scores
            loss = (ll_agent - ll_augmented).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if oracle.finish:
                break
    except KeyboardInterrupt:
        pass

    if not save_path.parent.exists():
        os.makedirs(save_path.parent)

    print("Saving model...")
    torch.save(model_agent.state_dict(), save_path)
    oracle.save_result(suffix=task + "_final")


if __name__ == "__main__":
    main()
