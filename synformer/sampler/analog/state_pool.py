import copy
import dataclasses
import itertools
import time
from collections.abc import Iterable
from functools import cached_property
from multiprocessing.synchronize import Lock

import pandas as pd
import torch
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import FingerprintOption, Molecule
from synformer.chem.stack import Stack
from synformer.data.collate import (
    apply_collate,
    collate_1d_features,
    collate_padding_masks,
    collate_tokens,
)
from synformer.data.common import TokenType, featurize_stack
from synformer.models.synformer import Synformer


@dataclasses.dataclass
class State:
    stack: Stack = dataclasses.field(default_factory=Stack)
    scores: list[float] = dataclasses.field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(self.scores)

    def featurize(self, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
        feats = featurize_stack(self.stack, end_token=False, fpindex=fpindex)
        return feats


@dataclasses.dataclass
class _ProductInfo:
    molecule: Molecule
    stack: Stack


class TimeLimit:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._start = time.time()

    def exceeded(self) -> bool:
        if self._seconds <= 0:
            return False
        return time.time() - self._start > self._seconds

    def check(self):
        if self.exceeded():
            raise TimeoutError()


class StatePool:
    def __init__(
        self,
        fpindex: FingerprintIndex,
        rxn_matrix: ReactantReactionMatrix,
        mol: Molecule,
        model: Synformer,
        factor: int = 16,
        max_active_states: int = 256,
        sort_by_score: bool = True,
    ) -> None:
        super().__init__()
        self._fpindex = fpindex
        self._rxn_matrix = rxn_matrix

        self._model = model
        self._mol = mol
        device = next(iter(model.parameters())).device
        atoms, bonds = mol.featurize_simple()
        self._atoms = atoms[None].to(device)
        self._bonds = bonds[None].to(device)
        num_atoms = atoms.size(0)
        self._atom_padding_mask = torch.zeros([1, num_atoms], dtype=torch.bool, device=device)

        smiles = mol.tokenize_csmiles()
        self._smiles = smiles[None].to(device)

        self._factor = factor
        self._max_active_states = max_active_states
        self._sort_by_score = sort_by_score

        self._active: list[State] = [State()]
        self._finished: list[State] = []
        self._aborted: list[State] = []

    @cached_property
    def device(self) -> torch.device:
        return self._atoms.device

    @cached_property
    def code(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            code, code_padding_mask, encoder_loss_dict = self._model.encode(
                {
                    "atoms": self._atoms,
                    "bonds": self._bonds,
                    "atom_padding_mask": self._atom_padding_mask,
                    "smiles": self._smiles,
                }
            )
            return code, code_padding_mask

    def _sort_states(self) -> None:
        if self._sort_by_score:
            self._active.sort(key=lambda s: s.score, reverse=True)
        self._active = self._active[: self._max_active_states]

    def _collate(self, feat_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }
        return apply_collate(spec_tokens, feat_list, feat_list[0]["token_types"].size(0))

    def evolve(
        self,
        gpu_lock: Lock | None = None,
        show_pbar: bool = False,
        time_limit: TimeLimit | None = None,
    ) -> None:
        if len(self._active) == 0:
            return
        feat_list = [
            featurize_stack(
                state.stack,
                end_token=False,
                fpindex=self._fpindex,
            )
            for state in self._active
        ]

        if gpu_lock is not None:
            gpu_lock.acquire()

        feat = {k: v.to(self.device) for k, v in self._collate(feat_list).items()}

        code, code_padding_mask = self.code
        code_size = list(code.size())
        code_size[0] = len(feat_list)
        code = code.expand(code_size)
        mask_size = list(code_padding_mask.size())
        mask_size[0] = len(feat_list)
        code_padding_mask = code_padding_mask.expand(mask_size)

        result = self._model.predict(
            code=code,
            code_padding_mask=code_padding_mask,
            token_types=feat["token_types"],
            rxn_indices=feat["rxn_indices"],
            reactant_fps=feat["reactant_fps"],
            rxn_matrix=self._rxn_matrix,
            fpindex=self._fpindex,
            topk=self._factor,
            result_device=torch.device("cpu"),
        )

        if gpu_lock is not None:
            gpu_lock.release()

        n = code.size(0)
        m = self._factor
        nm_iter: Iterable[tuple[int, int]] = itertools.product(range(n), range(m))
        if show_pbar:
            nm_iter = tqdm(nm_iter, total=n * m, desc="evolve", dynamic_ncols=True)

        best_token = result.best_token()
        top_reactants = result.top_reactants(topk=m)
        top_reactions = result.top_reactions(topk=m, rxn_matrix=self._rxn_matrix)

        next: list[State] = []
        for i, j in nm_iter:
            if time_limit is not None and time_limit.exceeded():
                break

            tok_next = best_token[i]
            base_state = self._active[i]
            if tok_next == TokenType.END:
                self._finished.append(base_state)

            elif tok_next == TokenType.REACTANT:
                reactant, mol_idx, score = top_reactants[i][j]
                new_state = copy.deepcopy(base_state)
                new_state.stack.push_mol(reactant, mol_idx)
                new_state.scores.append(score)
                next.append(new_state)

            elif tok_next == TokenType.REACTION:
                reaction, rxn_idx, score = top_reactions[i][j]
                new_state = copy.deepcopy(base_state)
                success = new_state.stack.push_rxn(reaction, rxn_idx, product_limit=None)
                if success:
                    rxn_score = max(
                        [self._mol.sim(m, fp_option=FingerprintOption.rdkit()) for m in new_state.stack.get_top()]
                    )
                    new_state.scores.append(rxn_score)
                    next.append(new_state)
                else:
                    self._aborted.append(new_state)

            else:
                self._aborted.append(base_state)

        del self._active
        self._active = next
        self._sort_states()

    def get_products(self) -> Iterable[_ProductInfo]:
        visited: set[Molecule] = set()
        for state in self._finished:
            for mol in state.stack.get_top():
                if mol in visited:
                    continue
                yield _ProductInfo(mol, state.stack)
                visited.add(mol)
        yield from []

    def get_dataframe(self, num_calc_extra_metrics: int = 10) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        smiles_to_mol: dict[str, Molecule] = {}
        for product in self.get_products():
            rows.append(
                {
                    "target": self._mol.smiles,
                    "smiles": product.molecule.smiles,
                    "score": self._mol.sim(product.molecule, FingerprintOption.morgan_for_tanimoto_similarity()),
                    "synthesis": product.stack.get_action_string(),
                    "num_steps": product.stack.count_reactions(),
                }
            )
            smiles_to_mol[product.molecule.smiles] = product.molecule
        rows.sort(key=lambda r: r["score"], reverse=True)
        for row in rows[:num_calc_extra_metrics]:
            mol = smiles_to_mol[str(row["smiles"])]
            row["scf_sim"] = self._mol.scaffold.tanimoto_similarity(
                mol.scaffold,
                fp_option=FingerprintOption.morgan_for_tanimoto_similarity(),
            )
            row["pharm2d_sim"] = self._mol.dice_similarity(mol, fp_option=FingerprintOption.gobbi_pharm2d())
            row["rdkit_sim"] = self._mol.tanimoto_similarity(mol, fp_option=FingerprintOption.rdkit())

        df = pd.DataFrame(rows)
        return df

    def print_stats(self) -> None:
        print(f"Active: {len(self._active)}")
        print(f"Finished: {len(self._finished)}")
        print(f"Aborted: {len(self._aborted)}")
