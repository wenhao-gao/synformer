import os
import pickle
import random
from typing import cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.stack import create_stack
from synformer.utils.train import worker_init_fn

from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
)
from .common import ProjectionBatch, ProjectionData, create_data


class Collater:
    def __init__(self, max_datapoints: int | None = None, max_num_atoms: int = 96, max_num_tokens: int = 24):
        super().__init__()
        self.max_datapoints = max_datapoints
        self.max_num_atoms = max_num_atoms
        self.max_num_tokens = max_num_tokens

        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[list[ProjectionData]]) -> ProjectionBatch:
        flat = [d for subl in data_list for d in subl]
        if self.max_datapoints is not None:
            random.shuffle(flat)
            flat = flat[: self.max_datapoints]
            if len(flat) < self.max_datapoints:
                flat += random.choices(flat, k=self.max_datapoints - len(flat))
        flat_t = cast(list[dict[str, torch.Tensor]], flat)
        batch = {
            **apply_collate(self.spec_atoms, flat_t, max_size=self.max_num_atoms),
            **apply_collate(self.spec_tokens, flat_t, max_size=self.max_num_tokens),
            "mol_seq": [d["mol_seq"] for d in flat],
            "rxn_seq": [d["rxn_seq"] for d in flat],
        }
        return cast(ProjectionBatch, batch)


class ProjectionDataset(Dataset):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_num_atoms: int = 80,
        max_num_reactions: int = 5,
        virtual_length: int = 65536,
        init_stack_weighted_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_num_reactions = max_num_reactions
        self._virtual_length = virtual_length
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio

    def __len__(self) -> int:
        return self._virtual_length

    def __getitem__(self, _: int):
        stack = create_stack(
            self._reaction_matrix,
            max_num_reactions=self._max_num_reactions,
            max_num_atoms=self._max_num_atoms,
            init_stack_weighted_ratio=self._init_stack_weighted_ratio,
        )

        data_list: list[ProjectionData] = []
        mol_seq_full = stack.mols
        mol_idx_seq_full = stack.get_mol_idx_seq()
        rxn_seq_full = stack.rxns
        rxn_idx_seq_full = stack.get_rxn_idx_seq()
        data_list.append(
            create_data(
                product=mol_seq_full[0],
                mol_seq=mol_seq_full[:1],
                mol_idx_seq=mol_idx_seq_full[:1],
                rxn_seq=rxn_seq_full[:1],
                rxn_idx_seq=rxn_idx_seq_full[:1],
                fpindex=self._fpindex,
            )
        )
        for i in range(1, len(mol_seq_full)):
            if rxn_idx_seq_full[i] is not None:
                data_list.append(
                    create_data(
                        product=mol_seq_full[i],
                        mol_seq=mol_seq_full[: i + 1],
                        mol_idx_seq=mol_idx_seq_full[: i + 1],
                        rxn_seq=rxn_seq_full[: i + 1],
                        rxn_idx_seq=rxn_idx_seq_full[: i + 1],
                        fpindex=self._fpindex,
                    )
                )

        return data_list


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        max_datapoints_per_sample: int | None = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        if max_datapoints_per_sample is not None:
            self.max_datapoints: int | None = batch_size * max_datapoints_per_sample
        else:
            self.max_datapoints = None
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            virtual_length=self.config.train.val_freq * self.batch_size * trainer.world_size,
            fpindex=fpindex,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            virtual_length=self.batch_size,
            fpindex=fpindex,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(max_datapoints=self.max_datapoints),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=Collater(max_datapoints=self.max_datapoints),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
