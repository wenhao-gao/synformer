import pickle
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.data.common import ProjectionBatch, draw_batch
from synformer.utils.train import get_optimizer, get_scheduler, sum_weighted_losses

from .base import BaseProjector, draw_generation_results
from .diffusion import DiffusionProjector
from .diffusion_bernoulli import BernoulliDiffusionProjector
from .diffusion_bernoulli_smiles import BernoulliDiffusionSMILESProjector
from .vanilla import VanillaProjector


def get_model(model_type: str, model_config) -> BaseProjector:
    if model_type == "vanilla":
        return VanillaProjector(model_config)
    elif model_type == "diffusion":
        return DiffusionProjector(model_config)
    elif model_type == "diffusion_bernoulli":
        return BernoulliDiffusionProjector(model_config)
    elif model_type == "diffusion_bernoulli_smiles":
        return BernoulliDiffusionSMILESProjector(model_config)
    raise ValueError(f"Unknown model type: {model_type}")


class ProjectorWrapper(pl.LightningModule):
    def __init__(self, config, args: dict | None = None):
        super().__init__()
        self.save_hyperparameters(
            {
                "config": OmegaConf.to_container(config),
                "args": args or {},
            }
        )
        self.model = get_model(config.model_type, config.model)

    @property
    def config(self):
        return OmegaConf.create(self.hparams["config"])

    @property
    def args(self):
        return OmegaConf.create(self.hparams.get("args", {}))

    def setup(self, stage: str) -> None:
        super().setup(stage)

        # Load chem data
        with open(self.config.chem.rxn_matrix, "rb") as f:
            self.rxn_matrix: ReactantReactionMatrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            self.fpindex: FingerprintIndex = pickle.load(f)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config.train.optimizer, self.model)
        if "scheduler" in self.config.train:
            scheduler = get_scheduler(self.config.train.scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }
        return optimizer

    def training_step(self, batch: ProjectionBatch, batch_idx: int):
        y_dict, loss_dict = self.model.get_loss_shortcut(batch, warmup=self.current_epoch == 0)
        loss_sum = sum_weighted_losses(loss_dict, self.config.train.loss_weights)

        self.log("train/loss", loss_sum, on_step=True, prog_bar=True, logger=True)
        self.log_dict({f"train/loss_{k}": v for k, v in loss_dict.items()}, on_step=True, logger=True)

        if "fp_select" in y_dict:
            fp_select: torch.Tensor = y_dict["fp_select"]
            fp_ratios: dict[str, float] = {}
            for i in range(self.model.cfg.dec.num_out_fingerprints):
                ratio = (fp_select == i).float().mean().nan_to_num(0.0)
                fp_ratios[f"fp_select/{i}"] = ratio.item()
            self.log_dict(fp_ratios, on_step=True, logger=True)
        return loss_sum

    def validation_step(self, batch: ProjectionBatch, batch_idx: int) -> Any:
        _, loss_dict = self.model.get_loss_shortcut(batch)
        loss_weight = self.config.train.get("val_loss_weights", self.config.train.loss_weights)
        loss_sum = sum_weighted_losses(loss_dict, loss_weight)

        self.log("val/loss", loss_sum, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log_dict({f"val/loss_{k}": v for k, v in loss_dict.items()}, on_step=False, logger=True, sync_dist=True)

        # Generate
        if self.args.get("visualize", True) and batch_idx == 0:
            result = self.model.generate_without_stack(batch=batch, rxn_matrix=self.rxn_matrix, fpindex=self.fpindex)
            images_gen = draw_generation_results(result)
            images_ref = draw_batch(batch)
            if self.logger is not None:
                tb_logger = self.logger.experiment
                for i, (image_gen, image_ref) in enumerate(zip(images_gen, images_ref)):
                    tb_logger.add_images(
                        f"val/{i}_generate",
                        np.array(image_gen) / 255,
                        self.current_epoch,
                        dataformats="HWC",
                    )
                    tb_logger.add_images(
                        f"val/{i}_reference",
                        np.array(image_ref) / 255,
                        self.current_epoch,
                        dataformats="HWC",
                    )

        return loss_sum
