import os

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies

from synformer.data.projection_dataset_new import ProjectionDataModule
from synformer.models.wrapper import SynformerWrapper
from synformer.utils.misc import (
    get_config_name,
    get_experiment_name,
    get_experiment_version,
)
from synformer.utils.vc import get_vc_info

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--seed", type=int, default=42)
@click.option("--debug", is_flag=True)
@click.option("--batch-size", "-b", type=int, default=196)
@click.option("--num-workers", type=int, default=8)
@click.option("--devices", type=int, default=4)
@click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--num-sanity-val-steps", type=int, default=1)
@click.option("--log-dir", type=click.Path(dir_okay=True, file_okay=False), default="./logs")
@click.option("--resume", type=click.Path(exists=True, dir_okay=False), default=None)
def main(
    config_path: str,
    seed: int,
    debug: bool,
    batch_size: int,
    num_workers: int,
    devices: int,
    num_nodes: int,
    num_sanity_val_steps: int,
    log_dir: str,
    resume: str | None,
):
    if batch_size % devices != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = batch_size // devices

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    config = OmegaConf.load(config_path)
    config_name = get_config_name(config_path)
    vc_info = get_vc_info()
    vc_info.disallow_changes(debug)
    exp_name = get_experiment_name(config_name, vc_info.display_version, vc_info.committed_at)
    exp_ver = get_experiment_version()

    # Dataloaders
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )

    # Model
    model = SynformerWrapper(
        config=config,
        args={
            "config_path": config_path,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "resume": resume,
        },
    )

    # Train
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True),
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=config.train.max_iters,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, monitor="val/loss", mode="min", save_top_k=5),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=[
            loggers.TensorBoardLogger(log_dir, name=exp_name, version=exp_ver),
        ],
        val_check_interval=config.train.val_freq,
        limit_val_batches=4,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)


if __name__ == "__main__":
    main()
