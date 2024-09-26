import os

import click
import pytorch_lightning as pl
import torch
from ipdb import set_trace as st
from omegaconf import OmegaConf

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
    get_experiment_name(config_name, vc_info.display_version, vc_info.committed_at)
    get_experiment_version()

    # Dataloaders
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )

    # Model
    wrapper = SynformerWrapper(config)

    st()

    result = wrapper.model.generate_without_stack(
        batch=datamodule, rxn_matrix=wrapper.rxn_matrix, fpindex=wrapper.fpindex
    )
    print(result)


if __name__ == "__main__":
    main()
