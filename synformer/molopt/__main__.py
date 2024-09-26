import copy
import datetime
import os
import pathlib
import pickle

import click
import torch
from omegaconf import OmegaConf

from synformer.chem.stack import Stack

from .model import create_dummy_feature, get_loss, load_model, sample, save_model
from .oracle import Oracle, get_oracle_evaluator
from .replay import Experience, ReplayBuffer


@click.command()
@click.option("model_path", "--model", type=click.Path(exists=True), required=True)
@click.option("oracle_name", "--oracle", type=str, required=True)
@click.option("--lr", type=float, default=1e-4)
@click.option("--batch-size", type=int, default=240)
@click.option("--max-call", type=int, default=10000)
@click.option("--device", type=str, default="cuda")
@click.option("--sigma", type=float, default=500.0)
@click.option("--log-dir", type=click.Path(), default="logs_molopt")
@click.option("--use-prior", is_flag=True)
@click.option("--use-replay-buffer", is_flag=True)
def main(
    model_path: pathlib.Path | str,
    device: str,
    oracle_name: str,
    lr: float,
    batch_size: int,
    max_call: int,
    sigma: float,
    log_dir: pathlib.Path | str,
    use_prior: bool,
    use_replay_buffer: bool,
):
    cfgs = OmegaConf.create(locals())
    model_path = pathlib.Path(model_path)
    log_dir = pathlib.Path(log_dir)

    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    task_name = f"{oracle_name.replace(':', '_')}_{timestamp}"
    task_dir = log_dir / task_name
    os.makedirs(task_dir, exist_ok=True)
    OmegaConf.save(cfgs, task_dir / "config.yaml")

    oracle = Oracle(get_oracle_evaluator(oracle_name), output_dir=task_dir, max_oracle_calls=max_call)
    model, config, fpindex, rxn_matrix = load_model(model_path)

    model_agent = model
    model_agent = model_agent.to(device)

    if use_prior:
        model_prior = copy.deepcopy(model)
        for param in model_prior.parameters():
            param.requires_grad_(False)
        model_prior = model_prior.to(device)
    else:
        model_prior = None

    optimizer = torch.optim.Adam(model_agent.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=100000)
    synthesis_buffer: dict[str, Stack] = {}

    try:
        while True:
            optimizer.zero_grad()

            if use_replay_buffer:
                replay = replay_buffer.sample(count=batch_size, device=device)
                if replay is not None:
                    loss_scale = replay.batch_size / batch_size
                    loss_replay = get_loss(model_agent, model_prior, replay, replay.scores, sigma) * loss_scale
                    print(f"Replay batch size: {replay.batch_size} average score: {replay.scores.mean().item()}")
                    loss_replay.backward()

            feat = create_dummy_feature(batch_size, device)
            result, stacks = sample(model_agent, feat, rxn_matrix, fpindex)
            tops = [stack.get_one_top().smiles for stack in stacks]
            scores = oracle(tops)
            scores = torch.tensor(scores, device=device, dtype=torch.float)
            # scores = torch.tensor([score_stack(oracle, stack) for stack in stacks], device=device, dtype=torch.float)
            loss = get_loss(model_agent, model_prior, result, scores, sigma)
            loss.backward()

            optimizer.step()

            for experience in Experience.from_samples(result, stacks, scores):
                replay_buffer.add(experience)

            for stack in stacks:
                if stack.get_stack_depth() == 1:
                    for mol in stack.get_top():
                        synthesis_buffer[mol.smiles] = stack

            if oracle.is_finished:
                oracle.log_intermediate(finish=True)
                break

    except KeyboardInterrupt:
        pass

    print("Saving model...")
    torch.save(model_agent.state_dict(), task_dir / "model.pt")
    save_model(model_agent, config, task_dir / "model.pt")
    oracle.save_buffer()
    with open(task_dir / "synthesis.pkl", "wb") as f:
        pickle.dump(synthesis_buffer, f)


if __name__ == "__main__":
    main()
