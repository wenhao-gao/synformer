import pathlib
import pickle

import click
import torch
from omegaconf import OmegaConf

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import Molecule
from synformer.models.synformer import Synformer


def load_model(model_path: pathlib.Path, config_path: pathlib.Path | None, device: torch.device):
    ckpt = torch.load(model_path, map_location="cpu")

    if config_path is None:
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = Synformer(config.model).to(device)
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    else:
        config = OmegaConf.load(config_path)
        model = Synformer(config.model).to(device)
        model.load_state_dict(ckpt)
    model.eval()

    fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return model, fpindex, rxn_matrix


def featurize_smiles(smiles: str, device: torch.device, repeat: int = 1):
    mol = Molecule(smiles)
    atoms, bonds = mol.featurize_simple()
    atoms = atoms[None].repeat(repeat, 1).to(device)
    bonds = bonds[None].repeat(repeat, 1, 1).to(device)
    num_atoms = atoms.size(0)
    atom_padding_mask = torch.zeros([1, num_atoms], dtype=torch.bool, device=device)

    smiles_t = mol.tokenize_csmiles()
    smiles_t = smiles_t[None].repeat(repeat, 1).to(device)
    feat = {
        "atoms": atoms,
        "bonds": bonds,
        "atom_padding_mask": atom_padding_mask,
        "smiles": smiles_t,
    }
    return mol, feat


@click.command()
@click.option("--smiles", type=str, default="COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1")
@click.option("--model-path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--config-path", type=click.Path(exists=True, path_type=pathlib.Path), required=False, default=None)
@click.option("--device", type=torch.device, default="cuda")
@click.option("--repeat", type=int, default=100)
def main(smiles, model_path: pathlib.Path, config_path: pathlib.Path | None, device: torch.device, repeat: int):
    model, fpindex, rxn_matrix = load_model(model_path, config_path, device)
    mol, feat = featurize_smiles(smiles, device, repeat=repeat)

    with torch.inference_mode():
        result = model.generate_without_stack(
            feat,
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            temperature_token=1.0,
            temperature_reactant=0.1,
            temperature_reaction=1.0,
        )
        ll = model.get_log_likelihood(
            code=result.code,
            code_padding_mask=result.code_padding_mask,
            token_types=result.token_types,
            rxn_indices=result.rxn_indices,
            reactant_fps=result.reactant_fps,
            token_padding_mask=result.token_padding_mask,
        )

    stacks = result.build()
    cnt = 0
    for i, stack in enumerate(stacks):
        if stack.get_stack_depth() == 1:
            analog = stack.get_one_top()
            ll_this = ll["total"][i].sum().item()
            cnt_rxn = stack.count_reactions()
            print(f"{analog.sim(mol):.2f} {cnt_rxn} {ll_this:.4f} {analog.smiles}")
            cnt += 1
    print(f"Total: {cnt} / {len(stacks)}")


if __name__ == "__main__":
    main()
