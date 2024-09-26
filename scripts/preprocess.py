import pathlib

import click
from omegaconf import DictConfig, OmegaConf

from synformer.chem.fpindex import create_fingerprint_index_cache
from synformer.chem.matrix import create_reactant_reaction_matrix_cache
from synformer.chem.mol import FingerprintOption


@click.command()
@click.option("--model-config", type=OmegaConf.load, required=True)
def preprocess(model_config: DictConfig):
    out_fp = pathlib.Path(model_config.chem.fpindex)
    out_rxn = pathlib.Path(model_config.chem.rxn_matrix)
    building_block_path = pathlib.Path(model_config.chem.building_block_path)
    reaction_path = pathlib.Path(model_config.chem.reaction_path)

    if not out_fp.exists() or click.confirm(f"{out_fp} already exists. Overwrite?"):
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        fp_option = FingerprintOption(**model_config.chem.fp_option)
        fpindex = create_fingerprint_index_cache(
            molecule_path=building_block_path,
            cache_path=out_fp,
            fp_option=fp_option,
        )
        print(f"Number of molecules: {len(fpindex.molecules)}")
        print(f"Saved index to {out_fp}")

    if not out_rxn.exists() or click.confirm(f"{out_rxn} already exists. Overwrite?"):
        out_rxn.parent.mkdir(parents=True, exist_ok=True)

        m = create_reactant_reaction_matrix_cache(
            reactant_path=building_block_path,
            reaction_path=reaction_path,
            cache_path=out_rxn,
        )
        print(f"Number of reactants: {len(m.reactants)}")
        print(f"Number of reactions: {len(m.reactions)}")
        print(f"Saved matrix to {out_rxn}")


if __name__ == "__main__":
    preprocess()
