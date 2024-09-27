from time import time

import pandas as pd

from synformer.chem.mol import Molecule
from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles


def decode_leads():
    # JNK3 leads
    smiles_list = [
        "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1",
        "Oc1ccc(Nc2nc(-c3ccc(Cl)cc3)cs2)cc1",
        "O=C(c1ccc(Cn2c(=O)[nH]c3ccccc3c2=O)cc1)N1CCN(c2ccccc2)CC1",
        "O=C1NN(c2cccc(Cl)c2)C(=O)/C1=C/c1ccc(O)c([N+](=O)[O-])c1",
        "N#Cc1c(NC(=O)c2ccccc2OC(F)F)sc2c1CCC2",
        "Cc1ccc(C)c(NC(=O)Cc2cccc3ccccc23)c1",
        "O=C(Nc1ccccc1)N[C@H]1CO[C@@H]2[C@@H](Nc3nccc(-c4ccc(-c5ccccc5)cc4)n3)CO[C@H]12",
        "C[NH+]1CCc2c(sc(NC(=O)c3ccccc3F)c2C#N)C1",
        "N#Cc1c(NC(=O)c2cccc3ncccc23)sc2c1CCC2",
        "CSc1ccccc1NC(=O)Cc1cccc2ccccc12",
    ]

    # GSK3B leads
    # smiles_list = [
    #     'COc1cccc(N/N=C2\\C(=O)NN=C2c2ccccc2)c1',
    #     'COc1ccc(-c2nn(-c3ccccc3)cc2/C=N\\O)cc1',
    #     'O=[N+]([O-])c1ccc(-c2nnc(SCc3ccccc3)o2)cc1',
    #     'Cc1ccc(Nc2nc(N)c(C(=O)c3ccccc3)s2)cc1',
    #     'COc1cc(/C=C2/S/C(=N/c3ccc(O)cc3)NC2=O)ccc1O',
    #     'c1cc2c(cc1-c1ccc3[nH]ncc3c1)OCCO2',
    #     'O=C1NC(=O)C(c2c[nH]c3ncccc23)=C1Cl',
    #     'Cc1cccc(Nc2nc(NCC[NH+](C)C)nc(N)c2[N+](=O)[O-])c1',
    #     'COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1',
    #     'Cc1c(-c2ccnc(Nc3cccc(Cl)c3)n2)nnc2nc(-c3ccco3)nn12'
    # ]

    input = [Molecule(s) for s in smiles_list]

    model_path = "logs/default/epoch=144-step=725000.ckpt"

    t0 = time()

    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=model_path,
        search_width=24,
        exhaustiveness=64,
        num_gpus=-1,
        num_workers_per_gpu=1,
        task_qsize=0,
        result_qsize=0,
        time_limit=180,
        sort_by_scores=True,
    )

    print(f"Time: {time() - t0:.2f} s")

    # print(result_df)

    result_df.to_csv("gsk_results_decode_timed.csv", index=False)


def decode_unsynth():
    df = pd.read_csv("/home/whgao/synformer_dev/experiments/unsynthesizable/goal_hard_cwo_unsynth.csv")
    smiles_list = df.SMILES.to_list()

    input = [Molecule(s) for s in smiles_list]

    model_path = "logs/default/epoch=144-step=725000.ckpt"

    t0 = time()

    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=model_path,
        search_width=24,
        exhaustiveness=64,
        num_gpus=-1,
        num_workers_per_gpu=1,
        task_qsize=0,
        result_qsize=0,
        time_limit=180,
        sort_by_scores=True,
    )

    print(f"Time: {time() - t0:.2f} s")

    result_df.to_csv("goal_hard_cwo_unsynth_decode.csv", index=False)


def decode_sbdd():
    df = pd.read_csv("/home/whgao/synformer_dev/experiments/sbdd/pocket2mol.csv")
    smiles_list = df.smiles.to_list()

    input = [Molecule(s) for s in smiles_list]

    model_path = "logs/default/epoch=144-step=725000.ckpt"

    t0 = time()

    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=model_path,
        search_width=24,
        exhaustiveness=64,
        num_gpus=-1,
        num_workers_per_gpu=1,
        task_qsize=0,
        result_qsize=0,
        time_limit=180,
        sort_by_scores=True,
    )

    print(f"Time: {time() - t0:.2f} s")

    result_df.to_csv("sbdd_decode.csv", index=False)


def decode_hits():
    df = pd.read_csv("/home/whgao/synformer_dev/experiments/processed_data/active_pubchem.csv")
    smiles_list = df.ligand_smiles.to_list()

    input = [Molecule(s) for s in smiles_list]

    model_path = "logs/default/epoch=144-step=725000.ckpt"

    t0 = time()

    result_df = run_parallel_sampling_return_smiles(
        input=input,
        model_path=model_path,
        search_width=100,
        exhaustiveness=100,
        num_gpus=-1,
        num_workers_per_gpu=1,
        task_qsize=0,
        result_qsize=0,
        time_limit=1800,
        sort_by_scores=True,
    )

    print(f"Time: {time() - t0:.2f} s")

    result_df.to_csv("hits_decode.csv", index=False)


if __name__ == "__main__":
    # decode_unsynth()
    # decode_sbdd()

    decode_hits()
