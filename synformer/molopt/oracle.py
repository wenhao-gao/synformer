import dataclasses
import os
import pathlib
import signal
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from typing import TypeAlias, overload

import joblib
import numpy as np
import tdc
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc.metadata import docking_target_info


@dataclasses.dataclass(frozen=True)
class MolBufferItem:
    score: float
    index: int

    def __iter__(self):
        return iter([self.score, self.index])

    def __getitem__(self, key: int):
        return [self.score, self.index][key]


_SmilesString: TypeAlias = str
MolBuffer: TypeAlias = OrderedDict[_SmilesString, MolBufferItem]


def top_auc(buffer: MolBuffer, top_n: int, finish: bool, freq_log: int, max_oracle_calls: int) -> float:
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
    def __init__(
        self,
        evaluator: tdc.Evaluator | tdc.Oracle | Callable[[_SmilesString], float],
        output_dir: str | pathlib.Path,
        max_oracle_calls=10000,
        freq_log=100,
        mol_buffer: MolBuffer | None = None,
        save_last_n=5,
        n_cpu=12,
    ):
        self.name = None
        self.task_label = None
        self.evaluator = evaluator
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log

        # self.mol_buffer = mol_buffer or OrderedDict()
        self.mol_buffer = mol_buffer or {}
        self.sa_scorer = tdc.Oracle(name="SA")
        self.diversity_evaluator = tdc.Evaluator(name="Diversity")

        self.last_log = 0
        self.output_dir = pathlib.Path(output_dir)
        self.save_last_n = save_last_n
        self.saved_paths: list[pathlib.Path] = []
        self.n_cpu = n_cpu

    @property
    def budget(self):
        return self.max_oracle_calls

    def sort_buffer(self):
        # self.mol_buffer = OrderedDict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1].score, reverse=True))
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_buffer(self):
        if len(self.saved_paths) >= self.save_last_n:
            path = self.saved_paths.pop(0)
            path.unlink()
        output_path = self.output_dir / f"oracle_{self.last_log}.yaml"
        self.sort_buffer()
        with open(output_path, "w") as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
        self.saved_paths.append(output_path)

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

        metrics: dict[str, float] = {
            "avg_top1": np.max(scores),
            "avg_top10": np.mean(sorted(scores, reverse=True)[:10]),
            "avg_top100": np.mean(scores),
            "avg_sa_top100": np.mean(self.sa_scorer(smis)),  # Top 100
            "div_top100": self.diversity_evaluator(smis),  # Top 100
        }
        if finish:
            metrics["auc_top1"] = top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls)
            metrics["auc_top10"] = top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls)
            metrics["auc_top100"] = top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls)
            metrics["n_oracle"] = float(n_calls)

        formatted = " | ".join(
            [f"{n_calls}/{self.max_oracle_calls}"] + [f"{key}: {value:.3f}" for key, value in metrics.items()]
        )
        print(formatted)

        with open(self.output_dir / "oracle_log.txt", "a") as f:
            f.write(formatted + "\n")

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi: _SmilesString) -> float:
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0.0

        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0.0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi not in self.mol_buffer:
                self.mol_buffer[smi] = [float(self.evaluator(smi)), len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0]

    @overload
    def __call__(self, smiles_lst: list[_SmilesString]) -> list[float]:
        ...

    @overload
    def __call__(self, smiles_lst: _SmilesString) -> float:
        ...

    def __call__(self, smiles_lst: list[_SmilesString] | _SmilesString) -> list[float] | float:
        """
        Score
        """
        is_input_list = isinstance(smiles_lst, list)
        if not isinstance(smiles_lst, list):
            smiles_lst = [smiles_lst]
        score_list: list[float] = []

        if self.n_cpu > 1:
            score_list = joblib.Parallel(n_jobs=self.n_cpu)(joblib.delayed(self.evaluator)(smi) for smi in smiles_lst)

            for idx, smi in enumerate(smiles_lst):
                if smi not in self.mol_buffer:
                    self.mol_buffer[smi] = [score_list[idx], len(self.mol_buffer) + 1]

            if len(self.mol_buffer) > self.freq_log:
                self.sort_buffer()
                self.log_intermediate()
                self.save_buffer()
        else:
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_buffer()

        return score_list if is_input_list else score_list[0]

    @property
    def is_finished(self):
        return len(self.mol_buffer) >= self.max_oracle_calls

    def add_to_buffer(self, smi: _SmilesString, score: float):
        if smi not in self.mol_buffer:
            self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_buffer()


class VinaSMILES:
    """Perform docking search from a conformer."""

    def __init__(
        self,
        receptor_pdbqt_file,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float],
        scorefunction="vina",
        cpu=1,
    ):
        self.scorefunction = scorefunction
        self.cpu = cpu
        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.center = center
        self.box_size = box_size

    def __call__(
        self,
        ligand_smiles: _SmilesString,
        output_file="out.pdbqt",
        exhaustiveness=4,
        n_poses=1,
        time_limit=180,
    ):
        def handler(signum, frame):
            print(f"Docking exceeded the time limit of {time_limit} seconds")
            raise TimeoutError()

        # Set the alarm
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(time_limit)  # Set the time limit

        try:
            from vina import Vina

            v = Vina(sf_name=self.scorefunction, cpu=self.cpu)
            v.set_receptor(rigid_pdbqt_filename=self.receptor_pdbqt_file)
            v.compute_vina_maps(center=self.center, box_size=self.box_size)

            # Generate random temporary files
            with (
                tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as temp_mol,
                tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as temp_pdbqt,
            ):
                # Create the molecule from SMILES
                m = Chem.MolFromSmiles(ligand_smiles)
                m = Chem.AddHs(m)
                AllChem.EmbedMolecule(m)
                AllChem.MMFFOptimizeMolecule(m)

                # Write the molecule to a temporary file
                print(Chem.MolToMolBlock(m), file=open(temp_mol.name, "w+"))

                # Prepare the ligand using the temporary file
                os.system(f"mk_prepare_ligand.py -i {temp_mol.name} -o {temp_pdbqt.name}")

                # Set the ligand for docking
                v.set_ligand_from_file(temp_pdbqt.name)

                # Perform docking
                v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

                # Write the poses to the output file
                v.write_poses(output_file, n_poses=n_poses, overwrite=True)

                # Get the energy score
                energy: float = v.score()[0]

            # Cleanup temporary files
            os.remove(temp_mol.name)
            os.remove(temp_pdbqt.name)

            # Disable the alarm
            signal.alarm(0)

        except TimeoutError:
            print("Docking process exceeded the time limit.")
            return float(0)
        except ImportError:
            raise ImportError(
                "Please install vina following guidance in \
                    https://github.com/ccsb-scripps/AutoDock-Vina/tree/develop/build/python"
            )
        except Exception as e:
            print(e)
            return float(0)

        if energy > 0:
            return float(0)
        else:
            return -energy


def get_vina_oracle(task_name: str):
    if task_name == "3pbl":
        pdbid = "3pbl"
        return VinaSMILES(
            receptor_pdbqt_file="./data/receptors/3pbl.pdbqt",
            center=docking_target_info[pdbid]["center"],
            box_size=docking_target_info[pdbid]["size"],
            scorefunction="vina",
            cpu=4,
        )
    else:
        raise ValueError(f"Unknown Vina task: {task_name}")


def get_oracle_evaluator(name: str) -> tdc.Evaluator | tdc.Oracle | Callable[[_SmilesString], float]:
    try:
        domain, task = name.split(":")
    except ValueError:
        raise ValueError(f"Invalid oracle name: {name}")

    if domain == "tdc":
        return tdc.Oracle(task)
    elif domain == "vina":
        return get_vina_oracle(task)
    else:
        raise ValueError(f"Unknown oracle domain: {domain}")
