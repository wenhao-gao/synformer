import dataclasses
import pathlib
import shutil
import subprocess
import tempfile

import click
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from synformer.chem.mol import Molecule


@dataclasses.dataclass
class QVinaOption:
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 16

    @classmethod
    def from_receptor_description(cls, path: pathlib.Path):
        desc = OmegaConf.load(path)
        return QVinaOption(
            center_x=desc.center_of_mass[0],
            center_y=desc.center_of_mass[1],
            center_z=desc.center_of_mass[2],
        )


@dataclasses.dataclass
class QVinaOutput:
    smiles: str
    path: pathlib.Path
    mode_id: int
    affinity: float
    receptor_name: str = ""


class QVinaDockingTask:
    def __init__(
        self,
        receptor_path: pathlib.Path,
        ligand_mol: Molecule,
        qvina_path: pathlib.Path,
        qvina_option: QVinaOption,
        obabel_path: pathlib.Path,
        output_path: pathlib.Path,
        overwrite: bool = False,
        receptor_name: str = "",
    ) -> None:
        super().__init__()
        if receptor_path.suffix != ".pdbqt":
            raise ValueError(f"Unsupported file format: {receptor_path.suffix}")
        self._receptor_path = receptor_path
        self._ligand_mol = ligand_mol
        self._qvina_path = qvina_path
        self._qvina_option = qvina_option
        self._obabel_path = obabel_path
        self._output_path = output_path
        self._overwrite = overwrite

        self.receptor_name = receptor_name

    def __call__(self, workdir_path: pathlib.Path | None = None) -> QVinaOutput | None:
        if workdir_path is None:
            with tempfile.TemporaryDirectory() as workdir:
                workdir_path = pathlib.Path(workdir)
                out = self._process(workdir_path)
        else:
            out = self._process(workdir_path)
        if len(out) == 0:
            return None
        return min(out, key=lambda x: x.affinity)

    def _process(self, workdir_path: pathlib.Path) -> list[QVinaOutput]:
        if self._overwrite or not self._output_path.exists():
            prep_success = self._prepare_ligand(workdir_path)
            if not prep_success:
                print(f"Ligand preparation failed: {self._output_path}")
                return []
            vina_out = self._run_docking(workdir_path)
            if vina_out.returncode != 0:
                print(f"Docking failed: {self._output_path}")
                print(vina_out.stdout)
                print(vina_out.stderr)
                return []
        return self._parse_qvina_outputs(self._output_path)

    def _prepare_ligand(self, workdir_path: pathlib.Path) -> bool:
        mol = Chem.AddHs(self._ligand_mol._rdmol, addCoords=True)
        Chem.AllChem.EmbedMultipleConfs(mol, numConfs=1)
        try:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except ValueError:
            print(f"Failed to optimize molecule: {self._ligand_mol.smiles}")
            return False
        sdf_writer = Chem.SDWriter(str(workdir_path / "ligand.sdf"))
        sdf_writer.write(mol)
        sdf_writer.close()

        out = subprocess.run(
            [str(self._obabel_path.absolute()), "ligand.sdf", "-Oligand.pdbqt"],
            cwd=workdir_path,
            check=False,
            capture_output=True,
        )
        return out.returncode == 0

    def _run_docking(self, workdir_path: pathlib.Path):
        shutil.copy(self._receptor_path, workdir_path / "receptor.pdbqt")
        # fmt: off
        cmd: list[str] = [
            str(self._qvina_path.absolute()),
            "--receptor", "receptor.pdbqt",
            "--ligand", "ligand.pdbqt",
            "--center_x", f"{self._qvina_option.center_x:.3f}",
            "--center_y", f"{self._qvina_option.center_y:.3f}",
            "--center_z", f"{self._qvina_option.center_z:.3f}",
            "--size_x", f"{self._qvina_option.size_x:.3f}",
            "--size_y", f"{self._qvina_option.size_y:.3f}",
            "--size_z", f"{self._qvina_option.size_z:.3f}",
            "--exhaustiveness", f"{self._qvina_option.exhaustiveness}",
        ]
        # fmt: on

        vina_out = subprocess.run(
            cmd,
            cwd=workdir_path,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [str(self._obabel_path.absolute()), "ligand_out.pdbqt", "-Oligand_out.sdf"],
            cwd=workdir_path,
            check=False,
            capture_output=True,
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(workdir_path / "ligand_out.sdf", self._output_path)

        return vina_out

    def _parse_qvina_outputs(self, docked_sdf_path: pathlib.Path) -> list[QVinaOutput]:
        if not docked_sdf_path.exists():
            return []
        sdf_lines = docked_sdf_path.read_text().splitlines()
        scores = [float(line.strip().split()[2]) for line in sdf_lines if line.startswith(" VINA RESULT:")]
        results: list[QVinaOutput] = []
        for i, score in enumerate(scores, start=1):
            results.append(
                QVinaOutput(
                    smiles=self._ligand_mol.smiles,
                    path=docked_sdf_path,
                    mode_id=i,
                    affinity=score,
                    receptor_name=self.receptor_name,
                )
            )
        return results

    def __repr__(self) -> str:
        return repr(
            {
                "receptor_path": self._receptor_path,
                "ligand_mol": self._ligand_mol,
                "qvina_path": self._qvina_path,
                "qvina_option": self._qvina_option,
                "obabel_path": self._obabel_path,
                "output_path": self._output_path,
            }
        )


@click.command()
@click.option(
    "--receptor-path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=pathlib.Path("data/LIT-PCBA-pockets/ADRB2/receptor.pdbqt"),
)
@click.option("--ligand-mol", type=Molecule, default="CC(C)Nc1nc(NC(C)C)nc(n1)n2nc(C)cc2C")
@click.option(
    "--qvina-path",
    type=click.Path(exists=True, dir_okay=False, executable=True, path_type=pathlib.Path),
    default=pathlib.Path("bin/qvina2.1"),
)
@click.option(
    "--obabel-path",
    type=click.Path(exists=True, dir_okay=False, executable=True, path_type=pathlib.Path),
    default=pathlib.Path("bin/obabel"),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    default=pathlib.Path("data/LIT-PCBA-pockets/ADRB2/docked.sdf"),
)
@click.option("--center-x", type=float, default=-9.84)
@click.option("--center-y", type=float, default=-4.32)
@click.option("--center-z", type=float, default=39.35)
def main(
    receptor_path: pathlib.Path,
    ligand_mol: Molecule,
    qvina_path: pathlib.Path,
    obabel_path: pathlib.Path,
    output_path: pathlib.Path,
    center_x: float,
    center_y: float,
    center_z: float,
):
    receptor_path = receptor_path.resolve()
    qvina_option = QVinaOption(center_x=center_x, center_y=center_y, center_z=center_z)
    task = QVinaDockingTask(
        receptor_path=receptor_path,
        ligand_mol=ligand_mol,
        qvina_path=qvina_path,
        qvina_option=qvina_option,
        obabel_path=obabel_path,
        output_path=output_path,
    )
    print(task())


if __name__ == "__main__":
    main()
