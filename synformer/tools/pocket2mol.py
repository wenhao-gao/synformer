import dataclasses
import multiprocessing as mp
import os
import pathlib
import subprocess
from typing import TypeAlias

from omegaconf import OmegaConf
from tqdm.auto import tqdm


@dataclasses.dataclass
class _ModelConfig:
    checkpoint: str = "./ckpt/pretrained_Pocket2Mol.pt"


@dataclasses.dataclass
class _SampleThresholdConfig:
    focal_threshold: float = 0.5
    pos_threshold: float = 0.25
    element_threshold: float = 0.3
    hasatom_threshold: float = 0.6
    bond_threshold: float = 0.4


@dataclasses.dataclass
class _SampleConfig:
    seed: int = 2024
    num_samples: int = 10000
    beam_size: int = 500
    max_steps: int = 30
    threshold: _SampleThresholdConfig = dataclasses.field(default_factory=_SampleThresholdConfig)


@dataclasses.dataclass
class SampleForPdbConfig:
    model: _ModelConfig = dataclasses.field(default_factory=_ModelConfig)
    sample: _SampleConfig = dataclasses.field(default_factory=_SampleConfig)


@dataclasses.dataclass
class Task:
    config: SampleForPdbConfig
    receptor_path: pathlib.Path
    center: tuple[float, float, float]
    out_dir: pathlib.Path


TaskQueueType: TypeAlias = "mp.JoinableQueue[Task | None]"


class Worker(mp.Process):
    def __init__(
        self,
        task_queue: TaskQueueType,
        gpu_id: str,
        python_path: pathlib.Path,
        pocket2mol_path: pathlib.Path,
    ):
        super().__init__()
        self._task_queue = task_queue
        self._gpu_id = gpu_id
        self._python_path = python_path
        self._pocket2mol_path = pocket2mol_path

    def run(self) -> None:
        try:
            while True:
                next_task = self._task_queue.get()
                if next_task is None:
                    self._task_queue.task_done()
                    break
                print(f"{self.name}: Processing {next_task.receptor_path}")
                self.process(next_task)
                self._task_queue.task_done()
                print(f"{self.name}: Finished {next_task.receptor_path}")
        except KeyboardInterrupt:
            print(f"{self.name}: Exiting due to KeyboardInterrupt")
            return

    def process(self, task: Task) -> None:
        conf = OmegaConf.structured(task.config)
        # Save config to a temporary file
        conf_path = task.out_dir / "config.yaml"
        OmegaConf.save(conf, conf_path)

        cmd: list[str] = [
            str(self._python_path.absolute()),
            str(self._pocket2mol_path.absolute() / "sample_for_pdb.py"),
            "--config",
            str(conf_path.absolute()),
            "--pdb_path",
            str(task.receptor_path.absolute()),
            "--center",
            f" {task.center[0]},{task.center[1]},{task.center[2]}",
            "--outdir",
            str(task.out_dir.absolute()),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._gpu_id

        subprocess.run(cmd, env=env, check=True, cwd=str(self._pocket2mol_path))


class WorkerPool:
    def __init__(
        self, num_workers: int, gpu_ids: list[str | int], python_path: pathlib.Path, pocket2mol_path: pathlib.Path
    ):
        self._task_queue: TaskQueueType = mp.JoinableQueue()
        self._workers = [
            Worker(self._task_queue, str(gpu_ids[i % len(gpu_ids)]), python_path, pocket2mol_path)
            for i in range(num_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, task: Task, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in tqdm(self._workers, desc="Terminating"):
            w.terminate()
        self._task_queue.close()
