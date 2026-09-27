"""Dynamic-queue coordinator for HPO + training run.

Schedules the benchmark suite across available GPUs using a shared work queue.
Datasets are ordered longest-first to minimise makespan.
Each worker: HPO -> training -> release GPU back to pool.

Usage:
    python run.py [--data_root <path>] [--n_gpus 1] [--dry_run] [--version v1]

Set PYG_DATA_ROOT to override the default data root.
"""
import argparse
import os
import subprocess
import sys
import threading
from queue import Empty, Queue

sys.stdout.reconfigure(line_buffering=True)

PY   = os.environ.get("CREST_PYTHON", sys.executable)
EXP  = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(EXP, "logs")


DATASETS_ORDERED = [
    "roman-empire",
    "pubmed",
    "cs",
    "dblp",
    "photo",
    "citeseer",
    "cora_ml",
    "squirrel-filtered",
]


def run_dataset(ds: str, gpu: int, data_root: str, dry_run: bool, version: str, n_trials: int) -> int:
    hpo_out   = f"{EXP}/results/hpo_{version}_{ds.replace('-', '_')}"
    train_out = f"{EXP}/results/training_{version}"
    log_path  = f"{LOGS}/{version}_{ds.replace('-', '_')}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    hpo_cmd = [
        PY, f"{EXP}/hpo.py",
        "--datasets", ds,
        "--n_trials", str(n_trials),
        "--data_root", data_root,
        "--out_dir", hpo_out,
    ]
    train_cmd = [
        PY, f"{EXP}/train.py",
        "--datasets", ds,
        "--hpo_params", f"{hpo_out}/all_best_params.json",
        "--data_root", data_root,
        "--out_dir", train_out,
    ]

    if dry_run:
        print(f"[DRY] GPU {gpu}: {ds} | HPO -> {hpo_out} | TRAIN -> {train_out}", flush=True)
        return 0

    os.makedirs(hpo_out, exist_ok=True)
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    with open(log_path, "w") as log:
        print(f"[GPU {gpu}] START {ds}", flush=True)

        hpo_proc = subprocess.run(hpo_cmd, env=env, stdout=log, stderr=log)
        if hpo_proc.returncode != 0:
            print(f"[GPU {gpu}] HPO FAILED {ds} (exit {hpo_proc.returncode})", flush=True)
            return hpo_proc.returncode

        train_proc = subprocess.run(train_cmd, env=env, stdout=log, stderr=log)

        rc = train_proc.returncode
        print(f"[GPU {gpu}] DONE {ds} (exit {rc})", flush=True)
        return rc


def worker(gpu: int, queue: Queue, data_root: str, dry_run: bool, version: str, n_trials: int) -> None:
    while True:
        try:
            ds = queue.get_nowait()
        except Empty:
            return
        run_dataset(ds, gpu, data_root, dry_run, version, n_trials)
        queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get("PYG_DATA_ROOT", "data"))
    parser.add_argument("--n_gpus",   type=int, default=1)
    parser.add_argument("--dry_run",  action="store_true")
    parser.add_argument("--version",  default="v1")
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    queue: Queue = Queue()
    for ds in DATASETS_ORDERED:
        queue.put(ds)

    threads = []
    for gpu in range(args.n_gpus):
        t = threading.Thread(target=worker, args=(gpu, queue, args.data_root, args.dry_run, args.version, args.n_trials), daemon=True)
        t.start()
        threads.append(t)

    queue.join()
    print("All datasets complete.", flush=True)


if __name__ == "__main__":
    main()
