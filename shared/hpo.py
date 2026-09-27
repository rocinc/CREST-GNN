"""CREST: Optuna HPO for {hidden, lr, dropout, wd}; depth and filter are fixed analytically."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from shared.data import load_dataset, MULTI_SPLIT_DATASETS
from shared.propagation import (compute_lambda_signal, select_aggregation_filter,
                                 compute_depth_from_gap)
from train import (
    GCN, GCN_HP, train_gcn, C0,
    _PLATONOV_DATASETS, _OGB_DATASETS, _PLANETOID_FIXED_DATASETS, _random_masks,
)

EPOCHS_HPO   = 300
N_SPLITS_HPO = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _suggest_params(trial: optuna.Trial) -> dict:
    return {
        "hidden":  trial.suggest_categorical("hidden",  [64, 128, 256, 512]),
        "lr":      trial.suggest_float("lr",      1e-4, 5e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0,  0.7),
        "wd":      trial.suggest_float("wd",      1e-6, 1e-2, log=True),
    }


def _objective(trial: optuna.Trial, dsname: str, data_root: str,
               train_ratio: float, val_ratio: float) -> float:
    params = _suggest_params(trial)

    x0, ei0, y0, tm0, vm0, te0 = load_dataset(dsname, data_root, split_idx=0)
    n, in_dim = x0.size(0), x0.size(1)
    n_classes = int(y0.max().item()) + 1

    lam   = compute_lambda_signal(ei0, y0, n, node_mask=None)
    h_phi = select_aggregation_filter(lam)
    lam_eff = (1.0 - lam / 2.0) if h_phi == 'hp' else lam
    k_star = compute_depth_from_gap(lam_eff, C0)

    if dsname.lower() in MULTI_SPLIT_DATASETS:
        from shared.data import load_dataset_all_splits
        splits = load_dataset_all_splits(dsname, data_root)[:N_SPLITS_HPO]
    elif dsname.lower() in _PLATONOV_DATASETS:
        splits = [load_dataset(dsname, data_root, split_idx=i) for i in range(N_SPLITS_HPO)]
    elif dsname.lower() in _OGB_DATASETS:
        splits = [(x0, ei0, y0, tm0, vm0, te0)]
    elif dsname.lower() in _PLANETOID_FIXED_DATASETS:
        splits = [(x0, ei0, y0, tm0, vm0, te0)] * N_SPLITS_HPO
    else:
        splits = [(x0, ei0, y0, *_random_masks(n, train_ratio, val_ratio, seed=i))
                  for i in range(N_SPLITS_HPO)]

    hp_cache = None
    if h_phi == 'hp':
        _tmp = GCN_HP(1, 1, 1, 1).to(DEVICE)
        _tmp.cache_graph(ei0.to(DEVICE), n)
        hp_cache = (_tmp._row, _tmp._col, _tmp._norm, _tmp._n)
        del _tmp

    val_accs = []
    for x, ei, y, tm, vm, te in splits:
        x  = x.to(DEVICE);  ei = ei.to(DEVICE)
        y  = y.to(DEVICE);  tm = tm.to(DEVICE);  vm = vm.to(DEVICE)

        if h_phi == 'hp':
            model = GCN_HP(in_dim, params["hidden"], n_classes, k_star,
                           dropout=params["dropout"]).to(DEVICE)
            model._row, model._col, model._norm, model._n = hp_cache
        else:
            model = GCN(in_dim, params["hidden"], n_classes, k_star,
                        dropout=params["dropout"]).to(DEVICE)

        val_acc = train_gcn(model, x, ei, y, tm, vm,
                            epochs=EPOCHS_HPO, lr=params["lr"], wd=params["wd"])
        val_accs.append(val_acc)

    return float(np.mean(val_accs))


def _hpo_n_jobs(dsname: str) -> int:
    _HUGE  = {"ogbn-arxiv"}
    _LARGE = {"roman-empire", "amazon-ratings", "cs", "pubmed", "dblp"}
    if dsname.lower() in _HUGE:
        return 1
    if dsname.lower() in _LARGE:
        return 2
    return 4


def run_hpo_dataset(dsname: str, args) -> dict:
    out_dir = os.path.join(args.out_dir, dsname)
    os.makedirs(out_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"crest_{dsname}",
    )

    def objective(trial):
        return _objective(trial, dsname, args.data_root, args.train_ratio, args.val_ratio)

    study.optimize(objective, n_trials=args.n_trials,
                   n_jobs=_hpo_n_jobs(dsname), show_progress_bar=False)

    best = study.best_trial
    entry = {
        "dataset":      dsname,
        "best_val_acc": best.value,
        "best_params":  best.params,
        "n_trials":     args.n_trials,
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(entry, f, indent=2)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=[
        "cora_ml", "citeseer", "pubmed", "dblp", "photo", "cs",
        "squirrel-filtered", "roman-empire",
    ])
    parser.add_argument("--data_root",   default=os.environ.get("PYG_DATA_ROOT", "data"))
    parser.add_argument("--n_trials",    type=int,   default=100)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio",   type=float, default=0.20)
    parser.add_argument("--out_dir",     default="results/hpo")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_entries = []

    for ds in args.datasets:
        print(f"\n{'='*60}\nDataset: {ds}", flush=True)
        entry = run_hpo_dataset(ds, args)
        all_entries.append(entry)
        p = entry["best_params"]
        print(f"  best_val={entry['best_val_acc']:.4f}  "
              f"hidden={p['hidden']}  lr={p['lr']:.5f}  "
              f"dropout={p['dropout']:.3f}  wd={p['wd']:.6f}", flush=True)

    out_path = os.path.join(args.out_dir, "all_best_params.json")
    with open(out_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
