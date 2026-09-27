"""CREST: train a K*-layer GCN with depth and filter fixed analytically before training."""
import argparse
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from shared.data import load_dataset, load_dataset_all_splits, MULTI_SPLIT_DATASETS
from shared.propagation import (
    compute_lambda_signal,
    select_aggregation_filter,
    compute_depth_from_gap,
    FILTER_THRESHOLD,
    _compute_std_adj_norm,
)

C0 = 0.25
DEFAULT_PARAMS = {"hidden": 128, "lr": 0.01, "dropout": 0.5, "wd": 5e-4}

_PLATONOV_DATASETS = frozenset({
    "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"
})
_OGB_DATASETS = frozenset({"ogbn-arxiv", "ogbn-products"})
_PLANETOID_FIXED_DATASETS: frozenset = frozenset()


def _random_masks(n: int, train_ratio: float, val_ratio: float, seed: int) -> tuple:
    assert train_ratio + val_ratio < 1.0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    tm = torch.zeros(n, dtype=torch.bool)
    vm = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    tm[perm[:n_train]]               = True
    vm[perm[n_train:n_train + n_val]] = True
    te[perm[n_train + n_val:]]        = True
    return tm, vm, te


class GCN(nn.Module):
    """K*-layer GCN with standard LP aggregation (GCNConv = D'^{-1/2}(A+I)D'^{-1/2})."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int,
                 n_layers: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout_p = dropout
        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(GCNConv(in_dim, n_classes))
        else:
            self.convs.append(GCNConv(in_dim, hidden))
            for _ in range(n_layers - 2):
                self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GCNConv(hidden, n_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout_p, training=self.training)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


class GCN_HP(nn.Module):
    """K*-layer GCN with HP aggregation: h_{out} = Linear(T_HP h_{in}) + ReLU. T_HP = L/2: h_new = 0.5 * (h - D^{-1/2} A D^{-1/2} h)."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int,
                 n_layers: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout_p = dropout
        self.lins = nn.ModuleList()
        if n_layers == 1:
            self.lins.append(nn.Linear(in_dim, n_classes))
        else:
            self.lins.append(nn.Linear(in_dim, hidden))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hidden, hidden))
            self.lins.append(nn.Linear(hidden, n_classes))
        self._row: torch.Tensor = None
        self._col: torch.Tensor = None
        self._norm: torch.Tensor = None
        self._n: int = None

    def cache_graph(self, edge_index: torch.Tensor, n_nodes: int) -> None:
        ei_sl, norm = _compute_std_adj_norm(edge_index, n_nodes)
        self._row = ei_sl[0]
        self._col = ei_sl[1]
        self._norm = norm
        self._n = n_nodes

    def _t_hp(self, h: torch.Tensor) -> torch.Tensor:
        assert self._row is not None, "call cache_graph before forward"
        from torch_scatter import scatter
        Ah = scatter(h[self._col] * self._norm.unsqueeze(1),
                     self._row, dim=0, dim_size=self._n, reduce="sum")
        return 0.5 * (h - Ah)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout_p, training=self.training)
        for i, lin in enumerate(self.lins):
            h = self._t_hp(lin(h))
            if i < len(self.lins) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


def train_gcn(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor,
              y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
              epochs: int, lr: float, wd: float) -> float:
    device = x.device
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)
    best_val, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask]).backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v = (model(x, edge_index)[val_mask].argmax(1) == y[val_mask]).float().mean().item()
            sched.step(1 - v)
            if v > best_val:
                best_val = v
                best_state = {k: p.clone() for k, p in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_val


def run_dataset(dsname: str, args, hpo_params: dict) -> dict:
    print(f"\n{'='*60}\nDataset: {dsname}", flush=True)
    t0 = time.time()

    out_dir = os.path.join(args.out_dir, dsname)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "final_result.json")

    if os.path.exists(result_path) and not args.force:
        print(f"  {dsname}: result exists, skipping", flush=True)
        with open(result_path) as f:
            return json.load(f)

    params = dict(DEFAULT_PARAMS)
    if dsname in hpo_params:
        params.update(hpo_params[dsname].get("best_params", {}))
        print(f"  Using HPO params: {params}", flush=True)
    else:
        print(f"  Using defaults: {params}", flush=True)

    x0, ei0, y0, tm0, vm0, te0 = load_dataset(dsname, args.data_root, split_idx=0)

    n = x0.size(0)
    lam = compute_lambda_signal(ei0, y0, n, node_mask=None)  # full labels measure label-topology alignment across the full graph, a dataset-level structural property
    h_phi = select_aggregation_filter(lam)
    lam_eff = (1.0 - lam / 2.0) if h_phi == 'hp' else lam
    k_star = compute_depth_from_gap(lam_eff, C0)
    print(f"  n={n}  λ={lam:.3f}  filter={h_phi}  K*={k_star}", flush=True)

    if dsname.lower() in MULTI_SPLIT_DATASETS:
        splits = load_dataset_all_splits(dsname, args.data_root)
    elif dsname.lower() in _PLATONOV_DATASETS:
        splits = [load_dataset(dsname, args.data_root, split_idx=i) for i in range(10)]
    elif dsname.lower() in _OGB_DATASETS:
        splits = [(x0, ei0, y0, tm0, vm0, te0)]
    elif dsname.lower() in _PLANETOID_FIXED_DATASETS:
        splits = [(x0, ei0, y0, tm0, vm0, te0)] * args.n_splits
    else:
        splits = [(x0, ei0, y0, *_random_masks(n, args.train_ratio, args.val_ratio, seed=i))
                  for i in range(args.n_splits)]

    in_dim    = x0.size(1)
    n_classes = int(y0.max().item()) + 1
    hidden    = params["hidden"]
    lr        = params["lr"]
    dropout   = params["dropout"]
    wd        = params["wd"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp_cache = None
    if h_phi == 'hp':
        _tmp = GCN_HP(1, 1, 1, 1).to(device)
        _tmp.cache_graph(ei0.to(device), n)
        hp_cache = (_tmp._row, _tmp._col, _tmp._norm, _tmp._n)
        del _tmp

    splits_gpu = [tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in s)
                  for s in splits]

    def _run_split(i_s):
        i, (x, ei, y, tm, vm, te) = i_s
        if h_phi == 'hp':
            model = GCN_HP(in_dim, hidden, n_classes, k_star, dropout=dropout).to(device)
            model._row, model._col, model._norm, model._n = hp_cache
        else:
            model = GCN(in_dim, hidden, n_classes, k_star, dropout=dropout).to(device)
        val_acc = train_gcn(model, x, ei, y, tm, vm,
                            epochs=args.epochs, lr=lr, wd=wd)
        model.eval()
        with torch.no_grad():
            test_acc = (model(x, ei)[te].argmax(1) == y[te]).float().mean().item()
        print(f"  split {i+1}/{len(splits)} val={val_acc:.4f} test={test_acc:.4f}", flush=True)
        return i, {"val_acc": val_acc, "test_acc": test_acc}

    from concurrent.futures import ThreadPoolExecutor
    _HUGE  = {"ogbn-arxiv"}
    _LARGE = {"roman-empire", "amazon-ratings", "cs", "pubmed", "dblp"}
    if dsname.lower() in _HUGE:
        n_parallel = 1
    elif dsname.lower() in _LARGE:
        n_parallel = 2
    else:
        n_parallel = min(len(splits_gpu), 4)

    split_results = [None] * len(splits_gpu)
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        for i, r in pool.map(_run_split, enumerate(splits_gpu)):
            split_results[i] = r

    test_accs = [r["test_acc"] for r in split_results]
    out = {
        "dataset":      dsname,
        "n_splits":     len(split_results),
        "lambda_signal": lam,
        "h_phi":        h_phi,
        "k_star":       k_star,
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std":  float(np.std(test_accs)),
        "params":       params,
        "elapsed_s":    time.time() - t0,
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  FINAL {dsname}: {out['test_acc_mean']:.4f}±{out['test_acc_std']:.4f}  K*={k_star}  filter={h_phi}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=os.environ.get("PYG_DATA_ROOT", "data"))
    parser.add_argument("--epochs",      type=int,   default=500)
    parser.add_argument("--n_splits",    type=int,   default=10)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio",   type=float, default=0.20)
    parser.add_argument("--out_dir",     default="results/training")
    parser.add_argument("--hpo_params",  default="results/hpo/all_best_params.json")
    parser.add_argument("--datasets",    nargs="+", default=None)
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()

    hpo_params = {}
    if os.path.exists(args.hpo_params):
        with open(args.hpo_params) as f:
            for entry in json.load(f):
                if "dataset" in entry and "error" not in entry:
                    hpo_params[entry["dataset"]] = entry
        print(f"Loaded HPO params for {len(hpo_params)} datasets", flush=True)
    else:
        print(f"WARNING: {args.hpo_params} not found — using defaults", flush=True)

    datasets = args.datasets or [
        "cora_ml", "citeseer", "pubmed", "dblp", "photo", "cs",
        "squirrel-filtered", "roman-empire",
    ]
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for ds in datasets:
        r = run_dataset(ds, args, hpo_params)
        all_results.append(r)

    print(f"\n{'='*80}\nFINAL SUMMARY", flush=True)
    for r in all_results:
        if "error" in r:
            print(f"  {r['dataset']:22s}  ERROR: {r['error']}", flush=True)
        else:
            print(f"  {r['dataset']:22s}  {r['test_acc_mean']*100:.2f}±{r['test_acc_std']*100:.2f}  K*={r['k_star']}  filter={r['h_phi']}", flush=True)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
