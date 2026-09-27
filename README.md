# CREST: Criterion for Representation Extremum via Spectral Threshold

A single spectral scalar, the task-effective spectral gap, determines the optimal GNN depth, selects between low-pass and high-pass filters, and predicts how a nonlinear activation shortens the depth. All three quantities follow from one sparse matrix–vector product evaluated before training. On eight benchmarks a GCN at the prescribed depth leads on five, and on the strongly heterophilic Roman-Empire the analytical filter switch gains 22.5 points over AD-GNN.

---

## Requirements

Python 3.10+. Tested on NVIDIA RTX 3090 (24 GB), CUDA 12.4.

PyTorch and PyG require CUDA-matched wheels. Install in this order:

```bash
conda create -n crest python=3.10
conda activate crest

# Step 1: PyTorch (adjust cu124 if your CUDA version differs)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Step 2: PyG scatter/sparse/cluster extensions (must match the torch version above)
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Step 3: torch-geometric and remaining packages
pip install torch-geometric==2.7.0
pip install -r requirements.txt
```

`requirements.txt` lists the non-CUDA packages. For CPU-only installs, replace Step 1 with the CPU wheel and skip Step 2.

---

## Datasets

All eight datasets download automatically through PyG on first run. Set the data root via `--data_root` or the environment variable `PYG_DATA_ROOT`.

| Dataset | Source | Splits |
|---|---|---|
| Cora-ML, DBLP | `CitationFull` (PyG) | 10× random 60/20/20 |
| Citeseer, Pubmed | `Planetoid` (PyG) | 10× random 60/20/20 |
| Amazon-Photo | `Amazon` (PyG) | 10× random 60/20/20 |
| Coauthor-CS | `Coauthor` (PyG) | 10× random 60/20/20 |
| Squirrel-filtered | Platonov et al. (2023) `.npz` | 10 fixed splits |
| Roman-Empire | `HeterophilousGraphDataset` (PyG) | 10 fixed splits |

Squirrel-filtered expects `.npz` files under `<data_root>/platonov_filtered/`. Download from [Platonov's repository](https://github.com/yandex-research/heterophilous-graphs) and place them there.

---

## Usage

### Quick start: full benchmark

The coordinator `experiments/run.py` queues all eight datasets across available GPUs:

```bash
python experiments/run.py --data_root data --n_gpus 1
```

This runs HPO (100 Optuna trials per dataset) followed by training (10 splits, 500 epochs each). Results write to `experiments/results/`. Add `--dry_run` to preview the schedule without executing.

### Step by step

**1. Hyperparameter search** — depth and filter are fixed analytically; HPO tunes hidden width, learning rate, dropout, and weight decay:

```bash
python shared/hpo.py --datasets cora_ml citeseer pubmed dblp photo cs squirrel-filtered roman-empire \
    --data_root data --n_trials 100 --out_dir results/hpo
```

**2. Training** — load HPO params, train 10 splits per dataset, report mean ± std:

```bash
python shared/train.py --datasets cora_ml citeseer pubmed dblp photo cs squirrel-filtered roman-empire \
    --data_root data --hpo_params results/hpo/all_best_params.json --out_dir results/training
```

Each dataset prints its prescribed depth, filter, and per-split accuracy as it runs.

---

## Results

Test accuracy (%) on eight datasets, mean ± std over 10 splits.

| Dataset | λ | Filter | K* | CREST | AD-GNN | BNA-GNN | BEC-GNN |
|---|---|---|---|---|---|---|---|
| Cora-ML | 0.255 | LP | 3 | **88.65 ± 1.01** | 87.32 ± 1.03 | 87.78 ± 1.03 | 54.25 ± 9.75 |
| Citeseer | 0.261 | LP | 3 | 76.52 ± 1.53 | **79.14 ± 0.92** | 75.24 ± 1.20 | 75.68 ± 2.08 |
| Pubmed | 0.329 | LP | 2 | **89.04 ± 0.36** | 88.39 ± 0.51 | 88.02 ± 0.57 | — |
| DBLP | 0.277 | LP | 3 | **86.06 ± 0.39** | 84.14 ± 0.68 | 85.43 ± 0.41 | — |
| Photo | 0.281 | LP | 3 | 93.73 ± 0.53 | **94.10 ± 0.55** | 93.30 ± 1.15 | 87.80 ± 3.18 |
| CS | 0.235 | LP | 3 | 92.91 ± 0.53 | **94.76 ± 0.29** | 93.83 ± 0.47 | — |
| Squirrel-filt. | 0.861 | HP | 1 | **41.38 ± 2.23** | 40.19 ± 1.08 | 39.35 ± 1.79 | 31.63 ± 21.14 |
| Roman-Empire | 0.742 | HP | 1 | **69.35 ± 0.47** | 46.86 ± 1.21 | 58.16 ± 1.92 | — |

λ is the task-effective spectral gap of the low-pass operator. K* is the depth the closed-form prescription assigns. Bold marks the highest mean per row.

---

## Repository Structure

```
crest_release/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── experiments/
│   └── run.py              # Dynamic-queue coordinator (HPO + training)
└── shared/
    ├── __init__.py
    ├── csbm.py             # CSBM graph generator (Assumption 1)
    ├── data.py             # PyG dataset loading for all benchmarks
    ├── fisher.py           # Fisher information, linear probe, TrFW curves
    ├── hpo.py              # Optuna HPO (depth/filter fixed, tunes hidden/lr/dropout/wd)
    ├── kstar.py            # K* computation: theory, empirical, IG-peak
    ├── propagation.py      # Spectral gap, filter selection, depth formula
    └── train.py            # GCN/GCN-HP training with prescribed depth
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
