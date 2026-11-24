# CREST: Criterion for Representation Extremum via Spectral Threshold

A single spectral scalar, the task-effective spectral gap $\lambda_{\mathrm{signal}}$, fixes GNN propagation depth, selects between low-pass and high-pass filters at the universal crossover $\lambda_{\mathrm{signal}} = 2/3$, and quantifies ReLU depth contraction before training. Estimated in linear time from training labels alone via a self-normalized Hájek estimator, CREST replaces multi-depth search with a pre-training spectral gate.

---

## Requirements

Python 3.10+. Tested on NVIDIA Tesla T4 (16 GB) and RTX 3090 (24 GB), CUDA 12.4.

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

Primary and auxiliary benchmarks download automatically through PyG on first run (except Squirrel-filtered). Set the data root via `--data_root` or the environment variable `PYG_DATA_ROOT`.

| Dataset | Source | Splits |
|---|---|---|
| Cora-ML, DBLP | `CitationFull` (PyG) | 10× deterministic 60/20/20 |
| Citeseer, Pubmed | `Planetoid` (PyG) | 10× deterministic 60/20/20 |
| Amazon-Photo | `Amazon` (PyG) | 10× deterministic 60/20/20 |
| Coauthor-CS | `Coauthor` (PyG) | 10× deterministic 60/20/20 |
| Squirrel-filtered | Platonov et al. (2023) `.npz` | 10 fixed splits |
| Roman-Empire | `HeterophilousGraphDataset` (PyG) | 10 fixed splits |
| Film, Wisconsin | `Actor` / `WebKB` (PyG) | 10 fixed Geom-GCN splits |

Squirrel-filtered expects `squirrel_filtered.npz` under `<data_root>/platonov_filtered/`. Download from [Platonov's repository](https://github.com/yandex-research/heterophilous-graphs) and place it there.

---

## Usage

This repository provides the core CREST pre-training spectral estimator, depth/filter prescription, Optuna HPO, and GCN / GCN-HP training pipeline.

### Quick start: core CREST benchmark

The coordinator `experiments/run.py` schedules all eight primary datasets across available GPUs:

```bash
python experiments/run.py --data_root data --n_gpus 1
```

This runs Optuna HPO (30 trials, 2 validation splits, 200 epochs) followed by 10-split evaluation (500 epochs each), computing the training-only Hájek spectral gap inside each split before any parameter is optimized. Results write to `experiments/results/`. Add `--dry_run` to preview the schedule without executing.

### Step by step

**1. Hyperparameter search** (depth and filter are fixed analytically per split; HPO tunes hidden width, learning rate, dropout, and weight decay):

```bash
python shared/hpo.py --datasets cora_ml citeseer pubmed dblp photo cs squirrel-filtered roman-empire \
    --data_root data --n_trials 30 --out_dir results/hpo
```

**2. Training** (load HPO params, evaluate 10 splits per dataset, report mean ± std):

```bash
python shared/train.py --datasets cora_ml citeseer pubmed dblp photo cs squirrel-filtered roman-empire \
    --data_root data --hpo_params results/hpo/all_best_params.json --out_dir results/training
```

Each dataset prints its per-split training-only spectral gap, filter polarity, retained depth `K*`, and accuracy.

---

## Results

Paper Table 2 benchmark test accuracy (%, mean ± std over 10 splits) under the matched 30-trial Optuna protocol. `CREST (Hájek)` is reproduced directly by this repository; competing baselines follow their respective official implementations under the matched protocol described in the manuscript. Bold marks the best depth-selection method per row (CREST versus AD-GNN and BEC-GNN).

| Dataset | $\hat{\lambda}_{\mathrm{sig}}$ | $(h_\phi, K^*)$ | CREST (Hájek) | AD-GNN | BEC-GNN | GCN-Grid | ACM-GCN |
|---|---|---|---|---|---|---|---|
| Cora-ML | 0.256 | (LP, 3) | **88.98 ± 0.98** | 87.70 ± 1.00 | 87.76 ± 1.02 | 88.31 ± 1.08 | 88.90 ± 0.84 |
| Citeseer | 0.261 | (LP, 3) | 76.10 ± 1.03 | **76.46 ± 1.46** | 76.05 ± 0.97 | 77.10 ± 1.17 | 76.23 ± 1.31 |
| Pubmed | 0.328 | (LP, 2) | **89.01 ± 0.37** | 88.55 ± 0.33 | OOM | 88.94 ± 0.53 | 90.29 ± 0.28 |
| DBLP | 0.279 | (LP, 3) | **86.00 ± 0.34** | 84.35 ± 0.40 | OOM | 86.02 ± 0.35 | 85.73 ± 0.33 |
| Photo | 0.283 | (LP, 3) | 93.61 ± 0.68 | **94.22 ± 0.55** | 93.33 ± 0.50 | 94.41 ± 0.42 | 95.42 ± 0.26 |
| CS | 0.235 | (LP, 3) | 92.84 ± 0.55 | **94.18 ± 0.27** | OOM | 94.16 ± 0.29 | 95.22 ± 0.30 |
| Squirrel-filt. | 0.861 | (HP, 1) | 40.77 ± 1.91 | **41.35 ± 2.82** | 40.16 ± 2.10 | 41.92 ± 1.80 | 35.66 ± 0.84 |
| Roman-Empire | 0.742 | (HP, 1) | **67.69 ± 0.33** | 53.52 ± 0.35 | OOM | 53.51 ± 0.44 | 84.56 ± 0.56 |
| Film | 0.752 | (HP, 1) | **33.68 ± 1.11** | 29.43 ± 1.25 | — | — | 35.51 ± 1.10 |
| Wisconsin | 0.779 | (HP, 1) | **79.22 ± 3.84** | 49.41 ± 5.93 | — | — | 85.49 ± 3.84 |

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
    ├── data.py             # PyG dataset loading utilities
    ├── fisher.py           # Posterior Fisher information and linear probe utilities
    ├── hpo.py              # Optuna HPO (depth/filter fixed per split, tunes hidden/lr/dropout/wd)
    ├── kstar.py            # Analytical and empirical depth computation utilities
    ├── propagation.py      # Spectral gap, training-only Hajek/HT estimators, filter & depth rules
    └── train.py            # GCN / GCN-HP training with per-split CREST configuration
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
