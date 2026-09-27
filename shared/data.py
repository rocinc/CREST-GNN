"""PyG dataset loading utilities for CREST experiments."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data


_DATASET_REGISTRY = {
    "cora": ("Planetoid", {"name": "Cora"}),
    "cora_ml": ("CitationFull", {"name": "cora_ml"}),
    "citeseer": ("Planetoid", {"name": "Citeseer"}),
    "pubmed": ("Planetoid", {"name": "Pubmed"}),
    "photo": ("Amazon", {"name": "Photo"}),
    "computers": ("Amazon", {"name": "Computers"}),
    "cs": ("Coauthor", {"name": "CS"}),
    "physics": ("Coauthor", {"name": "Physics"}),
    "squirrel": ("WikipediaNetwork", {"name": "squirrel", "geom_gcn_preprocess": True}),
    "chameleon": ("WikipediaNetwork", {"name": "chameleon", "geom_gcn_preprocess": True}),
    "squirrel-filtered": ("_platonov_npz", {"stem": "squirrel_filtered"}),
    "chameleon-filtered": ("_platonov_npz", {"stem": "chameleon_filtered"}),
    "texas": ("WebKB", {"name": "Texas"}),
    "cornell": ("WebKB", {"name": "Cornell"}),
    "wisconsin": ("WebKB", {"name": "Wisconsin"}),
    "film": ("Actor", {"_subdir": "actor"}),
    "roman-empire": ("HeterophilousGraphDataset", {"name": "Roman-empire"}),
    "amazon-ratings": ("HeterophilousGraphDataset", {"name": "Amazon-ratings"}),
    "minesweeper": ("HeterophilousGraphDataset", {"name": "Minesweeper"}),
    "tolokers": ("HeterophilousGraphDataset", {"name": "Tolokers"}),
    "questions": ("HeterophilousGraphDataset", {"name": "Questions"}),
    "genius": ("LINKXDataset", {"name": "genius"}),
    "dblp": ("CitationFull", {"name": "DBLP"}),
    "flickr": ("Flickr", {"_subdir": "Flickr"}),
    "reddit": ("Reddit", {"_subdir": "Reddit"}),
    "ogbn-arxiv": ("_ogbn", {"name": "ogbn-arxiv"}),
    "ogbn-products": ("_ogbn", {"name": "ogbn-products"}),
}


def load_dataset(
    name: str,
    data_root: str,
    split_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a named dataset; return (x, edge_index, y, train_mask, val_mask, test_mask)."""
    name_lower = name.lower()
    assert name_lower in _DATASET_REGISTRY, (
        f"Unknown dataset '{name}'. Registered: {sorted(_DATASET_REGISTRY)}"
    )

    cls_name, kwargs = _DATASET_REGISTRY[name_lower]
    root = Path(data_root)

    if cls_name == "_platonov_npz":
        return _load_platonov_npz(root, kwargs["stem"], split_idx)

    if cls_name == "_ogbn":
        return _load_ogbn(root, kwargs["name"])

    data = _load_pyg(cls_name, str(root), kwargs, split_idx)

    x = data.x
    edge_index = data.edge_index
    y = data.y

    if y.dim() == 2:
        y = y.squeeze(1)

    train_mask, val_mask, test_mask = _extract_masks(data, split_idx, x.size(0))

    assert x.dtype == torch.float or x.dtype == torch.float32, (
        f"Expected float32 features, got {x.dtype}. Convert before calling."
    )
    assert y.dtype == torch.long, f"Expected int64 labels, got {y.dtype}"
    assert edge_index.size(0) == 2
    assert train_mask.sum() >= 2, "Training set must have at least 2 nodes"
    assert val_mask.sum() >= 1, "Validation set must have at least 1 node"

    return x, edge_index, y, train_mask, val_mask, test_mask


def _load_platonov_npz(
    root: Path,
    stem: str,
    split_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Platonov 2023 squirrel-filtered / chameleon-filtered from .npz."""
    import numpy as np

    path = root / "platonov_filtered" / f"{stem}.npz"
    assert path.exists(), f"Platonov filtered file not found: {path}"

    d = np.load(str(path))
    x = torch.from_numpy(d["node_features"]).float()
    y = torch.from_numpy(d["node_labels"]).long()

    edges = torch.from_numpy(d["edges"]).long().t().contiguous()
    edge_index = torch.cat([edges, edges.flip(0)], dim=1)

    n_splits = d["train_masks"].shape[0]
    idx = split_idx % n_splits
    train_mask = torch.from_numpy(d["train_masks"][idx])
    val_mask = torch.from_numpy(d["val_masks"][idx])
    test_mask = torch.from_numpy(d["test_masks"][idx])

    assert train_mask.sum() >= 2
    assert val_mask.sum() >= 1
    return x, edge_index, y, train_mask, val_mask, test_mask


def _load_ogbn(
    root: Path,
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ogbn dataset via ogb; use canonical public split."""
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=name, root=str(root))
    data = dataset[0]
    split_idx_ogb = dataset.get_idx_split()

    from torch_geometric.utils import to_undirected

    x = data.x.float()
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    y = data.y.squeeze(1).long()

    n = x.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx_ogb["train"]] = True
    val_mask[split_idx_ogb["valid"]] = True
    test_mask[split_idx_ogb["test"]] = True

    assert train_mask.sum() >= 2
    assert val_mask.sum() >= 1
    return x, edge_index, y, train_mask, val_mask, test_mask


def _load_pyg(cls_name: str, root: str, kwargs: dict, split_idx: int) -> Data:
    """Instantiate a PyG dataset class and return the first Data object."""
    import os
    import torch_geometric.datasets as pyg_datasets

    subdir = kwargs.get("_subdir")
    clean_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    effective_root = os.path.join(root, subdir) if subdir else root

    assert hasattr(pyg_datasets, cls_name), (
        f"PyG class '{cls_name}' not found in torch_geometric.datasets. "
        f"Verify the correct class name (e.g. CitationFull not Coauthor for DBLP, "
        f"LINKXDataset not HeterophilousGraphDataset for Genius)."
    )
    cls = getattr(pyg_datasets, cls_name)
    dataset = cls(root=effective_root, **clean_kwargs)

    assert hasattr(dataset, "__getitem__"), (
        f"Dataset '{cls_name}' does not support indexing."
    )
    data = dataset[0]
    assert data.x is not None, (
        f"Dataset '{cls_name}' loaded but x is None. "
        f"Check that processed files exist at '{effective_root}' "
        f"and the root path is correct (not the central data_root)."
    )
    return data


def _extract_masks(
    data: Data,
    split_idx: int,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract train/val/test masks from a PyG Data object."""
    if hasattr(data, "train_mask") and data.train_mask.dim() == 1:
        return data.train_mask.bool(), data.val_mask.bool(), data.test_mask.bool()

    if hasattr(data, "train_mask") and data.train_mask.dim() == 2:
        n_splits = data.train_mask.size(1)
        idx = split_idx % n_splits
        return (
            data.train_mask[:, idx].bool(),
            data.val_mask[:, idx].bool(),
            data.test_mask[:, idx].bool(),
        )

    perm = torch.randperm(n_nodes)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    return train_mask, val_mask, test_mask


def load_dataset_all_splits(
    name: str,
    data_root: str,
) -> list:
    """Load all available splits for a dataset. For WebKB / WikipediaNetwork geom_gcn / Platonov filtered: returns all 10 splits. For Planetoid / Amazon / Platonov Heterophilous: returns single-element list. Replicates the 10-fold average protocol of AD-GNN (AAAI 2026) and BE-Curvature (KDD 2025) for multi-split datasets."""
    name_lower = name.lower()
    assert name_lower in _DATASET_REGISTRY
    cls_name, kwargs = _DATASET_REGISTRY[name_lower]
    root = Path(data_root)

    if cls_name == "_platonov_npz":
        import numpy as np
        path = root / "platonov_filtered" / f"{kwargs['stem']}.npz"
        d = np.load(str(path))
        x = torch.from_numpy(d["node_features"]).float()
        y = torch.from_numpy(d["node_labels"]).long()
        edges = torch.from_numpy(d["edges"]).long().t().contiguous()
        edge_index = torch.cat([edges, edges.flip(0)], dim=1)
        n_splits = d["train_masks"].shape[0]
        return [
            (x, edge_index, y,
             torch.from_numpy(d["train_masks"][i]),
             torch.from_numpy(d["val_masks"][i]),
             torch.from_numpy(d["test_masks"][i]))
            for i in range(n_splits)
        ]

    data = _load_pyg(cls_name, str(root), kwargs, split_idx=0)
    x = data.x
    edge_index = data.edge_index
    y = data.y.squeeze(1) if data.y.dim() == 2 else data.y
    assert x.dtype in (torch.float, torch.float32)
    assert y.dtype == torch.long

    if hasattr(data, "train_mask") and data.train_mask.dim() == 2:
        n_splits = data.train_mask.size(1)
        return [
            (x, edge_index, y,
             data.train_mask[:, i].bool(),
             data.val_mask[:, i].bool(),
             data.test_mask[:, i].bool())
            for i in range(n_splits)
        ]

    train_mask, val_mask, test_mask = _extract_masks(data, 0, x.size(0))
    return [(x, edge_index, y, train_mask, val_mask, test_mask)]


MULTI_SPLIT_DATASETS = {
    "texas", "cornell", "wisconsin",
    "squirrel-filtered", "chameleon-filtered",
}


def get_dataset_info(name: str, data_root: str) -> dict:
    """Return basic stats for a dataset without full loading. Useful for logging dataset properties at experiment start."""
    x, edge_index, y, train_mask, val_mask, test_mask = load_dataset(name, data_root)
    return {
        "name": name,
        "n_nodes": int(x.size(0)),
        "n_features": int(x.size(1)),
        "n_edges": int(edge_index.size(1)),
        "n_classes": int(y.max().item() + 1),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
