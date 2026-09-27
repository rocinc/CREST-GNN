import numpy as np
import torch
from typing import Tuple


def generate_csbm(
    n: int,
    d: int,
    gamma: float,
    sigma0: float,
    avg_degree: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a balanced CSBM instance (Assumption 1). Features: x_u = (2*y_u - 1) * mu + xi_u, xi_u ~ N(0, sigma0^2 I). Edge probabilities: p_in = (1+gamma)*p_base, p_out = (1-gamma)*p_base, p_base = avg_degree / (n - 1). Args: n: number of nodes (even, for balanced classes) d: feature dimension gamma: homophily in (0, 1] sigma0: feature noise std avg_degree: expected average degree seed: random seed Returns: edge_index: [2, E] COO (undirected, no self-loops) x: [N, d] node features y: [N] labels in {0, 1}"""
    assert n % 2 == 0
    assert 0.0 < gamma <= 1.0
    assert sigma0 > 0.0
    assert avg_degree > 0.0

    rng = np.random.default_rng(seed)
    y_np = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)

    p_base = avg_degree / (n - 1)
    p_in = float(np.clip((1.0 + gamma) * p_base, 0.0, 1.0))
    p_out = float(np.clip((1.0 - gamma) * p_base, 0.0, 1.0))

    edges_src, edges_dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            prob = p_in if y_np[i] == y_np[j] else p_out
            if rng.random() < prob:
                edges_src += [i, j]
                edges_dst += [j, i]

    if len(edges_src) == 0:
        for i in range(min(n - 1, 5)):
            edges_src += [i, i + 1]
            edges_dst += [i + 1, i]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    mu = np.zeros(d, dtype=np.float32)
    mu[0] = 1.0 / np.sqrt(d)
    labels_signed = (2 * y_np - 1).astype(np.float32)
    signal = np.outer(labels_signed, mu)
    noise = rng.normal(0.0, sigma0, size=(n, d)).astype(np.float32)
    x = torch.from_numpy(signal + noise)
    y = torch.from_numpy(y_np)

    return edge_index, x, y


def csbm_splits(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random train/val/test masks."""
    assert 0.0 < train_ratio < 1.0
    assert 0.0 < val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask
