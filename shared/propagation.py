import math

import torch
from typing import List, Optional, Tuple

from torch_geometric.utils import add_self_loops, degree as pyg_degree

FILTER_THRESHOLD: float = 2.0 / 3.0


def compute_adj_norm(edge_index: torch.Tensor, n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_index_with_self_loops, per-edge weights) for D'^{-1/2}(A+I)D'^{-1/2}."""
    ei_sl, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    row, col = ei_sl
    deg = pyg_degree(row, num_nodes=n_nodes, dtype=torch.float)
    assert (deg > 0).all(), "Isolated nodes detected"
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return ei_sl, norm


def propagate_once(h: torch.Tensor, ei_sl: torch.Tensor, norm: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """One LP aggregation step: h_new = D'^{-1/2}(A+I)D'^{-1/2} h."""
    from torch_scatter import scatter
    row, col = ei_sl
    return scatter(h[col] * norm.unsqueeze(1), row, dim=0, dim_size=n_nodes, reduce="sum")


def compute_propagation(x: torch.Tensor, edge_index: torch.Tensor, k_max: int) -> List[torch.Tensor]:
    """Return [h^(0), ..., h^(k_max)] under T_LP = D'^{-1/2}(A+I)D'^{-1/2}."""
    assert k_max >= 0
    n_nodes = x.size(0)
    ei_sl, norm = compute_adj_norm(edge_index, n_nodes)
    h_list: List[torch.Tensor] = [x]
    h = x
    for _ in range(k_max):
        h = propagate_once(h, ei_sl, norm, n_nodes)
        h_list.append(h)
    assert len(h_list) == k_max + 1
    return h_list


def compute_lambda_signal(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    node_mask: Optional[torch.Tensor] = None,
) -> float:
    """Class-averaged centered Rayleigh quotient of L: lambda_signal(I-L) (Definition 2)."""
    ei_sl, norm = compute_adj_norm(edge_index, n_nodes)
    row, col = ei_sl

    def rayleigh_quotient(indicator: torch.Tensor) -> float:
        v = indicator.float()
        if node_mask is not None:
            v = v * node_mask.float()
        v = v - v.mean()
        denom = float((v * v).sum())
        if denom == 0.0:
            return 0.0
        prop = torch.zeros_like(v)
        prop.index_add_(0, row, norm * v[col])
        return float((v * (v - prop)).sum()) / denom

    classes = torch.unique(y)
    lam = float(sum(rayleigh_quotient(y == c) for c in classes) / len(classes))
    return max(0.0, min(2.0, lam))


def _propagate_high_pass_once(
    h: torch.Tensor, row: torch.Tensor, col: torch.Tensor, norm: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """One HP step: h_new = (1/2)(h - D'^{-1/2}(A+I)D'^{-1/2} h)."""
    from torch_scatter import scatter
    Ah = scatter(h[col] * norm.unsqueeze(1), row, dim=0, dim_size=n_nodes, reduce="sum")
    return 0.5 * (h - Ah)


def compute_propagation_high_pass(x: torch.Tensor, edge_index: torch.Tensor, k_max: int) -> List[torch.Tensor]:
    """Return [h^(0), ..., h^(k_max)] under T_HP = L/2."""
    assert k_max >= 0
    n_nodes = x.size(0)
    ei_sl, norm = compute_adj_norm(edge_index, n_nodes)
    row, col = ei_sl
    h_list: List[torch.Tensor] = [x]
    h = x
    for _ in range(k_max):
        h = _propagate_high_pass_once(h, row, col, norm, n_nodes)
        h_list.append(h)
    assert len(h_list) == k_max + 1
    return h_list


def compute_lambda_signal_T(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    h_phi: str,
    node_mask: Optional[torch.Tensor] = None,
) -> float:
    """lambda_signal(T): LP = Rayleigh quotient of L; HP = 1 - lp/2."""
    assert h_phi in ('lp', 'hp'), f"h_phi must be 'lp' or 'hp', got {h_phi!r}"
    lam_lp = compute_lambda_signal(edge_index, y, n_nodes, node_mask=node_mask)
    if h_phi == 'lp':
        return lam_lp
    lam_hp = 1.0 - 0.5 * lam_lp
    assert 0.0 <= lam_hp <= 1.0
    return lam_hp


def compute_depth_from_gap(lambda_signal: float, c0: float = 0.25) -> int:
    """K* = ceil(log(1/c0) / (2 * nu)), nu = -log(1 - lambda_signal(T))."""
    assert 0.0 < lambda_signal < 1.0, f"lambda_signal must be in (0, 1), got {lambda_signal}"
    assert 0.0 < c0 < 1.0
    nu = -math.log1p(-lambda_signal)
    return int(math.ceil(math.log(1.0 / c0) / (2.0 * nu)))


def select_aggregation_filter(lambda_signal_lp: float) -> str:
    """Return 'hp' when lambda_signal(I-L) > 2/3, else 'lp' (Corollary 3)."""
    return 'hp' if lambda_signal_lp > FILTER_THRESHOLD else 'lp'


def _compute_std_adj_norm(edge_index: torch.Tensor, n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alias for compute_adj_norm."""
    return compute_adj_norm(edge_index, n_nodes)
