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
    estimator: str = "hajek",
) -> float:
    """Class-averaged centered Rayleigh quotient of L: lambda_signal(I-L).

    When node_mask is None, computes the full-label population oracle (Definition 1).
    When node_mask is provided (training nodes only), computes the partial-label
    estimator (Section 4.5): 'hajek' (self-normalized ratio) or 'ht'
    (Horvitz-Thompson inverse-probability weighted).
    """
    if node_mask is None:
        return compute_lambda_signal_oracle(edge_index, y, n_nodes)
    assert estimator in ("hajek", "ht")
    lam, _ = compute_lambda_signal_partial(edge_index, y, n_nodes, node_mask, estimator=estimator)
    return lam


def compute_lambda_signal_oracle(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
) -> float:
    """Full-label population oracle lambda_signal(I-L) (Definition 2)."""
    ei_sl, norm = compute_adj_norm(edge_index, n_nodes)
    row, col = ei_sl
    classes = torch.unique(y)
    assert classes.numel() >= 2

    total = 0.0
    for c in classes:
        v = (y == c).float()
        v = v - v.mean()
        denom = float((v * v).sum().item())
        assert denom > 0.0
        prop = torch.zeros_like(v)
        prop.index_add_(0, row, norm * v[col])
        total += float((v * (v - prop)).sum().item()) / denom
    lam = total / float(classes.numel())
    return max(0.0, min(2.0, lam))


def compute_lambda_signal_partial(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    train_mask: torch.Tensor,
    estimator: str = "hajek",
) -> Tuple[float, int]:
    """Partial-label estimator of lambda_signal(I-L) from training nodes (Section 4.5).

    Uses full-graph normalized weights A_hat_{uw} (topology only) and labels
    strictly on train_mask. Returns (lambda_signal_est, n_observed_train_edges).
    """
    assert estimator in ("hajek", "ht")
    tr = train_mask.bool()
    n_tr = int(tr.sum().item())
    assert n_tr >= 2
    assert n_nodes >= 2

    ei_sl, norm = compute_adj_norm(edge_index, n_nodes)
    row, col = ei_sl
    is_self = row == col
    self_row, self_norm = row[is_self], norm[is_self]
    off_row, off_col, off_norm = row[~is_self], col[~is_self], norm[~is_self]

    diag_w = torch.zeros(n_nodes, dtype=torch.float, device=edge_index.device)
    diag_w.index_add_(0, self_row, self_norm)

    obs_e = tr[off_row] & tr[off_col]
    n_obs_edges = int(obs_e.sum().item())
    o_r, o_c, o_w = off_row[obs_e], off_col[obs_e], off_norm[obs_e]

    p1 = float(n_tr) / float(n_nodes)
    p2 = float(n_tr * (n_tr - 1)) / float(n_nodes * (n_nodes - 1))

    if estimator == "ht":
        scale_diag = 1.0 / p1
        scale_off = 1.0 / p2
    else:
        w_diag_obs = float(diag_w[tr].sum().item())
        assert w_diag_obs > 0.0
        scale_diag = float(diag_w.sum().item()) / w_diag_obs
        w_off_obs = float(o_w.sum().item())
        assert w_off_obs > 0.0, "no_observed_train_edges"
        scale_off = float(off_norm.sum().item()) / w_off_obs

    y_tr = y[tr]
    classes = torch.unique(y_tr)
    assert classes.numel() >= 2

    total = 0.0
    for c in classes:
        pi_c = float((y_tr == c).float().mean().item())
        yc = (y == c).float() - pi_c
        yc_tr = yc[tr]
        denom = float((yc_tr * yc_tr).sum().item()) / p1
        assert denom > 0.0
        num_diag = float((diag_w[tr] * yc_tr * yc_tr).sum().item()) * scale_diag
        num_off = float((o_w * yc[o_r] * yc[o_c]).sum().item()) * scale_off
        total += 1.0 - (num_diag + num_off) / denom

    lam = total / float(classes.numel())
    return max(0.0, min(2.0, lam)), n_obs_edges


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
