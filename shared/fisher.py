import numpy as np
import torch
from typing import Optional, Tuple, List


def train_linear_probe(
    h: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    max_iter: int = -1,
    C: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a softmax logistic probe on propagated representations (Appendix, Computing the Criterion). Args: h: [N, d] node representations y: [N] integer class labels train_mask: [N] boolean training mask max_iter: max L-BFGS iterations; -1 = adaptive C: inverse regularization strength Returns: W_head: [n_classes, d] weight matrix b_head: [n_classes] bias"""
    device = h.device
    h_tr = h[train_mask].detach().float()
    y_tr = y[train_mask].long()
    assert h_tr.size(0) >= 2
    n_tr, d = h_tr.shape
    n_classes = int(y_tr.max().item()) + 1
    if max_iter < 0:
        max_iter = 150 if n_tr > 5000 else 300
    W = torch.zeros(n_classes, d, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    reg = 1.0 / (C * n_tr)
    optimizer = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-5)

    def closure():
        optimizer.zero_grad()
        logits = h_tr @ W.t() + b
        loss = torch.nn.functional.cross_entropy(logits, y_tr) + 0.5 * reg * (W * W).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    W_head = W.detach().cpu().numpy().astype(np.float32)
    b_head = b.detach().cpu().numpy().astype(np.float32)
    if n_classes == 2:
        W_head = np.vstack([-W_head[1:2] / 2.0, W_head[1:2] / 2.0])
        b_head = np.concatenate([-b_head[1:2] / 2.0, b_head[1:2] / 2.0])
    return W_head, b_head


def compute_softmax_fisher_trace(
    h: torch.Tensor,
    W_head: np.ndarray,
    b_head: np.ndarray,
) -> torch.Tensor:
    """Per-node posterior Fisher trace Tr(F_v^(k)) (eq. fisher_repspace, Section 3). Tr(F_v) = sum_c p_vc * ||w_c||^2 - ||W^T p_v||^2 Args: h: [N, d] node representations W_head: [C, d] probe weights b_head: [C] probe bias Returns: trf: [N] non-negative per-node Fisher traces"""
    W = torch.from_numpy(W_head).to(h.device)
    b = torch.from_numpy(b_head).to(h.device)
    logits = h @ W.t() + b
    p = torch.softmax(logits, dim=1)
    w_sq = (W * W).sum(dim=1)
    term1 = (p * w_sq.unsqueeze(0)).sum(dim=1)
    Wtp = p @ W
    term2 = (Wtp * Wtp).sum(dim=1)
    return (term1 - term2).clamp(min=0.0)


def compute_homophily_per_node(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Per-node 1-hop label homophily gamma_v = |same-class neighbors| / |neighbors|."""
    from torch_scatter import scatter
    device = y.device
    row, col = edge_index[0].to(device), edge_index[1].to(device)
    same_class = (y[row] == y[col]).float()
    same_count = torch.zeros(n_nodes, device=device).scatter_add_(0, row, same_class)
    total_count = torch.zeros(n_nodes, device=device).scatter_add_(0, row, torch.ones(row.size(0), device=device))
    return torch.where(total_count > 0, same_count / total_count, torch.zeros(n_nodes, device=device))


def compute_self_loop_weight(
    edge_index: torch.Tensor,
    n_nodes: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Self-loop weight a_vv = 1/(deg_v + 1) for T_LP = D'^{-1/2}(A+I)D'^{-1/2}."""
    from torch_geometric.utils import degree as pyg_degree
    row = edge_index[0]
    deg = pyg_degree(row, num_nodes=n_nodes, dtype=torch.float)
    a_vv = 1.0 / (deg + 1.0)
    if device is not None and a_vv.device != device:
        a_vv = a_vv.to(device)
    return a_vv


def compute_self_loop_weight_HP(
    n_nodes: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Self-loop weight for T_HP = L/2: constant 1/2 for all nodes."""
    return torch.full((n_nodes,), 0.5, dtype=torch.float, device=device)


def compute_trfw_hp(
    h: torch.Tensor,
    W_head: np.ndarray,
    b_head: np.ndarray,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Per-node Tr(F_v W_v) for T_HP = L/2: (1 - gamma_v) * (1/2) * Tr(F_v)."""
    device = h.device
    trf = compute_softmax_fisher_trace(h, W_head, b_head)
    gamma_v = compute_homophily_per_node(edge_index, y, n_nodes)
    b_vv = compute_self_loop_weight_HP(n_nodes, device=device)
    return ((1.0 - gamma_v) * b_vv * trf).clamp(min=0.0)


def compute_trfw(
    h: torch.Tensor,
    W_head: np.ndarray,
    b_head: np.ndarray,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    n_nodes: int,
    h_phi: str = 'lp',
) -> torch.Tensor:
    """Per-node Tr(F_v W_v) for the selected filter (Appendix, Computing the Criterion)."""
    assert h_phi in ('lp', 'hp'), f"h_phi must be 'lp' or 'hp', got {h_phi!r}"
    if h_phi == 'hp':
        return compute_trfw_hp(h, W_head, b_head, edge_index, y, n_nodes)
    device = h.device
    trf = compute_softmax_fisher_trace(h, W_head, b_head)
    gamma_v = compute_homophily_per_node(edge_index, y, n_nodes)
    a_vv = compute_self_loop_weight(edge_index, n_nodes, device=device)
    return (gamma_v * a_vv * trf).clamp(min=0.0)


def compute_trfw_curve(
    h_list: list,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    n_nodes: int,
    probe_max_iter: int = -1,
    h_phi: str = 'lp',
) -> Tuple[list, list]:
    """Tr(F_v W_v) curve across hops 0..K_max; return (mean_curve, per_node_curve_list)."""
    trfw_mean_curve = []
    trfw_all_curve = []
    for h in h_list:
        W_head, b_head = train_linear_probe(h, y, train_mask, max_iter=probe_max_iter)
        trfw_k = compute_trfw(h, W_head, b_head, edge_index, y, n_nodes, h_phi=h_phi)
        trfw_mean_curve.append(float(trfw_k.mean()))
        trfw_all_curve.append(trfw_k)
    return trfw_mean_curve, trfw_all_curve
