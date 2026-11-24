import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from .fisher import train_linear_probe
from .propagation import compute_lambda_signal, compute_lambda_signal_T, select_aggregation_filter


def compute_kstar_emp(
    h_list: List[torch.Tensor],
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    probe_max_iter: int = -1,
) -> Tuple[int, List[float]]:
    from sklearn.metrics import accuracy_score

    assert val_mask.sum() >= 1

    val_acc_curve = []
    y_val_np = y[val_mask].numpy()

    for h in h_list:
        W_head, b_head = train_linear_probe(h, y, train_mask, max_iter=probe_max_iter)
        W = torch.from_numpy(W_head)
        b = torch.from_numpy(b_head)
        logits_val = h[val_mask] @ W.t() + b
        preds = logits_val.argmax(dim=1).numpy()
        val_acc_curve.append(float(accuracy_score(y_val_np, preds)))

    k_star_emp = int(np.argmax(val_acc_curve))
    return k_star_emp, val_acc_curve


def compute_kstar_theory(
    e_signal: float,
    lambda_signal: float,
    C0: float,
) -> float:
    """Continuous K* = log(1/C0) / (2 * nu), nu = -log(1 - lambda_signal(T)) (Theorem 4)."""
    assert e_signal > 0, f"e_signal must be positive, got {e_signal}"
    assert 0.0 < lambda_signal < 1.0, f"lambda_signal must be in (0, 1), got {lambda_signal}"
    assert 0.0 < C0 < 1.0, f"C0 must be in (0, 1), got {C0}"

    nu = -math.log1p(-lambda_signal)
    return math.log(1.0 / C0) / (2.0 * nu)


def compute_kstar_ig_peak(
    trfw_mean_curve: List[float],
    C0: float,
    trf_mean_curve: Optional[List[float]] = None,
    h_list: Optional[List] = None,
    y: Optional = None,
    train_mask: Optional = None,
) -> Tuple[int, float]:
    assert len(trfw_mean_curve) >= 1
    peak_k = int(max(range(len(trfw_mean_curve)), key=lambda k: trfw_mean_curve[k]))
    e_signal_peak = float(trfw_mean_curve[peak_k])
    return peak_k, e_signal_peak


def compute_per_node_kstar_ig_peak(
    trfw_per_node_curve: List[torch.Tensor],
    C0: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(trfw_per_node_curve) >= 1
    curve = torch.stack(trfw_per_node_curve, dim=0)
    e_signal_v = curve[0, :]
    k_star_v = curve.argmax(dim=0).long()
    return k_star_v, e_signal_v


def compute_all_kstar(
    h_list: List[torch.Tensor],
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    C0: float,
    probe_max_iter: int = -1,
    restrict_lambda_to_train: bool = True,
    h_phi: Optional[str] = None,
    include_oracle_trfw: bool = False,
) -> dict:
    """Compute training-only K* (theory), empirical K*, and optional full-label oracle Tr(F W) diagnostics.

    When restrict_lambda_to_train=True (default), lambda_signal, h_phi, and k_star_theory use only
    train_mask labels. Set include_oracle_trfw=True explicitly to compute full-label oracle
    Tr(F_v W_v) diagnostics (k_star_ig_oracle, e_signal_oracle, trfw_mean_curve_oracle).
    """
    from .fisher import compute_trfw_curve

    n_nodes = h_list[0].size(0)
    node_mask_for_lambda = train_mask if restrict_lambda_to_train else None

    lambda_signal_lp = compute_lambda_signal(edge_index, y, n_nodes, node_mask=node_mask_for_lambda)

    if h_phi is None:
        h_phi = select_aggregation_filter(lambda_signal_lp)

    lambda_signal_t = compute_lambda_signal_T(
        edge_index, y, n_nodes, h_phi, node_mask=node_mask_for_lambda
    )

    k_star_emp, val_acc_curve = compute_kstar_emp(
        h_list, y, train_mask, val_mask, probe_max_iter
    )

    assert 0.0 < lambda_signal_t < 1.0
    k_star_theory = compute_kstar_theory(1.0, lambda_signal_t, C0)

    out = {
        "k_star_emp": k_star_emp,
        "k_star_theory": k_star_theory,
        "lambda_signal": lambda_signal_t,
        "lambda_signal_lp": lambda_signal_lp,
        "h_phi": h_phi,
        "val_acc_curve": val_acc_curve,
    }

    if include_oracle_trfw:
        trfw_mean_curve, _ = compute_trfw_curve(
            h_list, edge_index, y, train_mask, n_nodes, probe_max_iter, h_phi=h_phi
        )
        k_star_ig, e_signal_peak = compute_kstar_ig_peak(trfw_mean_curve, C0)
        out["k_star_ig_oracle"] = k_star_ig
        out["e_signal_oracle"] = e_signal_peak
        out["trfw_mean_curve_oracle"] = trfw_mean_curve

    return out
