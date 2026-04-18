from __future__ import annotations

import numpy as np
import torch


def l1(fitted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Mean per-vertex L1 distance, per frame. Requires vertex correspondence.

    Args:
        fitted: (T, V, 3) float32
        target: (T, V, 3) float32

    Returns:
        (T,) float32
    """
    return np.abs(fitted - target).sum(axis=-1).mean(axis=-1)


def l2(fitted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Mean per-vertex L2 (Euclidean) distance, per frame. Requires vertex correspondence.

    Args:
        fitted: (T, V, 3) float32
        target: (T, V, 3) float32

    Returns:
        (T,) float32
    """
    return np.linalg.norm(fitted - target, axis=-1).mean(axis=-1)


def _chamfer_one_sided(
    p: torch.Tensor, q: torch.Tensor, norm: int, chunk: int = 512
) -> torch.Tensor:
    """
    One-sided Chamfer: mean over p of min_j dist(p_i, q_j).
      norm=1: mean of sqrt(min squared distance)
      norm=2: mean of min squared distance
    p, q: (V, 3) float32 on the same device.
    Processes p in chunks to avoid OOM on large meshes.
    """
    min_d2_list = []
    for i in range(0, p.shape[0], chunk):
        p_chunk = p[i : i + chunk]                              # (C, 3)
        diff    = p_chunk.unsqueeze(1) - q.unsqueeze(0)         # (C, V, 3)
        dist2   = (diff ** 2).sum(-1)                           # (C, V)
        min_d2_list.append(dist2.min(dim=1).values)             # (C,)
    min_d2 = torch.cat(min_d2_list)                             # (V,)
    if norm == 2:
        return min_d2.mean()
    else:
        return min_d2.sqrt().mean()


def chamfer_l1(
    fitted: np.ndarray,
    target: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> np.ndarray:
    """
    Bidirectional Chamfer distance (L1 nearest-neighbour), per frame.

    Args:
        fitted: (T, V, 3) float32
        target: (T, V, 3) float32
        device: torch device

    Returns:
        (T,) float32
    """
    T   = fitted.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        p = torch.tensor(fitted[t], dtype=torch.float32, device=device)
        q = torch.tensor(target[t], dtype=torch.float32, device=device)
        out[t] = ((_chamfer_one_sided(p, q, 1) + _chamfer_one_sided(q, p, 1)) / 2).item()
    return out


def chamfer_l2(
    fitted: np.ndarray,
    target: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> np.ndarray:
    """
    Bidirectional Chamfer distance (L2 nearest-neighbour), per frame.

    Args:
        fitted: (T, V, 3) float32
        target: (T, V, 3) float32
        device: torch device

    Returns:
        (T,) float32
    """
    T   = fitted.shape[0]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        p = torch.tensor(fitted[t], dtype=torch.float32, device=device)
        q = torch.tensor(target[t], dtype=torch.float32, device=device)
        out[t] = ((_chamfer_one_sided(p, q, 2) + _chamfer_one_sided(q, p, 2)) / 2).item()
    return out
