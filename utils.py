# utils.py
import numpy as np

def wrap_xy(xy: np.ndarray, w: float, h: float) -> None:
    """Toroidal wrapping for (N,2) array in‑place."""
    xy[:, 0] %= w
    xy[:, 1] %= h

def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,M) matrix of Euclidean distances for 2‑D vectors."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))