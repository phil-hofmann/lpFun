import numpy as np
import numba as nb
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


# prange = nb.prange


@njit
def newton2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    nodes = np.asarray(nodes).astype(np.float64)
    x = nodes[:]
    n = len(x)
    Dx = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(i):
            if i == j + 1:
                Dx[i, j] = i
            else:
                Dx[i, j] = (x[j] - x[i - 1]) * Dx[i - 1, j] + Dx[i - 1, j - 1]
    return Dx.T


@njit
def chebyshev2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    # NOTE -- Matrix is independent of the nodes
    nodes = np.asarray(nodes).astype(np.float64)
    n = len(nodes) - 1
    Dx = np.zeros((n + 1, n + 1))
    for k in range(1, n + 1):
        for j in range(k - 1, -1, -2):
            Dx[j, k] = 2 * k
        Dx[0, k] *= 0.5
    return Dx


@njit
def legendre2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    # NOTE -- Matrix is independent of the nodes
    nodes = np.asarray(nodes).astype(np.float64)
    n = len(nodes) - 1
    Dx = np.zeros((n + 1, n + 1))
    for k in range(1, n + 1):
        for j in range(k - 1, -1, -2):
            Dx[j, k] = 2 * j + 1
    return Dx
