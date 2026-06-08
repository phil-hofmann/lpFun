import numpy as np
import numba as nb
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


# prange = nb.prange


@njit
def newton2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n))
    for i in range(n):
        monomials = np.ones(n, dtype=np.float64)
        for j in range(1, n):
            monomials[j] *= monomials[j - 1] * (x[i] - x[j - 1])
        Vx[i, :n] = monomials
    ###
    return Vx


@njit
def chebyshev2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Vx[i, 0] = 1.0
        if n > 0:
            Vx[i, 1] = x[i]
        for j in range(2, n + 1):
            Vx[i, j] = 2 * x[i] * Vx[i, j - 1] - Vx[i, j - 2]
    return Vx


@njit
def legendre2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Vx[i, 0] = 1.0
        if n > 1:
            Vx[i, 1] = x[i]
        for j in range(2, n):
            Vx[i, j] = ((2 * j - 1) * x[i] * Vx[i, j - 1] - (j - 1) * Vx[i, j - 2]) / j
    return Vx
