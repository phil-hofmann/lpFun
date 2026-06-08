import numpy as np
import numba as nb
from typing import Tuple
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


# prange = nb.prange


@njit
def binomial(n: int, m: int) -> int:
    if m < 0 or m > n:
        return 0
    result = 1
    for i in range(min(m, n - m)):
        result = result * (n - i) // (i + 1)
    return result


@njit
def classify(m: int, n: int, p: float) -> bool:
    m, n, p = int(m), int(n), float(p)
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (p <= 0.0 or p > 2.0) and (not p == np.inf):
        raise ValueError(f"The parameter p should be in the range (0, 2] or inf.")
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")
    return True


@njit(inline="always")
def apply_permutation(
    P: np.ndarray,
    x: np.ndarray,
    invert: bool,
) -> np.ndarray:
    """O(N)"""
    x_p = np.zeros_like(x)
    N = len(P)
    if invert:
        for i in range(N):
            x_p[i] = x[P[i]]
    else:
        for i in range(N):
            x_p[P[i]] = x[i]
    return x_p


@njit
def is_lower_triangular(
    M: np.ndarray,
    atol=1e-8,
) -> bool:
    """O(n^2)"""
    M = np.asarray(M).astype(np.float64)
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.abs(M[i, j]) < atol:
                return False
    return True


@njit
def get_lu(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """O(n^3)"""
    M = np.asarray(M).astype(np.float64)
    n = len(M)
    L = np.eye(n, dtype=np.float64)
    U = M[:, :]
    for j in range(n):
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    return L, U
