import numpy as np
import numba as nb
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


prange = nb.prange


@njit(parallel=True)
def newton2point(
    coefficients: np.ndarray,
    nodes: np.ndarray,
    points: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
) -> float:
    """O(Nmn)"""
    ### NOTE -- no type conversion
    len_points = len(points)
    values = np.zeros(len_points, dtype=np.float64)
    for l in prange(len_points):
        x = points[l]
        ###
        basis = np.ones((m, n + 1), dtype=np.float64)
        for i in range(m):
            for j in range(1, n + 1):
                basis[i, j] = basis[i, j - 1] * (x[i] - nodes[j - 1])
        ###
        value = 0.0
        for i in prange(len(A)):
            mi = A[i]
            prod = 1.0
            for j in range(m):
                prod *= basis[j, mi[j]]
            value += coefficients[i] * prod
        ###
        values[l] = value
    return values


@njit(parallel=True)
def chebyshev2point(
    coefficients: np.ndarray,
    points: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
) -> float:
    ### NOTE -- no type conversion
    len_points = len(points)
    values = np.zeros(len_points, dtype=np.float64)
    for l in prange(len_points):
        x = points[l]
        ###
        basis = np.empty((m, n + 1), dtype=np.float64)
        basis[:, 0] = 1.0
        if n >= 1:
            basis[:, 1] = x
        for j in range(1, n):
            basis[:, j + 1] = 2 * x * basis[:, j] - basis[:, j - 1]
        ###
        value = 0.0
        for i in prange(len(A)):
            mi = A[i]
            prod = 1.0
            for j in range(m):
                prod *= basis[j, mi[j]]
            value += coefficients[i] * prod
        ###
        values[l] = value
    return values


@njit(parallel=True)
def legendre2point(
    coefficients: np.ndarray,
    points: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
) -> float:
    """O(Nmn)"""
    ### NOTE -- no type conversion
    len_points = len(points)
    values = np.zeros(len_points, dtype=np.float64)
    for l in prange(len_points):
        x = points[l]
        ###
        basis = np.empty((m, n + 1), dtype=np.float64)
        basis[:, 0] = 1.0
        if n >= 1:
            basis[:, 1] = x
        for j in range(2, n + 1):
            for d in range(m):
                basis[d, j] = (
                    (2 * j - 1) * x[d] * basis[d, j - 1] - (j - 1) * basis[d, j - 2]
                ) / j
        ###
        value = 0.0
        for i in prange(len(A)):
            mi = A[i]
            prod = 1.0
            for j in range(m):
                prod *= basis[j, mi[j]]
            value += coefficients[i] * prod
        ###
        values[l] = value
    return values
