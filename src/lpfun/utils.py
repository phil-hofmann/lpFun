import itertools
import numpy as np
from typing import Tuple
from numba import njit, prange


"""Utility functions"""


@njit
def classify(m: int, n: int, p: float) -> bool:
    m, n, p = int(m), int(n), float(p)
    ###
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (p <= 0.0 or p > 2.0) and (not p == np.inf):
        raise ValueError(f"The parameter p should be in the range (0, 2] or inf.")
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")
    ###
    return True


# nodes


def cheb2nd_nodes(n: int) -> np.ndarray:
    """O(n)"""
    n = int(n)
    ###
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    return np.cos(np.arange(n, dtype=np.float64) * np.pi / (n - 1))


def leja_nodes(n: int, m: int = 1000) -> np.ndarray:
    """O(n^3)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    if n > m:
        raise (
            f"The amount of nodes {n} must be smaller or equal than the sample size {m}."
        )
    sample_nodes = cheb2nd_nodes(m)
    leja_order = get_leja_order(sample_nodes, limit=n)
    return sample_nodes[leja_order]


# vandermonde matrices


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


# differentiation matrices


@njit
def newton2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    nodes = np.asarray(nodes).astype(np.float64)
    x = nodes[:]
    n = len(x)
    ###
    Dx = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(i):
            if i == j + 1:
                Dx[i, j] = i
            else:
                Dx[i, j] = (x[j] - x[i - 1]) * Dx[i - 1, j] + Dx[i - 1, j - 1]
    ###
    return Dx.T


@njit
def chebyshev2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    ### NOTE -- Matrix is independent of the nodes
    nodes = np.asarray(nodes).astype(np.float64)
    n = len(nodes) - 1
    ###
    Dx = np.zeros((n + 1, n + 1))
    for k in range(1, n + 1):
        for j in range(k - 1, -1, -2):
            Dx[j, k] = 2 * k
        Dx[0, k] *= 0.5
    ###
    return Dx


# point evaluation


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


# Leja order


@njit
def get_leja_order(nodes: np.ndarray, limit: int = -1) -> np.ndarray:
    """O(n^3)"""
    """This function originates from minterpy."""
    ### NOTE -- no type conversion
    n = len(nodes)
    limit = n if limit == -1 else limit
    ord = np.arange(1, n, dtype=np.int64)
    lj = np.zeros(limit, dtype=np.int64)
    lj[0] = 0
    m = 0
    for k in range(0, limit - 1):
        jj = 0
        for i in range(0, n - k - 1):
            p = 1
            for j in range(k + 1):
                p = p * (nodes[lj[j]] - nodes[ord[i]])
            p = np.abs(p)
            if p >= m:
                jj = i
                m = p
        m = 0
        lj[k + 1] = ord[jj]
        ord = np.delete(ord, jj)
    return lj


# grid


def get_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    """O(N)"""
    nodes, m, n, p = (
        np.asarray(nodes).astype(np.float64),
        int(m),
        int(n),
        float(p),
    )
    if m == 1:
        return nodes.reshape(-1, 1)
    elif p == np.inf:
        return np.flip(
            np.asarray(list(itertools.product(nodes, repeat=m)), dtype=np.float64),
            axis=1,
        )
    else:
        A = np.asarray(A).astype(np.int64)
        return _get_grid(nodes, A, m)


@njit
def _get_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
) -> np.ndarray:
    ### NOTE -- no type conversion
    N = len(A)
    grid = np.zeros((N, m))
    for i in range(N):
        mi = A[i]
        grid_point = np.zeros(m, dtype=np.float64)
        for j in range(m):
            grid_point[j] = nodes[mi[j]]
        grid[i] = grid_point
    return grid


# row major ordering


@njit
def is_lower_triangular(
    M: np.ndarray,
    atol=1e-8,
) -> bool:
    """O(n^2)"""
    M = np.asarray(M).astype(np.float64)
    ###
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.abs(M[i, j]) < atol:
                return False
    ###
    return True


@njit
def get_rmo(L: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    L = np.asarray(L).astype(np.float64)
    ###
    n = len(L)
    N = int(n * (n + 1) / 2)
    result = np.zeros(N, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            result[k] = L[i, j]
            k += 1
    ###
    return result


# matrix operations


@njit
def get_lu(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """O(n^3)"""
    M = np.asarray(M).astype(np.float64)
    ###
    n = len(M)
    L = np.eye(n, dtype=np.float64)
    U = M[:, :]
    for j in range(n):
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    ###
    return L, U


# @njit # NOTE pivoting increases numerical stability
# def get_lu_pivot(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Stable LU with partial pivoting. Returns P, L, U so that P @ M = L @ U"""
#     M = np.asarray(M).astype(np.float64)
#     n = len(M)
#     L = np.zeros((n, n), dtype=np.float64)
#     U = M.copy()
#     P = np.eye(n, dtype=np.float64)

#     for j in range(n):
#         # Partial pivoting: find pivot row
#         pivot = j + np.argmax(np.abs(U[j:, j]))
#         if pivot != j:
#             # Swap rows in U
#             U[[j, pivot], :] = U[[pivot, j], :]
#             # Swap rows in P
#             P[[j, pivot], :] = P[[pivot, j], :]
#             # Swap rows in L (columns before j)
#             if j > 0:
#                 L[[j, pivot], :j] = L[[pivot, j], :j]

#         # Normal LU step
#         L[j, j] = 1.0
#         for i in range(j + 1, n):
#             L[i, j] = U[i, j] / U[j, j]
#             U[i, j:] -= L[i, j] * U[j, j:]

#     return P, L, U
