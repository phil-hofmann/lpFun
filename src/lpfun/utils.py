import itertools
import numpy as np
from numba import njit, prange
from typing import Tuple
from lpfun import NP_FLOAT, NP_INT
from lpfun.core.utils import apply_permutation


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


def cheb2nd(n: int) -> np.ndarray:
    """O(n)"""
    n = int(n)
    ###
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=NP_FLOAT)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=NP_FLOAT)
    return np.cos(np.arange(n, dtype=NP_FLOAT) * np.pi / (n - 1))


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


# inverse vandermonde matrices


@njit
def lagrange2newton(x: np.ndarray):
    x = np.asarray(x)
    n = len(x)
    inv_V = np.eye(n)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            inv_V[i, :] = (inv_V[i, :] - inv_V[i - 1, :]) / (x[i] - x[i - j])
    return inv_V


# differentiation matrices


@njit
def newton2derivative(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    x = nodes[:]
    n = len(x)
    ###
    Dx = np.zeros((n, n), dtype=NP_FLOAT)
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
    nodes = np.asarray(nodes).astype(NP_FLOAT)
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
) -> NP_FLOAT:
    """O(Nmn)"""
    len_points = len(points)
    values = np.zeros(len_points, dtype=NP_FLOAT)
    for l in prange(len_points):
        x = points[l]
        ###
        basis = np.ones((m, n + 1), dtype=NP_FLOAT)
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
    len_points = len(points)
    values = np.zeros(len_points, dtype=NP_FLOAT)
    for l in prange(len_points):
        x = points[l]
        ###
        basis = np.empty((m, n + 1), dtype=NP_FLOAT)
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


# Leja ordered nodes


def leja_nodes(nodes: np.ndarray) -> np.ndarray:
    """O(n^3)"""
    order = _leja_order(nodes)
    ordered_nodes = apply_permutation(order, nodes)
    return ordered_nodes


@njit
def _leja_order(nodes: np.ndarray) -> np.ndarray:
    """This function originates from minterpy."""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    n = len(nodes) - 1
    ord = np.arange(1, n + 1, dtype=NP_INT)
    lj = np.zeros(n + 1, dtype=NP_INT)
    lj[0] = 0
    m = 0
    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
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


def gen_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    """O(N)"""
    nodes, m, n, p = (
        np.asarray(nodes).astype(NP_FLOAT),
        int(m),
        int(n),
        float(p),
    )
    if m == 1:
        return nodes.reshape(-1, 1)
    elif p == np.inf:
        return np.flip(
            np.asarray(list(itertools.product(nodes, repeat=m)), dtype=NP_FLOAT), axis=1
        )
    else:
        A = np.asarray(A).astype(NP_INT)
        return _gen_grid(nodes, A, m)


@njit
def _gen_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
) -> np.ndarray:
    N = len(A)
    grid = np.zeros((N, m))
    for i in range(N):
        mi = A[i]
        grid_point = np.zeros(m, dtype=NP_FLOAT)
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
    M = np.asarray(M).astype(NP_FLOAT)
    ###
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.abs(M[i, j]) < atol:
                return False
    ###
    return True


@njit
def rmo(L: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    L = np.asarray(L).astype(NP_FLOAT)
    ###
    n = len(L)
    N = int(n * (n + 1) / 2)
    result = np.zeros(N, dtype=NP_FLOAT)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            result[k] = L[i, j]
            k += 1
    ###
    return result


# matrix operations


@njit
def lu(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """O(n^3)"""
    M = np.asarray(M).astype(NP_FLOAT)
    ###
    n = len(M)
    L = np.eye(n, dtype=NP_FLOAT)
    U = M[:, :]
    for j in range(n):
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]
    ###
    return L, U


# @njit
# def inv(L: np.ndarray) -> np.ndarray:
#     """O(n^3)"""
#     L = np.asarray(L).astype(np.float64)
#     n = len(L)
#     ###
#     invL = np.zeros_like(L)
#     for i in range(n):
#         invL[i, i] = 1 / L[i, i]
#         for j in range(i):
#             dotsum = np.sum(L[i, j : i + 1] * invL[j : i + 1, j])
#             invL[i, j] = -dotsum / L[i, i]
#     ###
#     return invL
