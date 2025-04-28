import itertools
import numpy as np
from numba import njit, prange
from typing import Tuple
from lpfun import NP_FLOAT, NP_INT
from lpfun.core.utils import classify, apply_permutation


"""Utility functions"""

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
def newton2lagrange(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    nodes = np.asarray(nodes).astype(np.float64)
    x = nodes[:]
    n = len(x)
    ###
    Qx = np.zeros((n, n))
    for i in range(n):
        monomials = np.ones(n, dtype=np.float64)
        for j in range(1, n):
            monomials[j] *= monomials[j - 1] * (x[i] - x[j - 1])
        Qx[i, :n] = monomials
    ###
    return Qx


@njit
def chebyshev2lagrange(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    nodes = np.asarray(nodes).astype(np.float64)
    x = nodes[:]
    n = len(x)
    ###
    Qx = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Qx[i, 0] = 1.0
        if n > 0:
            Qx[i, 1] = x[i]
        for j in range(2, n + 1):
            Qx[i, j] = 2 * x[i] * Qx[i, j - 1] - Qx[i, j - 2]
    return Qx


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


def newton2point(
    coefficients: np.ndarray,
    nodes: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
) -> NP_FLOAT:
    """O(Nmn)"""
    coefficients, nodes, x, m, n = (
        np.asarray(coefficients).astype(NP_FLOAT),
        np.asarray(nodes).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        int(m),
        int(n),
    )
    return _newton2point(coefficients, nodes, x, A, m, n)


@njit(parallel=True)
def _newton2point(
    coefficients: np.ndarray,
    nodes: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
) -> NP_FLOAT:
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
    return value


def chebyshev2point(
    coefficients: np.ndarray, x: np.ndarray, A: np.ndarray, m: int, n: int
) -> float:
    coefficients, x, A, m, n = (
        np.asarray(coefficients).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(A).astype(NP_INT),
        int(m),
        int(n),
    )
    return _chebyshev2point(coefficients, x, A, m, n)


@njit(parallel=True)
def _chebyshev2point(
    coefficients: np.ndarray, x: np.ndarray, A: np.ndarray, m: int, n: int
) -> float:
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
    return value


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
    classify(m, n, p)
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


@njit
def inv(L: np.ndarray) -> np.ndarray:
    """O(n^3)"""
    L = np.asarray(L).astype(np.float64)
    n = len(L)
    ###
    invL = np.zeros_like(L)
    for i in range(n):
        invL[i, i] = 1 / L[i, i]
        for j in range(i):
            dotsum = np.sum(L[i, j : i + 1] * invL[j : i + 1, j])
            invL[i, j] = -dotsum / L[i, i]
    ###
    return invL
