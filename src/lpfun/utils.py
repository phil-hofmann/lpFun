import warnings
import itertools
import numpy as np
from numba import njit
from math import gamma
from typing import Tuple
from lpfun import NP_FLOAT, NP_INT
from lpfun.core import MultiIndexSet

"""Utility functions"""

# Node Callables


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


# Interpolation Matrix Callables


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


# Differentiation Matrix Callables


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


# Newton Point Evaluation


@njit
def newton2point(
    coefficients: np.ndarray, nodes: np.ndarray, x: np.ndarray, m: int, p: float
) -> NP_FLOAT:
    """O(Nmn)"""
    coefficients = np.asarray(coefficients).astype(NP_FLOAT)
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    m, p = int(m), float(p)
    ###
    n = len(nodes) - 1
    monomials = np.ones((m, n + 1), dtype=NP_FLOAT)
    for i in range(m):
        for j in range(1, n + 1):
            monomials[i, j] = monomials[i, j - 1] * (x[i] - nodes[j - 1])
    result = 0.0
    mis = MultiIndexSet(m, n, p)
    while mis.next():
        mi = mis.multi_index
        prod = 1.0
        for i in range(m):
            prod *= monomials[i, mi[i]]
        result += coefficients[mis.i] * prod
    ###
    return result


# Classifications


@njit
def classify(m: int, n: int, p: float) -> bool:
    m, n, p = int(m), int(n), float(p)
    ###
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (p <= 0.0 or p > 2.0) and (not p == np.inf):
        raise ValueError(
            f"The parameter p should be in the range (0, 2] or p = np.inf."
        )
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")
    ###
    return True


@njit
def is_lower_triangular(M: np.ndarray, atol=1e-8) -> bool:
    M = np.asarray(M).astype(NP_FLOAT)
    ###
    n = len(M)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.abs(M[i, j]) < atol:
                return False
    ###
    return True


# Row major ordering


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


# Tube projections


@njit
def _memory_allocation(m: int, n: int, p: float) -> int:
    if p <= 1.0:
        singular = 1 + m * n
        if m < n:
            return int(singular * (1 - p) + _binomial(n + m, m) * p)
        else:
            return int(singular * (1 - p) + _binomial(n + m, n) * p)
    elif p <= 2.0:
        fac1 = (p * np.e / m) ** (1 / p)
        fac2 = np.sqrt(p / (2 * np.pi * m))
        return int(np.ceil((fac1 * (n + 2) * gamma(1 + 1 / p)) ** m * fac2))
    else:
        return int((n + 1) ** m)


@njit
def _binomial(n: int, m: int) -> int:
    if m < 0 or m > n:
        return 0
    result = 1
    for i in range(min(m, n - m)):
        result = result * (n - i) // (i + 1)
    return result


def tube(m: int, n: int, p: float) -> np.ndarray:
    classify(m, n, p)
    if m == 1:
        return np.array([n + 1], dtype=NP_INT)
    if p == np.inf:
        return np.array([n + 1] * (n + 1) ** (m - 1), dtype=NP_INT)
    return _tube(m, n, p)


@njit
def _tube(m: int, n: int, p: float) -> np.ndarray:
    mis = MultiIndexSet(m, n, p)
    memory_allocation = _memory_allocation(m - 1, n, p)
    tube = np.zeros(memory_allocation, dtype=NP_INT)

    while mis.next():
        if mis.k > 0:
            tube[mis.l] = mis.k
    return tube[: mis.l]


# Grid


def grid(nodes: np.ndarray, m: int, n: int, p: float) -> np.ndarray:
    classify(m, n, p)
    if p == np.inf:
        return np.flip(list(itertools.product(nodes, repeat=m)), axis=1)
    else:
        return _grid(nodes, m, n, p)


@njit
def _grid(nodes: np.ndarray, m: int, n: int, p: float) -> np.ndarray:
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    memory_allocation = _memory_allocation(m, n, p)
    grid = np.zeros((memory_allocation, m))
    mis = MultiIndexSet(m, n, p)
    while mis.next():
        mi = mis.multi_index
        grid[mis.i] = [nodes[mi[_]] for _ in range(m)]
    return grid[: mis.i + 1]


# Leja--ordered Nodes


def leja_nodes(nodes: np.ndarray) -> np.ndarray:
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


@njit
def permutation_max(
    m: int,
    n: int,
    i: int,
) -> np.ndarray:
    """O((n+1)^m)"""
    N = (n + 1) ** m
    mat = np.zeros(N, dtype=NP_INT)
    iter_len = (n + 1) ** i
    step_len = (n + 1) ** (m - i)
    k = 0
    for j in range(step_len):
        for i in range(iter_len):
            val = i * step_len + j
            mat[k] = val
            k += 1
    return mat


@njit
def permutation(
    T: np.ndarray,
    i: int,
) -> np.ndarray:
    n = np.max(T)
    if i == 0:
        return np.arange(n)
    else:
        Pi = transposition(T)
        if i == 1:
            return Pi
        P = Pi[:]
        for _ in range(i - 1):
            P = apply_permutation(P, Pi, invert=True)
        return P


@njit
def apply_permutation(
    P: np.ndarray,
    x: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    x_p = np.zeros_like(x)
    if invert:
        for i, j in enumerate(P):
            x_p[i] = x[j]
    else:
        for i, j in enumerate(P):
            x_p[j] = x[i]
    return x_p


@njit
def transposition(T: np.ndarray) -> np.ndarray:
    N, n = np.sum(T), np.max(T)
    permutation_vector = np.zeros(N, dtype=NP_INT)
    current_position = 0
    for j in range(n):
        for i in range(len(T)):
            if j < T[i]:
                permutation_vector[current_position] = sum(T[:i]) + j
                current_position += 1
    return permutation_vector


@njit
def reduceat(
    array: np.ndarray,
    split_indices: np.ndarray,
) -> np.ndarray:
    """O(len(array))"""
    sums = np.zeros(len(split_indices) - 1, dtype=array.dtype)
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        if start >= len(array):
            break
        end = min(end, len(array))
        sums[i] = np.sum(array[start:end])
    return sums[:i]


@njit
def entropy(T: np.ndarray) -> np.ndarray:
    cs_T = np.cumsum(T)
    e_T = np.array([cs_T[-1]], dtype=NP_INT)
    if e_T[0] == 1:
        return e_T
    while True:
        l = e_T[-1]
        temp = cs_T[0:l]
        index = np.where(temp == l)[0]
        if len(index) == 0:
            break
        else:
            index = index[0]
        e_T = np.append(e_T, index + 1)
    return e_T


@njit
def plusplus(
    m: int,
    i: np.ndarray,
    d: np.ndarray,
    T: np.ndarray,
    e_T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    j = 0
    max_j = 0
    while j < m - 1:
        if d[j] < T[: e_T[j + 2]][i[j + 1]] - 1:
            d[j] += 1
            i[j] += 1
            return i, d, max_j
        else:
            d[j] = 0
            i[j] += 1
            j += 1
        max_j = max(max_j, j)
    return None, None, -1


@njit
def ordinal_embedding(
    m: int,
    T: np.ndarray,
    T_prime: np.ndarray,
) -> np.ndarray:
    """O(||T_prime||_1)"""
    T = np.asarray(T, dtype=NP_INT)
    T_prime = np.asarray(T_prime, dtype=NP_INT)
    e_T = entropy(T)
    e_T_prime = entropy(T_prime)
    N = np.sum(T)
    phi = np.zeros(N, dtype=NP_INT)
    k, k_prime = 0, 0
    max_j, max_j_prime = 0, 0
    i, i_prime = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    d, d_prime = np.zeros(m, dtype=np.int64), np.zeros(m, dtype=np.int64)
    while max_j != -1:
        T0i0 = T[i[0]]
        T_prime0i0 = T_prime[i_prime[0]]
        if T0i0 <= T_prime0i0:
            for j in range(T0i0):
                phi[k + j] = k_prime + j
            k = k + T0i0
            i, d, max_j = plusplus(m, i, d, T, e_T)
            max_j_prime = -1
            while max_j_prime < max_j:
                k_prime = k_prime + T_prime[i_prime[0]]
                i_prime, d_prime, max_j_prime = plusplus(
                    m, i_prime, d_prime, T_prime, e_T_prime
                )
        else:
            raise ValueError("Undetermined condition encountered in the algorithm.")
    return phi


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
