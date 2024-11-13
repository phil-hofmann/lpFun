import itertools
import numpy as np
from numba import njit
from math import gamma
from typing import Tuple
from lpfun import NP_FLOAT, NP_INT
from lpfun.iterators import MultiIndexSet

"""Utility functions"""


@njit
def classify(m: int, n: int, p: float, allow_infty=False) -> np.ndarray:
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (not allow_infty) and (p <= 0.0 or p > 2.0):
        raise ValueError(" The parameter p should be in the range (0, 2].")
    if allow_infty and (p <= 0.0 or p > 2.0) and (not p == np.inf):
        raise ValueError(
            " The parameter p should be in the range (0, 2] or p = np.inf."
        )
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")


@njit
def n2l(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(nodes).astype(NP_FLOAT)
    n = len(nodes)
    lag_coeffs = np.zeros((n, n))
    for i in range(n):
        evals = _n_eval(x, x[i])
        lag_coeffs[i, :n] = evals
    return lag_coeffs


@njit
def _n_eval(nodes: np.ndarray, x: NP_FLOAT) -> NP_FLOAT:
    """O(n)"""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    n = len(nodes)
    monomials = np.ones(n, dtype=NP_FLOAT)
    # caution -- loop not parallelizable
    for i in range(1, n):
        monomials[i] *= monomials[i - 1] * (x - nodes[i - 1])
    return monomials


@njit
def n_eval_at_point(
    coefficients: np.ndarray, nodes: np.ndarray, x: np.ndarray, m: int, p: float
) -> NP_FLOAT:
    """O(m*k_m,n,p)"""
    coefficients = np.asarray(coefficients).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    n = len(nodes) - 1
    monomials = [_n_eval(nodes, x[i]) for i in range(m)]  # O(m*n)
    result = 0.0
    mis = MultiIndexSet(m, n, p)
    while mis.next():  #  O(k_mnp)
        mi = mis.multi_index
        result += coefficients[mis.i] * np.prod(
            np.array([monomials[i][mi[i]] for i in range(m)], dtype=NP_FLOAT)
        )  # O(m)
    return result


@njit
def l2n(nodes: np.ndarray) -> np.ndarray:
    """O(n^3)"""
    n = len(nodes)
    x = np.asarray(nodes)
    newton_coeffs = np.zeros((n, n))
    for i in range(n):
        y = np.zeros(n)
        y[i] = 1
        coeffs = _dds(x, y)
        newton_coeffs[i, :n] = coeffs
    return newton_coeffs.T


@njit
def _dds(x, y) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(NP_FLOAT)
    y = np.asarray(y).astype(NP_FLOAT)
    n = len(x)
    dd = np.zeros(n, dtype=NP_FLOAT)
    dd[0] = y[0]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            y[j] = (y[j] - y[j - 1]) / (x[j] - x[j - i])
        dd[i] = y[i]
    return dd


@njit
def n_dx(nodes: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    n = nodes.shape[0]
    dx = np.zeros((n, n), dtype=NP_FLOAT)
    for i in range(1, n):
        for j in range(i):
            if i == j + 1:
                dx[i, j] = i
            else:
                dx[i, j] = (nodes[j] - nodes[i - 1]) * dx[i - 1, j] + dx[i - 1, j - 1]
    return dx.T


@njit
def rmo(A: np.ndarray, mode: str = "lower") -> np.ndarray:
    if mode == "upper":
        return _rmo_upper(A)
    elif mode == "lower":
        return _rmo_lower(A)
    else:
        raise ValueError("The parameter mode should be either 'upper' or 'lower'.")


@njit
def _rmo_upper(A: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    A = np.asarray(A).astype(NP_FLOAT)
    n = A.shape[0]
    N = int(n * (n + 1) / 2)
    result = np.zeros(N, dtype=NP_FLOAT)
    k = 0
    for i in range(n):
        for j in range(n - i):
            result[k] = A[i, i + j]
            k += 1
    return result


@njit
def _rmo_lower(A: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    A = np.asarray(A).astype(NP_FLOAT)
    n = A.shape[0]
    N = int(n * (n + 1) / 2)
    result = np.zeros(N, dtype=NP_FLOAT)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            result[k] = A[i, j]
            k += 1
    return result


def lower_grid(nodes: np.ndarray, m: int, n: int, p: float) -> np.ndarray:
    classify(m, n, p, allow_infty=True)
    if p == np.inf:
        return np.flip(list(itertools.product(nodes, repeat=m)), axis=1)
    else:
        return _lower_grid(nodes, m, n, p)


@njit
def _lower_grid(nodes: np.ndarray, m: int, n: int, p: float) -> np.ndarray:
    """O(|A_{m, n, p}| + ... + |A_{1, n, p}|)"""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    memory_allocation = _memory_allocation(m, n, p)
    lower_grid = np.zeros((memory_allocation, m))
    mis = MultiIndexSet(m, n, p)
    while mis.next():
        mi = mis.multi_index
        lower_grid[mis.i] = [nodes[mi[_]] for _ in range(m)]
    return lower_grid[: mis.i + 1]


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
    """O(min(m, n-m))"""
    if m < 0 or m > n:
        return 0
    result = 1
    for i in range(min(m, n - m)):
        result = result * (n - i) // (i + 1)
    return result


@njit
def leja_nodes(
    n: int, nodes: callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _nodes = nodes(n)
    if len(np.unique(_nodes)) != len(_nodes):
        raise ValueError("The nodes are not distinct.")
    _leja_order = leja_order(_nodes)
    _leja_nodes = apply_permutation(_leja_order, _nodes)
    return _leja_nodes


@njit
def cheb(n: int) -> np.ndarray:
    """O(n)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=NP_FLOAT)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=NP_FLOAT)
    return np.cos(np.arange(n, dtype=NP_FLOAT) * np.pi / (n - 1))


@njit
def tube(m: int, n: int, p: NP_FLOAT) -> np.ndarray:
    classify(m, n, p)
    return _tube(m, n, p)


@njit
def _tube(m: int, n: int, p: float) -> np.ndarray:
    """O(k_m,n,p)"""
    mis = MultiIndexSet(m, n, p)
    memory_allocation = _memory_allocation(m - 1, n, p)
    tube = np.zeros(memory_allocation, dtype=NP_INT)

    while mis.next():
        if mis.k > 0:
            tube[mis.l] = mis.k
    return tube[: mis.l]


@njit
def leja_order(nodes: np.ndarray) -> np.ndarray:
    """This function originates from minterpy."""
    """O(n^2)"""
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
def permutation_maximal(m: int, n: int, i: int) -> np.ndarray:
    """O(N)"""
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
def permutation(T: np.ndarray, i: int) -> np.ndarray:
    """O(???)"""
    n = np.max(T)
    if i == 0:
        return np.arange(n)
    else:
        Pi = transposition(T)
        if i == 1:
            return Pi
        P = np.copy(Pi)
        for _ in range(i - 1):
            P = apply_permutation(P, Pi, invert=True)
        return P


@njit
def transposition(T) -> np.ndarray:
    """O(???)"""
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
def apply_permutation(P: np.ndarray, x: np.ndarray, invert: bool = False) -> np.ndarray:
    """O(n)"""
    x_p = np.zeros_like(x)
    if invert:
        for i, j in enumerate(P):
            x_p[i] = x[j]
    else:
        for i, j in enumerate(P):
            x_p[j] = x[i]
    return x_p


@njit
def rmo_transpose(rmo: np.ndarray) -> np.ndarray:
    N = rmo.size
    n = int((np.sqrt(1 + 8 * N) - 1) / 2)
    transposed_rmo = np.zeros(N, dtype=NP_FLOAT)
    for i in range(n):
        for j in range(i + 1):
            lower_idx = i * (i + 1) // 2 + j
            upper_idx = j * n - j * (j - 1) // 2 + i - j
            transposed_rmo[upper_idx] = rmo[lower_idx]
    return transposed_rmo


@njit
def concatenate_arrays(chunk_dot) -> np.ndarray:
    total_length = sum([len(arr) for arr in chunk_dot])
    result = np.empty(total_length, dtype=chunk_dot[0].dtype)
    start = 0
    for arr in chunk_dot:
        end = start + arr.size
        result[start:end] = arr
        start = end
    return result


@njit
def reduceat(array, split_indices) -> np.ndarray:
    """O(???)"""
    sums = np.zeros(len(split_indices) - 1, dtype=array.dtype)
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        if start >= len(array):
            break
        end = min(end, len(array))
        sums[i] = np.sum(array[start:end])
    return sums[:i]
