import itertools
import numpy as np
from numba import njit
from lpfun import NP_FLOAT, NP_ARRAY, NP_INT
from lpfun.iterators import MultiIndexSet

"""Utility functions"""


@njit
def classify(m: int, n: int, p: float, allow_infty=False) -> NP_ARRAY:
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (not allow_infty) and (p <= 0.0 or p > 2.0):
        raise ValueError(" The parameter p should be in the range (0, 2].")
    if allow_infty and (p <= 0.0 or p > 2.0) and (not p == np.infty):
        raise ValueError(
            " The parameter p should be in the range (0, 2] or p = np.infty."
        )
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")


@njit
def n2l(nodes: NP_ARRAY) -> NP_ARRAY:
    """O(n^2)"""
    x = np.asarray(nodes).astype(NP_FLOAT)
    n = len(nodes)
    lag_coeffs = np.zeros((n, n))
    for i in range(n):
        evals = _eval(x, x[i])
        lag_coeffs[i, :n] = evals
    return lag_coeffs


@njit
def _eval(nodes: NP_ARRAY, x: NP_FLOAT) -> NP_FLOAT:
    """O(n)"""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    n = len(nodes)
    monomials = np.ones(n, dtype=NP_FLOAT)
    # caution -- loop not parallelizable
    for i in range(1, n):
        monomials[i] *= monomials[i - 1] * (x - nodes[i - 1])
    return monomials


@njit
def eval_at_point(
    coefficients: NP_ARRAY, nodes: NP_ARRAY, x: NP_ARRAY, m: int, p: float
) -> NP_FLOAT:
    """O(m*k_m,n,p)"""
    coefficients = np.asarray(coefficients).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    n = len(nodes) - 1
    monomials = [_eval(nodes, x[i]) for i in range(m)]  # O(m*n)
    result = 0.0
    mis = MultiIndexSet(m, n, p)
    while mis.next():  #  O(k_mnp)
        mi = mis.multi_index
        result += coefficients[mis.i] * np.prod(
            np.array([monomials[i][mi[i]] for i in range(m)], dtype=NP_FLOAT)
        )  # O(m)
    return result


@njit
def l2n(nodes: NP_ARRAY) -> NP_ARRAY:
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
def _dds(x, y) -> NP_ARRAY:
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
def n_dx(nodes: NP_ARRAY) -> NP_ARRAY:
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
def l_dx(nodes: NP_ARRAY) -> NP_ARRAY:
    """O(n^2)"""
    n = len(nodes)
    w = _baryentric(nodes)
    dx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dx[i][i] = np.sum(
                    [1 / (nodes[i] - nodes[_]) for _ in range(n) if _ != i]
                )
            else:
                dx[i][j] = (w[j] / w[i]) * 1 / (nodes[i] - nodes[j])
    return dx


@njit
def _baryentric(nodes: NP_ARRAY) -> NP_ARRAY:
    return [
        1 / np.prod([(nodes[__] - nodes[_]) for _ in range(len(nodes)) if _ != __])
        for __ in range(len(nodes))
    ]


@njit
def rmo(A: NP_ARRAY, mode: str = "lower") -> NP_ARRAY:
    if mode == "upper":
        return _rmo_upper(A)
    elif mode == "lower":
        return _rmo_lower(A)
    else:
        raise ValueError("The parameter mode should be either 'upper' or 'lower'.")


@njit
def _rmo_upper(A: NP_ARRAY) -> NP_ARRAY:
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
def _rmo_lower(A: NP_ARRAY) -> NP_ARRAY:
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


def unisolvent_nodes(nodes: NP_ARRAY, m: int, n: int, p: float) -> NP_ARRAY:
    classify(m, n, p, allow_infty=True)
    if p == np.infty:
        return np.flip(list(itertools.product(nodes, repeat=m)), axis=1)
    else:
        return _unisolvent_nodes(nodes, m, n, p)


@njit
def _unisolvent_nodes(nodes: NP_ARRAY, m: int, n: int, p: float) -> NP_ARRAY:
    """O(|A_{m, n, p}| + ... + |A_{1, n, p}|)"""
    nodes = np.asarray(nodes).astype(NP_FLOAT)
    memory_allocation = _memory_allocation(m, n, p)
    unisolvent_nodes = np.zeros((memory_allocation, m))
    mis = MultiIndexSet(m, n, p)
    while mis.next():
        mi = mis.multi_index
        unisolvent_nodes[mis.i] = [nodes[mi[_]] for _ in range(m)]
    return unisolvent_nodes[: mis.i + 1]


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
        return int(np.ceil((fac1 * (n + 2) * np.math.gamma(1 + 1 / p)) ** m * fac2))
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
def unisolvent_nodes_1d(n: int, nodes: callable) -> NP_ARRAY:
    nodes_n = nodes(n)
    return nodes_n[leja_order(nodes_n)]


@njit
def cheb(n: int) -> NP_ARRAY:
    """O(n)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=NP_FLOAT)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=NP_FLOAT)
    return np.cos(np.arange(n, dtype=NP_FLOAT) * np.pi / (n - 1))


@njit
def tiling(m: int, n: int, p: NP_FLOAT) -> NP_ARRAY:
    classify(m, n, p)
    return _tiling(m, n, p)


@njit
def _tiling(m: int, n: int, p: float) -> NP_ARRAY:
    """O(k_m,n,p)"""
    mis = MultiIndexSet(m, n, p)
    memory_allocation = _memory_allocation(m - 1, n, p)
    tiling = np.zeros(memory_allocation, dtype=NP_INT)

    while mis.next():
        if mis.k > 0:
            tiling[mis.l] = mis.k
    return tiling[: mis.l]


@njit
def leja_order(nodes: NP_ARRAY) -> NP_ARRAY:
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
def permutation_maximal(m: int, n: int, i: int) -> NP_ARRAY:
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
def permutation(T: NP_ARRAY, i: int) -> NP_ARRAY:
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
def transposition(T) -> NP_ARRAY:
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
def apply_permutation(P: NP_ARRAY, x: NP_ARRAY, invert: bool = False) -> NP_ARRAY:
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
def rmo_transpose(rmo: NP_ARRAY) -> NP_ARRAY:
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
def concatenate_arrays(chunk_dot) -> NP_ARRAY:
    total_length = sum([len(arr) for arr in chunk_dot])
    result = np.empty(total_length, dtype=chunk_dot[0].dtype)
    start = 0
    for arr in chunk_dot:
        end = start + arr.size
        result[start:end] = arr
        start = end
    return result


@njit
def reduceat(array, split_indices) -> NP_ARRAY:
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
