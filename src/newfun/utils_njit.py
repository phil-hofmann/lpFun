import numpy as np
from numba import njit
from newfun.core import MultiIndexSet
from newfun import NP_INT, NP_FLOAT, NP_ARRAY

# TODO parallelize these loops

@njit
def n2l_subroutine(nodes: NP_ARRAY) -> NP_ARRAY:
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
    for i in range(1, n):
        monomials[i] *= monomials[i - 1] * (x - nodes[i - 1])
    return monomials


@njit
def l2n_subroutine(nodes: NP_ARRAY) -> NP_ARRAY:
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
def _dds(x, y):
    """O(n^2)"""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    dd = np.zeros(n)
    dd[0] = y[0]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            y[j] = (y[j] - y[j - 1]) / (x[j] - x[j - i])
        dd[i] = y[i]
    return dd


@njit
def unisolvent_nodes_subroutine(nodes: NP_ARRAY, m: int, n: int, p: float) -> NP_ARRAY:
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
def leja_order(nodes: NP_ARRAY):
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
def cheb_n(n: int) -> NP_ARRAY:
    """O(n)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=NP_FLOAT)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=NP_FLOAT)
    return np.cos(np.arange(n, dtype=NP_FLOAT) * np.pi / (n - 1))


@njit
def tiling_subroutine(m: int, n: int, p: float) -> NP_ARRAY:
    """O(|A_{m, n, p}| + ... + |A_{1, n, p}|)"""
    mis = MultiIndexSet(m, n, p)
    memory_allocation = _memory_allocation(m - 1, n, p)
    tiling = np.zeros(memory_allocation, dtype=NP_INT)

    while mis.next():
        if mis.k > 0:
            tiling[mis.l] = mis.k
    return tiling[: mis.l]


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
    if m < 0 or m > n:
        return 0
    result = 1
    for i in range(min(m, n - m)):
        result = result * (n - i) // (i + 1)
    return result


@njit
def rmo_subroutine(A: NP_ARRAY):
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

@njit
def concatenate_arrays(chunk_dot):
    total_length = sum([len(arr) for arr in chunk_dot])
    result = np.empty(total_length, dtype=chunk_dot[0].dtype)
    start = 0
    for arr in chunk_dot:
        end = start + arr.size
        result[start:end] = arr
        start = end
    return result

@njit
def reduceat(array, split_indices):
    sums = np.zeros(len(split_indices) - 1, dtype=array.dtype)
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        if start >= len(array):
            break
        end = min(end, len(array))
        sums[i] = np.sum(array[start:end])
    return sums[:i]
