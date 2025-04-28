import numpy as np
from typing import Tuple
from numba import njit
from itertools import product
from lpfun import NP_INT, NP_FLOAT
from lpfun.core.utils import classify, apply_permutation, memory_estimate

"""
- This file is not parallelized.
"""


@njit
def _lp_set(
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    memory = memory_estimate(m, n, p)
    multi_index_set, multi_index, multi_index_p, num_p = (
        np.zeros((memory, m), dtype=NP_INT),
        np.zeros(m, dtype=NP_INT),
        np.zeros(m, dtype=NP_FLOAT),
        np.arange(0, n + 1).astype(NP_FLOAT) ** p,
    )
    sum_multi_index_p, n_p, i, j = 0.0, n**p, 0, 0
    while True:
        while True:
            if j >= m:
                return multi_index_set[: i + 1]
            elif multi_index[j] < n:
                sum_multi_index_p = sum_multi_index_p - multi_index_p[j]
                multi_index[j] += 1
                multi_index_p[j] = num_p[multi_index[j]]
                sum_multi_index_p = sum_multi_index_p + multi_index_p[j]
                break
            else:
                sum_multi_index_p = sum_multi_index_p - multi_index_p[j]
                multi_index[j] = 0
                multi_index_p[j] = 0
                j += 1
        if sum_multi_index_p <= n_p:
            i += 1
            j = 0
            multi_index_set[i] = multi_index
        else:
            sum_multi_index_p = np.sum(multi_index**p)
            if sum_multi_index_p <= n_p:
                i += 1
                j = 0
                multi_index_set[i] = multi_index
            else:
                sum_multi_index_p = sum_multi_index_p - multi_index_p[j]
                multi_index[j] = 0
                multi_index_p[j] = 0
                j += 1


def lp_set(
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    m, n, p = (
        int(m),
        int(n),
        float(p),
    )
    classify(m, n, p)
    ###
    if m == 1:
        return np.array(range(n + 1), dtype=NP_INT).reshape(-1, 1)
    elif p == np.inf:
        return np.flip(
            np.asarray(list(product(range(n + 1), repeat=m))).astype(np.int64), axis=1
        )
    return _lp_set(m, n, p)


# tube projection


@njit
def _lp_tube(A: np.ndarray) -> np.ndarray:
    N, A0 = len(A), A[:, 0]
    T = np.zeros(len(A), dtype=NP_INT)
    i, j = 1, 0
    for k in range(1, N):
        if A0[k] > 0:
            i += 1
        else:
            T[j] = i
            j += 1
            i = 1
    T[j] = i
    return T[: j + 1]


def lp_tube(A: np.ndarray, m: int, n: int, p: float) -> np.ndarray:
    A, m, n, p = (
        np.asarray(A).astype(NP_INT),
        int(m),
        int(n),
        float(p),
    )
    classify(m, n, p)
    if m == 1:
        return np.array([n + 1], dtype=NP_INT)
    elif p == np.inf:
        return np.array([n + 1] * (n + 1) ** (m - 1), dtype=NP_INT)
    return _lp_tube(A)


# permutations


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
def transposition(T: np.ndarray) -> np.ndarray:
    N, n = np.sum(T), np.max(T)
    permutation_vector = np.zeros(N, dtype=NP_INT)
    current_position = 0
    for j in range(n):
        for i in range(len(T)):
            if j < T[i]:
                permutation_vector[current_position] = np.sum(T[:i]) + j
                current_position += 1
    return permutation_vector


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


# ordinal embedding


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
def _plusplus(
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
            i, d, max_j = _plusplus(m, i, d, T, e_T)
            max_j_prime = -1
            while max_j_prime < max_j:
                k_prime = k_prime + T_prime[i_prime[0]]
                i_prime, d_prime, max_j_prime = _plusplus(
                    m, i_prime, d_prime, T_prime, e_T_prime
                )
        else:
            raise ValueError("Undetermined condition encountered in the algorithm.")
    return phi
