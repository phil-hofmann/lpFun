import numpy as np
import numba as nb
from math import gamma
from typing import Tuple
from itertools import product

from lpfun import CACHE
from lpfun.utils import apply_permutation, binomial


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


"""
- Nothing in this file is parallelized.
"""


@njit
def memory_estimate(m: int, n: int, p: float) -> int:
    """O(1)"""
    if p <= 1.0:
        singular = 1 + m * n
        if m < n:
            return int(singular * (1 - p) + binomial(n + m, m) * p)
        else:
            return int(singular * (1 - p) + binomial(n + m, n) * p)
    elif p <= 2.0:
        fac1 = (p * np.e / m) ** (1 / p)
        fac2 = np.sqrt(p / (2 * np.pi * m))
        return int(np.ceil((fac1 * (n + 2) * gamma(1 + 1 / p)) ** m * fac2))
    else:
        return int((n + 1) ** m)


# set


@njit
def _lp_set(
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    """O(m|A|)"""
    memory = memory_estimate(m, n, p)
    index_set, index, index_p, num_p = (
        np.zeros((memory, m), dtype=np.int64),
        np.zeros(m, dtype=np.int64),
        np.zeros(m, dtype=np.float64),
        np.arange(0, n + 1).astype(np.float64) ** p,
    )
    sum_index_p, n_p, i, j = 0.0, n**p, 0, m - 1
    while True:
        while True:
            if j < 0:
                return index_set[: i + 1]
            elif index[j] < n:
                sum_index_p -= index_p[j]
                index[j] += 1
                index_p[j] = num_p[index[j]]
                sum_index_p += index_p[j]
                break
            else:
                sum_index_p -= index_p[j]
                index[j] = 0
                index_p[j] = 0
                j -= 1
        if sum_index_p <= n_p:
            i += 1
            j = m - 1
            index_set[i] = index
        else:
            sum_index_p = np.sum(index**p)
            if sum_index_p <= n_p:
                i += 1
                j = m - 1
                index_set[i] = index
            else:
                sum_index_p -= index_p[j]
                index[j] = 0
                index_p[j] = 0
                j -= 1


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
    if m == 1:
        return np.array(range(n + 1), dtype=np.int64).reshape(-1, 1)
    elif p == np.inf:
        return np.asarray(list(product(range(n + 1), repeat=m))).astype(np.int64)
    return _lp_set(m, n, p)


# tube size projections


@njit
def _tube(A: np.ndarray) -> np.ndarray:
    """O(|A|)"""
    N, A0 = len(A), A[:, -1]
    T = np.zeros(len(A), dtype=np.int64)
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
        np.asarray(A).astype(np.int64),
        int(m),
        int(n),
        float(p),
    )
    if m == 1:
        return np.array([n + 1], dtype=np.int64)
    elif p == np.inf:
        return np.array([n + 1] * (n + 1) ** (m - 1), dtype=np.int64)
    return _tube(A)


# permutations


@njit
def transposition(T: np.ndarray) -> np.ndarray:
    """O(|A|)"""
    N = len(T)
    if N == 0:
        return np.empty(0, dtype=np.int64)

    total = np.sum(T)
    M = np.max(T)
    tau = np.empty(total, dtype=np.int64)

    offsets = np.empty(N, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(T[:-1])

    # Buckets
    bucket_head = -np.ones(M + 1, dtype=np.int64)
    bucket_next = -np.ones(N, dtype=np.int64)

    for l in range(N):
        k = T[l]
        bucket_next[l] = bucket_head[k]
        bucket_head[k] = l

    # Active ordered linked list
    prev_active = np.empty(N, dtype=np.int64)
    next_active = np.empty(N, dtype=np.int64)

    for l in range(N):
        prev_active[l] = l - 1
        next_active[l] = l + 1

    next_active[N - 1] = -1
    head = 0

    j = 0

    # Level-wise traversal
    for k in range(1, M + 1):
        # Process all currently active fibers.
        l = head
        while l != -1:
            tau[j] = offsets[l] + (k - 1)
            j += 1
            l = next_active[l]

        # Remove fibers whose size is exactly k.
        l = bucket_head[k]
        while l != -1:
            nxt_bucket = bucket_next[l]

            p = prev_active[l]
            q = next_active[l]

            if p != -1:
                next_active[p] = q
            else:
                head = q

            if q != -1:
                prev_active[q] = p

            l = nxt_bucket

    return tau


@njit
def permutations(
    N: int,
    m: int,
    pi: np.ndarray,
) -> np.ndarray:
    """O(m|A|)"""
    perm = np.empty((m, N), dtype=np.int64)
    sig = np.arange(N)
    for i in range(m):
        perm[i] = sig
        sig = apply_permutation(sig, pi, invert=True)
    return perm[::-1]


@njit
def permutations_max(
    m: int,
    n: int,
) -> np.ndarray:
    """O(m(n+1)^m)"""
    ###
    perm = np.empty((m, (n + 1) ** m), dtype=np.int64)
    for i in range(m):
        iter_len = (n + 1) ** (m - i)
        step_len = (n + 1) ** i
        ###
        j = 0
        for k in range(step_len):
            ###
            for l in range(iter_len):
                perm[i, j] = l * step_len + k
                j += 1
            ###
        ###
    ###
    return perm


# rank embedding


@njit
def entropy(T: np.ndarray) -> np.ndarray:
    cs_T = np.cumsum(T)
    e_T = np.array([cs_T[-1]], dtype=np.int64)
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
def rank_embedding(
    m: int,
    T: np.ndarray,
    T_prime: np.ndarray,
) -> np.ndarray:
    """O(|A'|)"""
    e_T = entropy(T)
    e_T_prime = entropy(T_prime)
    N = np.sum(T)
    phi = np.zeros(N, dtype=np.int64)
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
