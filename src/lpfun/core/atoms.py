import numpy as np
from numba import njit, prange
from lpfun import NP_FLOAT, NP_INT
from lpfun.utils import reduceat, ordinal_embedding, entropy

"""
Comments:
    - This module contains numba jit-compiled functions for the transformation of a vector by a matrix.
    - The functions are divided into the following categories: 1d, maximal, 2d, 3d and md transformations.
    - The functions are used in the transform methods in the molecules.py module.
"""

# TODO:
# - Replace np.sums(...) wherever possible by volumes etc.

# 1d


@njit
def transform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(n^2)"""
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = len(x)
    ### indexing: j, k
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        j_next = j + k + 1
        dot[k] = (x[k] - np.sum(L[j : j_next - 1] * dot[:k])) / L[j_next - 1]
        j = j_next
    ###
    return dot


@njit
def transform_ut_1d(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(n^2)"""
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = len(x)
    ### indexing: j, k
    ###
    dot, j = np.zeros_like(x), n * (n + 1) // 2
    for k in range(n):
        k_prime = n - k - 1
        j_next = j - k - 1
        dot[k_prime] = (x[k_prime] - np.sum(U[j_next:j] * dot[k_prime:])) / U[j_next]
        j = j_next
    ###
    return dot


@njit
def itransform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(n^2)"""
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    if parallel:
        return _itransform_lt_1d_parallel(L, x)
    else:
        return _itransform_lt_1d(L, x)


@njit
def _itransform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    n = len(x)
    ### indexing: j, k
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
        j = j_next
    ###
    return dot


@njit(parallel=True)
def _itransform_lt_1d_parallel(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    n = len(x)
    ### indexing: k > j
    ### NOTE: loop runs in parallel
    dot = np.zeros_like(x)
    for k in prange(n):
        j = (k * (k + 1)) // 2
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
    ###
    return dot


@njit
def itransform_ut_1d(
    U: np.ndarray,
    x: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(n^2)"""
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    if parallel:
        return _itransform_ut_1d_parallel(U, x)
    else:
        return _itransform_ut_1d(U, x)


@njit
def _itransform_ut_1d(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    ###
    n = len(x)
    ### indexing: j, k
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        k_prime = n - k - 1
        j_next = j + k_prime + 1
        dot[k] = np.sum(U[j:j_next] * x[k:])
        j = j_next
    ###
    return dot


@njit(parallel=True)
def _itransform_ut_1d_parallel(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    ###
    n = len(x)
    ### indexing: k > j
    ### NOTE: loop runs in parallel
    dot = np.zeros_like(x)
    for k in prange(n):
        k_prime = n - k - 1
        j = k * n - k * (k - 1) // 2
        j_next = j + k_prime + 1
        dot[k] = np.sum(U[j:j_next] * x[k:])
    ###
    return dot


# maximal


@njit
def transform_lt_max(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r > j, k > l
    ###
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ###
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ###
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos_k = pos_k + s
                j_next = j + k + 1
                ###
                dot_row, pos_l = np.zeros(s, dtype=NP_FLOAT), 0
                for l in range(j, j_next - 1):
                    next_pos_l = pos_l + s
                    dot_row += L[l] * dot_block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                next_pos_l = pos_l + s
                dot_block[pos_k:next_pos_k] = (block[pos_l:next_pos_l] - dot_row) / L[
                    j_next - 1
                ]
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos:next_pos] = dot_block
            pos = next_pos
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit
def transform_ut_max(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(U)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s,r > j, k > l
    ###
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ###
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ###
            dot_block, pos_k, j = (
                np.zeros((s_next), dtype=NP_FLOAT),
                s_next,
                n * (n + 1) // 2,
            )
            for k in range(n):
                next_pos_k, j_next = pos_k - s, j - k - 1
                ###
                dot_row, pos_l = np.zeros(s, dtype=NP_FLOAT), s_next
                for l in range(j - 1, j_next, -1):
                    next_pos_l = pos_l - s
                    dot_row += U[l] * dot_block[next_pos_l:pos_l]
                    pos_l = next_pos_l
                next_pos_l = pos_l - s
                dot_block[next_pos_k:pos_k] = (block[next_pos_l:pos_l] - dot_row) / U[
                    j_next
                ]
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos:next_pos] = dot_block
            pos = next_pos
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit
def itransform_lt_max(
    L: np.ndarray,
    x: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nmn)"""
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    if parallel:
        return _itransform_lt_max_parallel(L, x)
    else:
        return _itransform_lt_max(L, x)


@njit
def _itransform_lt_max(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r > j, k > l
    ###
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ###
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ###
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos_k, j_next = pos_k + s, j + k + 1
                ###
                pos_l = 0
                for l in range(j, j_next):
                    next_pos_l = pos_l + s
                    dot_block[pos_k:next_pos_k] += L[l] * block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos:next_pos] = dot_block
            pos = next_pos
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit(parallel=True)
def _itransform_lt_max_parallel(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r > i > j, k > l
    ### NOTE: loop not parallelizable, runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs in parallel
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos_k, j_next = pos_k + s, j + k + 1
                ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
                pos_l = 0
                for l in range(j, j_next):
                    next_pos_l = pos_l + s
                    dot_block[pos_k:next_pos_k] += L[l] * block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos_i:next_pos_i] = dot_block
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit
def itransform_ut_max(
    U: np.ndarray,
    x: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nmn)"""
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    if parallel:
        return _itransform_ut_max_parallel(U, x)
    else:
        return _itransform_ut_max(U, x)


@njit
def _itransform_ut_max(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(U)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r > j, k > l
    ###
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ###
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ###
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                k_prime = n - k - 1
                next_pos_k, j_next = pos_k + s, j + k_prime + 1
                ###
                pos_l = int(pos_k)
                for l in range(j, j_next):
                    next_pos_l = pos_l + s
                    dot_block[pos_k:next_pos_k] += U[l] * block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos:next_pos] = dot_block
            pos = next_pos
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit(parallel=True)
def _itransform_ut_max_parallel(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(U)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r > i > j, k > l
    ### NOTE: loop not parallelizable, runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs in parallel
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                k_prime = n - k - 1
                next_pos_k, j_next = pos_k + s, j + k_prime + 1
                ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
                pos_l = pos_k
                for l in range(j, j_next):
                    next_pos_l = pos_l + s
                    dot_block[pos_k:next_pos_k] += U[l] * block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                ###
                pos_k, j = next_pos_k, j_next
            ###
            dot[pos_i:next_pos_i] = dot_block
        ###
        s, r = s_next, r_next
    ###
    return dot


@njit
def dtransform_max(
    L: np.ndarray,
    x: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nn)"""
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    if parallel:
        return _dtransform_max_parallel(L, x)
    else:
        return _dtransform_max(L, x)


@njit
def _dtransform_max(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    ### indexing: s > j, k
    ###
    pos, s, dot = 0, N // n, np.copy(x)
    for _ in range(s):
        next_pos = pos + n
        block = dot[pos:next_pos]
        ###
        dot_block, j = np.zeros(n, dtype=NP_FLOAT), 0
        for k in range(n):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot[pos:next_pos] = dot_block
        pos = next_pos
    ###
    return dot


@njit(parallel=True)
def _dtransform_max_parallel(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    ###
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    ### indexing: s > i > j, k
    ### NOTE: loop runs in parallel
    s, dot = N // n, np.copy(x)
    for i in prange(s):
        pos = i * n
        next_pos = pos + n
        block = dot[pos:next_pos]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(n, dtype=NP_FLOAT), 0
        for k in range(n):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot[pos:next_pos] = dot_block
    ###
    return dot


# 2d


@njit
def transform_lt_2d(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2Nn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N_1 = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        chunk = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = (chunk[k] - np.sum(L[j : j_next - 1] * dot_block[:k])) / L[
                j_next - 1
            ]
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, pos_i, j = np.zeros_like(x), 0, 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        ###
        pos_k, dot_row = 0, np.zeros(t_i, dtype=NP_FLOAT)
        for k in range(i):
            t_k = T[k]
            next_pos_k = pos_k + t_k
            dot_row += L[j] * dot_2d[pos_k : pos_k + t_i]
            pos_k = next_pos_k
            j += 1
        dot_2d[pos_i:next_pos_i] = (dot_1d[pos_i:next_pos_i] - dot_row) / L[j]
        j += 1
        ###
        pos_i = next_pos_i
    ###
    return dot_2d


@njit
def transform_ut_2d(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2Nn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N_0, N_1 = (
        np.sum(T),
        len(T),
    )
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_1 - t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=NP_FLOAT),
            t_i * N_1 - t_i * (t_i - 1) // 2 - delta,
        )
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1 - delta
            dot_block[k_prime] = (
                block[k_prime] - np.sum(U[j_next + delta : j] * dot_block[k_prime:])
            ) / U[j_next + delta]
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, pos_i, j = np.zeros_like(x), N_0, N_1 * (N_1 + 1) // 2
    for i in range(N_1):
        i_prime = N_1 - i - 1
        t_i = T[i_prime]
        next_pos_i = pos_i - t_i
        ###
        pos_k, dot_row = N_0, np.zeros(t_i, dtype=NP_FLOAT)
        for k in range(i):
            j -= 1
            k_prime = N_1 - k - 1
            t_k = T[k_prime]
            next_pos_k = pos_k - t_k
            dot_row[:t_k] += U[j] * dot_2d[next_pos_k:pos_k]
            pos_k = next_pos_k
        j -= 1
        dot_2d[next_pos_i:pos_i] = (dot_1d[next_pos_i:pos_i] - dot_row) / U[j]
        ###
        pos_i = next_pos_i
    ###
    return dot_2d


@njit
def itransform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(2Nn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _itransform_lt_2d_parallel(L, x, T)
    else:
        return _itransform_lt_2d(L, x, T)


@njit
def _itransform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N_1 = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        block = x[pos_i:next_pos_i]
        ### NOTE: loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, j, pos_i = np.zeros_like(x), 0, 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        ###
        pos_k = 0
        for k in range(i + 1):
            t_k = T[k]
            next_pos_k = pos_k + t_k
            dot_2d[pos_i:next_pos_i] += L[j] * dot_1d[pos_k : pos_k + t_i]
            pos_k = next_pos_k
            j += 1
        ###
        pos_i = next_pos_i
    ###
    return dot_2d


@njit(parallel=True)
def _itransform_lt_2d_parallel(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    zero = np.array([0], dtype=NP_INT)
    N_1, cs_T = (
        len(T),
        np.concatenate((zero, np.cumsum(T))),
    )
    ### 1d
    ### indexing: i > j, k
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        chunk = x[pos_i:next_pos_i]
        ### caution -- possible overhead or numerical instability
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k
    dot_2d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ### caution -- possible overhead or numerical instability
        pos_k, j = 0, i * (i + 1) // 2
        for k in range(i + 1):
            t2 = T[k]
            next_pos_k = pos_k + t2
            dot_2d[pos_i:next_pos_i] += L[j] * dot_1d[pos_k : pos_k + t_i]
            pos_k = next_pos_k
            j += 1
        ###
    ###
    return dot_2d


@njit
def itransform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(2Nn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _itransform_ut_2d_parallel(U, x, T)
    else:
        return _itransform_ut_2d(U, x, T)


@njit
def _itransform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N_1 = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_1 - t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j + k_prime + 1 + delta
            dot_block[k] = np.sum(U[j : j_next - delta] * block[k:])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, pos_i, j = np.zeros_like(x), 0, 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        ###
        pos_k = int(pos_i)
        for k in range(N_1 - i):
            t_k = T[i + k]
            next_pos_k = pos_k + t_k
            dot_2d[pos_i : pos_i + t_k] += U[j] * dot_1d[pos_k:next_pos_k]
            pos_k = next_pos_k
            j += 1
        ###
        pos_i = next_pos_i
    ###
    return dot_2d


@njit(parallel=True)
def _itransform_ut_2d_parallel(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ####
    zero = np.array([0], dtype=NP_INT)
    N_1, cs_T = (
        len(T),
        np.concatenate((zero, np.cumsum(T))),
    )
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, delta = x[pos_i:next_pos_i], N_1 - t_i
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j + k_prime + 1 + delta
            dot_block[k] = np.sum(U[j : j_next - delta] * block[k:])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    dot_2d = np.zeros_like(x)
    for i in prange(N_1):
        pos_i = cs_T[i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos_k, j = int(pos_i), i * N_1 - i * (i - 1) // 2
        for k in range(N_1 - i):
            t_k = T[i + k]
            next_pos_k = pos_k + t_k
            dot_2d[pos_i : pos_i + t_k] += U[j] * dot_1d[pos_k:next_pos_k]
            pos_k = next_pos_k
            j += 1
        ###
    ###
    return dot_2d


# 3d


@njit
def transform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N_1, N_2 = len(T), T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = (block[k] - np.sum(L[j : j_next - 1] * dot_block[:k])) / L[
                j_next - 1
            ]
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d, V_2, pos_i, vol_i = np.zeros_like(x), np.zeros(N_2, dtype=NP_INT), 0, 0
    for i in range(N_2):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k, j = int(vol_i), 0, 0
        for k in range(t_i):
            t_k = sub_t_i[k]
            next_pos_k = pos_k + t_k
            ###
            pos_l, dot_row = int(vol_i), np.zeros(t_k, dtype=NP_FLOAT)
            for l in range(k):
                t_l = sub_t_i[l]
                next_pos_l = pos_l + t_l
                dot_row += L[j] * dot_2d[pos_l : pos_l + t_k]
                pos_l = next_pos_l
                j += 1
            dot_2d[pos_k:next_pos_k] = (dot_1d[pos_k:next_pos_k] - dot_row) / L[j]
            j += 1
            ###
            pos_k = next_pos_k
            vol_k += t_k
        ###
        pos_i = next_pos_i
        vol_i += vol_k
        V_2[i] = vol_k
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, pos_i, vol_i, j = np.zeros_like(x), 0, 0, 0
    for i in range(N_2):
        t_i, v_i = T[i], V_2[i]
        next_pos_i, next_vol_i = pos_i + t_i, vol_i + v_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k, dot_row = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
        for k in range(i):
            t_k, v_k = T[k], V_2[k]
            next_pos_k, next_vol_k = pos_k + t_k, vol_k + v_k
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_3d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_i):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1:next_pos_l_1] = block[pos_l_2 : pos_l_2 + t_l_1]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_row += L[j] * sub
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
            j += 1
        dot_3d[vol_i:next_vol_i] = (dot_2d[vol_i:next_vol_i] - dot_row) / L[j]
        j += 1
        ###
        pos_i, vol_i = next_pos_i, next_vol_i
    ###
    return dot_3d


@njit
def transform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N_0, N_1, N_2 = np.sum(T), len(T), T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_2 - t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=NP_FLOAT),
            t_i * N_2 - t_i * (t_i - 1) // 2 - delta,
        )
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1 - delta
            dot_block[k_prime] = (
                block[k_prime] - np.sum(U[j_next + delta : j] * dot_block[k_prime:])
            ) / U[j_next + delta]
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d, V_2, pos_i, vol_i = np.zeros_like(x), np.zeros(N_2, dtype=NP_INT), 0, 0
    for i in range(N_2):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_2 - t_i
        sub_t_i = T[pos_i:next_pos_i]
        sub_vol_i = np.sum(sub_t_i)
        vol_i += sub_vol_i
        V_2[i] = sub_vol_i
        ###
        pos_k, j = (vol_i, t_i * N_2 - t_i * (t_i - 1) // 2 - delta)
        for k in range(t_i):
            k_prime = t_i - k - 1
            t_k = sub_t_i[k_prime]
            next_pos_k = pos_k - t_k
            ###
            pos_l, dot_row = int(vol_i), np.zeros(t_k, dtype=NP_FLOAT)
            for l in range(k):
                j -= 1
                l_prime = t_i - l - 1
                t_l = sub_t_i[l_prime]
                next_pos_l = pos_l - t_l
                dot_row[:t_l] += U[j] * dot_2d[next_pos_l:pos_l]
                pos_l = next_pos_l
            j -= 1
            dot_2d[next_pos_k:pos_k] = (dot_1d[next_pos_k:pos_k] - dot_row) / U[j]
            j -= delta
            ###
            pos_k = next_pos_k
        ###
        pos_i = next_pos_i
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, pos_i, vol_i, j = np.zeros_like(x), int(V_2[0]), N_0, N_2 * (N_2 + 1) // 2
    for i in range(N_2):
        i_prime = N_2 - i - 1
        t_i, v_i = T[i_prime], V_2[i_prime]
        next_pos_i, next_vol_i = pos_i - t_i, vol_i - v_i
        sub_t_i = T[next_pos_i:pos_i]
        ###
        pos_k, vol_k, dot_row = int(V_2[0]), N_0, np.zeros(v_i, dtype=NP_FLOAT)
        for k in range(i):
            j -= 1
            k_prime = N_2 - k - 1
            t_k, v_k = T[k_prime], V_2[k_prime]
            next_pos_k, next_vol_k = pos_k - t_k, vol_k - v_k
            sub_t2 = T[next_pos_k:pos_k]
            block = dot_3d[next_vol_k:vol_k]
            ###
            pos_l_1, pos_l_2, ext = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_k):
                t_l_1, t_l_2 = sub_t_i[l], sub_t2[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                ext[pos_l_1 : pos_l_1 + t_l_2] = block[pos_l_2:next_pos_l_2]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_row += U[j] * ext
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
        j -= 1
        dot_3d[next_vol_i:vol_i] = (dot_2d[next_vol_i:vol_i] - dot_row) / U[j]
        ###
        pos_i, vol_i = next_pos_i, next_vol_i
    ###
    return dot_3d


@njit
def itransform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(3Nn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _itransform_lt_3d_parallel(L, x, T)
    else:
        return _itransform_lt_3d(L, x, T)


@njit
def _itransform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ##
    N_1, N_2 = (
        len(T),
        T[0],
    )
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d, V_2, pos_i, vol_i = np.zeros_like(x), np.zeros(N_2, dtype=NP_INT), 0, 0
    for i in range(N_2):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k, j = int(vol_i), 0, 0
        for k in range(t_i):
            t_k = sub_t_i[k]
            next_pos_k = pos_k + t_k
            pos_l = int(vol_i)
            for l in range(k + 1):
                t_l = sub_t_i[l]
                next_pos_l = pos_l + t_l
                dot_2d[pos_k:next_pos_k] += L[j] * dot_1d[pos_l : pos_l + t_k]
                pos_l = next_pos_l
                j += 1
            pos_k = next_pos_k
            vol_k += t_k
        ###
        pos_i = next_pos_i
        vol_i += vol_k
        V_2[i] = vol_k
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, pos_i, vol_i, j = np.zeros_like(x), 0, 0, 0
    for i in range(N_2):
        t_i, v_i = T[i], V_2[i]
        next_pos_i, next_vol_i = pos_i + t_i, vol_i + v_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k = 0, 0
        for k in range(i + 1):
            t_k, v_k = T[k], V_2[k]
            next_pos_k, next_vol_k = pos_k + t_k, vol_k + v_k
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_2d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_i):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1:next_pos_l_1] = block[pos_l_2 : pos_l_2 + t_l_1]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_3d[vol_i:next_vol_i] += L[j] * sub
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
            j += 1
        ###
        pos_i, vol_i = next_pos_i, next_vol_i
    ###
    return dot_3d


@njit(parallel=True)
def _itransform_lt_3d_parallel(
    L: np.ndarray, x: np.ndarray, T: np.ndarray
) -> np.ndarray:
    ###
    N_1, N_2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    V_2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N_2)], dtype=NP_INT)
    cs_V_2 = np.concatenate((np.array([0]), np.cumsum(V_2)))
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ### NOTE: loop runs in parallel
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, vol_i, next_pos_i = T[i], cs_T[i], cs_V_2[i], cs_T[i + 1]
        sub_t1 = T[pos_i:next_pos_i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos_k, _vol1, j = vol_i, 0, 0
        for k in range(t_i):
            t_k = sub_t1[k]
            next_pos_k = pos_k + t_k
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            pos_l = int(vol_i)
            for l in range(k + 1):
                t_l = sub_t1[l]
                next_pos_l = pos_l + t_l
                dot_2d[pos_k:next_pos_k] += L[j] * dot_1d[pos_l : pos_l + t_k]
                pos_l = next_pos_l
                j += 1
            pos_k = next_pos_k
            _vol1 += t_k
        ###
    ###
    ### 3d
    ### indexing: i, j > k > l
    ### NOTE: loop runs in parallel
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * (i + 1) // 2
        t_i, v_i = T[i], V_2[i]
        pos_i, vol_i = cs_T[i], cs_V_2[i]
        next_pos_i, next_vol_i = cs_T[i + 1], cs_V_2[i + 1]
        sub_t_i = T[pos_i:next_pos_i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos_k, vol_k = 0, 0
        for k in range(i + 1):
            t_k, v_k = T[k], V_2[k]
            next_pos_k, next_vol_k = pos_k + t_k, vol_k + v_k
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_2d[vol_k:next_vol_k]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            pos_l_1, _pos2, sub = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_i):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, _pos2 + t_l_2
                sub[pos_l_1:next_pos_l_1] = block[_pos2 : _pos2 + t_l_1]
                pos_l_1, _pos2 = next_pos_l_1, next_pos_l_2
            dot_3d[vol_i:next_vol_i] += L[j] * sub
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
            j += 1
        ###
    ###
    return dot_3d


@njit
def itransform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(3Nn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _itransform_ut_3d_parallel(U, x, T)
    else:
        return _itransform_ut_3d(U, x, T)


@njit
def _itransform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N_1, N_2 = (
        len(T),
        T[0],
    )
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_2 - t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j + k_prime + 1
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
            j = j_next + delta
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
        pos_i = next_pos_i
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d, V_2, pos_i, vol_i = np.zeros_like(x), np.zeros(N_2, dtype=NP_INT), 0, 0
    for i in range(N_2):
        t_i = T[i]
        next_pos_i, delta = pos_i + t_i, N_2 - t_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k, j = int(vol_i), 0, 0
        for k in range(t_i):
            t_k = sub_t_i[k]
            next_pos_k = pos_k + t_k
            ###
            pos_l = int(vol_i + vol_k)
            for l in range(t_i - k):
                t_l = sub_t_i[k + l]
                next_pos_l = pos_l + t_l
                dot_2d[pos_k : pos_k + t_l] += U[j] * dot_1d[pos_l:next_pos_l]
                pos_l = next_pos_l
                j += 1
            j += delta
            ###
            pos_k = next_pos_k
            vol_k += t_k
        ###
        pos_i, vol_i, V_2[i] = next_pos_i, pos_k, vol_k
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, pos_i, vol_i, j = np.zeros_like(x), 0, 0, 0
    for i in range(N_2):
        t_i, v_i = T[i], V_2[i]
        next_pos_i, next_vol_i = pos_i + t_i, vol_i + v_i
        sub_t_i = T[pos_i:next_pos_i]
        ###
        pos_k, vol_k = int(pos_i), int(vol_i)
        for k in range(N_2 - i):
            i_plus_k = i + k
            t_k, v_k = T[i_plus_k], V_2[i_plus_k]
            next_pos_k, next_vol_k = pos_k + t_k, vol_k + v_k
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_2d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_k):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1 : pos_l_1 + t_l_2] = block[pos_l_2:next_pos_l_2]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_3d[vol_i:next_vol_i] += U[j] * sub
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
            j += 1
        ###
        pos_i, vol_i = next_pos_i, next_vol_i
    ###
    return dot_3d


@njit(parallel=True)
def _itransform_ut_3d_parallel(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N_1, N_2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    V_2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N_2)], dtype=NP_INT)
    cs_V2 = np.concatenate((np.array([0]), np.cumsum(V_2)))
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        chunk, delta = x[pos_i:next_pos_i], N_2 - t_i
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j + k_prime + 1 + delta
            dot_block[k] = np.sum(U[j : j_next - delta] * chunk[k:])
            j = j_next
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ### NOTE: loop runs in parallel
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, vol_i, next_pos_i = T[i], cs_T[i], cs_V2[i], cs_T[i + 1]
        sub_t_i, delta = T[pos_i:next_pos_i], N_2 - t_i
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos_k, vol_k, j = int(vol_i), 0, 0
        for k in range(t_i):
            t_k = sub_t_i[k]
            next_pos_k = pos_k + t_k
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            pos_l = int(vol_i + vol_k)
            for l in range(t_i - k):
                t_l = sub_t_i[k + l]
                next_pos_l = pos_l + t_l
                dot_2d[pos_k : pos_k + t_l] += U[j] * dot_1d[pos_l:next_pos_l]
                pos_l = next_pos_l
                j += 1
            j += delta
            ###
            pos_k = next_pos_k
            vol_k += t_k
        ###
        vol_i += vol_k
        V_2[i] = vol_k
    ###
    ### 3d
    ### indexing: i, j > k > l
    ### NOTE: loop runs in parallel
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * N_2 - i * (i - 1) // 2
        v_i = V_2[i]
        pos_i, vol_i = cs_T[i], cs_V2[i]
        next_pos_i, next_vol_i = cs_T[i + 1], cs_V2[i + 1]
        sub_t_i = T[pos_i:next_pos_i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos_k, vol_k = int(pos_i), int(vol_i)
        for k in range(N_2 - i):
            i_plus_k = i + k
            t_k, v_k = T[i_plus_k], V_2[i_plus_k]
            next_pos_k, next_vol_k = pos_k + t_k, vol_k + v_k
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_2d[vol_k:next_vol_k]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=NP_FLOAT)
            for l in range(t_k):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1 : pos_l_1 + t_l_2] = block[pos_l_2:next_pos_l_2]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_3d[vol_i:next_vol_i] += U[j] * sub
            ###
            pos_k, vol_k = next_pos_k, next_vol_k
            j += 1
        ###
    ###
    return dot_3d


# md


@njit # NOTE refactor me!
def itransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nmn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _itransform_lt_md_parallel(L, x, T)
    else:
        return _itransform_lt_md(L, x, T)


@njit
def _itransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    zero = np.array([0], dtype=NP_INT)
    dot, cs_T, V0, e_T = (
        np.copy(x),
        np.concatenate((zero, np.cumsum(T))),
        np.copy(T),
        entropy(T),
    )
    m = len(e_T) - 1
    ### 1d: i,j,k
    ### NOTE: loop runs sequentially
    pos = 0
    for i in range(e_T[1]):
        slot = V0[i]
        next_pos = pos + slot
        chunk = dot[pos:next_pos]
        chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
        for k in range(slot):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        pos = next_pos
    ###
    ### md: h,i,j,k,l
    ### NOTE: no parallelization
    V1 = reduceat(V0, cs_T)
    for h in range(1, m):
        pos = 0
        cs_V0 = np.concatenate((zero, np.cumsum(V0)))
        ### Outer loop
        ### NOTE: loop runs sequentially
        for i in range(e_T[h + 1]):
            slot = V1[i]
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_V0 = V0[
                np.searchsorted(cs_V0, pos) : np.searchsorted(cs_V0, next_pos)
            ]
            len_chunk_V0 = len(chunk_V0)
            cs_chunk_V0 = np.concatenate((zero, np.cumsum(chunk_V0)))
            chunk_T = T[np.searchsorted(cs_T, pos) : np.searchsorted(cs_T, next_pos)]
            cs_chunk_T = np.concatenate((zero, np.cumsum(chunk_T)))
            chunk_V0_Ts = [
                chunk_T[
                    np.searchsorted(cs_chunk_T, cs_chunk_V0[g]) : np.searchsorted(
                        cs_chunk_T, cs_chunk_V0[g + 1]
                    )
                ]
                for g in range(len_chunk_V0)
            ]
            ### Inner loop
            # BEGIN LP DOT
            # NOTE: loop runs sequentially
            chunk_dot, j = [np.zeros((size), dtype=NP_FLOAT) for size in chunk_V0], 0
            for k in range(len_chunk_V0):
                j_next = j + k + 1
                follower = chunk_V0_Ts[k]
                chunk_pos = 0
                for l in range(k + 1):
                    a, leader = L[j + l], chunk_V0_Ts[l]
                    next_chunk_pos = chunk_pos + np.sum(leader)
                    sub = np.copy(chunk[chunk_pos:next_chunk_pos])
                    rc = ordinal_embedding(h, follower, leader)
                    sub_rc = sub[rc]
                    chunk_dot[k] += a * sub_rc
                    chunk_pos = next_chunk_pos
                j = j_next
            # END LP DOT
            # START CONCATENATE ARRAYS
            # NOTE: loop runs sequentially
            concat_chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
            for k in range(len_chunk_V0):
                j_next = j + chunk_V0[k]
                concat_chunk_dot[j:j_next] = np.copy(chunk_dot[k])
                j = j_next
            dot[pos:next_pos] = concat_chunk_dot
            # END CONCATENATE ARRAYS
            ###
            pos = next_pos
        V0 = np.copy(V1)
        V1 = reduceat(V1, cs_T)
        ###
    ###
    return dot


@njit(parallel=True)
def _itransform_lt_md_parallel(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    zero = np.array([0], dtype=NP_INT)
    dot, cs_T, V0, e_T = (
        np.copy(x),
        np.concatenate((zero, np.cumsum(T))),
        np.copy(T),
        entropy(T),
    )
    m = len(e_T) - 1
    ### 1d: i,j,k
    ### NOTE: loop runs in parallel
    cs_V0 = np.concatenate((zero, np.cumsum(V0)))
    for i in prange(e_T[1]):
        pos, slot = cs_V0[i], V0[i]
        next_pos = pos + slot
        chunk = dot[pos:next_pos]
        chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
        for k in range(slot):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
    ###
    ### md: h, i, j, k, l
    ### NOTE: no parallelization
    V1 = reduceat(V0, cs_T)
    for h in range(1, m):
        cs_V1 = np.concatenate((zero, np.cumsum(V1)))
        ### Outer loop
        ### NOTE: loop runs in parallel
        for i in prange(e_T[h + 1]):
            pos, slot = cs_V1[i], V1[i]
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_V0 = V0[
                np.searchsorted(cs_V0, pos) : np.searchsorted(cs_V0, next_pos)
            ]
            len_chunk_V0 = len(chunk_V0)
            cs_chunk_V0 = np.concatenate((zero, np.cumsum(chunk_V0)))
            chunk_T = T[np.searchsorted(cs_T, pos) : np.searchsorted(cs_T, next_pos)]
            cs_chunk_T = np.concatenate((zero, np.cumsum(chunk_T)))
            chunk_V0_Ts = [
                chunk_T[
                    np.searchsorted(cs_chunk_T, cs_chunk_V0[g]) : np.searchsorted(
                        cs_chunk_T, cs_chunk_V0[g + 1]
                    )
                ]
                for g in range(len_chunk_V0)
            ]
            ### Inner loop
            # BEGIN LP DOT
            # NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            chunk_dot, j = [np.zeros((size), dtype=NP_FLOAT) for size in chunk_V0], 0
            for k in range(len_chunk_V0):
                j_next = j + k + 1
                follower = chunk_V0_Ts[k]
                chunk_pos = 0
                for l in range(k + 1):
                    a, leader = L[j + l], chunk_V0_Ts[l]
                    next_chunk_pos = chunk_pos + np.sum(leader)
                    sub = np.copy(chunk[chunk_pos:next_chunk_pos])
                    rc = ordinal_embedding(h, follower, leader)
                    sub_rc = sub[rc]
                    chunk_dot[k] += a * sub_rc
                    chunk_pos = next_chunk_pos
                j = j_next
            # END LP DOT
            # START CONCATENATE ARRAYS
            # NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            concat_chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
            for k in range(len_chunk_V0):
                j_next = j + chunk_V0[k]
                concat_chunk_dot[j:j_next] = np.copy(chunk_dot[k])
                j = j_next
            dot[pos:next_pos] = concat_chunk_dot
            # END CONCATENATE ARRAYS
            ###
        V0 = np.copy(V1)
        cs_V0 = np.copy(cs_V1)
        V1 = reduceat(V1, cs_T)
        ###
    ###
    return dot


# TODO CONTINUE ...
# @njit # TODO CHANGE!
def itransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nmn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if not parallel:
        return _itransform_ut_md_parallel(U, x, T)
    else:
        return _itransform_ut_md(U, x, T)


# @njit # TODO
def _itransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    zero = np.array([0], dtype=NP_INT)
    dot, cs_T, V0, e_T, Nm = (
        np.copy(x),
        np.concatenate((zero, np.cumsum(T))),
        np.copy(T),
        entropy(T),
        T[0],
    )
    m = len(e_T) - 1
    ### 1d: i,j,k
    ### NOTE: loop runs sequentially
    pos = 0
    for i in range(e_T[1]):
        slot = V0[i]
        next_pos, delta = pos + slot, Nm - slot
        chunk = dot[pos:next_pos]
        chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
        for k in range(slot):
            k_prime = slot - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        dot[pos:next_pos] = chunk_dot
        pos = next_pos
    ###
    ### md: h, i, j, k, l
    ### NOTE: no parallelization
    V1 = reduceat(V0, cs_T)
    for h in range(1, m):
        pos = 0
        cs_V0 = np.concatenate((zero, np.cumsum(V0)))
        ### Outer loop
        ### NOTE: loop runs sequentially
        for i in range(e_T[h + 1]):
            slot = V1[i]
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_V0 = V0[
                np.searchsorted(cs_V0, pos) : np.searchsorted(cs_V0, next_pos)
            ]
            len_chunk_V0 = len(chunk_V0)
            cs_chunk_V0 = np.concatenate((zero, np.cumsum(chunk_V0)))
            chunk_T = T[np.searchsorted(cs_T, pos) : np.searchsorted(cs_T, next_pos)]
            cs_chunk_T = np.concatenate((zero, np.cumsum(chunk_T)))
            chunk_V0_Ts = [
                chunk_T[
                    np.searchsorted(cs_chunk_T, cs_chunk_V0[g]) : np.searchsorted(
                        cs_chunk_T, cs_chunk_V0[g + 1]
                    )
                ]
                for g in range(len_chunk_V0)
            ]
            delta = Nm - len_chunk_V0
            ### Inner loop
            # BEGIN LP DOT
            # NOTE: loop runs sequentially
            chunk_dot, j = [np.zeros((size), dtype=NP_FLOAT) for size in chunk_V0], 0
            for k in range(len_chunk_V0):
                k_prime = len_chunk_V0 - k
                j_next = j + k_prime
                follower = chunk_V0_Ts[k]
                chunk_pos = 0
                for l in range(k_prime):
                    a, leader = U[j + l], chunk_V0_Ts[k + l]
                    next_chunk_pos = chunk_pos + np.sum(leader)
                    sub = np.copy(chunk[chunk_pos:next_chunk_pos])
                    rc = ordinal_embedding(h, leader, follower)
                    sub_rc = np.zeros(np.sum(follower), dtype=NP_INT)
                    sub_rc[rc] = sub
                    chunk_dot[k] += a * sub_rc
                    chunk_pos = next_chunk_pos
                j = j_next + delta
            # END LP DOT
            # START CONCATENATE ARRAYS
            # NOTE: loop runs sequentially
            concat_chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
            for k in range(len_chunk_V0):
                j_next = j + chunk_V0[k]
                concat_chunk_dot[j:j_next] = np.copy(chunk_dot[k])
                j = j_next
            dot[pos:next_pos] = concat_chunk_dot
            # END CONCATENATE ARRAYS
            ###
            pos = next_pos
        V0 = np.copy(V1)
        V1 = reduceat(V1, cs_T)
        ###
    ###
    return dot


@njit(parallel=True)  # TODO
def _itransform_ut_md_parallel(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("Not implemented yet.")


@njit
def dtransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nn)"""
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _dtransform_lt_md_parallel(L, x, T)
    else:
        return _dtransform_lt_md(L, x, T)


@njit
def _dtransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    ###
    N1 = len(T)
    ### NOTE: loop runs sequentially
    dot, pos = np.zeros_like(x), 0
    for i in range(N1):
        t = T[i]
        next_pos = pos + t
        chunk = x[pos:next_pos]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t, dtype=NP_FLOAT), 0
        for k in range(t):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        ###
        pos = next_pos
    ###
    return dot


@njit(parallel=True)
def _dtransform_lt_md_parallel(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    ###
    N1, cs_T = (
        len(T),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ### NOTE: loop runs in parallel
    dot = np.zeros_like(x)
    for i in prange(N1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        chunk = x[pos:next_pos]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        chunk_dot, j = np.zeros(t, dtype=NP_FLOAT), 0
        for k in range(t):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        ###
        pos = next_pos
    ###
    return dot


@njit
def dtransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    """O(Nn)"""
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    if parallel:
        return _dtransform_ut_md_cpu(U, x, T)
    else:
        return _dtransform_ut_md_seq(U, x, T)


@njit
def _dtransform_ut_md_seq(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    ###
    N1, Nm = (
        len(T),
        int(T[0]),
    )
    ### NOTE: loop runs sequentially
    pos, dot = 0, np.zeros_like(x)
    for i in range(N1):
        t = T[i]
        next_pos, delta = pos + t, Nm - t
        chunk = x[pos:next_pos]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t, dtype=np.float64), 0
        for k in range(t):
            k_prime = t - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        dot[pos:next_pos] = chunk_dot
        ###
        pos = next_pos
    ###
    return dot


@njit(parallel=True)
def _dtransform_ut_md_cpu(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    ###
    N1, Nm, cs_T = (
        len(T),
        int(T[0]),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ### NOTE: loop runs in parallel
    dot = np.zeros_like(x)
    for i in prange(N1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        chunk, delta = x[pos:next_pos], Nm - t
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        chunk_dot, j = np.zeros(t, dtype=np.float64), 0
        for k in range(t):
            k_prime = t - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        dot[pos:next_pos] = chunk_dot
        ###
        pos = next_pos
    ###
    return dot
