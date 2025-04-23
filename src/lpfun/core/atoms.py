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
    ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs sequentially
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ### NOTE: loop runs sequentially
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos_k = pos_k + s
                j_next = j + k + 1
                ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs sequentially
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ### NOTE: loop runs sequentially
            dot_block, pos_k, j = (
                np.zeros((s_next), dtype=NP_FLOAT),
                s_next,
                n * (n + 1) // 2,
            )
            for k in range(n):
                next_pos_k, j_next = pos_k - s, j - k - 1
                ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs sequentially
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ### NOTE: loop runs sequentially
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos_k, j_next = pos_k + s, j + k + 1
                ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    dot, s, r = np.copy(x), 1, N
    for _ in range(m):
        s_next, r_next = s * n, r // n
        ### NOTE: loop runs sequentially
        pos = 0
        for _ in range(r_next):
            next_pos = pos + s_next
            block = dot[pos:next_pos]
            ### NOTE: loop runs sequentially
            dot_block, pos_k, j = np.zeros((s_next), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                k_prime = n - k - 1
                next_pos_k, j_next = pos_k + s, j + k_prime + 1
                ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    pos, s, dot = 0, N // n, np.copy(x)
    for _ in range(s):
        next_pos = pos + n
        block = dot[pos:next_pos]
        ### NOTE: loop runs sequentially
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
    ###
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N_1 = len(T)
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs sequentially
    dot_1d, pos_i = np.zeros_like(x), 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        chunk = x[pos_i:next_pos_i]
        ### NOTE: loop runs sequentially
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
    ### NOTE: loop runs sequentially
    dot_2d, pos_i, j = np.zeros_like(x), 0, 0
    for i in range(N_1):
        t_i = T[i]
        next_pos_i = pos_i + t_i
        ### NOTE: loop runs sequentially
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


@njit # NOTE REFACTORING
def transform_ut_2d(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2Nn)"""
    ### NOTE: parallelization partially possible, not recommended
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N0, N1 = (
        np.sum(T),
        len(T),
    )
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N1 - t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = (
            np.zeros(t1, dtype=NP_FLOAT),
            t1 * N1 - t1 * (t1 - 1) // 2 - delta,
        )
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j - k - 1
            dotsum = np.sum(U[j_next:j] * chunk_dot[k_prime:])
            chunk_dot[k_prime] = (chunk[k_prime] - dotsum) / U[j_next]
            j = j_next - delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    dot2, pos1, j = np.zeros_like(x), N0, N1 * (N1 + 1) // 2
    for i1 in range(N1):
        i1_prime = N1 - i1 - 1
        t1 = T[i1_prime]
        next_pos1 = pos1 - t1
        ### NOTE: loop runs sequentially
        pos2, dotsum = N0, np.zeros(t1, dtype=NP_FLOAT)
        for i2 in range(i1):
            j -= 1
            i2_prime = N1 - i2 - 1
            t2 = T[i2_prime]
            next_pos2 = pos2 - t2
            chunk = dot2[next_pos2:pos2]
            dotsum[:t2] += U[j] * chunk
            pos2 = next_pos2
        j -= 1
        dot2[next_pos1:pos1] = (dot1[next_pos1:pos1] - dotsum) / U[j]
        ###
        pos1 = next_pos1
    ###
    return dot2


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
    N1 = len(T)
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    dot2, j, pos1 = np.zeros_like(x), 0, 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ### NOTE: loop runs sequentially
        pos2 = 0
        for i2 in range(i1 + 1):
            t2 = T[i2]
            next_pos2 = pos2 + t2
            chunk = dot1[pos2 : pos2 + t1]
            dot2[pos1:next_pos1] += L[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
        pos1 = next_pos1
    ###
    return dot2


@njit(parallel=True)
def _itransform_lt_2d_parallel(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N1 = len(T)
    cs_T = np.concatenate((np.array([0]), np.cumsum(T)))
    ### 1d
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk = x[pos1:next_pos1]
        ### caution -- possible overhead or numerical instability
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    dot2 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        ### caution -- possible overhead or numerical instability
        pos2, j = 0, i1 * (i1 + 1) // 2
        for i2 in range(i1 + 1):
            t2 = T[i2]
            next_pos2 = pos2 + t2
            chunk = dot1[pos2 : pos2 + t1]
            dot2[pos1:next_pos1] += L[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
    ###
    return dot2


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
    N1 = len(T)
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N1 - t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    dot2, pos1, j = np.zeros_like(x), 0, 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ### NOTE: loop runs sequentially
        pos2 = pos1
        for i2 in range(N1 - i1):
            t2 = T[i1 + i2]
            next_pos2 = pos2 + t2
            chunk = dot1[pos2:next_pos2]
            dot2[pos1 : pos1 + t2] += U[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
        pos1 = next_pos1
    ###
    return dot2


@njit(parallel=True)
def _itransform_ut_2d_parallel(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ####
    N1, cs_T = (
        len(T),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ### 1d
    ### NOTE: loop runs in parallel
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk, delta = x[pos1:next_pos1], N1 - t1
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    ### NOTE: loop runs in parallel
    dot2 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos2, j = pos1, i1 * N1 - i1 * (i1 - 1) // 2
        for i2 in range(N1 - i1):
            t2 = T[i1 + i2]
            next_pos2 = pos2 + t2
            chunk = dot1[pos2:next_pos2]
            dot2[pos1 : pos1 + t2] += U[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
    ###
    return dot2


# 3d


@njit
def transform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    # NOTE: parallelization partially possible, not recommended
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, N2 = len(T), T[0]
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            dotsum = np.sum(L[j : j_next - 1] * chunk_dot[:k])
            chunk_dot[k] = (chunk[k] - dotsum) / L[j_next - 1]
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            ### NOTE: loop runs sequentially
            _pos2, dotsum = vol1, np.zeros(_t1, dtype=NP_FLOAT)
            for _i2 in range(_i1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot2[_pos2 : _pos2 + _t1]
                dotsum += L[j] * chunk
                _pos2 = _next_pos2
                j += 1
            dot2[_pos1:_next_pos1] = (dot1[_pos1:_next_pos1] - dotsum) / L[j]
            j += 1
            ###
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        pos1 = next_pos1
        vol1 += _vol1
        V2[i1] = _vol1
    ###
    ### 3d
    ### NOTE: loop runs sequentially
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        pos2, vol2, dotsum = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
        for i2 in range(i1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot3[vol2:next_vol2]
            ### NOTE: loop runs sequentially
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dotsum += L[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        dot3[vol1:next_vol1] = (dot2[vol1:next_vol1] - dotsum) / L[j]
        j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit
def transform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    # NOTE: parallelization partially possible, not recommended
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N0, N1, N2 = np.sum(T), len(T), T[0]
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = (
            np.zeros(t1, dtype=NP_FLOAT),
            t1 * N2 - t1 * (t1 - 1) // 2 - delta,
        )
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j - k - 1
            dotsum = np.sum(U[j_next:j] * chunk_dot[k_prime:])
            chunk_dot[k_prime] = (chunk[k_prime] - dotsum) / U[j_next]
            j = j_next - delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1
        sub_t1 = T[pos1:next_pos1]
        _vol1 = np.sum(sub_t1)
        vol1 += _vol1
        V2[i1] = _vol1
        ### NOTE: loop runs sequentially
        _pos1, j = (vol1, t1 * N2 - t1 * (t1 - 1) // 2 - delta)
        for _i1 in range(t1):
            _i1_prime = t1 - _i1 - 1
            _t1 = sub_t1[_i1_prime]
            _next_pos1 = _pos1 - _t1
            ### NOTE: loop runs sequentially
            _pos2, dotsum = vol1, np.zeros(_t1, dtype=NP_FLOAT)
            for _i2 in range(_i1):
                j -= 1
                _i2_prime = t1 - _i2 - 1
                _t2 = sub_t1[_i2_prime]
                _next_pos2 = _pos2 - _t2
                chunk = dot2[_next_pos2:_pos2]
                dotsum[:_t2] += U[j] * chunk
                _pos2 = _next_pos2
            j -= 1
            dot2[_next_pos1:_pos1] = (dot1[_next_pos1:_pos1] - dotsum) / U[j]
            j -= delta
            ###
            _pos1 = _next_pos1
        ###
        pos1 = next_pos1
    ###
    ### 3d :: TODO
    ### NOTE: loop runs sequentially..
    dot3, pos1, vol1, j = np.zeros_like(x), V2[0], N0, N2 * (N2 + 1) // 2
    for i1 in range(N2):
        i1_prime = N2 - i1 - 1
        t1, v1 = T[i1_prime], V2[i1_prime]
        next_pos1, next_vol1 = pos1 - t1, vol1 - v1
        sub_t1 = T[next_pos1:pos1]
        ### NOTE: loop runs sequentially
        pos2, vol2, dotsum = V2[0], N0, np.zeros(v1, dtype=NP_FLOAT)
        for i2 in range(i1):
            j -= 1
            i2_prime = N2 - i2 - 1
            t2, v2 = T[i2_prime], V2[i2_prime]
            next_pos2, next_vol2 = pos2 - t2, vol2 - v2
            sub_t2 = T[next_pos2:pos2]
            chunk = dot3[next_vol2:vol2]
            ### NOTE: loop runs sequentially
            _pos1, _pos2, ext = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            # TODO: swap t1 with t2
            for i in range(t2):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                ext[_pos1 : _pos1 + _t2] = chunk[_pos2:_next_pos2]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dotsum += U[j] * ext
            ###
            pos2, vol2 = next_pos2, next_vol2
        j -= 1
        dot3[next_vol1:vol1] = (dot2[next_vol1:vol1] - dotsum) / U[j]
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


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
    N1, N2 = (
        len(T),
        T[0],
    )
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1
            for _i2 in range(_i1 + 1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += L[j] * chunk
                _pos2 = _next_pos2
                j += 1
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        pos1 = next_pos1
        vol1 += _vol1
        V2[i1] = _vol1
    ###
    ### 3d
    ### NOTE: loop runs sequentially
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        pos2, vol2 = 0, 0
        for i2 in range(i1 + 1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ### NOTE: loop runs sequentially
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += L[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit(parallel=True)
def _itransform_lt_3d_parallel(
    L: np.ndarray, x: np.ndarray, T: np.ndarray
) -> np.ndarray:
    ###
    N1, N2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    V2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N2)], dtype=NP_INT)
    cs_V2 = np.concatenate((np.array([0]), np.cumsum(V2)))
    ### 1d
    ### NOTE: loop runs in parallel
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk = x[pos1:next_pos1]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    ### NOTE: loop runs in parallel
    dot2 = np.zeros_like(x)
    for i1 in prange(N2):
        t1, pos1, vol1, next_pos1 = T[i1], cs_T[i1], cs_V2[i1], cs_T[i1 + 1]
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            for _i2 in range(_i1 + 1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += L[j] * chunk
                _pos2 = _next_pos2
                j += 1
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
    ###
    ### 3d
    ### NOTE: loop runs in parallel
    dot3 = np.zeros_like(x)
    for i1 in prange(N2):
        j = i1 * (i1 + 1) // 2
        t1, v1 = T[i1], V2[i1]
        pos1, vol1 = cs_T[i1], cs_V2[i1]
        next_pos1, next_vol1 = cs_T[i1 + 1], cs_V2[i1 + 1]
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos2, vol2 = 0, 0
        for i2 in range(i1 + 1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += L[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


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
    N1, N2 = (
        len(T),
        T[0],
    )
    ### 1d
    ### NOTE: loop runs sequentially
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1
        chunk = x[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    ### NOTE: loop runs sequentially
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1 + _vol1
            for _i2 in range(t1 - _i1):
                _i1_i2 = _i1 + _i2
                _t2 = sub_t1[_i1_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2:_next_pos2]
                dot2[_pos1 : _pos1 + _t2] += U[j] * chunk
                _pos2 = _next_pos2
                j += 1
            j += delta
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        pos1, vol1, V2[i1] = next_pos1, _pos1, _vol1
    ###
    ### 3d
    ### NOTE: loop runs sequentially
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: loop runs sequentially
        pos2, vol2 = pos1, vol1
        for i2 in range(N2 - i1):
            i1_i2 = i1 + i2
            t2, v2 = T[i1_i2], V2[i1_i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ### NOTE: loop runs sequentially
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t2):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1 : _pos1 + _t2] = chunk[_pos2:_next_pos2]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += U[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit(parallel=True)  # TODO Check!
def _itransform_ut_3d_parallel(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    ###
    N1, N2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    V2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N2)], dtype=NP_INT)
    cs_V2 = np.concatenate((np.array([0]), np.cumsum(V2)))
    ### 1d
    ### NOTE: loop runs in parallel
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk, delta = x[pos1:next_pos1], N2 - t1
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            k_prime = t1 - k - 1
            j_next = j + k_prime + 1
            chunk_dot[k] = np.sum(U[j:j_next] * chunk[k:])
            j = j_next + delta
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    ### NOTE: loop runs in parallel
    dot2 = np.zeros_like(x)
    for i1 in prange(N2):
        t1, pos1, vol1, next_pos1 = T[i1], cs_T[i1], cs_V2[i1], cs_T[i1 + 1]
        sub_t1, delta = T[pos1:next_pos1], N2 - t1
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1 + _vol1
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            for _i2 in range(t1 - _i1):
                _i1_i2 = _i1 + _i2
                _t2 = sub_t1[_i1_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2:_next_pos2]
                dot2[_pos1 : _pos1 + _t2] += U[j] * chunk
                _pos2 = _next_pos2
                j += 1
            j += delta
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        pos1 = next_pos1
        vol1 += _vol1
        V2[i1] = _vol1
    ###
    ### 3d
    ### NOTE: loop runs in parallel
    dot3 = np.zeros_like(x)
    for i1 in prange(N2):
        j = i1 * N2 - i1 * (i1 - 1) // 2
        t1, v1 = T[i1], V2[i1]
        pos1, vol1 = cs_T[i1], cs_V2[i1]
        next_pos1, next_vol1 = cs_T[i1 + 1], cs_V2[i1 + 1]
        sub_t1 = T[pos1:next_pos1]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        pos2, vol2 = pos1, vol1
        for i2 in range(N2 - i1):
            i1_i2 = i1 + i2
            t2, v2 = T[i1_i2], V2[i1_i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t2):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1 : _pos1 + _t2] = chunk[_pos2:_next_pos2]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += U[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit
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
