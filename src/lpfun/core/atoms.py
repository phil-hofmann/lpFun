import numpy as np
from numba import njit, prange
from lpfun import PARALLEL
from lpfun.core.set import ordinal_embedding, entropy

"""
- This module contains numba jit-compiled functions for the transformation of a vector by a matrix.
- The functions are divided into the following categories: 1d, maximal, 2d, 3d and md transformations.
- The functions are used in the transform methods in the molecules.py module.
"""

# cs_T, N_0, N_1, V_2, V_1 ?


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


# 1d


@njit
def transform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """O(n^2)"""
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
    n: int,
) -> np.ndarray:
    """O(n^2)"""
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


@njit(parallel=PARALLEL)
def itransform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """O(n^2)"""
    ### indexing: j, k
    ###
    dot = np.zeros_like(x)
    for k in prange(n):
        j = k * (k + 1) // 2
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
    ###
    return dot


@njit(parallel=PARALLEL)
def itransform_ut_1d(
    U: np.ndarray,
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """O(n^2)"""
    ### indexing: j, k
    ###
    dot = np.zeros_like(x)
    for k in prange(n):
        j, k_prime = k * (2 * n - k + 1) // 2, n - k - 1
        j_next = j + k_prime + 1
        dot[k] = np.sum(U[j:j_next] * x[k:])
    ###
    return dot


# maximal


@njit(parallel=PARALLEL)  # NOTE: deactivated
def transform_lt_max(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(L)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r, h > i > j, k > l
    ###
    dot = x.copy()
    for h in range(m):
        s, r_next = n**h, n ** (m - h - 1)
        s_next = s * n
        ###
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ###
            dot_block, pos_k, j = np.zeros((s_next), dtype=np.float64), 0, 0
            for k in range(n):
                next_pos_k = pos_k + s
                j_next = j + k + 1
                ###
                dot_row, pos_l = np.zeros(s, dtype=np.float64), 0
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
            dot[pos_i:next_pos_i] = dot_block
        ###
    ###
    return dot


@njit(parallel=PARALLEL)  # NOTE: deactivated
def transform_ut_max(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    N, n = (
        len(x),
        int((np.sqrt(1 + 8 * len(U)) - 1) / 2),
    )
    m = int(np.log(N) / np.log(n))
    ### indexing: s, r, h > i > j, k > l
    ###
    dot = x.copy()
    for h in range(m):
        s, r_next = n**h, n ** (m - h - 1)
        s_next = s * n
        ###
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ###
            dot_block, pos_k, j = (
                np.zeros((s_next), dtype=np.float64),
                s_next,
                n * (n + 1) // 2,
            )
            for k in range(n):
                next_pos_k, j_next = pos_k - s, j - k - 1
                ###
                dot_row, pos_l = np.zeros(s, dtype=np.float64), s_next
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
            dot[pos_i:next_pos_i] = dot_block
        ###
    ###
    return dot


@njit(parallel=PARALLEL)
def itransform_lt_max(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    ### indexing: s, r, h > i > j, k > l
    ###
    dot = x.copy()
    for h in range(m):
        s, r_next = n**h, n ** (m - h - 1)
        s_next = s * n
        ###
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ###
            dot_block, pos_k = np.zeros((s_next), dtype=np.float64), 0
            for k in range(n):
                pos_k, j = k * s, k * (k + 1) // 2
                next_pos_k, j_next = pos_k + s, j + k + 1
                ###
                pos_l = 0
                for l in range(j, j_next):
                    next_pos_l = pos_l + s
                    dot_block[pos_k:next_pos_k] += L[l] * block[pos_l:next_pos_l]
                    pos_l = next_pos_l
                ###
            ###
            dot[pos_i:next_pos_i] = dot_block
        ###
    ###
    return dot


@njit(parallel=PARALLEL)
def itransform_ut_max(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    ### indexing: s, r, h > i > j, k > l
    ###
    dot = x.copy()
    for h in range(m):
        s, r_next = n**h, n ** (m - h - 1)
        s_next = s * n
        ###
        for i in prange(r_next):
            pos_i = i * s_next
            next_pos_i = pos_i + s_next
            block = dot[pos_i:next_pos_i]
            ###
            dot_block, pos_k, j = np.zeros((s_next), dtype=np.float64), 0, 0
            for k in range(n):
                k_prime = n - k - 1
                next_pos_k, j_next = pos_k + s, j + k_prime + 1
                ###
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
    ###
    return dot


@njit(parallel=PARALLEL)
def dtransform_max(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    ### indexing: s > i > j, k
    ###
    dot = x.copy()
    s = n ** (m - 1)
    for i in prange(s):
        pos = i * n
        next_pos = pos + n
        block = dot[pos:next_pos]
        ###
        dot_block = np.zeros(n, dtype=np.float64)
        for k in prange(n):
            j = k * (k + 1) // 2
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
        ###
        dot[pos:next_pos] = dot_block
    ###
    return dot


# 2d


@njit(parallel=PARALLEL)
def transform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(2Nn)"""
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=np.float64), 0
        for k in range(t_i):
            j_next = j + k
            dot_block[k] = (block[k] - np.sum(L[j:j_next] * dot_block[:k])) / L[j_next]
            j = j_next + 1
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, pos_i, j = np.zeros_like(x), 0, 0
    for i in range(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(i):
            pos_k = cs_T[k]
            dot_block += L[j] * dot_2d[pos_k : pos_k + t_i]
            j = j + 1
        ###
        dot_2d[pos_i:next_pos_i] = (dot_1d[pos_i:next_pos_i] - dot_block) / L[j]
        j = j + 1
    ###
    return dot_2d


@njit(parallel=PARALLEL)
def transform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(2Nn)"""
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, delta = x[pos_i:next_pos_i], N_1 - t_i
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=np.float64),
            t_i * N_1 - t_i * (t_i - 1) // 2 - delta,
        )
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1
            dot_block[k_prime] = (
                block[k_prime] - np.sum(U[j_next:j] * dot_block[k_prime:])
            ) / U[j_next]
            j = j_next - delta
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, j = np.zeros_like(x), N_1 * (N_1 + 1) // 2
    for i in range(N_1):
        i_prime = N_1 - i - 1
        t_i, pos_i, next_pos_i = T[i_prime], cs_T[i_prime], cs_T[i_prime + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(i):
            j, k_prime = j - 1, N_1 - k - 1
            t_k, pos_k, next_pos_k = T[k_prime], cs_T[k_prime], cs_T[k_prime + 1]
            dot_block[:t_k] += U[j] * dot_2d[pos_k:next_pos_k]
        ###
        j = j - 1
        dot_2d[pos_i:next_pos_i] = (dot_1d[pos_i:next_pos_i] - dot_block) / U[j]
    ###
    return dot_2d


@njit(parallel=PARALLEL)
def itransform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(2Nn)"""
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * (k + 1) // 2
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(i + 1):
            j, pos_k = i * (i + 1) // 2 + k, cs_T[k]
            dot_block += L[j] * dot_1d[pos_k : pos_k + t_i]
        ###
        dot_2d[pos_i:next_pos_i] = dot_block
    ###
    return dot_2d


@njit(parallel=PARALLEL)
def itransform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(2Nn)"""
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * N_1 - k * (k - 1) // 2
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(N_1 - i):
            j = i * N_1 - i * (i - 1) // 2 + k
            t_k, pos_k, next_pos_k = T[i + k], cs_T[i + k], cs_T[i + k + 1]
            dot_block[:t_k] += U[j] * dot_1d[pos_k:next_pos_k]
        ###
        dot_2d[pos_i:next_pos_i] = dot_block
    ###
    return dot_2d


# 3d


@njit(parallel=PARALLEL)
def transform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(3Nn)"""
    N_2 = T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=np.float64), 0
        for k in range(t_i):
            j_next = j + k
            dot_block[k] = (block[k] - np.sum(L[j:j_next] * dot_block[:k])) / L[j_next]
            j = j_next + 1
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        j = 0
        for k in prange(t_i):
            pk = pos_i + k
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            ###
            dot_block = np.zeros(t_k, dtype=np.float64)
            for l in range(k):
                pos_l = cs_T[pos_i + l]
                dot_block += L[j] * dot_2d[pos_l : pos_l + t_k]
                j = j + 1
            ###
            dot_2d[pos_k:next_pos_k] = (dot_1d[pos_k:next_pos_k] - dot_block) / L[j]
            j = j + 1
        ###
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, j = np.zeros_like(x), 0
    for i in range(N_2):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        v_i, vol_i, next_vol_i = V_2[i], cs_V_2[i], cs_V_2[i + 1]
        ###
        dot_block = np.zeros(v_i, dtype=np.float64)
        for k in range(i):
            t_k, pos_k, next_pos_k = T[k], cs_T[k], cs_T[k + 1]
            vol_k, next_vol_k = cs_V_2[k], cs_V_2[k + 1]
            block = dot_3d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=np.float64)
            for l in range(t_i):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1:next_pos_l_1] = block[pos_l_2 : pos_l_2 + t_l_1]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_block += L[j] * sub
            ###
            j = j + 1
        ###
        dot_3d[vol_i:next_vol_i] = (dot_2d[vol_i:next_vol_i] - dot_block) / L[j]
        j = j + 1
    ###
    return dot_3d


@njit(parallel=PARALLEL)
def transform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(3Nn)"""
    N_2 = T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        delta = N_2 - t_i
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=np.float64),
            t_i * N_2 - t_i * (t_i - 1) // 2 - delta,
        )
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1
            dot_block[k_prime] = (
                block[k_prime] - np.sum(U[j_next:j] * dot_block[k_prime:])
            ) / U[j_next]
            j = j_next - delta
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        delta = N_2 - t_i
        ###
        j = t_i * N_2 - t_i * (t_i - 1) // 2 - delta
        for k in prange(t_i):
            k_prime = t_i - k - 1
            pk = pos_i + k_prime
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            ###
            dot_block = np.zeros(t_k, dtype=np.float64)
            for l in range(k):
                j -= 1
                l_prime = t_i - l - 1
                pl = pos_i + l_prime
                t_l, pos_l, next_pos_l = T[pl], cs_T[pl], cs_T[pl + 1]
                dot_block[:t_l] += U[j] * dot_2d[pos_l:next_pos_l]
            j = j - 1
            ###
            dot_2d[pos_k:next_pos_k] = (dot_1d[pos_k:next_pos_k] - dot_block) / U[j]
            j = j - delta
        ###
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, j = np.zeros_like(x), N_2 * (N_2 + 1) // 2
    for i in range(N_2):
        i_prime = N_2 - i - 1
        pos_i, next_pos_i = cs_T[i_prime], cs_T[i_prime + 1]
        v_i, vol_i, next_vol_i = V_2[i_prime], cs_V_2[i_prime], cs_V_2[i_prime + 1]
        ###
        dot_block = np.zeros(v_i, dtype=np.float64)
        for k in range(i):
            j = j - 1
            k_prime = N_2 - k - 1
            t_k, pos_k = T[k_prime], cs_T[k_prime]
            vol_k, next_vol_k = cs_V_2[k_prime], cs_V_2[k_prime + 1]
            block = dot_3d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, ext = 0, 0, np.zeros(v_i, dtype=np.float64)
            for l in range(t_k):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                ext[pos_l_1 : pos_l_1 + t_l_2] = block[pos_l_2:next_pos_l_2]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_block += U[j] * ext
            ###
        j = j - 1
        dot_3d[vol_i:next_vol_i] = (dot_2d[vol_i:next_vol_i] - dot_block) / U[j]
        ###
    ###
    return dot_3d


@njit(parallel=PARALLEL)
def itransform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(3Nn)"""
    N_2 = T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * (k + 1) // 2
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, vol_i, next_pos_i = T[i], cs_T[i], cs_V_2[i], cs_T[i + 1]
        ###
        for k in prange(t_i):
            j = k * (k + 1) // 2
            pk = pos_i + k
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            ###
            for l in prange(k + 1):
                pos_l = cs_T[pos_i + l]
                dot_2d[pos_k:next_pos_k] += L[j + l] * dot_1d[pos_l : pos_l + t_k]
            ###
        ###
    ###
    ### 3d
    ### indexing: i, j > k > l
    ###
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * (i + 1) // 2
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        v_i, vol_i, next_vol_i = V_2[i], cs_V_2[i], cs_V_2[i + 1]
        sub_t_i = T[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(v_i, dtype=np.float64)
        for k in prange(i + 1):
            t_k, pos_k, next_pos_k = T[k], cs_T[k], cs_T[k + 1]
            vol_k, next_vol_k = cs_V_2[k], cs_V_2[k + 1]
            sub_t_k = T[pos_k:next_pos_k]
            block = dot_2d[vol_k:next_vol_k]
            ###
            pos_l_1, _pos2, sub = 0, 0, np.zeros(v_i, dtype=np.float64)
            for l in range(t_i):
                t_l_1, t_l_2 = sub_t_i[l], sub_t_k[l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, _pos2 + t_l_2
                sub[pos_l_1:next_pos_l_1] = block[_pos2 : _pos2 + t_l_1]
                pos_l_1, _pos2 = next_pos_l_1, next_pos_l_2
            ###
            dot_block += L[j + k] * sub
        ###
        dot_3d[vol_i:next_vol_i] = dot_block
    ###
    return dot_3d


@njit(parallel=PARALLEL)
def itransform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
    N_1: int,
) -> np.ndarray:
    """O(3Nn)"""
    N_2 = T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * N_2 - k * (k - 1) // 2
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        t_i, pos_i, vol_i, next_pos_i = T[i], cs_T[i], cs_V_2[i], cs_T[i + 1]
        ###
        for k in prange(t_i):
            pk = pos_i + k
            t_k, pos_k = T[pk], cs_T[pk]
            ###
            j = k * N_2 - k * (k - 1) // 2
            for l in prange(t_i - k):
                pkl = pos_i + k + l
                t_l, pos_l, next_pos_l = T[pkl], cs_T[pkl], cs_T[pkl + 1]
                dot_2d[pos_k : pos_k + t_l] += U[j + l] * dot_1d[pos_l:next_pos_l]
            ###
        ###
    ###
    ### 3d
    ### indexing: i, j > k > l
    ###
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * N_2 - i * (i - 1) // 2
        pos_i, next_pos_i = cs_T[i], cs_T[i + 1]
        v_i, vol_i, next_vol_i = V_2[i], cs_V_2[i], cs_V_2[i + 1]
        ###
        for k in prange(N_2 - i):
            ik = i + k
            t_k, pos_k = T[ik], cs_T[ik]
            vol_k, next_vol_k = cs_V_2[ik], cs_V_2[ik + 1]
            block = dot_2d[vol_k:next_vol_k]
            ###
            pos_l_1, pos_l_2, sub = 0, 0, np.zeros(v_i, dtype=np.float64)
            for l in range(t_k):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                next_pos_l_1, next_pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
                sub[pos_l_1 : pos_l_1 + t_l_2] = block[pos_l_2:next_pos_l_2]
                pos_l_1, pos_l_2 = next_pos_l_1, next_pos_l_2
            dot_3d[vol_i:next_vol_i] += U[j + k] * sub
            ###
        ###
    ###
    return dot_3d


# md


@njit(parallel=PARALLEL)
def transform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    e_T: np.ndarray,
    m: int,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=np.int64)
    dot, V_0 = np.zeros_like(x), T.copy()
    ### 1d
    ### indexing: i > j, k
    ###
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * (k + 1) // 2
            j_next = j + k
            dot_block[k] = (block[k] - np.sum(L[j:j_next] * dot_block[:k])) / L[j_next]
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ###
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else, unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### outer loop
        ###
        for i in prange(e_T[h + 1]):
            pos, next_pos = cs_V_1[i], cs_V_1[i + 1]
            block = dot[pos:next_pos]
            interval = np.array([pos, next_pos])
            pos_T, next_pos_T = np.searchsorted(cs_T, interval)
            pos_V_0, next_pos_V_0 = np.searchsorted(cs_V_0, interval)
            block_T, block_V_0 = (
                T[pos_T:next_pos_T],
                V_0[pos_V_0:next_pos_V_0],
            )
            cs_block_T, cs_block_V_0 = (
                np.concatenate((zero, np.cumsum(block_T))),
                np.concatenate((zero, np.cumsum(block_V_0))),
            )
            ids_block_T = np.searchsorted(cs_block_T, cs_block_V_0)
            len_block_V_0 = len(block_V_0)
            ### start inner loop
            ###
            dot_block, j = np.zeros(cs_block_V_0[-1], dtype=np.float64), 0
            for k in range(len_block_V_0):
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k] : ids_block_T[k + 1]],
                    cs_block_V_0[k],
                    cs_block_V_0[k + 1],
                )
                ###
                for l in range(k + 1):
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[l] : ids_block_T[l + 1]],
                        cs_block_V_0[l],
                        cs_block_V_0[l + 1],
                    )
                    phi = ordinal_embedding(h, follower, leader)
                    if l < k:
                        vec = dot_block[pos_leader:next_pos_leader]
                        dot_block[pos_follower:next_pos_follower] += L[j] * vec[phi]
                    else:
                        vec = block[pos_leader:next_pos_leader]
                        dot_block[pos_follower:next_pos_follower] = (
                            vec[phi] - dot_block[pos_follower:next_pos_follower]
                        ) / L[j]
                    j = j + 1
                ###
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=PARALLEL)
def transform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    e_T: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=np.int64)
    dot, V_0 = np.zeros_like(x), T.copy()
    ### 1d
    ### indexing: i > j, k
    ###
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        delta, block = n - t_i, x[pos_i:next_pos_i]
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=np.float64),
            t_i * n - t_i * (t_i - 1) // 2 - delta,
        )
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1
            dot_block[k_prime] = (
                block[k_prime] - np.sum(U[j_next:j] * dot_block[k_prime:])
            ) / U[j_next]
            j = j_next - delta
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ###
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### start outer loop
        ###
        for i in prange(e_T[h + 1]):
            t_i, pos, next_pos = T[i], cs_V_1[i], cs_V_1[i + 1]
            block = dot[pos:next_pos]
            interval = np.array([pos, next_pos])
            pos_T, next_pos_T = np.searchsorted(cs_T, interval)
            pos_V_0, next_pos_V_0 = np.searchsorted(cs_V_0, interval)
            block_T, block_V_0 = (
                T[pos_T:next_pos_T],
                V_0[pos_V_0:next_pos_V_0],
            )
            cs_block_T, cs_block_V_0 = (
                np.concatenate((zero, np.cumsum(block_T))),
                np.concatenate((zero, np.cumsum(block_V_0))),
            )
            ids_block_T = np.searchsorted(cs_block_T, cs_block_V_0)
            len_block_V_0 = len(block_V_0)
            delta = n - len_block_V_0
            ### start inner loop
            ###
            dot_block, j = (
                np.zeros(cs_block_V_0[-1], dtype=np.float64),
                t_i * n - t_i * (t_i - 1) // 2 - delta,
            )
            for k in range(len_block_V_0):
                k_prime = len_block_V_0 - k - 1
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k_prime] : ids_block_T[k_prime + 1]],
                    cs_block_V_0[k_prime],
                    cs_block_V_0[k_prime + 1],
                )
                sum_follower = next_pos_follower - pos_follower
                ###
                for l in range(k + 1):
                    l_prime = len_block_V_0 - l - 1
                    j = j - 1
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[l_prime] : ids_block_T[l_prime + 1]],
                        cs_block_V_0[l_prime],
                        cs_block_V_0[l_prime + 1],
                    )
                    phi = ordinal_embedding(h, leader, follower)
                    ext = np.zeros(sum_follower, dtype=np.float64)
                    if l < k:
                        vec = dot_block[pos_leader:next_pos_leader]
                        ext[phi] = vec
                        dot_block[pos_follower:next_pos_follower] += U[j] * ext
                    else:
                        vec = block[pos_leader:next_pos_leader]
                        ext[phi] = vec
                        dot_block[pos_follower:next_pos_follower] = (
                            ext - dot_block[pos_follower:next_pos_follower]
                        ) / U[j]
                ###
                j = j - delta
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=PARALLEL)
def itransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    e_T: np.ndarray,
    m: int,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=np.int64)
    dot, V_0 = np.zeros_like(x), T.copy()
    ### 1d
    ### indexing: i > j, k
    ###
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * (k + 1) // 2
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ###
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else, unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### outer loop
        ###
        for i in prange(e_T[h + 1]):
            pos, next_pos = cs_V_1[i], cs_V_1[i + 1]
            block = dot[pos:next_pos]
            interval = np.array([pos, next_pos])
            pos_T, next_pos_T = np.searchsorted(cs_T, interval)
            pos_V_0, next_pos_V_0 = np.searchsorted(cs_V_0, interval)
            block_T, block_V_0 = (
                T[pos_T:next_pos_T],
                V_0[pos_V_0:next_pos_V_0],
            )
            cs_block_T, cs_block_V_0 = (
                np.concatenate((zero, np.cumsum(block_T))),
                np.concatenate((zero, np.cumsum(block_V_0))),
            )
            ids_block_T = np.searchsorted(cs_block_T, cs_block_V_0)
            len_block_V_0 = len(block_V_0)
            ### start inner loop
            ###
            dot_block = np.zeros(cs_block_V_0[-1], dtype=np.float64)
            for k in prange(len_block_V_0):
                j = k * (k + 1) // 2
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k] : ids_block_T[k + 1]],
                    cs_block_V_0[k],
                    cs_block_V_0[k + 1],
                )
                ###
                for l in prange(k + 1):
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[l] : ids_block_T[l + 1]],
                        cs_block_V_0[l],
                        cs_block_V_0[l + 1],
                    )
                    vec = block[pos_leader:next_pos_leader]
                    phi = ordinal_embedding(h, follower, leader)
                    dot_block[pos_follower:next_pos_follower] += L[j + l] * vec[phi]
                ###
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=PARALLEL)
def itransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    e_T: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=np.int64)
    dot, V_0 = np.zeros_like(x), T.copy()
    ### 1d
    ### indexing: i > j, k
    ###
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in prange(t_i):
            j = k * n - k * (k - 1) // 2
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ###
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### start outer loop
        ###
        for i in prange(e_T[h + 1]):
            pos, next_pos = cs_V_1[i], cs_V_1[i + 1]
            block = dot[pos:next_pos]
            interval = np.array([pos, next_pos])
            pos_T, next_pos_T = np.searchsorted(cs_T, interval)
            pos_V_0, next_pos_V_0 = np.searchsorted(cs_V_0, interval)
            block_T, block_V_0 = (
                T[pos_T:next_pos_T],
                V_0[pos_V_0:next_pos_V_0],
            )
            cs_block_T, cs_block_V_0 = (
                np.concatenate((zero, np.cumsum(block_T))),
                np.concatenate((zero, np.cumsum(block_V_0))),
            )
            ids_block_T = np.searchsorted(cs_block_T, cs_block_V_0)
            len_block_V_0 = len(block_V_0)
            ### start inner loop
            ###
            dot_block = np.zeros(cs_block_V_0[-1], dtype=np.float64)
            for k in prange(len_block_V_0):
                j = k * n - k * (k - 1) // 2
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k] : ids_block_T[k + 1]],
                    cs_block_V_0[k],
                    cs_block_V_0[k + 1],
                )
                sum_follower = next_pos_follower - pos_follower
                ###
                for l in prange(len_block_V_0 - k):
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[k + l] : ids_block_T[k + l + 1]],
                        cs_block_V_0[k + l],
                        cs_block_V_0[k + l + 1],
                    )
                    vec = block[pos_leader:next_pos_leader]
                    phi = ordinal_embedding(h, leader, follower)
                    ext = np.zeros(sum_follower, dtype=np.float64)
                    ext[phi] = vec
                    dot_block[pos_follower:next_pos_follower] += U[j + l] * ext
                ###
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=PARALLEL)  # TODO: create test
def dtransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    """O(Nn)"""
    N1, cs_T = (
        len(T),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ###
    dot = np.zeros_like(x)
    for i in prange(N1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        chunk = x[pos:next_pos]
        ###
        chunk_dot, j = np.zeros(t, dtype=np.float64), 0
        for k in range(t):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        ###
        pos = next_pos
    ###
    return dot


@njit(parallel=PARALLEL)
def dtransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    N_1: int,
    n: int,
):
    """O(Nn)"""
    ###
    dot = np.zeros_like(x)
    for i in prange(N_1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos:next_pos]
        ###
        dot_block = np.zeros(t, dtype=np.float64)
        for k in prange(t):
            j = k * n - k * (k - 1) // 2
            j_next = j + t - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
        dot[pos:next_pos] = dot_block
        ###
        pos = next_pos
    ###
    return dot
