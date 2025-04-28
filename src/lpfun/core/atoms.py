import numpy as np
from numba import njit, prange
from lpfun import NP_FLOAT, NP_INT
from lpfun.core.set import ordinal_embedding, entropy

"""
- This module contains numba jit-compiled functions for the transformation of a vector by a matrix.
- The functions are divided into the following categories: 1d, maximal, 2d, 3d and md transformations.
- The functions are used in the transform methods in the molecules.py module.
"""


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


@njit(parallel=True)
def itransform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(n^2)"""
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


@njit(parallel=True)
def itransform_ut_1d(
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """O(n^2)"""
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


@njit(parallel=True)
def itransform_lt_max(
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


@njit(parallel=True)
def itransform_ut_max(
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


@njit(parallel=True)
def dtransform_max(
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


@njit(parallel=True)
def itransform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(2Nn)"""
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


@njit(parallel=True)
def itransform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(2Nn)"""
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
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
            j = j_next + delta
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


@njit(parallel=True)
def itransform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
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


@njit(parallel=True)
def itransform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
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
        block, delta = x[pos_i:next_pos_i], N_2 - t_i
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
            j = j_next + delta
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


@njit(parallel=True)
def itransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=NP_INT)
    dot, cs_T, V_0, e_T = (
        np.zeros_like(x),
        np.concatenate((zero, np.cumsum(T))),
        np.copy(T),
        entropy(T),
    )
    m = len(e_T) - 1
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + k + 1
            dot_block[k] = np.sum(L[j:j_next] * block[: k + 1])
            j = j_next
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ### NOTE: no parallelization
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else, unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### outer loop
        ### NOTE: loop runs in parallel
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
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            dot_block, j = np.zeros(cs_block_V_0[-1], dtype=NP_FLOAT), 0
            for k in range(len_block_V_0):
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k] : ids_block_T[k + 1]],
                    cs_block_V_0[k],
                    cs_block_V_0[k + 1],
                )
                ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
                for l in range(k + 1):
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[l] : ids_block_T[l + 1]],
                        cs_block_V_0[l],
                        cs_block_V_0[l + 1],
                    )
                    vec = block[pos_leader:next_pos_leader]
                    phi = ordinal_embedding(h, follower, leader)
                    dot_block[pos_follower:next_pos_follower] += L[j] * vec[phi]
                    j += 1
                ###
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=True)
def itransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """O(Nmn)"""
    zero = np.array([0], dtype=NP_INT)
    dot, cs_T, V_0, e_T, n = (
        np.zeros_like(x),
        np.concatenate((zero, np.cumsum(T))),
        np.copy(T),
        entropy(T),
        T[0],
    )
    m = len(e_T) - 1
    ### 1d
    ### indexing: i > j, k
    ### NOTE: loop runs in parallel
    for i in prange(e_T[1]):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        delta, block = n - t_i, x[pos_i:next_pos_i]
        ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
        dot_block, j = np.zeros(t_i, dtype=NP_FLOAT), 0
        for k in range(t_i):
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
            j = j_next + delta
        ###
        dot[pos_i:next_pos_i] = dot_block
    ###
    ### md
    ### indexing: h > i > j, k > l
    ### NOTE: no parallelization
    V_1 = reduceat(V_0, cs_T)
    for h in range(1, m):
        cs_V_0, cs_V_1 = (
            np.concatenate((zero, np.cumsum(V_0))),  # NOTE: else unexpected behaviour
            np.concatenate((zero, np.cumsum(V_1))),
        )
        ### start outer loop
        ### NOTE: loop runs in parallel
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
            delta = n - len_block_V_0
            ### start inner loop
            ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
            dot_block, j = np.zeros(cs_block_V_0[-1], dtype=NP_FLOAT), 0
            for k in range(len_block_V_0):
                follower, pos_follower, next_pos_follower = (
                    block_T[ids_block_T[k] : ids_block_T[k + 1]],
                    cs_block_V_0[k],
                    cs_block_V_0[k + 1],
                )
                sum_follower = next_pos_follower - pos_follower
                ### NOTE: possible overhead or numerical instability when parallelized, loop runs sequentially
                for l in range(len_block_V_0 - k):
                    leader, pos_leader, next_pos_leader = (
                        block_T[ids_block_T[k + l] : ids_block_T[k + l + 1]],
                        cs_block_V_0[k + l],
                        cs_block_V_0[k + l + 1],
                    )
                    vec = block[pos_leader:next_pos_leader]
                    phi = ordinal_embedding(h, leader, follower)
                    ext = np.zeros(sum_follower, dtype=NP_FLOAT)
                    ext[phi] = vec
                    dot_block[pos_follower:next_pos_follower] += U[j] * ext
                    j += 1
                ###
                j += delta
            dot[pos:next_pos] = dot_block
            ### end inner loop
        ### end outer loop
        V_0 = V_1
        V_1 = reduceat(V_1, cs_T)
    ###
    return dot


@njit(parallel=True)
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


@njit(parallel=True)
def dtransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
):
    """O(Nn)"""
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
