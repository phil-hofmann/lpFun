import numpy as np
import numba as nb
from lpfun import CACHE, PARALLEL
from lpfun.utils import apply_permutation


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


prange = nb.prange

# Public functions


def transform_lt(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    if m == 1:
        return _transform_lt_1d(L, x, n)

    if m == 2:
        return _transform_lt_2d(L, x, T, cs_T)

    if m == 3:
        return _transform_lt_3d(L, x, T, cs_T, V_2, cs_V_2)

    return _transform_lt_md(L, x, m, pi, T, cs_T)


def transform_ut(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    if m == 1:
        return _transform_ut_1d(U, x, n)

    if m == 2:
        return _transform_ut_2d(U, x, T, cs_T)

    if m == 3:
        return _transform_ut_3d(U, x, T, cs_T, V_2, cs_V_2)

    return _transform_ut_md(U, x, m, n, pi, T, cs_T)


def itransform_lt(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    if m == 1:
        return _itransform_lt_1d(L, x, n)

    if m == 2:
        return _itransform_lt_2d(L, x, T, cs_T)

    if m == 3:
        return _itransform_lt_3d(L, x, T, cs_T, V_2, cs_V_2)

    return _itransform_lt_md(L, x, m, pi, T, cs_T)


def itransform_ut(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    if m == 1:
        return _itransform_ut_1d(U, x, n)

    if m == 2:
        return _itransform_ut_2d(U, x, T, cs_T)

    if m == 3:
        return _itransform_ut_3d(U, x, T, cs_T, V_2, cs_V_2)

    return _itransform_ut_md(U, x, m, n, pi, T, cs_T)


@njit(parallel=PARALLEL)
def dtransform_lt(
    D: np.ndarray,
    x: np.ndarray,
    perm: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(n|A|)"""
    x = apply_permutation(perm, x, invert=True)
    dot = np.zeros_like(x)
    for i in prange(len(T)):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos:next_pos]
        ###
        block_dot, j = np.zeros(t, dtype=np.float64), 0
        for k in range(t):
            j_next = j + k + 1
            block_dot[k] = np.sum(D[j:j_next] * block[: k + 1])
            j = j_next
        dot[pos:next_pos] = block_dot
        ###
    dot = apply_permutation(perm, dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def dtransform_ut(
    D: np.ndarray,
    x: np.ndarray,
    n: int,
    perm: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(n|A|)"""
    x = apply_permutation(perm, x, invert=True)
    dot = np.zeros_like(x)
    for i in prange(len(T)):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos:next_pos]
        ###
        dot_block = np.zeros(t, dtype=np.float64)
        for k in range(t):
            j = k * n - k * (k - 1) // 2
            j_next = j + t - k
            dot_block[k] = np.sum(D[j:j_next] * block[k:])
        dot[pos:next_pos] = dot_block
        ###
    return apply_permutation(perm, dot, invert=False)


# 1d


@njit
def _itransform_lt_1d(
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
def _itransform_ut_1d(
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


@njit
def _transform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """O(n^2)"""
    ### indexing: j, k
    ###
    dot = np.zeros_like(x)
    for k in range(n):
        j = k * (k + 1) // 2
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
    ###
    return dot


@njit
def _transform_ut_1d(
    U: np.ndarray,
    x: np.ndarray,
    n: int,
) -> np.ndarray:
    """O(n^2)"""
    ### indexing: j, k
    ###
    dot = np.zeros_like(x)
    for k in range(n):
        j, k_prime = k * (2 * n - k + 1) // 2, n - k - 1
        j_next = j + k_prime + 1
        dot[k] = np.sum(U[j:j_next] * x[k:])
    ###
    return dot


# 2d


@njit(parallel=PARALLEL)
def _itransform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
) -> np.ndarray:
    """O(2n|A|)"""
    N = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N):
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
    for i in range(N):
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
def _itransform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
) -> np.ndarray:
    """O(2n|A|)"""
    N = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, delta = x[pos_i:next_pos_i], N - t_i
        ###
        dot_block, j = (
            np.zeros(t_i, dtype=np.float64),
            t_i * N - t_i * (t_i - 1) // 2 - delta,
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
    dot_2d, j = np.zeros_like(x), N * (N + 1) // 2
    for i in range(N):
        i_prime = N - i - 1
        t_i, pos_i, next_pos_i = T[i_prime], cs_T[i_prime], cs_T[i_prime + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(i):
            j, k_prime = j - 1, N - k - 1
            t_k, pos_k, next_pos_k = T[k_prime], cs_T[k_prime], cs_T[k_prime + 1]
            dot_block[:t_k] += U[j] * dot_2d[pos_k:next_pos_k]
        ###
        j = j - 1
        dot_2d[pos_i:next_pos_i] = (dot_1d[pos_i:next_pos_i] - dot_block) / U[j]
    ###
    return dot_2d


@njit(parallel=PARALLEL)
def _transform_lt_2d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
) -> np.ndarray:
    """O(2n|A|)"""
    N = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(t_i):
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
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(i + 1):
            j, pos_k = i * (i + 1) // 2 + k, cs_T[k]
            dot_block += L[j] * dot_1d[pos_k : pos_k + t_i]
        ###
        dot_2d[pos_i:next_pos_i] = dot_block
    ###
    return dot_2d


@njit(parallel=PARALLEL)
def _transform_ut_2d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
) -> np.ndarray:
    """O(2n|A|)"""
    N = len(T)
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(t_i):
            j = k * N - k * (k - 1) // 2
            j_next = j + t_i - k
            dot_block[k] = np.sum(U[j:j_next] * block[k:])
        ###
        dot_1d[pos_i:next_pos_i] = dot_block
    ###
    ### 2d
    ### indexing: i > j, k
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(N - i):
            j = i * N - i * (i - 1) // 2 + k
            t_k, pos_k, next_pos_k = T[i + k], cs_T[i + k], cs_T[i + k + 1]
            dot_block[:t_k] += U[j] * dot_1d[pos_k:next_pos_k]
        ###
        dot_2d[pos_i:next_pos_i] = dot_block
    ###
    return dot_2d


# 3d


@njit(parallel=PARALLEL)
def _itransform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    N_1, N_2 = len(T), T[0]
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
        for k in range(t_i):
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
def _itransform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    N_1, N_2 = len(T), T[0]
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
        for k in range(t_i):
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
def _transform_lt_3d(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    N_1, N_2 = len(T), T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(t_i):
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
        for k in range(t_i):
            j = k * (k + 1) // 2
            pk = pos_i + k
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            ###
            for l in range(k + 1):
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
        for k in range(i + 1):
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
def _transform_ut_3d(
    U: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
    V_2: np.ndarray,
    cs_V_2: np.ndarray,
) -> np.ndarray:
    """O(3Nn)"""
    N_1, N_2 = len(T), T[0]
    ### 1d
    ### indexing: i > j, k
    ###
    dot_1d = np.zeros_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block = x[pos_i:next_pos_i]
        ###
        dot_block = np.zeros(t_i, dtype=np.float64)
        for k in range(t_i):
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
        for k in range(t_i):
            pk = pos_i + k
            t_k, pos_k = T[pk], cs_T[pk]
            ###
            j = k * N_2 - k * (k - 1) // 2
            for l in range(t_i - k):
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
        for k in range(N_2 - i):
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
def _transform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(mn|A|)"""
    dot = x.copy()
    for _ in range(m):
        ###
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
            block = dot[pos:next_pos]
            ###
            block_dot, j = np.zeros(t, dtype=np.float64), 0
            for k in range(t):
                j_next = j + k + 1
                block_dot[k] = np.sum(L[j:j_next] * block[: k + 1])
                j = j_next
            new_dot[pos:next_pos] = block_dot
            ###
        dot = new_dot
        ###
        if m > 1:
            dot = apply_permutation(pi, dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def _transform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(mn|A|)"""
    dot = x.copy()
    for _ in range(m):
        ###
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
            block = dot[pos:next_pos]
            ###
            dot_block = np.zeros(t, dtype=np.float64)
            for k in range(t):
                j = k * n - k * (k - 1) // 2
                j_next = j + t - k
                dot_block[k] = np.sum(U[j:j_next] * block[k:])
            new_dot[pos:next_pos] = dot_block
            ###
        dot = new_dot
        ###
        if m > 1:
            dot = apply_permutation(pi, dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def _itransform_lt_md(
    L: np.ndarray,
    x: np.ndarray,
    m: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(mn|A|)"""
    dot = x.copy()
    for _ in range(m):
        if m > 1:
            dot = apply_permutation(pi, dot, invert=True)
        ###
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
            block = dot[pos_i:next_pos_i]
            ###
            dot_block = np.zeros(t_i, dtype=np.float64)
            for k in range(t_i):
                j = k * (k + 1) // 2
                j_next = j + k
                dot_block[k] = (block[k] - np.sum(L[j:j_next] * dot_block[:k])) / L[
                    j_next
                ]
            ###
            new_dot[pos_i:next_pos_i] = dot_block
        dot = new_dot
        ###
    return dot


@njit(parallel=PARALLEL)
def _itransform_ut_md(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(mn|A|)"""
    dot = x.copy()
    for _ in range(m):
        if m > 1:
            dot = apply_permutation(pi, dot, invert=True)
        ###
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
            delta, block = n - t_i, dot[pos_i:next_pos_i]
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
            new_dot[pos_i:next_pos_i] = dot_block
        dot = new_dot
        ###
    return dot
