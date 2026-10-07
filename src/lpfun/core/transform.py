import numpy as np
import numba as nb
from lpfun import CACHE, PARALLEL


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
    ### the permutation is fused into the tube loop: gather on read, scatter on write
    xp, dot = np.empty_like(x), np.empty_like(x)
    for i in prange(len(T)):
        t, pos = T[i], cs_T[i]
        for k in range(t):
            xp[pos + k] = x[perm[pos + k]]
        block = xp[pos : pos + t]
        ###
        j = 0
        for k in range(t):
            s = 0.0
            for l in range(k + 1):
                s += D[j + l] * block[l]
            dot[perm[pos + k]] = s
            j = j + k + 1
        ###
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
    ### the permutation is fused into the tube loop: gather on read, scatter on write
    xp, dot = np.empty_like(x), np.empty_like(x)
    for i in prange(len(T)):
        t, pos = T[i], cs_T[i]
        for k in range(t):
            xp[pos + k] = x[perm[pos + k]]
        block = xp[pos : pos + t]
        ###
        for k in range(t):
            j = k * n - k * (k - 1) // 2
            s = 0.0
            for l in range(t - k):
                s += D[j + l] * block[k + l]
            dot[perm[pos + k]] = s
        ###
    return dot


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
        s = 0.0
        for l in range(k):
            s += L[j + l] * dot[l]
        dot[k] = (x[k] - s) / L[j + k]
        j = j + k + 1
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
        s = 0.0
        for l in range(1, k + 1):
            s += U[j_next + l] * dot[k_prime + l]
        dot[k_prime] = (x[k_prime] - s) / U[j_next]
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
        s = 0.0
        for l in range(k + 1):
            s += L[j + l] * x[l]
        dot[k] = s
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
        j = k * (2 * n - k + 1) // 2
        s = 0.0
        for l in range(n - k):
            s += U[j + l] * x[k + l]
        dot[k] = s
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
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        j = 0
        for k in range(t_i):
            s = 0.0
            for l in range(k):
                s += L[j + l] * out[l]
            out[k] = (block[k] - s) / L[j + k]
            j = j + k + 1
        ###
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, j = np.zeros_like(x), 0
    for i in range(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        out = dot_2d[pos_i:next_pos_i]
        ###
        for k in range(i):
            c, pos_k = L[j], cs_T[k]
            src = dot_2d[pos_k : pos_k + t_i]
            for q in range(t_i):
                out[q] += c * src[q]
            j = j + 1
        ###
        c, block = L[j], dot_1d[pos_i:next_pos_i]
        for q in range(t_i):
            out[q] = (block[q] - out[q]) / c
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
        block, out, delta = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i], N - t_i
        ###
        j = t_i * N - t_i * (t_i - 1) // 2 - delta
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1
            s = 0.0
            for l in range(1, k + 1):
                s += U[j_next + l] * out[k_prime + l]
            out[k_prime] = (block[k_prime] - s) / U[j_next]
            j = j_next - delta
        ###
    ###
    ### 2d
    ### indexing: j, i > k
    ###
    dot_2d, j = np.zeros_like(x), N * (N + 1) // 2
    for i in range(N):
        i_prime = N - i - 1
        t_i, pos_i, next_pos_i = T[i_prime], cs_T[i_prime], cs_T[i_prime + 1]
        out = dot_2d[pos_i:next_pos_i]
        ###
        for k in range(i):
            j, k_prime = j - 1, N - k - 1
            t_k, pos_k = T[k_prime], cs_T[k_prime]
            c, src = U[j], dot_2d[pos_k : pos_k + t_k]
            for q in range(t_k):
                out[q] += c * src[q]
        ###
        j = j - 1
        c, block = U[j], dot_1d[pos_i:next_pos_i]
        for q in range(t_i):
            out[q] = (block[q] - out[q]) / c
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
    dot_1d = np.empty_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        for k in range(t_i):
            j = k * (k + 1) // 2
            s = 0.0
            for l in range(k + 1):
                s += L[j + l] * block[l]
            out[k] = s
        ###
    ###
    ### 2d
    ### indexing: i > j, k
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        out = dot_2d[pos_i:next_pos_i]
        ###
        for k in range(i + 1):
            j, pos_k = i * (i + 1) // 2 + k, cs_T[k]
            c, src = L[j], dot_1d[pos_k : pos_k + t_i]
            for q in range(t_i):
                out[q] += c * src[q]
        ###
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
    dot_1d = np.empty_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        for k in range(t_i):
            j = k * N - k * (k - 1) // 2
            s = 0.0
            for l in range(t_i - k):
                s += U[j + l] * block[k + l]
            out[k] = s
        ###
    ###
    ### 2d
    ### indexing: i > j, k
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        out = dot_2d[pos_i:next_pos_i]
        ###
        for k in range(N - i):
            j = i * N - i * (i - 1) // 2 + k
            t_k, pos_k = T[i + k], cs_T[i + k]
            c, src = U[j], dot_1d[pos_k : pos_k + t_k]
            for q in range(t_k):
                out[q] += c * src[q]
        ###
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
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        j = 0
        for k in range(t_i):
            s = 0.0
            for l in range(k):
                s += L[j + l] * out[l]
            out[k] = (block[k] - s) / L[j + k]
            j = j + k + 1
        ###
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i = T[i], cs_T[i]
        ###
        j = 0
        for k in range(t_i):
            pk = pos_i + k
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            out = dot_2d[pos_k:next_pos_k]
            ###
            for l in range(k):
                pos_l = cs_T[pos_i + l]
                c, src = L[j], dot_2d[pos_l : pos_l + t_k]
                for q in range(t_k):
                    out[q] += c * src[q]
                j = j + 1
            ###
            c, block = L[j], dot_1d[pos_k:next_pos_k]
            for q in range(t_k):
                out[q] = (block[q] - out[q]) / c
            j = j + 1
        ###
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, j = np.zeros_like(x), 0
    for i in range(N_2):
        t_i, pos_i = T[i], cs_T[i]
        v_i, vol_i, next_vol_i = V_2[i], cs_V_2[i], cs_V_2[i + 1]
        out = dot_3d[vol_i:next_vol_i]
        ###
        for k in range(i):
            pos_k, vol_k, next_vol_k = cs_T[k], cs_V_2[k], cs_V_2[k + 1]
            block, c = dot_3d[vol_k:next_vol_k], L[j]
            ###
            pos_l_1, pos_l_2 = 0, 0
            for l in range(t_i):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                for q in range(t_l_1):
                    out[pos_l_1 + q] += c * block[pos_l_2 + q]
                pos_l_1, pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
            ###
            j = j + 1
        ###
        c, block = L[j], dot_2d[vol_i:next_vol_i]
        for q in range(v_i):
            out[q] = (block[q] - out[q]) / c
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
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        j = t_i * N_2 - t_i * (t_i - 1) // 2 - delta
        for k in range(t_i):
            k_prime = t_i - k - 1
            j_next = j - k - 1
            s = 0.0
            for l in range(1, k + 1):
                s += U[j_next + l] * out[k_prime + l]
            out[k_prime] = (block[k_prime] - s) / U[j_next]
            j = j_next - delta
        ###
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i = T[i], cs_T[i]
        delta = N_2 - t_i
        ###
        j = t_i * N_2 - t_i * (t_i - 1) // 2 - delta
        for k in range(t_i):
            k_prime = t_i - k - 1
            pk = pos_i + k_prime
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            out = dot_2d[pos_k:next_pos_k]
            ###
            for l in range(k):
                j -= 1
                l_prime = t_i - l - 1
                pl = pos_i + l_prime
                t_l, pos_l = T[pl], cs_T[pl]
                c, src = U[j], dot_2d[pos_l : pos_l + t_l]
                for q in range(t_l):
                    out[q] += c * src[q]
            j = j - 1
            ###
            c, block = U[j], dot_1d[pos_k:next_pos_k]
            for q in range(t_k):
                out[q] = (block[q] - out[q]) / c
            j = j - delta
        ###
    ###
    ### 3d
    ### indexing: j, i > k > l
    ###
    dot_3d, j = np.zeros_like(x), N_2 * (N_2 + 1) // 2
    for i in range(N_2):
        i_prime = N_2 - i - 1
        pos_i = cs_T[i_prime]
        v_i, vol_i, next_vol_i = V_2[i_prime], cs_V_2[i_prime], cs_V_2[i_prime + 1]
        out = dot_3d[vol_i:next_vol_i]
        ###
        for k in range(i):
            j = j - 1
            k_prime = N_2 - k - 1
            t_k, pos_k = T[k_prime], cs_T[k_prime]
            vol_k, next_vol_k = cs_V_2[k_prime], cs_V_2[k_prime + 1]
            block, c = dot_3d[vol_k:next_vol_k], U[j]
            ###
            pos_l_1, pos_l_2 = 0, 0
            for l in range(t_k):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                for q in range(t_l_2):
                    out[pos_l_1 + q] += c * block[pos_l_2 + q]
                pos_l_1, pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
            ###
        j = j - 1
        c, block = U[j], dot_2d[vol_i:next_vol_i]
        for q in range(v_i):
            out[q] = (block[q] - out[q]) / c
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
    dot_1d = np.empty_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        for k in range(t_i):
            j = k * (k + 1) // 2
            s = 0.0
            for l in range(k + 1):
                s += L[j + l] * block[l]
            out[k] = s
        ###
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i = T[i], cs_T[i]
        ###
        for k in range(t_i):
            j = k * (k + 1) // 2
            pk = pos_i + k
            t_k, pos_k, next_pos_k = T[pk], cs_T[pk], cs_T[pk + 1]
            out = dot_2d[pos_k:next_pos_k]
            ###
            for l in range(k + 1):
                pos_l = cs_T[pos_i + l]
                c, src = L[j + l], dot_1d[pos_l : pos_l + t_k]
                for q in range(t_k):
                    out[q] += c * src[q]
            ###
        ###
    ###
    ### 3d
    ### indexing: i, j > k > l
    ###
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * (i + 1) // 2
        t_i, pos_i = T[i], cs_T[i]
        vol_i, next_vol_i = cs_V_2[i], cs_V_2[i + 1]
        out = dot_3d[vol_i:next_vol_i]
        ###
        for k in range(i + 1):
            pos_k, vol_k, next_vol_k = cs_T[k], cs_V_2[k], cs_V_2[k + 1]
            block, c = dot_2d[vol_k:next_vol_k], L[j + k]
            ###
            pos_l_1, pos_l_2 = 0, 0
            for l in range(t_i):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                for q in range(t_l_1):
                    out[pos_l_1 + q] += c * block[pos_l_2 + q]
                pos_l_1, pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
            ###
        ###
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
    dot_1d = np.empty_like(x)
    for i in prange(N_1):
        t_i, pos_i, next_pos_i = T[i], cs_T[i], cs_T[i + 1]
        block, out = x[pos_i:next_pos_i], dot_1d[pos_i:next_pos_i]
        ###
        for k in range(t_i):
            j = k * N_2 - k * (k - 1) // 2
            s = 0.0
            for l in range(t_i - k):
                s += U[j + l] * block[k + l]
            out[k] = s
        ###
    ###
    ### 2d
    ### indexing: i > j, k > l
    ###
    dot_2d = np.zeros_like(x)
    for i in prange(N_2):
        t_i, pos_i = T[i], cs_T[i]
        ###
        for k in range(t_i):
            pk = pos_i + k
            t_k, pos_k = T[pk], cs_T[pk]
            out = dot_2d[pos_k : pos_k + t_k]
            ###
            j = k * N_2 - k * (k - 1) // 2
            for l in range(t_i - k):
                pkl = pos_i + k + l
                t_l, pos_l = T[pkl], cs_T[pkl]
                c, src = U[j + l], dot_1d[pos_l : pos_l + t_l]
                for q in range(t_l):
                    out[q] += c * src[q]
            ###
        ###
    ###
    ### 3d
    ### indexing: i, j > k > l
    ###
    dot_3d = np.zeros_like(x)
    for i in prange(N_2):
        j = i * N_2 - i * (i - 1) // 2
        pos_i, vol_i, next_vol_i = cs_T[i], cs_V_2[i], cs_V_2[i + 1]
        out = dot_3d[vol_i:next_vol_i]
        ###
        for k in range(N_2 - i):
            ik = i + k
            t_k, pos_k = T[ik], cs_T[ik]
            vol_k, next_vol_k = cs_V_2[ik], cs_V_2[ik + 1]
            block, c = dot_2d[vol_k:next_vol_k], U[j + k]
            ###
            pos_l_1, pos_l_2 = 0, 0
            for l in range(t_k):
                t_l_1, t_l_2 = T[pos_i + l], T[pos_k + l]
                for q in range(t_l_2):
                    out[pos_l_1 + q] += c * block[pos_l_2 + q]
                pos_l_1, pos_l_2 = pos_l_1 + t_l_1, pos_l_2 + t_l_2
            ###
        ###
    ###
    return dot_3d


# md
#
# Each pass applies the 1d operator along every tube and then rotates the
# directions with the permutation pi. The rotation is fused into the tube loop
# (double buffered): the forward kernels scatter each result to its permuted
# position, dst[pi[pos + k]] (== apply_permutation(pi, ., invert=False)), and the
# inverse kernels gather their input from src[pi[pos + k]]
# (== apply_permutation(pi, ., invert=True)).


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
    permute = m > 1
    src, dst = x.copy(), np.empty_like(x)
    for _ in range(m):
        ###
        for i in prange(len(T)):
            t, pos = T[i], cs_T[i]
            block = src[pos : pos + t]
            ###
            j = 0
            for k in range(t):
                s = 0.0
                for l in range(k + 1):
                    s += L[j + l] * block[l]
                if permute:
                    dst[pi[pos + k]] = s
                else:
                    dst[pos + k] = s
                j = j + k + 1
            ###
        src, dst = dst, src
        ###
    return src


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
    permute = m > 1
    src, dst = x.copy(), np.empty_like(x)
    for _ in range(m):
        ###
        for i in prange(len(T)):
            t, pos = T[i], cs_T[i]
            block = src[pos : pos + t]
            ###
            for k in range(t):
                j = k * n - k * (k - 1) // 2
                s = 0.0
                for l in range(t - k):
                    s += U[j + l] * block[k + l]
                if permute:
                    dst[pi[pos + k]] = s
                else:
                    dst[pos + k] = s
            ###
        src, dst = dst, src
        ###
    return src


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
    permute = m > 1
    src, dst = x.copy(), np.empty_like(x)
    for _ in range(m):
        ###
        for i in prange(len(T)):
            t_i, pos_i = T[i], cs_T[i]
            out = dst[pos_i : pos_i + t_i]
            ###
            for k in range(t_i):
                j = k * (k + 1) // 2
                s = 0.0
                for l in range(k):
                    s += L[j + l] * out[l]
                b = src[pi[pos_i + k]] if permute else src[pos_i + k]
                out[k] = (b - s) / L[j + k]
            ###
        src, dst = dst, src
        ###
    return src


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
    permute = m > 1
    src, dst = x.copy(), np.empty_like(x)
    for _ in range(m):
        ###
        for i in prange(len(T)):
            t_i, pos_i = T[i], cs_T[i]
            delta, out = n - t_i, dst[pos_i : pos_i + t_i]
            ###
            j = t_i * n - t_i * (t_i - 1) // 2 - delta
            for k in range(t_i):
                k_prime = t_i - k - 1
                j_next = j - k - 1
                s = 0.0
                for l in range(1, k + 1):
                    s += U[j_next + l] * out[k_prime + l]
                b = src[pi[pos_i + k_prime]] if permute else src[pos_i + k_prime]
                out[k_prime] = (b - s) / U[j_next]
                j = j_next - delta
            ###
        src, dst = dst, src
        ###
    return src
