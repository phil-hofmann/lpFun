import numpy as np
from numba import njit, prange
from lpfun import NP_FLOAT, NP_INT, PARALLEL
from lpfun.utils import concatenate_arrays, reduceat, phi
from lpfun.iterators import CompatibleIntegerList as CIL

"""
This module contains the Numba JIT-compiled functions for the transformation of a vector by a matrix.

The functions are divided into the following categories:
- 1d transformations
- Maximal transformations
- 2d transformations
- 3d transformations
- md Transformations

The functions are used in the Transform methods in the molecules.py module.
"""

# 1d Transformations


@njit  #
def transform_1d(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    ### NOTE -- parallelization not possible
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    n = x.shape[0]
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        j_next = j + k + 1
        dotsum = np.sum(L[j : j_next - 1] * dot[:k])
        dot[k] = (x[k] - dotsum) / L[j_next - 1]
        j = j_next
    ###
    return dot


@njit  #
def itransform_1d(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    return (
        _itransform_1d_parallel(L, x) if PARALLEL else _itransform_1d_sequential(L, x)
    )


@njit  #
def _itransform_1d_sequential(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    n = x.shape[0]
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
        j = j_next
    ###
    return dot


@njit(parallel=True)  #
def _itransform_1d_parallel(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    n = x.shape[0]
    ###
    dot = np.zeros_like(x)
    for k in prange(n):
        j = (k * (k + 1)) // 2
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
    ###
    return dot


# Maximal Transformations


@njit  #
def transform_maximal(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(N*n*m)"""
    ### NOTE -- parallelization recommended -- but becomes less effective at deeper levels of the divide-and-conquer tree
    return (
        _transform_maximal_parallel(Q, x)
        if PARALLEL
        else _transform_maximal_sequential(Q, x)
    )


@njit  #
def _transform_maximal_sequential(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    ###
    dot, slot, remainder = x[:], 1, N
    for _ in range(m):
        remainder //= n
        slot *= n
        splits = slot // n
        ###
        pos1 = 0
        for _ in range(remainder):
            next_pos1 = pos1 + slot
            chunk = dot[pos1:next_pos1]
            ###
            chunk_dot, pos2, j = np.zeros((slot), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos2 = pos2 + splits
                j_next = j + k + 1
                ###
                dotsum, pos3 = np.zeros(splits, dtype=NP_FLOAT), 0
                for l in range(j, j_next - 1):
                    next_pos3 = pos3 + splits
                    dotsum += L[l] * chunk_dot[pos3:next_pos3]
                    pos3 = next_pos3
                next_pos3 = pos3 + splits
                chunk_dot[pos2:next_pos2] = (chunk[pos3:next_pos3] - dotsum) / L[
                    j_next - 1
                ]
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
            pos1 = next_pos1
        ###
    ###
    return dot


@njit(parallel=True)  #
def _transform_maximal_parallel(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    ### caution -- loop not parallelizable
    dot, slot, remainder = x[:], 1, N
    for _ in range(m):
        remainder //= n
        slot *= n
        splits = slot // n
        ###
        for i in prange(remainder):
            pos = i * slot
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            ### caution -- loop not parallelizable
            chunk_dot, pos2, j = np.zeros((slot), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos2 = pos2 + splits
                j_next = j + k + 1
                ### caution -- possible overhead or numerical instability
                dotsum, pos3 = np.zeros(splits, dtype=NP_FLOAT), 0
                for l in range(j, j_next - 1):
                    next_pos3 = pos3 + splits
                    dotsum += L[l] * chunk_dot[pos3:next_pos3]
                    pos3 = next_pos3
                next_pos3 = pos3 + splits
                chunk_dot[pos2:next_pos2] = (chunk[pos3:next_pos3] - dotsum) / L[
                    j_next - 1
                ]
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos:next_pos] = chunk_dot
            pos = next_pos
        ###
    ###
    return dot


@njit  #
def itransform_maximal(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(N*n*m)"""
    return (
        _itransform_maximal_parallel(L, x)
        if PARALLEL
        else _itransform_maximal_sequential(L, x)
    )


@njit  #
def _itransform_maximal_sequential(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    ###
    dot, slot, remainder = x[:], 1, N
    for _ in range(m):
        remainder //= n
        slot *= n
        splits = slot // n
        ###
        pos1 = 0
        for _ in range(remainder):
            next_pos1 = pos1 + slot
            chunk = dot[pos1:next_pos1]
            ###
            chunk_dot, pos2, j = np.zeros((slot), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos2 = pos2 + splits
                j_next = j + k + 1
                Lk = L[j:j_next]
                ###
                pos3 = 0
                for l in range(k + 1):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += Lk[l] * chunk[pos3:next_pos3]
                    pos3 = next_pos3
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
            pos1 = next_pos1
        ###
    ###
    return dot


@njit(parallel=True)  #
def _itransform_maximal_parallel(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    ### caution -- loop not parallelizable
    dot, slot, remainder = x[:], 1, N
    for _ in range(m):
        remainder //= n
        slot *= n
        splits = slot // n
        ###
        for i1 in prange(remainder):
            pos1 = i1 * slot
            next_pos1 = pos1 + slot
            chunk = dot[pos1:next_pos1]
            # ### caution -- possible overhead or numerical instability
            chunk_dot, pos2, j = np.zeros((slot), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos2 = pos2 + splits
                j_next = j + k + 1
                Lk = L[j:j_next]
                ### caution -- possible overhead or numerical instability
                pos3 = 0
                for l in range(k + 1):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += Lk[l] * chunk[pos3:next_pos3]
                    pos3 = next_pos3
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
        ###
    ###
    return dot


@njit  #
def dtransform_maximal(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(N*n)"""
    return (
        _dtransform_maximal_parallel(L, x)
        if PARALLEL
        else _dtransform_maximal_sequential(L, x)
    )


@njit  #
def _dtransform_maximal_sequential(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    ###
    pos, splits, dot = 0, N // n, x[:]
    for _ in range(splits):
        next_pos = pos + n
        chunk = dot[pos:next_pos]
        chunk_dot, j = np.zeros(n, dtype=NP_FLOAT), 0
        for i in range(n):
            j_next = j + i + 1
            chunk_dot[i] = np.sum(L[j:j_next] * chunk[: i + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        pos = next_pos
    ###
    return dot


@njit(parallel=True)  #
def _dtransform_maximal_parallel(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    N, n = x.shape[0], int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2)
    ###
    splits, dot = N // n, x[:]
    for i in prange(splits):
        pos = i * n
        next_pos = pos + n
        chunk = dot[pos:next_pos]
        ### caution -- possible overhead or numerical instability
        chunk_dot, j = np.zeros(n, dtype=NP_FLOAT), 0
        for k in range(n):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot[pos:next_pos] = chunk_dot
    ###
    return dot


# 2D Transformations


@njit  #
def transform_2d(Q: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2*N*n)"""
    ### NOTE -- parallelization possible but not recommended
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1 = len(T)
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ###
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            dotsum = np.sum(Q[j : j_next - 1] * chunk_dot[:k])
            chunk_dot[k] = (chunk[k] - dotsum) / Q[j_next - 1]
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    pos1, dot2, j = 0, np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ###
        pos2, dotsum = 0, np.zeros(t1, dtype=NP_FLOAT)
        for i2 in range(i1):
            t2 = T[i2]
            next_pos2 = pos2 + t2
            chunk = dot2[pos2 : pos2 + t1]
            dotsum += Q[j] * chunk
            pos2 = next_pos2
            j += 1
        dot2[pos1:next_pos1] = (dot1[pos1:next_pos1] - dotsum) / Q[j]
        j += 1
        ###
        pos1 = next_pos1
    ###
    return dot2


@njit  #
def itransform_2d(Q: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2*N*n)"""
    return (
        _itransform_2d_parallel(Q, x, T)
        if PARALLEL
        else _itransform_2d_sequential(Q, x, T)
    )


@njit  #
def _itransform_2d_sequential(
    Q: np.ndarray, x: np.ndarray, T: np.ndarray
) -> np.ndarray:
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1 = len(T)
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ###
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(Q[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    pos1, dot2, j = 0, np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ###
        pos2 = 0
        for i2 in range(i1 + 1):
            t2 = T[i2]
            next_pos2 = pos2 + t2
            chunk = dot1[pos2 : pos2 + t1]
            dot2[pos1:next_pos1] += Q[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
        pos1 = next_pos1
    ###
    return dot2


@njit(parallel=True)  #
def _itransform_2d_parallel(Q: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
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
            chunk_dot[k] = np.sum(Q[j:j_next] * chunk[: k + 1])
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
            dot2[pos1:next_pos1] += Q[j] * chunk
            pos2 = next_pos2
            j += 1
        ###
    ###
    return dot2


# 3D Transformations

# TODO
def transform_3d(Q: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(3*N*n)"""
    # NOTE -- parallelization possible but not recommended
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1, N2 = len(T), T[0]
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ###
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            dotsum = np.sum(Q[j : j_next - 1] * chunk_dot[:k])
            chunk_dot[k] = (chunk[k] - dotsum) / Q[j_next - 1]
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        sub_t1 = T[pos1:next_pos1]
        ###
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1
            for _i2 in range(_i1 + 1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += Q[j] * chunk
                _pos2 = _next_pos2
                j += 1
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        # ###
        # pos2, dotsum = 0, np.zeros(t1, dtype=NP_FLOAT)
        # for i2 in range(i1):
        #     t2 = T[i2]
        #     next_pos2 = pos2 + t2
        #     chunk = dot2[pos2 : pos2 + t1]
        #     dotsum += Q[j] * chunk
        #     pos2 = next_pos2
        #     j += 1
        # dot2[pos1:next_pos1] = (dot1[pos1:next_pos1] - dotsum) / Q[j]
        # j += 1
        # ###
        pos1 = next_pos1
        vol1 += _vol1
        V2[i1] = _vol1
    ###
    ### 3d
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ###
        pos2, vol2 = 0, 0
        for i2 in range(i1 + 1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ###
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += Q[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit  #
def itransform_3d(A: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(3*N*n)"""
    return (
        _itransform_3d_parallel(A, x, T)
        if PARALLEL
        else _itransform_3d_sequential(A, x, T)
    )


@njit  #
def _itransform_3d_sequential(Q: np.ndarray, x: np.ndarray, T: np.ndarray):
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1, N2 = len(T), T[0]
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        chunk = x[pos1:next_pos1]
        ###
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(Q[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        sub_t1 = T[pos1:next_pos1]
        ###
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1
            for _i2 in range(_i1 + 1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += Q[j] * chunk
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
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ###
        pos2, vol2 = 0, 0
        for i2 in range(i1 + 1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ###
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += Q[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit(parallel=True)  #
def _itransform_3d_parallel(Q: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1, N2 = len(T), T[0]
    cs_T = np.concatenate((np.array([0]), np.cumsum(T)))
    V2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N2)], dtype=NP_INT)
    cs_V2 = np.concatenate((np.array([0]), np.cumsum(V2)))
    ### 1d
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk = x[pos1:next_pos1]
        ### caution -- possible overhead or numerical instability
        chunk_dot, j = np.zeros(t1, dtype=NP_FLOAT), 0
        for k in range(t1):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(Q[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    dot2 = np.zeros_like(x)
    for i1 in prange(N2):
        vol1 = cs_V2[i1]
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        sub_t1 = T[pos1:next_pos1]
        ###
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1
            for _i2 in range(_i1 + 1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += Q[j] * chunk
                _pos2 = _next_pos2
                j += 1
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
    ###
    ### 3d
    dot3 = np.zeros_like(x)
    for i1 in prange(N2):
        j = i1 * (i1 + 1) // 2
        t1, v1 = T[i1], V2[i1]
        pos1, vol1 = cs_T[i1], cs_V2[i1]
        next_pos1, next_vol1 = cs_T[i1 + 1], cs_V2[i1 + 1]
        sub_t1 = T[pos1:next_pos1]
        ###
        pos2, vol2 = 0, 0
        for i2 in range(i1 + 1):
            t2, v2 = T[i2], V2[i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ###
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t1):
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1:_next_pos1] = chunk[_pos2 : _pos2 + _t1]
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += Q[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    return dot3


# md Transformations


@njit
def transform_md(A: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(sum(T^2)*m)"""
    return (
        _transform_md_parallel(A, x, T)
        if PARALLEL
        else _transform_md_sequential(A, x, T)
    )


@njit
def _transform_md_sequential(A: np.ndarray, x: np.ndarray, T: np.ndarray):
    """O(sum(T^2)*m)"""
    N = x.shape[0]
    (
        dot,
        cs_T,
        V0,  # Volume Projection
        V1,  # Next Volume Projection
    ) = (
        x[:],
        np.concatenate((np.array([0]), np.cumsum(T))),
        np.ones(N, dtype=NP_INT),
        T[:],
    )
    # caution -- loop not parallelizable
    l = 0
    while len(V0) > 1:  # O(m)
        pos = 0
        cs_V0 = np.concatenate((np.array([0]), np.cumsum(V0)))
        for _, slot in enumerate(V1):  # O(V1)
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_V0 = V0[
                np.searchsorted(cs_V0, pos) : np.searchsorted(cs_V0, next_pos)
            ]
            len_chunk_V0 = len(chunk_V0)
            cs_chunk_V0 = np.concatenate((np.array([0]), np.cumsum(chunk_V0)))
            chunk_T = T[np.searchsorted(cs_T, pos) : np.searchsorted(cs_T, next_pos)]
            cs_chunk_T = np.concatenate((np.array([0]), np.cumsum(chunk_T)))
            if l > 0:  # Case mD
                chunk_V0_Ts = [
                    chunk_T[
                        np.searchsorted(cs_chunk_T, cs_chunk_V0[i]) : np.searchsorted(
                            cs_chunk_T, cs_chunk_V0[i + 1]
                        )
                    ]
                    for i in range(len_chunk_V0)
                ]
                # BEGIN DOT_LP #
                chunk_dot, j = [
                    np.zeros((size), dtype=NP_FLOAT) for size in chunk_V0
                ], 0
                for i in range(len(chunk_V0_Ts)):  # O(n)
                    j_next = j + i + 1
                    follower = chunk_V0_Ts[i]
                    chunk_pos = 0
                    for k, a in enumerate(A[j:j_next]):
                        leader = chunk_V0_Ts[k]
                        next_chunk_pos = chunk_pos + np.sum(leader)
                        sub = chunk[chunk_pos:next_chunk_pos][:]
                        rc = phi(l, follower, leader)
                        sub_rc = sub[rc]
                        chunk_dot[i] += a * sub_rc
                        chunk_pos = next_chunk_pos
                    j = j_next
                dot[pos:next_pos] = concatenate_arrays(chunk_dot)
                # END DOT_LP #
            else:  # Case 1D
                chunk_dot_1D, k = np.zeros(slot, dtype=NP_FLOAT), 0
                for j in range(slot):  # O(T_i)
                    k_next = k + j + 1
                    chunk_dot_1D[j] = np.sum(A[k:k_next] * chunk[: j + 1])  # O(2*j)
                    k = k_next
                dot[pos:next_pos] = chunk_dot_1D
            pos = next_pos
        V0 = V1[:]
        V1 = reduceat(V1, cs_T)
        l += 1
    return dot


# TODO: Deprecated!
@njit
def _rightchoices(x: np.ndarray, leader: np.ndarray, follower: np.ndarray, depth: int):
    """O(N*(1+kappa))"""
    cil_l = CIL()
    cil_l.init(leader, depth)
    cil_f = CIL()
    cil_f.init(follower, depth)
    y = np.zeros(np.sum(follower), dtype=NP_FLOAT)
    pos_l, pos_f, go = 0, 0, True
    while go:
        indices = cil_l.indices
        next_pos_l = pos_l + cil_l.element
        if cil_f.eval(indices):
            next_pos_f = pos_f + cil_f.element
            y[pos_f:next_pos_f] = x[pos_l : pos_l + cil_f.element]
            pos_f = next_pos_f
        else:
            pass
        if next_pos_f == len(y):
            break
        pos_l = next_pos_l
        go = cil_l.next()
    return y


@njit  # (parallel=True)
def _transform_md_parallel(A: np.ndarray, x: np.ndarray, T: np.ndarray):
    """O(Nmn)"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


@njit  #
def dtransform_md(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    """O(N*n)"""
    return (
        _dtransform_md_parallel(L, x, T)
        if PARALLEL
        else _dtransform_md_sequential(L, x, T)
    )


@njit  #
def _dtransform_md_sequential(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1 = len(T)
    ###
    pos, dot = 0, np.zeros_like(x)
    for i in range(N1):
        t = T[i]
        next_pos = pos + t
        chunk = x[pos:next_pos]
        ###
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


@njit(parallel=True)  #
def _dtransform_md_parallel(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    N1 = len(T)
    cs_T = np.concatenate((np.array([0]), np.cumsum(T)))
    ###
    dot = np.zeros_like(x)
    for i in prange(N1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        next_pos = pos + t
        chunk = x[pos:next_pos]
        ###
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
