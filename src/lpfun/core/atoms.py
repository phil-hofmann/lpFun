import numpy as np
from typing import Literal
from numba import njit, prange
from lpfun import NP_FLOAT, NP_INT
from lpfun.iterators import CompatibleIntegerList as CIL  # TODO Remove
from lpfun.utils import concatenate_arrays, reduceat, phi

"""
Comments:
    - This module contains numba jit-compiled functions for the transformation of a vector by a matrix.
    - The functions are divided into the following categories: 1d, maximal, 2d, 3d and md transformations.
    - The functions are used in the transform methods in the molecules.py module.
"""

# 1d


@njit
def transform_lt_1d(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(n**2)"""
    ### NOTE -- parallelization impossible
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
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


@njit
def transform_ut_1d(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(n**2)"""
    ### NOTE -- parallelization impossible
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = x.shape[0]
    ###
    dot, j = np.zeros_like(x), n * (n + 1) // 2
    for k in range(n):
        k_prime = n - k - 1
        j_next = j - k - 1
        dotsum = np.sum(U[j_next:j] * dot[k_prime:])
        dot[k_prime] = (x[k_prime] - dotsum) / U[j_next]
        j = j_next
    ###
    return dot


@njit
def itransform_lt_1d(
    L: np.ndarray,
    x: np.ndarray,
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """O(n**2)"""
    if parallel == "cpu":
        return _itransform_lt_1d_cpu(L, x)
    elif parallel == "seq":
        return _itransform_lt_1d_seq(L, x)


@njit
def _itransform_lt_1d_seq(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = x.shape[0]
    ###
    dot, j = np.zeros_like(x), 0
    for k in range(n):
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
        j = j_next
    ###
    return dot


@njit(parallel=True)
def _itransform_lt_1d_cpu(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = x.shape[0]
    ###
    dot = np.zeros_like(x)
    for k in prange(n):
        j = (k * (k + 1)) // 2
        j_next = j + k + 1
        dot[k] = np.sum(L[j:j_next] * x[: k + 1])
    ###
    return dot


@njit
def itransform_ut_1d(
    L: np.ndarray,
    x: np.ndarray,
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """O(n**2)"""
    if parallel == "cpu":
        return _itransform_ut_1d_cpu(L, x)
    elif parallel == "seq":
        return _itransform_ut_1d_seq(L, x)


@njit
def _itransform_ut_1d_seq(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = x.shape[0]
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
def _itransform_ut_1d_cpu(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    n = x.shape[0]
    ###
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
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """O(N*n*m)"""
    ### NOTE -- parallelization recommended -- but becomes less effective at deeper levels of the divide-and-conquer tree
    if parallel == "cpu":
        return _transform_lt_max_cpu(L, x)
    elif parallel == "seq":
        return _transform_lt_max_seq(L, x)


@njit
def _transform_lt_max_seq(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
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


@njit(parallel=True)
def _transform_lt_max_cpu(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
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


@njit
def transform_ut_max(
    U: np.ndarray, x: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n*m)"""
    ### NOTE -- parallelization recommended -- but becomes less effective at deeper levels of the divide-and-conquer tree
    if parallel == "cpu":
        return _transform_ut_max_cpu(U, x)
    elif parallel == "seq":
        return _transform_ut_max_seq(U, x)


@njit
def _transform_ut_max_seq(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * U.shape[0]) - 1) / 2),
    )
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
            chunk_dot, pos2, j = (
                np.zeros((slot), dtype=NP_FLOAT),
                slot,
                n * (n + 1) // 2,
            )
            for k in range(n):
                next_pos2 = pos2 - splits
                j_next = j - k - 1
                ###
                dotsum, pos3 = np.zeros(splits, dtype=NP_FLOAT), slot
                for l in range(j - 1, j_next, -1):
                    next_pos3 = pos3 - splits
                    dotsum += U[l] * chunk_dot[next_pos3:pos3]
                    pos3 = next_pos3
                next_pos3 = pos3 - splits
                chunk_dot[next_pos2:pos2] = (chunk[next_pos3:pos3] - dotsum) / U[j_next]
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
            pos1 = next_pos1
        ###
    ###
    return dot


@njit(parallel=True)
def _transform_ut_max_cpu(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * U.shape[0]) - 1) / 2),
    )
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
            chunk_dot, pos2, j = (
                np.zeros((slot), dtype=NP_FLOAT),
                slot,
                n * (n + 1) // 2,
            )
            for k in range(n):
                next_pos2 = pos2 - splits
                j_next = j - k - 1
                ###
                dotsum, pos3 = np.zeros(splits, dtype=NP_FLOAT), slot
                for l in range(j - 1, j_next, -1):
                    next_pos3 = pos3 - splits
                    dotsum += U[l] * chunk_dot[next_pos3:pos3]
                    pos3 = next_pos3
                next_pos3 = pos3 - splits
                chunk_dot[next_pos2:pos2] = (chunk[next_pos3:pos3] - dotsum) / U[j_next]
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos:next_pos] = chunk_dot
            pos = next_pos
        ###
    ###
    return dot


@njit
def itransform_lt_max(
    L: np.ndarray, x: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n*m)"""
    if parallel == "cpu":
        return _itransform_lt_max_cpu(L, x)
    elif parallel == "seq":
        return _itransform_lt_max_seq(L, x)


@njit
def _itransform_lt_max_seq(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
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
                pos3 = 0
                for l in range(j, j_next):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += L[l] * chunk[pos3:next_pos3]
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


@njit(parallel=True)
def _itransform_lt_max_cpu(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
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
                ### caution -- possible overhead or numerical instability
                pos3 = 0
                for l in range(j, j_next):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += L[l] * chunk[pos3:next_pos3]
                    pos3 = next_pos3
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
        ###
    ###
    return dot


@njit
def itransform_ut_max(
    U: np.ndarray, x: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n*m)"""
    if parallel == "cpu":
        return _itransform_ut_max_cpu(U, x)
    elif parallel == "seq":
        return _itransform_ut_max_seq(U, x)


@njit
def _itransform_ut_max_seq(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * U.shape[0]) - 1) / 2),
    )
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
                k_prime = n - k - 1
                j_next = j + k_prime + 1
                ###
                pos3 = pos2
                for l in range(j, j_next):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += U[l] * chunk[pos3:next_pos3]
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


@njit(parallel=True)
def _itransform_ut_max_cpu(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    U, x = np.asarray(U).astype(NP_FLOAT), np.asarray(x).astype(NP_FLOAT)
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * U.shape[0]) - 1) / 2),
    )
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
            ### caution -- possible overhead or numerical instability
            chunk_dot, pos2, j = np.zeros((slot), dtype=NP_FLOAT), 0, 0
            for k in range(n):
                next_pos2 = pos2 + splits
                k_prime = n - k - 1
                j_next = j + k_prime + 1
                ### caution -- possible overhead or numerical instability
                pos3 = pos2
                for l in range(j, j_next):
                    next_pos3 = pos3 + splits
                    chunk_dot[pos2:next_pos2] += U[l] * chunk[pos3:next_pos3]
                    pos3 = next_pos3
                ###
                pos2 = next_pos2
                j = j_next
            ###
            dot[pos1:next_pos1] = chunk_dot
        ###
    ###
    return dot


@njit
def dtransform_max(
    L: np.ndarray, x: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n)"""
    if parallel == "cpu":
        return _dtransform_max_cpu(L, x)
    elif parallel == "seq":
        return _dtransform_max_seq(L, x)


@njit
def _dtransform_max_seq(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
    ###
    pos, splits, dot = 0, N // n, x[:]
    for _ in range(splits):
        next_pos = pos + n
        chunk = dot[pos:next_pos]
        chunk_dot, j = np.zeros(n, dtype=NP_FLOAT), 0
        for k in range(n):
            j_next = j + k + 1
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        dot[pos:next_pos] = chunk_dot
        pos = next_pos
    ###
    return dot


@njit(parallel=True)
def _dtransform_max_cpu(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    L, x = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
    )
    N, n = (
        x.shape[0],
        int((np.sqrt(1 + 8 * L.shape[0]) - 1) / 2),
    )
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


# 2d


@njit
def transform_lt_2d(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2*N*n)"""
    ### NOTE -- parallelization possible but not recommended
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
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
            dotsum = np.sum(L[j : j_next - 1] * chunk_dot[:k])
            chunk_dot[k] = (chunk[k] - dotsum) / L[j_next - 1]
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    dot2, pos1, j = np.zeros_like(x), 0, 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ###
        pos2, dotsum = 0, np.zeros(t1, dtype=NP_FLOAT)
        for i2 in range(i1):
            t2 = T[i2]
            next_pos2 = pos2 + t2
            chunk = dot2[pos2 : pos2 + t1]
            dotsum += L[j] * chunk
            pos2 = next_pos2
            j += 1
        dot2[pos1:next_pos1] = (dot1[pos1:next_pos1] - dotsum) / L[j]
        j += 1
        ###
        pos1 = next_pos1
    ###
    return dot2


@njit
def transform_ut_2d(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(2*N*n)"""
    ### NOTE -- parallelization possible but not recommended
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
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N1 - t1
        chunk = x[pos1:next_pos1]
        ###
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
    dot2, pos1, j = np.zeros_like(x), N0, N1 * (N1 + 1) // 2
    for i1 in range(N1):
        i1_prime = N1 - i1 - 1
        t1 = T[i1_prime]
        next_pos1 = pos1 - t1
        ###
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
    L: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(2*N*n)"""
    if parallel == "cpu":
        return _itransform_lt_2d_cpu(L, x, T)
    elif parallel == "seq":
        return _itransform_lt_2d_seq(L, x, T)


@njit
def _itransform_lt_2d_seq(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
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
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
        pos1 = next_pos1
    ###
    ### 2d
    dot2, j, pos1 = np.zeros_like(x), 0, 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ###
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
def _itransform_lt_2d_cpu(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    L = np.asarray(L).astype(NP_FLOAT)
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
    U: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(2*N*n)"""
    if parallel == "cpu":
        return _itransform_ut_2d_cpu(U, x, T)
    elif parallel == "seq":
        return _itransform_ut_2d_seq(U, x, T)


@njit
def _itransform_ut_2d_seq(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1 = len(T)
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N1 - t1
        chunk = x[pos1:next_pos1]
        ###
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
    dot2, pos1, j = np.zeros_like(x), 0, 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1 = pos1 + t1
        ###
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
def _itransform_ut_2d_cpu(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, cs_T = (
        len(T),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ### 1d
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk, delta = x[pos1:next_pos1], N1 - t1
        ### caution -- possible overhead or numerical instability
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
    dot2 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        ### caution -- possible overhead or numerical instability
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


@njit  # TODO
def transform_lt_3d(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    """O(3*N*n)"""
    # NOTE -- parallelization possible but not recommended
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
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
            dotsum = np.sum(L[j : j_next - 1] * chunk_dot[:k])
            chunk_dot[k] = (chunk[k] - dotsum) / L[j_next - 1]
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
            _pos2, dotsum = vol1, np.zeros(_t1, dtype=NP_FLOAT)
            for _i2 in range(_i1):
                _t2 = sub_t1[_i2]
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2 : _pos2 + _t1]
                dot2[_pos1:_next_pos1] += L[j] * chunk
                _pos2 = _next_pos2
                j += 1
            dot2[_pos1:_next_pos1] = (dot1[_pos1:_next_pos1] - dotsum) / L[j]
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
        pos2, vol2, dotsum = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
        for i2 in range(i1):
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


@njit  # TODO
def transform_ut_3d(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Not implemented yet.")


@njit
def itransform_lt_3d(
    L: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(3*N*n)"""
    if parallel == "cpu":
        return _itransform_lt_3d_cpu(L, x, T)
    elif parallel == "seq":
        return _itransform_lt_3d_seq(L, x, T)


@njit
def _itransform_lt_3d_seq(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, N2 = (
        len(T),
        T[0],
    )
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
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
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
            dot3[vol1:next_vol1] += L[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit(parallel=True)
def _itransform_lt_3d_cpu(L: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, N2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
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
            chunk_dot[k] = np.sum(L[j:j_next] * chunk[: k + 1])
            j = j_next
        ###
        dot1[pos1:next_pos1] = chunk_dot
    ###
    ### 2d
    dot2 = np.zeros_like(x)
    for i1 in prange(N2):
        t1, pos1, vol1, next_pos1 = T[i1], cs_T[i1], cs_V2[i1], cs_T[i1 + 1]
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
                dot2[_pos1:_next_pos1] += L[j] * chunk
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
    U: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(3*N*n)"""
    if parallel == "cpu":
        return _itransform_ut_3d_cpu(U, x, T)
    elif parallel == "seq":
        return _itransform_ut_3d_seq(U, x, T)


@njit
def _itransform_ut_3d_seq(U: np.ndarray, x: np.ndarray, T: np.ndarray):
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, N2 = (
        len(T),
        T[0],
    )
    ### 1d
    dot1, pos1 = np.zeros_like(x), 0
    for i1 in range(N1):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1
        chunk = x[pos1:next_pos1]
        ###
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
    pos1, vol1, dot2, V2 = 0, 0, np.zeros_like(x), np.zeros(N2, dtype=NP_INT)
    for i1 in range(N2):
        t1 = T[i1]
        next_pos1, delta = pos1 + t1, N2 - t1  # changed!
        sub_t1 = T[pos1:next_pos1]
        ###
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1 + _vol1  # changed!
            for _i2 in range(t1 - _i1):  # changed!
                _i1_i2 = _i1 + _i2
                _t2 = sub_t1[_i1_i2]  # changed!
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2:_next_pos2]  # changed!
                dot2[_pos1 : _pos1 + _t2] += U[j] * chunk  # changed!
                _pos2 = _next_pos2
                j += 1
            j += delta
            _pos1 = _next_pos1
            _vol1 += _t1
        ###
        pos1, vol1, V2[i1] = next_pos1, _pos1, _vol1
    ###
    ### 3d
    dot3, pos1, vol1, j = np.zeros_like(x), 0, 0, 0
    for i1 in range(N2):
        t1, v1 = T[i1], V2[i1]
        next_pos1, next_vol1 = pos1 + t1, vol1 + v1
        sub_t1 = T[pos1:next_pos1]
        ###
        pos2, vol2 = pos1, vol1  # changed!
        for i2 in range(N2 - i1):
            i1_i2 = i1 + i2
            t2, v2 = T[i1_i2], V2[i1_i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ###
            _pos1, _pos2, sub = 0, 0, np.zeros(v1, dtype=NP_FLOAT)
            for i in range(t2):  # changed!
                _t1, _t2 = sub_t1[i], sub_t2[i]
                _next_pos1, _next_pos2 = _pos1 + _t1, _pos2 + _t2
                sub[_pos1 : _pos1 + _t2] = chunk[_pos2:_next_pos2]  # changed!
                _pos1, _pos2 = _next_pos1, _next_pos2
            dot3[vol1:next_vol1] += U[j] * sub
            ###
            pos2, vol2 = next_pos2, next_vol2
            j += 1
        ###
        pos1, vol1 = next_pos1, next_vol1
    ###
    return dot3


@njit(parallel=True)
def _itransform_ut_3d_cpu(U: np.ndarray, x: np.ndarray, T: np.ndarray) -> np.ndarray:
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, N2, cs_T = (
        len(T),
        T[0],
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    V2 = np.array([np.sum(T[cs_T[i] : cs_T[i + 1]]) for i in range(N2)], dtype=NP_INT)
    cs_V2 = np.concatenate((np.array([0]), np.cumsum(V2)))
    ### 1d
    dot1 = np.zeros_like(x)
    for i1 in prange(N1):
        t1, pos1, next_pos1 = T[i1], cs_T[i1], cs_T[i1 + 1]
        chunk, delta = x[pos1:next_pos1], N2 - t1
        ### caution -- possible overhead or numerical instability
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
    dot2 = np.zeros_like(x)
    for i1 in prange(N2):
        t1, pos1, vol1, next_pos1 = T[i1], cs_T[i1], cs_V2[i1], cs_T[i1 + 1]
        sub_t1, delta = T[pos1:next_pos1], N2 - t1
        ###
        _pos1, _vol1, j = vol1, 0, 0
        for _i1 in range(t1):
            _t1 = sub_t1[_i1]
            _next_pos1 = _pos1 + _t1
            _pos2 = vol1 + _vol1  # changed!
            for _i2 in range(t1 - _i1):  # changed!
                _i1_i2 = _i1 + _i2
                _t2 = sub_t1[_i1_i2]  # changed!
                _next_pos2 = _pos2 + _t2
                chunk = dot1[_pos2:_next_pos2]  # changed!
                dot2[_pos1 : _pos1 + _t2] += U[j] * chunk  # changed!
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
    dot3 = np.zeros_like(x)
    for i1 in prange(N2):
        j = i1 * N2 - i1 * (i1 - 1) // 2  # changed!
        t1, v1 = T[i1], V2[i1]
        pos1, vol1 = cs_T[i1], cs_V2[i1]  # changed!
        next_pos1, next_vol1 = cs_T[i1 + 1], cs_V2[i1 + 1]  # changed!
        sub_t1 = T[pos1:next_pos1]
        ###
        pos2, vol2 = pos1, vol1
        for i2 in range(N2 - i1):
            i1_i2 = i1 + i2
            t2, v2 = T[i1_i2], V2[i1_i2]
            next_pos2, next_vol2 = pos2 + t2, vol2 + v2
            sub_t2 = T[pos2:next_pos2]
            chunk = dot2[vol2:next_vol2]
            ###
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


# md


@njit
def itransform_lt_md(
    L: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(sum(T^2)*m)"""
    if parallel == "cpu":
        return _itransform_lt_md_cpu(L, x, T)
    elif parallel == "seq":
        return _itransform_lt_md_seq(L, x, T)


@njit
def _itransform_lt_md_seq(A: np.ndarray, x: np.ndarray, T: np.ndarray):
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
def _itransform_lt_md_cpu(A: np.ndarray, x: np.ndarray, T: np.ndarray):
    """O(Nmn)"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


@njit
def dtransform_lt_md(
    L: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n)"""
    if parallel == "cpu":
        return _dtransform_lt_md_cpu(L, x, T)
    elif parallel == "seq":
        return _dtransform_lt_md_seq(L, x, T)


@njit
def _dtransform_lt_md_seq(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1 = len(T)
    ###
    dot, pos = np.zeros_like(x), 0
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


@njit(parallel=True)
def _dtransform_lt_md_cpu(L: np.ndarray, x: np.ndarray, T: np.ndarray):
    L, x, T = (
        np.asarray(L).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
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
    U: np.ndarray, x: np.ndarray, T: np.ndarray, parallel: Literal["seq", "cpu"]
) -> np.ndarray:
    """O(N*n)"""
    if parallel == "cpu":
        return _dtransform_ut_md_cpu(U, x, T)
    elif parallel == "seq":
        return _dtransform_ut_md_seq(U, x, T)


@njit
def _dtransform_ut_md_seq(U: np.ndarray, x: np.ndarray, T: np.ndarray):
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, Nm = (
        len(T),
        int(T[0]),
    )
    ###
    pos, dot = 0, np.zeros_like(x)
    for i in range(N1):
        t = T[i]
        next_pos, delta = pos + t, Nm - t
        chunk = x[pos:next_pos]
        ###
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
def _dtransform_ut_md_cpu(U: np.ndarray, x: np.ndarray, T: np.ndarray):
    U, x, T = (
        np.asarray(U).astype(NP_FLOAT),
        np.asarray(x).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    N1, Nm, cs_T = (
        len(T),
        int(T[0]),
        np.concatenate((np.array([0]), np.cumsum(T))),
    )
    ###
    dot = np.zeros_like(x)
    for i in prange(N1):
        t, pos, next_pos = T[i], cs_T[i], cs_T[i + 1]
        chunk, delta = x[pos:next_pos], Nm - t
        ###
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
