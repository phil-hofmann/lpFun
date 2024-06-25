import numpy as np
from numba import njit, prange
from lpfun import NP_FLOAT, NP_INT, NP_ARRAY, PARALLEL
from lpfun.utils import concatenate_arrays, reduceat
from lpfun.iterators import CompatibleIntegerList as CIL

"""
This module contains the Numba JIT-compiled functions for the transformation of a vector x by a matrix A.

The functions are divided into the following categories:
- 1D transformations
- Maximal transformations
- 2D transformations
- Transformations

The functions are used in the Transform methods in the molecules.py module.
"""

# 1D transformations


@njit(parallel=PARALLEL)
def lt_transform(A: NP_ARRAY, x: NP_ARRAY) -> NP_ARRAY:
    """O(n^2)"""
    return _lt_transform_parallel(A, x) if PARALLEL else _lt_transform_sequential(A, x)


@njit
def _lt_transform_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(n^2)"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, j, n = np.zeros_like(x), 0, x.shape[0]
    for i in range(n):  # O(n)
        j_next = j + i + 1
        dot[i] = np.sum(A[j:j_next] * x[: i + 1])  # O(2i)
        j = j_next
    return dot


@njit(parallel=True)
def _lt_transform_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(n^2)"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, n = np.zeros_like(x), x.shape[0]
    for i in prange(n):  # O(n)
        j = (i * (i + 1)) // 2
        dot[i] = np.sum(A[j : j + i + 1] * x[: i + 1])  # O(2i)
    return dot


@njit(parallel=PARALLEL)
def ut_transform(A: NP_ARRAY, x: NP_ARRAY) -> NP_ARRAY:
    """O(n^2)"""
    return _ut_transform_parallel(A, x) if PARALLEL else _ut_transform_sequential(A, x)


@njit
def _ut_transform_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(n^2)"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, j, n = np.zeros_like(x), 0, x.shape[0]
    for i in range(n):  # O(n)
        i_prime = n - i - 1
        j_next = j + i_prime + 1
        dot[i] = np.sum(A[j:j_next] * x[i:n])  # O(2i)
        j = j_next
    return dot


@njit(parallel=True)
def _ut_transform_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(n^2)"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, n = np.zeros_like(x), x.shape[0]
    for i in prange(n):  # O(n)
        i_prime = n - i - 1
        j = (i_prime * (i_prime + 1)) // 2
        j_next = j + i_prime + 1
        dot[i] = np.sum(A[j:j_next] * x[i:n])  # O(2i)
    return dot


# Maximal transformations


@njit(parallel=PARALLEL)
def n_transform_maximal(A: NP_ARRAY, x: NP_ARRAY) -> NP_ARRAY:
    """O(N*n*m)"""
    return (
        _n_transform_maximal_parallel(A, x)
        if PARALLEL
        else _n_transform_maximal_sequential(A, x)
    )


@njit
def _n_transform_maximal_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n*m)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    dot, slot, remainder = np.copy(x), 1, N
    for _ in range(m):  # O(m)
        slot *= n  # = n^(l+1)
        remainder //= n  # = n^(m-l-1)
        splits = slot // n  # = n^l
        pos = 0
        for _ in range(remainder):  # O(remainder)
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_splits = np.array_split(chunk, n)
            chunk_dot, j = np.zeros((n, splits), dtype=NP_FLOAT), 0
            for i in range(n):  # O(n)
                j_next = j + i + 1
                chunk_splits_i = chunk_splits[0 : i + 1]
                for k, a in enumerate(A[j:j_next]):  # O(i)
                    chunk_dot[i] += a * chunk_splits_i[k]  # O(2splits)
                j = j_next
            dot[pos:next_pos] = chunk_dot.reshape(-1)
            pos = next_pos
    return dot


@njit(parallel=True)
def _n_transform_maximal_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n*m)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    dot, slot, remainder = np.copy(x), 1, N
    # caution -- loop not parallelizable
    for _ in range(m):  # O(m)
        slot *= n  # = n^(l+1)
        remainder //= n  # = n^(m-l-1)
        splits = slot // n  # = n^l
        for f in prange(remainder):  # O(remainder)
            pos = f * slot
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_splits = np.array_split(chunk, n)
            chunk_dot = np.zeros((n, splits), dtype=NP_FLOAT)
            # caution -- possible overhead
            for i in range(n):  # O(n)
                j = (i * (i + 1)) // 2
                chunk_splits_i = chunk_splits[: i + 1]
                # caution -- possible overhead
                for k, a in enumerate(A[j : j + i + 1]):  # O(i)
                    chunk_dot[i] += a * chunk_splits_i[k]  # O(2splits)
            dot[pos:next_pos] = chunk_dot.reshape(-1)
            pos = next_pos
    return dot


@njit(parallel=PARALLEL)
def ut_diag_transform_maximal(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    return (
        _ut_diag_transform_maximal_parallel(A, x)
        if PARALLEL
        else _ut_diag_transform_maximal_sequential(A, x)
    )


@njit
def _ut_diag_transform_maximal_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    pos, splits, result = 0, N // n, np.copy(x)
    for _ in range(splits):  # O(n^m)
        next_pos = pos + n
        chunk = result[pos:next_pos]
        chunk_dot, j, s, k = np.zeros(n, dtype=NP_FLOAT), 0, 0, n
        for i in range(n):  # O(n)
            j_next = j + k
            chunk_dot[i] = np.sum(A[j:j_next] * chunk[s:n])  # O(2*i)
            j, k, s = j_next, k - 1, s + 1
        result[pos:next_pos] = chunk_dot
        pos = next_pos
    return result


@njit(parallel=True)
def _ut_diag_transform_maximal_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


@njit(parallel=PARALLEL)
def lt_diag_transform_maximal(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    return (
        _lt_diag_transform_maximal_parallel(A, x)
        if PARALLEL
        else _lt_diag_transform_maximal_sequential(A, x)
    )


@njit
def _lt_diag_transform_maximal_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    pos, splits, result = 0, N // n, np.copy(x)
    for _ in range(splits):  # O(n^m)
        next_pos = pos + n
        chunk = result[pos:next_pos]
        chunk_dot, j = np.zeros(n, dtype=NP_FLOAT), 0
        for i in range(n):  # O(n)
            j_next = j + i + 1
            chunk_dot[i] = np.sum(A[j:j_next] * chunk[: i + 1])  # O(2*i)
            j = j_next
        result[pos:next_pos] = chunk_dot
        pos = next_pos
    return result


@njit(parallel=True)
def _lt_diag_transform_maximal_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(N*n)"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


# 2D transformations


@njit(parallel=PARALLEL)
def n_transform_2d(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY) -> NP_ARRAY:
    """O(sum(T^2)*2)"""
    return (
        _n_transform_2d_parallel(A, x, T)
        if PARALLEL
        else _n_transform_2d_sequential(A, x, T)
    )


@njit
def _n_transform_2d_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(2*sum(T^2))"""
    dot_1d = np.zeros_like(x)
    dot_2d = np.zeros_like(x)
    l, pos = 0, 0
    for i, slot in enumerate(T):  # O(T) = O(n) (in 2d)
        next_pos = pos + slot
        chunk = x[pos : pos + slot]
        chunk_dot, k = np.zeros(slot, dtype=NP_FLOAT), 0
        for j in range(slot):  # O(T_i)
            k_next = k + j + 1
            chunk_dot[j] = np.sum(A[k:k_next] * chunk[: j + 1])  # O(2j)
            k = k_next
        dot_1d[pos:next_pos] = chunk_dot
        inner_pos = 0
        for j in range(i + 1):  # O(i)
            dot_2d[pos:next_pos] += (
                A[l] * dot_1d[inner_pos : inner_pos + slot]
            )  # O(2T_i)
            inner_pos += T[j]
            l += 1
        pos = next_pos
    return dot_2d


@njit(parallel=True)
def _n_transform_2d_parallel(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(2*sum(T^2))"""
    length = T.shape[0]
    dot_1d = np.zeros_like(x)
    for i in prange(length):  # O(T)
        slot, pos = T[i], np.sum(T[:i])
        next_pos = pos + slot
        chunk = x[pos : pos + slot]
        chunk_dot = np.zeros(slot, dtype=NP_FLOAT)
        # caution -- possible overhead
        for j in range(slot):  # O(T_i)
            k = (j * (j + 1)) // 2
            chunk_dot[j] = np.sum(A[k : k + j + 1] * chunk[0 : j + 1])  # O(2j)
        dot_1d[pos:next_pos] = chunk_dot
    dot_2d = np.zeros_like(x)
    for i in prange(length):  # O(T)
        slot, pos = T[i], np.sum(T[:i])
        next_pos = pos + slot
        # caution -- possible overhead
        for j in range(i + 1):  # O(i)
            inner_pos = np.sum(T[:j])
            dot_2d[pos:next_pos] += (
                A[(i * (i + 1)) // 2 + j] * dot_1d[inner_pos : inner_pos + slot]
            )  # O(2T_i)
        pos = next_pos
    return dot_2d


# transformations


@njit(parallel=PARALLEL)
def n_transform_md(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY) -> NP_ARRAY:
    """O(sum(T^2)*m)"""
    return (
        _n_transform_md_parallel(A, x, T) if PARALLEL else _n_transform_md_sequential(A, x, T)
    )


@njit
def _n_transform_md_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2)*m)"""
    N = x.shape[0]
    (
        dot,
        cumsum_T,
        T0,
        T1,
    ) = (
        np.copy(x),
        np.concatenate((np.array([0]), np.cumsum(T))),
        np.ones(N, dtype=NP_INT),
        np.copy(T),
    )
    # caution -- loop not parallelizable
    l = 0
    while True:
        if len(T0) == 1:
            break
        pos = 0
        cumsum_T0 = np.concatenate((np.array([0]), np.cumsum(T0)))
        for _, slot in enumerate(T1):
            next_pos = pos + slot
            chunk = dot[pos:next_pos]
            chunk_T0 = T0[
                np.searchsorted(cumsum_T0, pos) : np.searchsorted(cumsum_T0, next_pos)
            ]
            len_chunk_T0 = len(chunk_T0)
            cumsum_chunk_T0 = np.concatenate((np.array([0]), np.cumsum(chunk_T0)))
            chunk_T = T[
                np.searchsorted(cumsum_T, pos) : np.searchsorted(cumsum_T, next_pos)
            ]
            cumsum_chunk_T = np.concatenate((np.array([0]), np.cumsum(chunk_T)))
            if l > 0:  # Case mD
                chunk_T0_Ts = [
                    chunk_T[
                        np.searchsorted(
                            cumsum_chunk_T, cumsum_chunk_T0[i]
                        ) : np.searchsorted(cumsum_chunk_T, cumsum_chunk_T0[i + 1])
                    ]
                    for i in range(len_chunk_T0)
                ]
                # BEGIN DOT_LP #
                chunk_dot, j = [
                    np.zeros((size), dtype=NP_FLOAT) for size in chunk_T0
                ], 0
                for i in range(len(chunk_T0_Ts)):  # O(n)
                    j_next = j + i + 1
                    follower = chunk_T0_Ts[i]
                    chunk_pos = 0
                    for k, a in enumerate(A[j:j_next]):
                        leader = chunk_T0_Ts[k]
                        next_chunk_pos = chunk_pos + np.sum(leader)
                        sub = np.copy(chunk[chunk_pos:next_chunk_pos])
                        sub_rc = _rightchoices(sub, leader, follower, l + 1)
                        chunk_dot[i] += a * sub_rc
                        chunk_pos = next_chunk_pos
                    j = j_next
                dot[pos:next_pos] = concatenate_arrays(chunk_dot)
                # END DOT_LP #
            else:  # Case 1D
                chunk_dot_1D, k = np.zeros(slot, dtype=NP_FLOAT), 0
                for j in range(slot):  # O(T_i)
                    k_next = k + j + 1
                    chunk_dot_1D[j] = np.sum(A[k:k_next] * chunk[: j + 1])  # O(2j)
                    k = k_next
                dot[pos:next_pos] = chunk_dot_1D
            pos = next_pos
        T0 = np.copy(T1)
        T1 = reduceat(T1, cumsum_T)
        l += 1
    return dot


@njit
def _rightchoices(x: NP_ARRAY, leader: NP_ARRAY, follower: NP_ARRAY, depth: int):
    """O(???)"""
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


@njit(parallel=True)
def _n_transform_md_parallel(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2)*m)"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


@njit(parallel=PARALLEL)
def ut_diag_transform(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    return (
        _ut_diag_transform_parallel(A, x, T)
        if PARALLEL
        else _ut_diag_transform_sequential(A, x, T)
    )


@njit
def _ut_diag_transform_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    n, pos, result = np.max(T), 0, np.zeros_like(x)
    for slot in T:  # O(len(T))
        next_pos = pos + slot
        chunk = x[pos:next_pos]
        chunk_dot, j, s, k = np.zeros(slot, dtype=NP_FLOAT), 0, 0, n
        for i in range(slot):  # O(slot)
            j_next, j_inner = j + k, j + slot - s
            chunk_dot[i] = np.sum(A[j:j_inner] * chunk[s:slot])  # O(2*slot)
            j, k, s = j_next, k - 1, s + 1
        result[pos:next_pos] = chunk_dot
        pos = next_pos
    return result


@njit
def _ut_diag_transform_parallel(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    # TODO
    raise NotImplementedError("Not implemented yet.")


@njit(parallel=PARALLEL)
def lt_diag_transform(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    return (
        _lt_diag_transform_parallel(A, x, T)
        if PARALLEL
        else _lt_diag_transform_sequential(A, x, T)
    )


@njit
def _lt_diag_transform_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    pos, result = 0, np.zeros_like(x)
    for slot in T:  # O(len(T))
        next_pos = pos + slot
        chunk = x[pos:next_pos]
        chunk_dot, j = np.zeros(slot, dtype=NP_FLOAT), 0
        for i in range(slot):  # O(slot)
            j_next = j + i + 1
            chunk_dot[i] = np.sum(A[j:j_next] * chunk[: i + 1])  # O(2*slot)
            j = j_next
        result[pos:next_pos] = chunk_dot
        pos = next_pos
    return result


@njit
def _lt_diag_transform_parallel(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(sum(T^2))"""
    # TODO
    raise NotImplementedError("Not implemented yet.")
