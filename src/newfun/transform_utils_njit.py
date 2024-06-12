import numpy as np
from numba import njit, prange
from newfun import NP_FLOAT, NP_INT, NP_ARRAY
from newfun.core import CompatibleIntegerList as CIL
from newfun.utils_njit import concatenate_arrays, reduceat


@njit
def transform_1d_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(n(n+1))"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, j, n = np.zeros_like(x), 0, x.shape[0]
    for i in range(n):  # O(n)
        j_next = j + i + 1
        dot[i] = np.sum(A[j:j_next] * x[0 : i + 1])  # O(2i)
        j = j_next
    return dot


@njit(parallel=True)
def transform_1d_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(n(n+1))"""
    x = np.asarray(x).astype(NP_FLOAT)
    dot, n = np.zeros_like(x), x.shape[0]
    for i in prange(n):  # O(n)
        j = (i * (i + 1)) // 2
        j_next = j + i + 1
        dot[i] = np.sum(A[j:j_next] * x[0 : i + 1])  # O(2i)
    return dot


@njit
def transform_maximal_sequential(A: NP_ARRAY, x: NP_ARRAY):
    """O(m(n+1)N)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    result, slot, remainder = np.copy(x), 1, N
    for _ in range(m):  # O(m)
        slot *= n  # = n^(l+1)
        remainder //= n  # = n^(m-l-1)
        splits = slot // n  # = n^l
        pos = 0
        for _ in range(remainder):  # O(remainder)
            next_pos = pos + slot
            chunk = result[pos:next_pos]
            chunk_splits = np.array_split(chunk, n)
            chunk_dot, j = np.zeros((n, splits), dtype=NP_FLOAT), 0
            for i in range(n):  # O(n)
                j_next = j + i + 1
                chunk_splits_i = chunk_splits[0 : i + 1]
                for k, a in enumerate(A[j:j_next]):  # O(i)
                    chunk_dot[i] += a * chunk_splits_i[k]  # O(2splits)
                j = j_next
            result[pos:next_pos] = chunk_dot.reshape(-1)
            pos = next_pos
    return result


@njit(parallel=True)
def transform_maximal_parallel(A: NP_ARRAY, x: NP_ARRAY):
    """O(m(n+1)N)"""
    N, n = x.shape[0], int((np.sqrt(1 + 8 * A.shape[0]) - 1) / 2)
    m = int(np.log(N) / np.log(n))
    result, slot, remainder = np.copy(x), 1, N
    # caution -- loop not parallelizable
    for _ in range(m):  # O(m)
        slot *= n  # = n^(l+1)
        remainder //= n  # = n^(m-l-1)
        splits = slot // n  # = n^l
        for f in prange(remainder):  # O(remainder)
            pos = f * slot
            next_pos = pos + slot
            chunk = result[pos:next_pos]
            chunk_splits = np.array_split(chunk, n)
            chunk_dot = np.zeros((n, splits), dtype=NP_FLOAT)
            for i in prange(n):  # O(n)
                j = (i * (i + 1)) // 2
                j_next = j + i + 1
                chunk_splits_i = chunk_splits[0 : i + 1]
                # caution -- possible overhead
                for k, a in enumerate(A[j:j_next]):  # O(i)
                    chunk_dot[i] += a * chunk_splits_i[k]  # O(2splits)
            result[pos:next_pos] = chunk_dot.reshape(-1)
            pos = next_pos
    return result


@njit
def transform_2d_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(2|A_{3,n,p}|))"""
    helper = np.zeros_like(x)
    result = np.zeros_like(x)
    l, pos = 0, 0
    for i, slot in enumerate(T):  # O(T) = O(n) (in 2d)
        next_pos, chunk = pos + slot, x[pos : pos + slot]
        chunk_dot, k = np.zeros(slot, dtype=NP_FLOAT), 0
        for j in range(slot):  # O(T_i)
            k_next = k + j + 1
            chunk_dot[j] = np.sum(A[k:k_next] * chunk[0 : j + 1])  # O(2j)
            k = k_next
        helper[pos:next_pos] = chunk_dot
        inner_pos = 0
        for j in range(i + 1):  # O(i)
            result[pos:next_pos] += (
                A[l] * helper[inner_pos : inner_pos + slot]
            )  # O(2T_i)
            inner_pos += T[j]
            l += 1
        pos = next_pos
    return result


@njit(parallel=True)
def transform_2d_parallel(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(2|A_{3,n,p}|))"""
    length = T.shape[0]
    helper = np.zeros_like(x)
    for i in prange(length):  # O(T)
        slot = T[i]
        pos = np.sum(T[:i])
        next_pos, chunk = pos + slot, x[pos : pos + slot]
        chunk_dot = np.zeros(slot, dtype=NP_FLOAT)
        for j in prange(slot):  # O(T_i)
            k = (j * (j + 1)) // 2
            k_next = k + j + 1
            chunk_dot[j] = np.sum(A[k:k_next] * chunk[0 : j + 1])  # O(2j)
        helper[pos:next_pos] = chunk_dot
    result = np.zeros_like(x)
    for i in prange(length):  # O(T)
        slot = T[i]
        pos = np.sum(T[:i])
        next_pos = pos + slot
        for j in prange(i + 1):  # O(i)
            inner_pos = np.sum(T[:j])
            l = (i * (i + 1)) // 2 + j
            result[pos:next_pos] += (
                A[l] * helper[inner_pos : inner_pos + slot]
            )  # O(2T_i)
        pos = next_pos
    return result


@njit
def transform_md_sequential(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(T^2*m)"""
    N = x.shape[0]
    (
        result,
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
            chunk = result[pos:next_pos]
            chunk_T0 = T0[
                np.searchsorted(cumsum_T0, pos) : np.searchsorted(cumsum_T0, next_pos)
            ]
            len_chunk_T0 = len(chunk_T0)
            cumsum_chunk_T0 = np.concatenate((np.array([0]), np.cumsum(chunk_T0)))
            chunk_T = T[
                np.searchsorted(cumsum_T, pos) : np.searchsorted(cumsum_T, next_pos)
            ]
            cumsum_chunk_T = np.concatenate((np.array([0]), np.cumsum(chunk_T)))
            # ? For non parallel version please put this in the for loop ?
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
                        sub_rc = rightchoices(sub, leader, follower, l)
                        chunk_dot[i] += a * sub_rc
                        chunk_pos = next_chunk_pos
                    j = j_next
                result[pos:next_pos] = concatenate_arrays(chunk_dot)
                # END DOT_LP #
            else:  # Case 1D
                chunk_dot_1D, k = np.zeros(slot, dtype=NP_FLOAT), 0
                for j in range(slot):  # O(T_i)
                    k_next = k + j + 1
                    chunk_dot_1D[j] = np.sum(A[k:k_next] * chunk[0 : j + 1])  # O(2j)
                    k = k_next
                result[pos:next_pos] = chunk_dot_1D
            pos = next_pos
        T0 = np.copy(T1)
        T1 = reduceat(T1, cumsum_T)
        l += 1
    return result


@njit
def rightchoices(x: NP_ARRAY, leader: NP_ARRAY, follower: NP_ARRAY, depth: int):
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
        pos_l = next_pos_l
        go = cil_l.next()
    return y


@njit
def transform_diag_maximal(A: NP_ARRAY, x: NP_ARRAY):
    pass


@njit
def transform_diag(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    """O(|A_{m + 1, n, p}|)"""
    result = np.zeros_like(x)
    pos = 0
    for _, slot in enumerate(T):
        next_pos, chunk = pos + slot, x[pos : pos + slot]
        chunk_dot, k = np.zeros(slot, dtype=NP_FLOAT), 0
        for j in range(slot):  # TC = O(slot^2)
            k_next = k + j + 1
            chunk_dot[j] = np.sum(A[k:k_next] * chunk[0 : j + 1])
            k = k_next
        result[pos:next_pos] = chunk_dot
