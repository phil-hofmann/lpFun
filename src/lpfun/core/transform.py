import numpy as np
import numba as nb
from lpfun import CACHE, PARALLEL
from lpfun.utils import apply_permutation


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


prange = nb.prange


@njit(parallel=PARALLEL)
def transform_lt(
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
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t = T[i]
            pos = cs_T[i]
            for row in range(t):
                acc = 0.0
                for col in range(row + 1):
                    acc += L[row, col] * dot[pos + col]
                new_dot[pos + row] = acc
        if m == 1:
            dot = new_dot
        else:
            dot = apply_permutation(pi, new_dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def transform_ut(
    U: np.ndarray,
    x: np.ndarray,
    m: int,
    pi: np.ndarray,
    T: np.ndarray,
    cs_T: np.ndarray,
):
    """O(mn|A|)"""
    dot = x.copy()
    for _ in range(m):
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t = T[i]
            pos = cs_T[i]
            for row in range(t):
                acc = 0.0
                for col in range(row, t):
                    acc += U[row, col] * dot[pos + col]
                new_dot[pos + row] = acc
        if m == 1:
            dot = new_dot
        else:
            dot = apply_permutation(pi, new_dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def itransform_lt(
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
            rhs = apply_permutation(pi, dot, invert=True)
        else:
            rhs = dot
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t = T[i]
            pos = cs_T[i]
            for row in range(t):
                acc = rhs[pos + row]
                for col in range(row):
                    acc -= L[row, col] * new_dot[pos + col]
                new_dot[pos + row] = acc / L[row, row]
        dot = new_dot
    return dot


@njit(parallel=PARALLEL)
def itransform_ut(
    U: np.ndarray,
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
            rhs = apply_permutation(pi, dot, invert=True)
        else:
            rhs = dot
        new_dot = np.zeros_like(dot)
        for i in prange(len(T)):
            t = T[i]
            pos = cs_T[i]
            for row in range(t - 1, -1, -1):
                acc = rhs[pos + row]
                for col in range(row + 1, t):
                    acc -= U[row, col] * new_dot[pos + col]
                new_dot[pos + row] = acc / U[row, row]
        dot = new_dot
    return dot


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
        t = T[i]
        pos = cs_T[i]
        for row in range(t):
            acc = 0.0
            for col in range(row + 1):
                acc += D[row, col] * x[pos + col]
            dot[pos + row] = acc
    dot = apply_permutation(perm, dot, invert=False)
    return dot


@njit(parallel=PARALLEL)
def dtransform_ut(
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
        t = T[i]
        pos = cs_T[i]
        for row in range(t):
            acc = 0.0
            for col in range(row, t):
                acc += D[row, col] * x[pos + col]
            dot[pos + row] = acc
    return apply_permutation(perm, dot, invert=False)
