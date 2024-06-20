import numpy as np
from typing import Optional
from newfun.utils import classify
from newfun.utils_njit import (
    permutation_maximal,
    permutation,
    apply_permutation,
)
from newfun.transform_utils_njit import (
    transform_1d_sequential,
    transform_1d_parallel,
    transform_maximal_sequential,
    transform_maximal_parallel,
    transform_2d_sequential,
    transform_2d_parallel,
    # transform_3d_sequential,
    # transform_3d_parallel,
    transform_md_sequential,
    # transform_md_parallel,
    transform_diag_1d_sequential,
    transform_diag_1d_parallel,
    transform_diag_sequential,
    transform_diag_parallel,
    transform_diag_maximal_sequential,
    transform_diag_maximal_parallel,
)
from newfun import NP_INT, NP_FLOAT, NP_ARRAY, PARALLEL


def transform(A: NP_ARRAY, x: NP_ARRAY, T: Optional[NP_ARRAY], m: int, p: float):
    """
    Fast Newton Transformation (FNT) : Forward / Backward
    -----------------------------------------------------
    A: NP_ARRAY
        Row major ordering matrix
    x: NP_ARRAY
        Input vector
    T: NP_ARRAY
        Tiling of the transformation
    m: int
        Dimension of the transformation
    p: float
        Parameter of the lp space

    Returns
    -------
    NP_ARRAY:
        Transformed vector

    Time Complexity
    ---------------
    O(|A_{m + 1, n, p}|*gamma_p*m)
    """
    if (not p == np.infty) and (T is None) and (not m == 1):
        raise ValueError("Tiling is required for p != np.infty.")
    classify(m, 0, p, allow_infty=True)
    if m == 1:
        return _transform_1d(A, x)
    elif p == np.infty:
        return _transform_maximal(A, x)
    elif m == 2:
        return _transform_2d(A, x, T)
    else:
        return _transform_md(A, x, T)


def transform_dx(
    A: NP_ARRAY, x: NP_ARRAY, T: Optional[NP_ARRAY], m: int, n: int, p: float, i: int
):
    """
    Fast Newton Transformation (FNT) : Differentiation
    --------------------------------------------------
    A: NP_ARRAY
        Row major ordering matrix
    x: NP_ARRAY
        Input vector
    T: NP_ARRAY
        Tiling of the transformation
    m: int
        Dimension of the transformation
    p: float
        Parameter of the lp space
    i: int
        Coordinate of differentiation

    Returns
    -------
    NP_ARRAY:
        Transformed vector

    Time Complexity
    ---------------
    O(|A_{m + 1, n, p}|*gamma_p)
    """
    if (not p == np.infty) and (T is None) and (not m == 1):
        raise ValueError("Tiling is required for p != np.infty.")
    classify(m, 0, p, allow_infty=True)
    if m == 1:
        return _transform_diag_1d(A, x)
    elif p == np.infty:
        P = permutation_maximal(m, n, i)
        if not i == 0:
            x = apply_permutation(P, x)
        x = _transform_diag_maximal(A, x)
        if not i == 0:
            x = apply_permutation(P, x, invert=True)
        return x
    else:
        P = permutation(m, n, T, i)
        if not i == 0:
            x = apply_permutation(P, x)
        x = _transform_diag(A, x, T)
        if not i == 0:
            x = apply_permutation(P, x, invert=True)
        return x


def _transform_1d(A: NP_ARRAY, x: NP_ARRAY):
    return transform_1d_parallel(A, x) if PARALLEL else transform_1d_sequential(A, x)


def _transform_maximal(A: NP_ARRAY, x: NP_ARRAY):
    # O(N*n*m)
    return (
        transform_maximal_parallel(A, x)
        if PARALLEL
        else transform_maximal_sequential(A, x)
    )


def _transform_2d(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    # O(|A_{3,n,p}|)*2)
    return (
        transform_2d_parallel(A, x, T) if PARALLEL else transform_2d_sequential(A, x, T)
    )


def _transform_md(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    # O(|A_{m+1,n,p}|*m)
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    return transform_md_sequential(A, x, T)


def _transform_diag_1d(A: NP_ARRAY, x: NP_ARRAY):
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    # O(n^2)
    return (
        transform_diag_1d_parallel(A, x)
        if PARALLEL
        else transform_diag_1d_sequential(A, x)
    )


def _transform_diag_maximal(A: NP_ARRAY, x: NP_ARRAY):
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    # O(N*m)
    return (
        transform_diag_maximal_parallel(A, x)
        if PARALLEL
        else transform_diag_maximal_sequential(A, x)
    )


def _transform_diag(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY):
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    # O(|A_{m+1,n,p}|*gamma_p)
    return (
        transform_diag_parallel(A, x) if PARALLEL else transform_diag_sequential(A, x)
    )
