import numpy as np
from numba import njit
from lpfun import NP_INT, NP_FLOAT, NP_ARRAY, PARALLEL
from lpfun.utils import (
    classify,
    permutation_maximal,
    permutation,
    apply_permutation,
    rmo_transpose,
)
from lpfun.core.atoms import (
    lt_transform,
    ut_transform,
    n_transform_maximal,
    n_transform_2d,
    n_transform_md,
    ut_diag_transform_maximal,
    lt_diag_transform_maximal,
    ut_diag_transform,
    lt_diag_transform,
    diag_transform_maximal,
)


@njit(parallel=PARALLEL)
def n_transform(A: NP_ARRAY, x: NP_ARRAY, T: NP_ARRAY, m: int, p: float) -> NP_ARRAY:
    """
    Fast Newton Transformation
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
    O(sum(T^2)*m)
    """
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    m, p = int(m), float(p)
    if (not p == np.inf) and (T is None) and (not m == 1):
        raise ValueError("Tiling is required for p != np.inf.")
    classify(m, 0, p, allow_infty=True)
    if m == 1:
        return lt_transform(A, x)
    elif p == np.inf:
        return n_transform_maximal(A, x)
    elif m == 2:
        return n_transform_2d(A, x, T)
    else:
        return n_transform_md(A, x, T)


@njit(parallel=PARALLEL)
def n_dx_transform(
    A: NP_ARRAY,
    x: NP_ARRAY,
    T: NP_ARRAY,
    m: int,
    n: int,
    p: float,
    i: int,
    transpose: bool,
) -> NP_ARRAY:
    """
    Fast Spectral Differentiation
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
    transpose: bool
        Whether to transpose the matrix

    Returns
    -------
    NP_ARRAY:
        Transformed vector

    Time Complexity
    ---------------
    O(sum(T^2))
    """
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    m, n, p = int(m), int(n), float(p)
    if (not p == np.inf) and (T is None) and (not m == 1):
        raise ValueError("Tiling is required for p != np.inf.")
    classify(m, 0, p, allow_infty=True)
    if m == 1:
        if transpose:
            At = rmo_transpose(A)
            return lt_transform(At, x)
        else:
            return ut_transform(A, x)
    elif p == np.inf:
        P = permutation_maximal(m, n, i)
        if not i == 0:
            x = apply_permutation(P, x)
        if transpose:
            At = rmo_transpose(A)
            x = lt_diag_transform_maximal(At, x)
        else:
            x = ut_diag_transform_maximal(A, x)
        if not i == 0:
            x = apply_permutation(P, x, invert=True)
        return x
    else:
        P = permutation(T, i)
        if not i == 0:
            x = apply_permutation(P, x, invert=True)
        if transpose:
            At = rmo_transpose(A)
            x = lt_diag_transform(A, x, T)
        else:
            x = ut_diag_transform(A, x, T)
        if not i == 0:
            x = apply_permutation(P, x)
        return x


@njit(parallel=PARALLEL)
def l_dx_transform(
    A: NP_ARRAY,
    x: NP_ARRAY,
    m: int,
    n: int,
    i: int,
    transpose: bool,
) -> NP_ARRAY:
    """
    Fast Spectral Differentiation
    --------------------------------------------------
    A: NP_ARRAY
        Matrix
    x: NP_ARRAY
        Input vector
    m: int
        Dimension of the transformation
    i: int
        Coordinate of differentiation
    transpose: bool
        Whether to transpose the matrix

    Returns
    -------
    NP_ARRAY:
        Transformed vector

    Time Complexity
    ---------------
    O(N*n)
    """
    A = np.asarray(A).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    m, n = int(m), int(n)
    classify(m, 0, np.inf, allow_infty=True)
    if m == 1:
        if transpose:
            At = A.T
            return At @ x
        else:
            return A @ x
    P = permutation_maximal(m, n, i)
    if not i == 0:
        x = apply_permutation(P, x)
    if transpose:
        At = A.T
        x = diag_transform_maximal(At, x)
    else:
        x = diag_transform_maximal(A, x)
    if not i == 0:
        x = apply_permutation(P, x, invert=True)
    return x
