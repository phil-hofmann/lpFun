import numpy as np
from numba import njit
from lpfun import NP_INT, NP_FLOAT, PARALLEL
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
    transform_maximal,
    transform_2d,
    transform_md,
    ut_diag_transform_maximal,
    lt_diag_transform_maximal,
    ut_diag_transform,
    lt_diag_transform,
)


@njit(parallel=PARALLEL)
def transform(A: np.ndarray, x: np.ndarray, T: np.ndarray, m: int, p: float) -> np.ndarray:
    """
    Fast Newton Transform
    ---------------------
    A: np.ndarray
        Row major ordering matrix
    x: np.ndarray
        Input vector
    T: np.ndarray
        Tube of the transformation
    m: int
        Dimension of the transformation
    p: float
        Parameter of the lp space

    Returns
    -------
    np.ndarray:
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
        raise ValueError("Tube is required for p != np.inf.")
    classify(m, 0, p, allow_infty=True)
    if m == 1:
        return lt_transform(A, x)
    elif p == np.inf:
        return transform_maximal(A, x)
    elif m == 2:
        return transform_2d(A, x, T)
    else:
        return transform_md(A, x, T)


@njit(parallel=PARALLEL)
def dx_transform(
    A: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    m: int,
    n: int,
    p: float,
    i: int,
    transpose: bool,
) -> np.ndarray:
    """
    Fast Spectral Differentiation
    -----------------------------
    A: np.ndarray
        Row major ordering matrix
    x: np.ndarray
        Input vector
    T: np.ndarray
        Tube of the transformation
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
    np.ndarray:
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
        raise ValueError("Tube is required for p != np.inf.")
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