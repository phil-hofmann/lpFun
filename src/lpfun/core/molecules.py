import numpy as np
from numba import njit
from lpfun import NP_INT, NP_FLOAT
from lpfun.utils import (
    classify,
    permutation_maximal,
    permutation,
    apply_permutation,
)
from lpfun.core.atoms import (
    itransform_maximal,
    itransform_1d,
    itransform_2d,
    itransform_3d,
    transform_maximal,
    transform_1d,
    transform_2d,
    # transform_3d,
    transform_md,
    ###
    dtransform_maximal,
    dtransform_md,
)


# @njit # NOTE optional
def transform(
    L: np.ndarray, x: np.ndarray, T: np.ndarray, m: int, p: float
) -> np.ndarray:
    """
    Fast Newton Transform
    ---------------------
    L: np.ndarray
        Matrix (lower triangular)
    x: np.ndarray
        Input vector
    T: np.ndarray
        Tube projection
    m: int
        Spatial dimension
    p: float
        Parameter of the lp space

    Returns
    -------
    np.ndarray:
        Transformed vector

    Time Complexity
    ---------------
    O(N*m*n)
    """
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    m, p = int(m), float(p)
    if (T is None) and p != np.inf and m != 1:
        raise ValueError("Tube projection is required for p != np.inf and m != 1.")
    classify(m, 0, p)
    if m == 1:
        return transform_1d(L, x)
    elif p == np.inf:
        return transform_maximal(L, x)
    elif m == 2:
        return transform_2d(L, x, T)
    # elif m == 3:
    #     return transform_3d(L, x, T)
    return transform_md(L, x, T)


# @njit # NOTE optional
def itransform(
    L: np.ndarray, x: np.ndarray, T: np.ndarray, m: int, p: float
) -> np.ndarray:
    """
    Inverse Fast Newton Transform
    ---------------------
    L: np.ndarray
        Row major ordering (lower triangular)
    x: np.ndarray
        Input vector
    T: np.ndarray
        Tube projection
    m: int
        Spatial dimension
    p: float
        Parameter of the lp space

    Returns
    -------
    np.ndarray:
        Inverse transformed vector

    Time Complexity
    ---------------
    O(N*m*n)
    """
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    m, p = int(m), float(p)
    if (T is None) and p != np.inf and m != 1:
        raise ValueError("Tube projection is required for p != np.inf and m != 1.")
    classify(m, 0, p)
    if m == 1:
        return itransform_1d(L, x)
    elif p == np.inf:
        return itransform_maximal(L, x)
    elif m == 2:
        return itransform_2d(L, x, T)
    elif m == 3:
        return itransform_3d(L, x, T)
    return transform_md(L, x, T)  # TODO: return itransform_md(L, x, T)


# @njit # NOTE optional
def dtransform(
    L: np.ndarray,
    x: np.ndarray,
    T: np.ndarray,
    m: int,
    n: int,
    p: float,
    i: int,
) -> np.ndarray:
    """
    Fast Diagonal Newton Transformation
    -----------------------------
    L: np.ndarray
       Matrix (lower triangular)
    x: np.ndarray
        Input vector
    T: np.ndarray
        Tube projection
    m: int
        Spatial dimension
    p: float
        Parameter of the lp space
    i: int
        Coordinate permutation

    Returns
    -------
    np.ndarray:
        Transformed vector

    Time Complexity
    ---------------
    O(N*n)
    """
    L = np.asarray(L).astype(NP_FLOAT)
    x = np.asarray(x).astype(NP_FLOAT)
    T = np.asarray(T).astype(NP_INT)
    m, n, p = int(m), int(n), float(p)
    if (T is None) and p != np.inf and m != 1:
        raise ValueError("Tube projection is required for p != np.inf and m != 1.")
    classify(m, 0, p)
    if m == 1:
        return itransform_1d(L, x)
    elif p == np.inf:
        P = permutation_maximal(m, n, i)
        x = x if i == 0 else apply_permutation(P, x)
        x = dtransform_maximal(L, x)
        x = x if i == 0 else apply_permutation(P, x, invert=True)
        return x
    P = permutation(T, i)
    x = x if i == 0 else apply_permutation(P, x, invert=True)
    x = dtransform_md(L, x, T)
    x = x if i == 0 else apply_permutation(P, x)
    return x
