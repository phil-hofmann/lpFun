import numpy as np

# from numba import njit # NOTE optional
from lpfun import NP_INT, NP_FLOAT
from lpfun.utils import (
    classify,
    permutation_max,
    permutation,
    apply_permutation,
)
from lpfun.core.atoms import (
    itransform_lt_1d,
    itransform_ut_1d,
    itransform_lt_max,
    itransform_ut_max,
    itransform_lt_2d,
    itransform_ut_2d,
    itransform_lt_3d,
    itransform_ut_3d,
    itransform_lt_md,
    # itransform_ut_md,
    ###
    transform_lt_1d,
    transform_ut_1d,
    transform_lt_max,
    transform_ut_max,
    transform_lt_2d,
    transform_ut_2d,
    transform_lt_3d,
    # transform_ut_3d,
    ###
    dtransform_max,
    dtransform_lt_md,
    dtransform_ut_md,
)
from typing import Literal


def validate(
    mode: str,
    parallel: str,
    T: np.ndarray,
    p: float,
    m: int,
) -> None:
    if mode not in ("upper", "lower"):
        raise ValueError('Mode must be either "upper" or "lower".')

    if parallel not in ("seq", "cpu"):
        raise ValueError('Parallel must be either "seq" or "cpu".')

    if (T is None) and p != np.inf and m != 1:
        raise ValueError("Tube projection is required for p != np.inf and m != 1.")


# @njit # NOTE optional
def transform(
    Qx: np.ndarray,
    f: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    mode: Literal["lower", "upper"],
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """
    Fast Newton Transform
    ---------------------
    Qx: np.ndarray
        Row major ordering
    f: np.ndarray
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
    Qx, f, T = (
        np.asarray(Qx).astype(NP_FLOAT),
        np.asarray(f).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    m, p = (
        int(m),
        float(p),
    )
    mode, parallel = (
        str(mode),
        str(parallel),
    )
    validate(mode, parallel, T, p, m)
    classify(m, 0, p)
    mode = True if mode == "lower" else False

    if m == 1:
        return transform_lt_1d(Qx, f) if mode else transform_ut_1d(Qx, f)
    elif p == np.inf:
        return (
            transform_lt_max(Qx, f, parallel)
            if mode
            else transform_ut_max(Qx, f, parallel)
        )
    elif m == 2:
        return transform_lt_2d(Qx, f, T) if mode else transform_ut_2d(Qx, f, T)
    elif m == 3:
        return (
            transform_lt_3d(Qx, f, T) if mode else None
        )  # TODO: return transform_ut_3d(Qx, f, T)
    return (
        itransform_lt_md(Qx, f, T) if mode else None
    )  # TODO: return itransform_ut_md(Qx, f, T)


# @njit # NOTE optional
def itransform(
    Qx: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    mode: Literal["lower", "upper"],
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """
    Inverse Fast Newton Transform
    ---------------------
    Qx: np.ndarray
        Row major ordering
    c: np.ndarray
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
    Qx, c, T = (
        np.asarray(Qx).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    m, p = (
        int(m),
        float(p),
    )
    mode, parallel = (
        str(mode),
        str(parallel),
    )
    validate(mode, parallel, T, p, m)
    classify(m, 0, p)
    if m > 3:
        raise NotImplementedError("Spatial dimensions m > 3 are not supported.")
    mode = True if mode == "lower" else False

    if m == 1:
        return (
            itransform_lt_1d(Qx, c, parallel)
            if mode
            else itransform_ut_1d(Qx, c, parallel)
        )
    elif p == np.inf:
        return (
            itransform_lt_max(Qx, c, parallel)
            if mode
            else itransform_ut_max(Qx, c, parallel)
        )
    elif m == 2:
        return (
            itransform_lt_2d(Qx, c, T, parallel)
            if mode
            else itransform_ut_2d(Qx, c, T, parallel)
        )
    elif m == 3:
        return (
            itransform_lt_3d(Qx, c, T, parallel)
            if mode
            else itransform_ut_3d(Qx, c, T, parallel)
        )


# @njit # NOTE optional
def dtransform(
    Dx: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    i: int,
    mode: Literal["lower", "upper"],
    parallel: Literal["seq", "cpu"],
) -> np.ndarray:
    """
    Fast Diagonal Newton Transformation
    -----------------------------
    Dx: np.ndarray
       Row major ordering (lower triangular)
    c: np.ndarray
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
    Dx, c, T = (
        np.asarray(Dx).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
    )
    m, p, i = (
        int(m),
        float(p),
        int(i),
    )
    mode, parallel = (
        str(mode),
        str(parallel),
    )
    validate(mode, parallel, T, p, m)
    classify(m, 0, p)
    if i < 0 or i >= m:
        raise ValueError(f"Choose a coordinate between i=0 and i={m - 1}.")
    mode = True if mode == "lower" else False

    n = int(T[0] - 1)
    if m == 1:
        return (
            itransform_lt_1d(Dx, c, parallel)
            if mode
            else itransform_ut_1d(Dx, c, parallel)
        )
    elif p == np.inf:
        Perm = None
        if not i == 0:
            Perm = permutation_max(m, n, i)
            c = apply_permutation(Perm, c)
        c = (
            dtransform_max(Dx, c, parallel)
            if mode
            else dtransform_max(Dx[::-1], c[::-1], parallel)[::-1]
        )
        if not i == 0:
            c = apply_permutation(Perm, c, invert=True)
    else:
        Perm = None
        if not i == 0:
            Perm = permutation(T, i)
            c = apply_permutation(Perm, c, invert=True)
        c = (
            dtransform_lt_md(Dx, c, T, parallel)
            if mode
            else dtransform_ut_md(Dx, c, T, parallel)
        )
        if not i == 0:
            c = apply_permutation(Perm, c)
    return c
