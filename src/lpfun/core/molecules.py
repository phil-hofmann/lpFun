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
    itransform_lt_1d,  # 1.
    itransform_ut_1d,  # 2.
    itransform_lt_max,  # 3.
    itransform_ut_max,  # 4.
    itransform_lt_2d,  # 5.
    itransform_ut_2d,  # 6.
    itransform_lt_3d,  # 7.
    itransform_ut_3d,  # 8.
    itransform_lt_md,  # 9.
    itransform_ut_md,  # 10.
    ###
    transform_lt_1d,  # 1.
    transform_ut_1d,  # 2.
    transform_lt_max,  # 3.
    transform_ut_max,  # 4.
    transform_lt_2d,  # 5.
    transform_ut_2d,  # 6.
    transform_lt_3d,  # 7.
    transform_ut_3d,  # 8.
    # transform_lt_md, # TODO
    # transform_ut_md, # TODO
    ###
    dtransform_max,
    dtransform_lt_md,
    dtransform_ut_md,
)
from typing import Literal


def validate(
    mode: str,
    T: np.ndarray,
    p: float,
    m: int,
) -> None:
    if mode not in ("upper", "lower"):
        raise ValueError('Mode must be either "upper" or "lower".')

    if (T is None) and p != np.inf and m != 1:
        raise ValueError("Tube projection is required for p != np.inf and m != 1.")


# @njit # NOTE optional
def transform(
    V: np.ndarray,
    f: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    mode: Literal["lower", "upper"],
) -> np.ndarray:
    """
    Fast Newton Transform
    ---------------------
    V: np.ndarray
        Row major ordering
    f: np.ndarray
        Input vector
    T: np.ndarray
        Tube projection
    m: int
        Spatial dimension
    p: float
        Parameter of the lp space
    mode: str
        Lower or upper triangular

    Returns
    -------
    np.ndarray:
        Transformed vector

    Time Complexity
    ---------------
    O(Nmn)
    """
    V, f, T, m, p, mode = (
        np.asarray(V).astype(NP_FLOAT),
        np.asarray(f).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
        int(m),
        float(p),
        str(mode),
    )
    validate(mode, T, p, m)
    classify(m, 0, p)
    mode = True if mode == "lower" else False

    if m == 1:
        return transform_lt_1d(V, f) if mode else transform_ut_1d(V, f)
    elif p == np.inf:
        return transform_lt_max(V, f) if mode else transform_ut_max(V, f)
    elif m == 2:
        return transform_lt_2d(V, f, T) if mode else transform_ut_2d(V, f, T)
    elif m == 3:
        return transform_lt_3d(V, f, T) if mode else transform_ut_3d(V, f, T)
    # TODO:
    # return (
    #     transform_lt_md(V, f, T) if mode else transform_ut_md(V, f, T)
    # )


# @njit # NOTE optional
def itransform(
    V: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    mode: Literal["lower", "upper"],
    parallel: bool,
) -> np.ndarray:
    """
    Inverse Fast Newton Transform
    ---------------------
    V: np.ndarray
        Row major ordering
    c: np.ndarray
        Input vector
    T: np.ndarray
        Tube projection
    m: int
        Spatial dimension
    p: float
        Parameter of the lp space
    mode: str
        Lower or upper triangular
    parallel: bool
        CPU parallelization enabled or not

    Returns
    -------
    np.ndarray:
        Inverse transformed vector

    Time Complexity
    ---------------
    O(Nmn)
    """
    V, c, T, m, p, mode, parallel = (
        np.asarray(V).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
        int(m),
        float(p),
        str(mode),
        bool(parallel),
    )
    validate(mode, T, p, m)
    classify(m, 0, p)
    mode = True if mode == "lower" else False

    if m == 1:
        return (
            itransform_lt_1d(V, c, parallel)
            if mode
            else itransform_ut_1d(V, c, parallel)
        )
    elif p == np.inf:
        return (
            itransform_lt_max(V, c, parallel)
            if mode
            else itransform_ut_max(V, c, parallel)
        )
    elif m == 2:
        return (
            itransform_lt_2d(V, c, T, parallel)
            if mode
            else itransform_ut_2d(V, c, T, parallel)
        )
    elif m == 3:
        return (
            itransform_lt_3d(V, c, T, parallel)
            if mode
            else itransform_ut_3d(V, c, T, parallel)
        )
    return (
        itransform_lt_md(V, c, T, parallel)
        if mode
        else itransform_ut_md(V, c, T, parallel)
    )


# @njit # NOTE optional
def dtransform(
    D: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    i: int,
    mode: Literal["lower", "upper"],
    parallel: bool,
) -> np.ndarray:
    """
    Fast Diagonal Newton Transformation
    -----------------------------
    D: np.ndarray
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
    mode: str
        Lower or upper triangular
    parallel: bool
        CPU parallelization enabled or not

    Returns
    -------
    np.ndarray:
        Transformed vector

    Time Complexity
    ---------------
    O(Nn)
    """
    D, c, T, m, p, i, mode, parallel = (
        np.asarray(D).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
        int(m),
        float(p),
        int(i),
        str(mode),
        bool(parallel),
    )
    validate(mode, T, p, m)
    classify(m, 0, p)
    if i < 0 or i >= m:
        raise ValueError(f"Choose a coordinate between i=0 and i={m - 1}.")
    mode = True if mode == "lower" else False

    n = int(T[0] - 1)
    if m == 1:
        return (
            itransform_lt_1d(D, c, parallel)
            if mode
            else itransform_ut_1d(D, c, parallel)
        )
    elif p == np.inf:
        Perm = None
        if not i == 0:
            Perm = permutation_max(m, n, i)
            c = apply_permutation(Perm, c)
        c = (
            dtransform_max(D, c, parallel)
            if mode
            else dtransform_max(D[::-1], c[::-1], parallel)[::-1]
        )
        if not i == 0:
            c = apply_permutation(Perm, c, invert=True)
    else:
        Perm = None
        if not i == 0:
            Perm = permutation(T, i)
            c = apply_permutation(Perm, c, invert=True)
        c = (
            dtransform_lt_md(D, c, T, parallel)
            if mode
            else dtransform_ut_md(D, c, T, parallel)
        )
        if not i == 0:
            c = apply_permutation(Perm, c)
    return c
