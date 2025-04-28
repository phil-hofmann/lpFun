import numpy as np

from numba import njit
from lpfun import NP_INT, NP_FLOAT
from lpfun.core.utils import classify, apply_permutation
from lpfun.core.set import permutation_max, permutation
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
    itransform_ut_md,
    dtransform_max,
    dtransform_lt_md,
    dtransform_ut_md,
)
from typing import Literal

@njit
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


@njit
def itransform(
    Vx: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    mode: Literal["lower", "upper"],
) -> np.ndarray:
    """
    Inverse Fast Newton Transform
    ---------------------
    Vx: np.ndarray
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

    Returns
    -------
    np.ndarray:
        Inverse transformed vector

    Time Complexity
    ---------------
    O(Nmn)
    """
    Vx, c, T, m, p, mode = (
        np.asarray(Vx).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
        int(m),
        float(p),
        str(mode),
    )
    validate(mode, T, p, m)
    classify(m, 0, p)
    lt = mode == "lower"

    if m == 1:
        return itransform_lt_1d(Vx, c) if lt else itransform_ut_1d(Vx, c)
    # elif p == np.inf:
    #     return itransform_lt_max(Vx, c) if lt else itransform_ut_max(Vx, c)
    elif m == 2:
        return itransform_lt_2d(Vx, c, T) if lt else itransform_ut_2d(Vx, c, T)
    elif m == 3:
        return itransform_lt_3d(Vx, c, T) if lt else itransform_ut_3d(Vx, c, T)
    return itransform_lt_md(Vx, c, T) if lt else itransform_ut_md(Vx, c, T)


@njit
def dtransform(
    Dx: np.ndarray,
    c: np.ndarray,
    T: np.ndarray,
    m: int,
    p: float,
    i: int,
    mode: Literal["lower", "upper"],
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
    mode: str
        Lower or upper triangular

    Returns
    -------
    np.ndarray:
        Transformed vector

    Time Complexity
    ---------------
    O(Nn)
    """
    Dx, c, T, m, p, i, mode = (
        np.asarray(Dx).astype(NP_FLOAT),
        np.asarray(c).astype(NP_FLOAT),
        np.asarray(T).astype(NP_INT),
        int(m),
        float(p),
        int(i),
        str(mode),
    )
    validate(mode, T, p, m)
    classify(m, 0, p)
    if i < 0 or i >= m:
        raise ValueError(f"Choose a coordinate between i=0 and i={m - 1}.")
    lt = mode == "lower"

    n = int(T[0] - 1)
    if m == 1:
        return itransform_lt_1d(Dx, c) if lt else itransform_ut_1d(Dx, c)
    # elif p == np.inf:
    #     Perm = None
    #     if not i == 0:
    #         Perm = permutation_max(m, n, i)
    #         c = apply_permutation(Perm, c)
    #     c = dtransform_max(Dx, c) if lt else dtransform_max(Dx[::-1], c[::-1])[::-1]
    #     if not i == 0:
    #         c = apply_permutation(Perm, c, invert=True)
    else:
        Perm = None
        if not i == 0:
            Perm = permutation(T, i)
            c = apply_permutation(Perm, c, invert=True)
        c = dtransform_lt_md(Dx, c, T) if lt else dtransform_ut_md(Dx, c, T)
        if not i == 0:
            c = apply_permutation(Perm, c)
    return c
