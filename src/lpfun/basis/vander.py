import numpy as np
import numba as nb
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


# prange = nb.prange


@njit
def newton2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n))
    for i in range(n):
        monomials = np.ones(n, dtype=np.float64)
        for j in range(1, n):
            monomials[j] *= monomials[j - 1] * (x[i] - x[j - 1])
        Vx[i, :n] = monomials
    ###
    return Vx


@njit
def chebyshev2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Vx[i, 0] = 1.0
        if n > 1:
            Vx[i, 1] = x[i]
        for j in range(2, n):  # columns 0..n-1 (the former `n + 1` wrote one past the row)
            Vx[i, j] = 2 * x[i] * Vx[i, j - 1] - Vx[i, j - 2]
    return Vx


@njit
def legendre2lagrange(x: np.ndarray) -> np.ndarray:
    """O(n^2)"""
    x = np.asarray(x).astype(np.float64)
    n = len(x)
    ###
    Vx = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Vx[i, 0] = 1.0
        if n > 1:
            Vx[i, 1] = x[i]
        for j in range(2, n):
            Vx[i, j] = ((2 * j - 1) * x[i] * Vx[i, j - 1] - (j - 1) * Vx[i, j - 2]) / j
    return Vx


@njit
def chebyshev2lagrange_lobatto(lidx: np.ndarray, n: int) -> np.ndarray:
    """O(n^2)

    Chebyshev Vandermonde matrix at Chebyshev-Lobatto nodes given by their integer
    indices, V[i, k] = T_k(cos(l_i pi / n)) = cos(k l_i pi / n), i, k = 0..len(lidx)-1.

    ``chebyshev2lagrange`` evaluates T_k by the three-term recurrence at the rounded node
    fl(cos(l pi / n)), so its entries carry |T_k'| * |dx| of node rounding (typically
    k * eps, up to k^2 * eps near the nodes +-1) on top of the recurrence's own rounding.
    Here the angle index k * l is reduced modulo 2n in
    integers and folded into the first octant, so each entry is a cosine or sine of an
    angle in [0, pi/4]: the matrix is that of the exact Lobatto nodes to about one ulp per
    entry.
    """
    m = len(lidx)
    Vx = np.zeros((m, m), dtype=np.float64)
    # angles in units of u = pi / (4n): full period 8n, pi = 4n, pi/2 = 2n, pi/4 = n
    n = np.int64(n)
    period, half, quarter, eighth = 8 * n, 4 * n, 2 * n, n
    u = np.pi / float(half)
    for i in range(m):
        l = np.int64(lidx[i])
        for k in range(m):
            r = (4 * np.int64(k) * l) % period
            if r > half:  # cos(2 pi - t) = cos(t)
                r = period - r
            sign = 1.0
            if r > quarter:  # cos(pi - t) = -cos(t)
                r = half - r
                sign = -1.0
            if r > eighth:  # cos(t) = sin(pi/2 - t)
                Vx[i, k] = sign * np.sin(float(quarter - r) * u)
            else:
                Vx[i, k] = sign * np.cos(float(r) * u)
    return Vx
