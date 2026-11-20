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

    The three-term recurrence of ``chebyshev2lagrange`` introduces numerical errors: it
    starts from the rounded node fl(cos(l pi / n)), and this rounding grows to
    |T_k'| * |dx| in column k (typically k * eps, up to k^2 * eps near the nodes +-1), on
    top of the recurrence's own rounding. Here the entries are computed directly from the
    angles instead. The angle index k * l is reduced modulo 2n in integers and folded into
    the first octant, so each entry is a cosine or sine of an angle in [0, pi/4]: the
    matrix is that of the exact Lobatto nodes to about one ulp per entry.
    """
    m = len(lidx)
    Vx = np.zeros((m, m), dtype=np.float64)
    # Vx[i, k] = cos(t) with t = r pi / n and r = k * l. The integer r is reduced modulo 2n
    # and folded by the symmetries of the cosine until t is in [0, pi/2], so the angle
    # carries no rounding. In the upper half of that range the cosine is evaluated as a
    # sine, which stays accurate where the entry is close to zero.
    for i in range(m):
        l = np.int64(lidx[i])
        for k in range(m):
            r = (k * l) % (2 * n)  # t in [0, 2 pi)
            if r > n:  # cos(2 pi - t) = cos(t)
                r = 2 * n - r
            sign = 1.0
            if 2 * r > n:  # cos(pi - t) = -cos(t)
                r = n - r
                sign = -1.0
            if 4 * r > n:  # cos(t) = sin(pi / 2 - t)
                Vx[i, k] = sign * np.sin((n - 2 * r) * np.pi / (2 * n))
            else:
                Vx[i, k] = sign * np.cos(r * np.pi / n)
    return Vx
