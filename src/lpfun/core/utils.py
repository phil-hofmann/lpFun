import numpy as np
from math import gamma
from numba import njit


@njit
def apply_permutation(
    P: np.ndarray,
    x: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    x_p = np.zeros_like(x)
    N = len(P)
    if invert:
        for i in range(N):
            x_p[i] = x[P[i]]
    else:
        for i in range(N):
            x_p[P[i]] = x[i]
    return x_p


@njit
def binomial(n: int, m: int) -> int:
    if m < 0 or m > n:
        return 0
    result = 1
    for i in range(min(m, n - m)):
        result = result * (n - i) // (i + 1)
    return result


@njit
def memory_estimate(m: int, n: int, p: float) -> int:
    if p <= 1.0:
        singular = 1 + m * n
        if m < n:
            return int(singular * (1 - p) + binomial(n + m, m) * p)
        else:
            return int(singular * (1 - p) + binomial(n + m, n) * p)
    elif p <= 2.0:
        fac1 = (p * np.e / m) ** (1 / p)
        fac2 = np.sqrt(p / (2 * np.pi * m))
        return int(np.ceil((fac1 * (n + 2) * gamma(1 + 1 / p)) ** m * fac2))
    else:
        return int((n + 1) ** m)
