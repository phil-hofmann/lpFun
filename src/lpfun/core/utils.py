import numpy as np
from math import gamma
from numba import njit, prange

# from lpfun import NP_INT, NP_FLOAT


@njit
def classify(m: int, n: int, p: float) -> bool:
    m, n, p = int(m), int(n), float(p)
    ###
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (p <= 0.0 or p > 2.0) and (not p == np.inf):
        raise ValueError(f"The parameter p should be in the range (0, 2] or inf.")
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")
    ###
    return True


@njit
def apply_permutation(
    P: np.ndarray,
    x: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    # P, x, invert = (
    #     np.asarray(P).astype(NP_INT),
    #     np.asarray(x).astype(NP_FLOAT),
    #     bool(invert),
    # )
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
