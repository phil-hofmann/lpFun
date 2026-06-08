import numpy as np
from lpfun.core.grid import get_leja_order


def cheb2nd_nodes(n: int) -> np.ndarray:
    """O(n)"""
    n = int(n)
    ###
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    return np.cos(np.arange(n, dtype=np.float64) * np.pi / (n - 1))


def leja_nodes(n: int, m: int = 25_000) -> np.ndarray:
    """O(n^3)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)
    if n > m:
        raise (
            f"The amount of nodes {n} must be smaller or equal than the sample size {m}."
        )
    sample_nodes = cheb2nd_nodes(m)
    leja_order = get_leja_order(sample_nodes, limit=n)
    return sample_nodes[leja_order]
