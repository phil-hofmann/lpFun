import itertools
import numpy as np
from newfun.utils_njit import (
    n2l_subroutine,
    l2n_subroutine,
    rmo_subroutine,
    leja_order,
    unisolvent_nodes_subroutine,
    tiling_subroutine,
)
from newfun import NP_FLOAT, NP_ARRAY


def classify(m: int, n: int, p: float, allow_infty=False) -> NP_ARRAY:
    if m < 1:
        raise ValueError("The parameter dim should be at least 1.")
    if (not allow_infty) and (p <= 0.0 or p > 2.0):
        raise ValueError(" The parameter p should be in the range (0, 2].")
    if allow_infty and (p <= 0.0 or p > 2.0) and (not p == np.infty):
        raise ValueError(
            " The parameter p should be in the range (0, 2] or p = np.infty."
        )
    if n < 0:
        raise ValueError("The parameter degree should be non-negative.")


def n2l(nodes: NP_ARRAY) -> NP_ARRAY:
    return n2l_subroutine(nodes)


def l2n(nodes: NP_ARRAY) -> NP_ARRAY:
    return l2n_subroutine(nodes)


def rmo(A: NP_ARRAY) -> NP_ARRAY:
    return rmo_subroutine(A)


def unisolvent_nodes(nodes: NP_ARRAY, m: int, n: int, p: float) -> NP_ARRAY:
    classify(m, n, p, allow_infty=True)
    if p == np.infty:
        return np.flip(list(itertools.product(nodes, repeat=m)), axis=1)
    else:
        return unisolvent_nodes_subroutine(nodes, m, n, p)


def unisolvent_nodes_1d(n: int, nodes: callable) -> NP_ARRAY:
    nodes_n = nodes(n)
    return nodes_n[leja_order(nodes_n)]


def cheb(n: int) -> NP_ARRAY:
    """O(n)"""
    if n < 0:
        raise ValueError("The parameter ``n`` should be non-negative.")
    if n == 0:
        return np.zeros(1, dtype=NP_FLOAT)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=NP_FLOAT)
    return np.cos(np.arange(n, dtype=NP_FLOAT) * np.pi / (n - 1))


def tiling(m: int, n: int, p: NP_FLOAT) -> NP_ARRAY:
    classify(m, n, p)
    return tiling_subroutine(m, n, p)
