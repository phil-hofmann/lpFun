import itertools
import numpy as np
import numba as nb
from lpfun import CACHE


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


# prange = nb.prange

# tolerance on the log-product of distances below which two candidates tie
LEJA_TOL = 1e-9


@njit
def get_leja_order(nodes: np.ndarray, limit: int = -1) -> np.ndarray:
    """O(n^2)

    Greedy Leja order: start at the node of largest modulus, then repeatedly pick
    the node maximizing the product of distances to the nodes chosen so far. The
    product is accumulated as a sum of logarithms, so it cannot underflow. Ties
    (e.g. x and -x on a symmetric grid, equal up to rounding) are broken towards
    the first candidate in array order, i.e. the larger node on a descending grid.
    """
    n = nodes.shape[0]
    if n == 0:
        return None
    if limit == -1:
        limit = n

    # preallocate
    order = np.empty(n, dtype=np.int64)
    chosen = np.zeros(n, dtype=np.bool_)

    # pick the first node as the one with largest absolute value
    max_idx = 0
    max_val = np.abs(nodes[0])
    for i in range(1, n):
        val = np.abs(nodes[i])
        if val > max_val:
            max_val = val
            max_idx = i

    order[0] = max_idx
    chosen[max_idx] = True

    # log-product of distances for unchosen nodes
    log_dist = np.zeros(n, dtype=np.float64)
    for i in range(n):
        log_dist[i] = np.log(np.abs(nodes[i] - nodes[max_idx]))

    # iterate to choose remaining Leja points
    for k in range(1, limit):
        # best score among unchosen nodes
        best_val = -np.inf
        for i in range(n):
            if not chosen[i] and log_dist[i] > best_val:
                best_val = log_dist[i]

        # first unchosen node within tolerance of the best score
        best_idx = -1
        for i in range(n):
            if not chosen[i] and log_dist[i] >= best_val - LEJA_TOL:
                best_idx = i
                break

        # assign chosen
        order[k] = best_idx
        chosen[best_idx] = True

        # update scores only for unchosen nodes
        for i in range(n):
            if not chosen[i]:
                log_dist[i] += np.log(np.abs(nodes[i] - nodes[best_idx]))

    return order[:limit]


@njit
def _get_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
) -> np.ndarray:
    """O(m|A|)"""
    N = len(A)
    grid = np.zeros((N, m))
    for i in range(N):
        mi = A[i]
        grid_point = np.zeros(m, dtype=np.float64)
        for j in range(m):
            grid_point[j] = nodes[mi[j]]
        grid[i] = grid_point
    return grid


def get_grid(
    nodes: np.ndarray,
    A: np.ndarray,
    m: int,
    n: int,
    p: float,
) -> np.ndarray:
    """O(|A|)"""
    nodes, m, n, p = (
        np.asarray(nodes).astype(np.float64),
        int(m),
        int(n),
        float(p),
    )
    if m == 1:
        return nodes.reshape(-1, 1)
    elif p == np.inf:
        return np.asarray(list(itertools.product(nodes, repeat=m)), dtype=np.float64)
    else:
        A = np.asarray(A).astype(np.int64)
        return _get_grid(nodes, A, m)
