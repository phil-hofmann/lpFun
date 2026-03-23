from numba import njit
import numpy as np


@njit
def get_leja_order(nodes: np.ndarray, limit: int = -1) -> np.ndarray:
    """O(n^3)"""
    """This function originates from minterpy."""
    ### NOTE -- no type conversion
    n = len(nodes)
    limit = n if limit == -1 else limit
    ord = np.arange(1, n, dtype=np.int64)
    lj = np.zeros(limit, dtype=np.int64)
    lj[0] = 0
    m = 0
    for k in range(0, limit - 1):
        jj = 0
        for i in range(0, n - k - 1):
            p = 1
            for j in range(k + 1):
                p = p * (nodes[lj[j]] - nodes[ord[i]])
            p = np.abs(p)
            # if p >= m: # NOTE altered
            if p > m:
                jj = i
                m = p
        m = 0
        lj[k + 1] = ord[jj]
        ord = np.delete(ord, jj)
    return lj


@njit
def get_leja_order_new(nodes: np.ndarray) -> np.ndarray:
    """O(n^3)"""
    n = nodes.shape[0]
    if n == 0:
        return None

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

    # product of distances for unchosen nodes (float64)
    prod_dist = np.zeros(n, dtype=np.float64)
    for i in range(n):
        prod_dist[i] = np.abs(nodes[i] - nodes[max_idx])

    # iterate to choose remaining Leja points
    for k in range(1, n):
        best_idx = -1
        best_val = -1.0

        # find best unchosen index
        for i in range(n):
            if not chosen[i]:
                if prod_dist[i] > best_val:
                    best_val = prod_dist[i]
                    best_idx = i

        # assign chosen
        order[k] = best_idx
        chosen[best_idx] = True

        # update products only for unchosen nodes
        for i in range(n):
            if not chosen[i]:
                prod_dist[i] = prod_dist[i] * np.abs(nodes[i] - nodes[best_idx])

    return order


if __name__ == "__main__":
    import time

    nodes = np.array([1, *np.random.rand(1000)])

    # Precompile
    get_leja_order_new(nodes)
    get_leja_order(nodes)

    # old version
    t0 = time.perf_counter()
    leja_order = get_leja_order(nodes)
    t1 = time.perf_counter()
    time_old = t1 - t0

    # Numba version
    t0 = time.perf_counter()
    leja_order_new = get_leja_order_new(nodes)
    t1 = time.perf_counter()
    time_new = t1 - t0

    diff = np.sum(np.abs(leja_order - leja_order_new))

    print(f"Difference between implementations: {diff:.2e}")
    print(f"Old time: {time_old * 1e3:.3f} ms")
    print(f"New time: {time_new * 1e3:.3f} ms")
    print(f"Speedup: {time_old / time_new:.2f}x")
