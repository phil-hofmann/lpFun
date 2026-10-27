import lpfun
import pytest
import numpy as np
from itertools import product

# Parameters

ms = [1, 2, 3, 4, 5, 6]
ps = [1.0, 2.0, np.inf]
bases = ["newton", "chebyshev", "legendre"]
precomputation = [True, False]
m_p_ba_pr = list(product(ms, ps, bases, precomputation))
NS = [4, 5, 6, 7, 8]


# Tests


@pytest.mark.parametrize("m", ms)
def test_tube_absolute_degree(m: int):
    for n in NS:
        A = lpfun.core.set.lp_set(m, n, 1.0)
        tube = lpfun.core.set.lp_tube(A, m, n, 1.0)
        tube_sum = np.sum(tube)
        cardinality = lpfun.utils.binomial(n + m, m)
        assert tube_sum == cardinality


@pytest.mark.parametrize("m", ms)
def test_tube_euclidean_degree(m: int):
    for n in NS:
        A = lpfun.core.set.lp_set(m, n, 2.0)
        tube = lpfun.core.set.lp_tube(A, m, n, 2.0)
        tube_sum = np.sum(tube)
        cardinality = len(
            [
                point
                for point in product(range(n + 1), repeat=m)
                if np.linalg.norm(point) <= n
            ]
        )
        assert tube_sum == cardinality


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_interp_eval(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )
        function_values = np.random.rand(len(fun))
        reconstruction = fun.eval(fun.interp(function_values))
        eps = np.linalg.norm(reconstruction - function_values)
        assert eps < 1e-8


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_diff(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        def f(x):
            val = 0.0
            for j in range(m):
                val += (j + 1) * x[j] ** (n - 1)
            return val

        def df(dim, order, x):
            c = dim + 1

            if order == 0:
                return f(x)

            if order == 1:
                return c * (n - 1) * x[dim] ** (n - 2)

            elif order == 2:
                return c * (n - 1) * (n - 2) * x[dim] ** (n - 3)

            elif order == 3:
                return c * (n - 1) * (n - 2) * (n - 3) * x[dim] ** (n - 4)

        function_values = np.array([f(x) for x in fun.grid])
        coeffs = fun.interp(function_values)

        for order in [0, 1, 2, 3]:
            for dim in range(m):
                dx_function_values = np.array([df(dim, order, x) for x in fun.grid])

                dx_reconstruction = (
                    fun.eval(coeffs)
                    if order == 0
                    else fun.eval(fun.diff(coeffs, dim, order))
                )

                eps = np.linalg.norm(dx_reconstruction - dx_function_values)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_diffT(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        for order in [1, 2, 3]:
            for dim in range(m):
                x = np.random.randn(len(fun))
                y = np.random.randn(len(fun))

                Dx = fun.diff(x, dim, order)
                DTx = fun.diffT(y, dim, order)

                lhs = np.dot(Dx, y)
                rhs = np.dot(x, DTx)

                eps = np.abs(lhs - rhs)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_call(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        fun = lpfun.Function(
            m,
            n + 1,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            report=False,
        )

        def f(x):
            return np.sum(x**3)

        function_values = np.array([f(x) for x in fun.grid])
        coeffs = fun.interp(function_values)
        points = np.random.rand(10, m)
        function_value = np.array([f(x) for x in points])
        reconstruction = fun(coeffs, points)
        eps = np.max(np.abs(reconstruction - function_value))
        assert eps < 1e-6


# Leja ordering: deterministic tie-breaking, no underflow for large node sets

LEJA_ORDER_REFERENCE = {
    9: [0, 8, 4, 2, 6, 3, 5, 1, 7],
    17: [0, 16, 8, 5, 12, 3, 10, 14, 6, 2, 11, 7, 13, 4, 9, 1, 15],
    33: [0, 32, 16, 10, 23, 6, 27, 19, 13, 4, 25, 21, 8, 29, 14, 2, 18, 30, 11, 5, 24, 17, 9, 28, 3, 22, 15, 26, 7, 12, 20, 1, 31],
}


@pytest.mark.parametrize("n", sorted(LEJA_ORDER_REFERENCE))
def test_leja_order_deterministic(n: int):
    x = lpfun.basis.nodes.cheb2nd_nodes(n)
    order = lpfun.core.grid.get_leja_order(x)
    assert list(order) == LEJA_ORDER_REFERENCE[n]
    # ties between x and -x are broken towards the larger node
    assert x[order[3]] > 0


def test_leja_order_no_underflow():
    # the raw product of distances underflows after ~1000 picks on this grid,
    # after which a naive implementation degenerates into ascending array order
    x = lpfun.basis.nodes.cheb2nd_nodes(25_000)
    order = lpfun.core.grid.get_leja_order(x, limit=2000)
    assert len(np.unique(order)) == 2000
    assert not np.all(np.diff(order[1100:]) > 0)


def test_leja_order_large_degree_interp():
    # beyond ~1082 nodes the old ordering degenerated and the (non-pivoted) LU of the
    # Vandermonde matrix blew up: at n = 1500 the 1D interp error was ~1e18
    n = 1500
    fun = lpfun.Function(1, n, basis="chebyshev", precompilation=False, report=False)
    lidx = np.rint(np.arccos(fun.nodes) * n / np.pi).astype(np.int64)
    V = np.cos((np.outer(lidx, np.arange(n + 1)) % (2 * n)) * np.pi / n)  # T_k at the exact nodes
    c = np.random.default_rng(n).uniform(-1, 1, n + 1)
    assert np.abs(fun.interp(V @ c) - c).max() <= 5e-12
