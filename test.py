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


# @pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
# def test_diff_transpose(m: int, p: float, ba: str, pr: bool):
#     for n in NS:
#         fun = lpfun.Function(
#             m,
#             n,
#             p,
#             basis=ba,
#             precomputation=pr,
#             precompilation=False,
#             report=False,
#         )

#         for k in [1, 2, 3]:
#             for i in range(m):
#                 x = np.random.randn(len(t))
#                 y = np.random.randn(len(t))

#                 Dx = t.diff(x, i, k)  # D x
#                 DTx = t.diff(y, i, k, transpose=True)  # D^T y

#                 lhs = np.dot(Dx, y)  # <D x, y>
#                 rhs = np.dot(x, DTx)  # <x, D^T y>

#                 eps = np.abs(lhs - rhs)
#                 assert eps < 1e-6


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
