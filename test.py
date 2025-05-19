import lpfun
import pytest
import numpy as np
from itertools import product

# Parameters

ms = [1, 2, 3, 4, 5, 6]
ps = [1.0, 2.0, np.inf]
bases = ["newton", "chebyshev"]
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
        cardinality = lpfun.core.utils.binomial(n + m, m)
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
def test_fnt_ifnt(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        t = lpfun.Transform(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            colex_order=False,
            report=False,
        )
        function_values = np.random.rand(len(t))
        reconstruction = t.ifnt(t.fnt(function_values))
        eps = np.linalg.norm(reconstruction - function_values)
        assert eps < 1e-9


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_dx(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        t = lpfun.Transform(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            colex_order=False,
            report=False,
        )

        def f(x):
            return np.sum(x ** (n - 1))

        def df(k, x):
            if k == 1:
                return (n - 1) * x[i] ** (n - 2)
            elif k == 2:
                return (n - 1) * (n - 2) * x[i] ** (n - 3)
            elif k == 3:
                return (n - 1) * (n - 2) * (n - 3) * x[i] ** (n - 4)

        function_values = np.array([f(x) for x in t.grid])
        coeffs = t.fnt(function_values)
        for k in [1, 2, 3]:
            for i in range(m):
                dx_function_values = np.array([df(k, x) for x in t.grid])
                dx_reconstruction = t.ifnt(t.dx(coeffs, i, k))
                eps = np.linalg.norm(dx_reconstruction - dx_function_values)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_dxT(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        t = lpfun.Transform(
            m,
            n,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            colex_order=False,
            report=False,
        )

        for k in [1, 2, 3]:
            for i in range(m):
                x = np.random.randn(len(t))
                y = np.random.randn(len(t))

                Dx = t.dx(x, i, k)  # D x
                DTx = t.dxT(y, i, k)  # D^T y

                lhs = np.dot(Dx, y)  # <D x, y>
                rhs = np.dot(x, DTx)  # <x, D^T y>

                eps = np.abs(lhs - rhs)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba, pr", m_p_ba_pr)
def test_eval(m: int, p: float, ba: str, pr: bool):
    for n in NS:
        t = lpfun.Transform(
            m,
            n + 1,
            p,
            basis=ba,
            precomputation=pr,
            precompilation=False,
            colex_order=False,
            report=False,
        )

        def f(x):
            return np.sum(x**3)

        function_values = np.array([f(x) for x in t.grid])
        coeffs = t.fnt(function_values)
        points = np.random.rand(10, m)
        function_value = np.array([f(x) for x in points])
        reconstruction = t.eval(coeffs, points)
        eps = np.max(np.abs(reconstruction - function_value))
        assert eps < 1e-6
