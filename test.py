import lpfun
import pytest
import numpy as np
from itertools import product

# Parameters

ms = [1, 2, 3, 4, 5, 6]
ps = [1.0, 2.0, np.inf]
bases = ["newton", "chebyshev"]
m_p_ba = list(product(ms, ps, bases))

# Prerequisites


def ns(m: int) -> np.ndarray:
    # Generate random ns
    n_max_m, default = {1: 35, 2: 30, 3: 25, 4: 20, 5: 15, 6: 10}, 5
    n_max = n_max_m.get(m, default)
    ns = np.random.randint(3, n_max, max(int(n_max * 0.2), 1))
    return ns


# Tests


def test_newton2lagrange():
    for n in ns(1):
        nodes = lpfun.utils.leja_nodes(lpfun.utils.cheb2nd(n))
        n2l = lpfun.utils.newton2lagrange(nodes)
        l2n = lpfun.utils.inv(n2l)
        identity = n2l @ l2n
        eps = np.linalg.norm(identity - np.eye(n))
        assert eps < 1e-8


def test_chebyshev2lagrange():
    for n in ns(1):
        nodes = lpfun.utils.leja_nodes(lpfun.utils.cheb2nd(n))
        c2l = lpfun.utils.chebyshev2lagrange(nodes)
        L, U = lpfun.utils.lu(c2l)
        L_inv, U_inv = lpfun.utils.inv(L), lpfun.utils.inv(U[::-1, ::-1])[::-1, ::-1]
        l2c = U_inv @ L_inv
        identity = c2l @ l2c
        eps = np.linalg.norm(identity - np.eye(n))
        assert eps < 1e-8


@pytest.mark.parametrize("m", ms)
def test_tube_absolute_degree(m: int):
    for n in ns(m):
        A = lpfun.core.set.lp_set(m, n, 1.0)
        tube = lpfun.core.set.lp_tube(A, m, n, 1.0)
        tube_sum = np.sum(tube)
        cardinality = lpfun.core.utils.binomial(n + m, m)
        assert tube_sum == cardinality


@pytest.mark.parametrize("m", ms)
def test_tube_euclidean_degree(m: int):
    for n in ns(m):
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


@pytest.mark.parametrize("m, p, ba", m_p_ba)
def test_fnt_ifnt(m: int, p: float, ba: str):
    for n in ns(m):
        t = lpfun.Transform(m, n, p, basis=ba, precompilation=False, report=False)
        function_values = np.random.rand(len(t))
        reconstruction = t.ifnt(t.fnt(function_values))
        eps = np.linalg.norm(reconstruction - function_values)
        assert eps < 1e-9


@pytest.mark.parametrize("m, p, ba", m_p_ba)
def test_dx(m: int, p: float, ba: str):
    for n in ns(m):
        t = lpfun.Transform(m, n, p, basis=ba, report=False)
        seed = np.random.rand(m)

        def f(x):
            return np.sum(seed * x**n)

        def df(i, k, x):
            if k == 1:
                return n * seed[i] * x[i] ** (n - 1)
            elif k == 2:
                return n * (n - 1) * seed[i] * x[i] ** (n - 2)
            elif k == 3:
                return n * (n - 1) * (n - 2) * seed[i] * x[i] ** (n - 3)

        function_values = np.array([f(x) for x in t.grid])
        coeffs = t.fnt(function_values)
        for k in [1, 2, 3]:
            for i in range(m):
                dx_function_values = np.array([df(i, k, x) for x in t.grid])
                dx_reconstruction = t.ifnt(t.dx(coeffs, i, k))
                eps = np.linalg.norm(dx_reconstruction - dx_function_values)
                assert eps < 1e-6


@pytest.mark.parametrize("m, p, ba", m_p_ba)
def test_eval(m: int, p: float, ba: str):
    for n in ns(m):
        t = lpfun.Transform(m, n, p, basis=ba, report=False)
        seed = np.random.rand(m)

        def f(x):
            return np.sum(seed * x**n)

        function_values = np.array([f(x) for x in t.grid])
        coeffs = t.fnt(function_values)
        x = np.random.rand(m)
        function_value = f(x)
        reconstruction = t.eval(coeffs, x)
        eps = np.abs(reconstruction - function_value)
        assert eps < 1e-6
