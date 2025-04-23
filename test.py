import pytest
import numpy as np
import lpfun as lp
from itertools import product
from lpfun import NP_FLOAT, NP_INT

# Parameters

ms = [1, 2, 3, 4, 5, 6] # [2]  # 
ps = [1.0 , 2.0, np.inf] # [2.0]  # 
bases = ["newton", "chebyshev"] # ["newton"] # 
parallel = [True, False] # [False] # 
mpbapa = list(product(ms, ps, bases, parallel))


# Prerequisites


def ns(m: int) -> np.ndarray:
    # Generate random ns
    n_max_m, default = {1: 60, 2: 50, 3: 40, 4: 30, 5: 20}, 10
    n_max = n_max_m.get(m, default)
    num = max(int(n_max * 0.2), 1)
    ns = np.random.randint(1, n_max, num)
    return ns


def monomial(m, n):
    exponents = np.random.randint(0, n, size=m, dtype=NP_INT)

    def f(*x):
        return np.prod(x**exponents, dtype=NP_FLOAT)

    def df(i, *x):
        if exponents[i] == 0:
            return 0.0
        else:
            return exponents[i] * np.prod(
                np.array(x, dtype=NP_FLOAT)
                ** np.array(
                    [exp if j != i else exp - 1 for j, exp in enumerate(exponents)],
                    dtype=NP_FLOAT,
                ),
                dtype=NP_FLOAT,
            )

    return f, df


# Tests


def test_newton2lagrange():
    for n in ns(1):
        nodes = lp.utils.leja_nodes(lp.utils.cheb2nd(n))
        n2l = lp.utils.newton2lagrange(nodes)
        l2n = lp.utils.inv(n2l)
        identity = n2l @ l2n
        eps = np.linalg.norm(identity - np.eye(n))
        assert eps < 1e-8


def test_chebyshev2lagrange():
    for n in ns(1):
        nodes = lp.utils.leja_nodes(lp.utils.cheb2nd(n))
        c2l = lp.utils.chebyshev2lagrange(nodes)
        L, U = lp.utils.lu(c2l)
        L_inv, U_inv = lp.utils.inv(L), lp.utils.inv(U[::-1, ::-1])[::-1, ::-1]
        l2c = U_inv @ L_inv
        identity = c2l @ l2c
        eps = np.linalg.norm(identity - np.eye(n))
        assert eps < 1e-8


@pytest.mark.parametrize("m", ms)
def test_tube_absolute_degree(m: int):
    for n in ns(m):
        tube = lp.utils.tube(m, n, 1.0)
        tube_sum = np.sum(tube)
        cardinality = lp.utils._binomial(n + m, m)
        assert tube_sum == cardinality


@pytest.mark.parametrize("m", ms)
def test_tube_euclidean_degree(m: int):
    for n in ns(m):
        tube = lp.utils.tube(m, n, 2.0)
        tube_sum = np.sum(tube)
        cardinality = len(
            [
                point
                for point in product(range(n + 1), repeat=m)
                if np.linalg.norm(point) <= n
            ]
        )
        assert tube_sum == cardinality


@pytest.mark.parametrize("m, p, ba, pa", mpbapa)
def test_fnt_ifnt(m: int, p: float, ba: str, pa: bool):
    for n in ns(m):
        t = lp.Transform(
            m,
            n,
            p,
            basis=ba,
            parallel=pa,
            report=False,
            precompilation=True,
        )
        function_values = np.random.rand(len(t))
        reconstruction = t.ifnt(t.fnt(function_values))
        eps = np.linalg.norm(reconstruction - function_values)
        assert eps < 1e-9
    # assert False # TODO UNDO!


@pytest.mark.parametrize("m, p, ba, pa", mpbapa)
def test_dx(m: int, p: float, ba: str, pa: bool):
    for n in range(1, 5):
        f, df = monomial(m, n)
        t = lp.Transform(
            m,
            int(1 + 2 * m / p) * n + 1,
            p,
            basis=ba,
            parallel=pa,
            report=False,
        )
        function_values = np.array([f(*x) for x in t.grid])
        coeffs = t.fnt(function_values)
        for i in range(m):
            dx_function_values = np.array([df(i, *x) for x in t.grid])
            dx_reconstruction = t.ifnt(t.dx(coeffs, i))
            eps = np.linalg.norm(dx_reconstruction - dx_function_values)
            assert eps < 1e-7
    # assert False # TODO UNDO!
        
