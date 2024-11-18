import pytest
import numpy as np
import lpfun as lp

# Parameters

ms = [1, 2, 3, 4, 5, 6]
ps = [1.0, 2.0, np.inf]


# Prerequisites


def ns(m: int) -> np.ndarray:
    # Generate random ns
    n_max_m = {1: 120, 2: 80, 3: 40, 4: 20, 5: 10}
    n_max = n_max_m.get(m, 5)
    num = max(int(n_max * 0.2), 1)
    ns = np.random.randint(1, n_max, num)
    return ns


def monomial(m=2, n=3):
    # Generate random coefficient for the monomial
    c = 2 * np.random.rand() - 1

    # Generate random exponents for each dimension
    exponents = np.random.randint(0, n + 1, size=m)

    # Generate the monomial and its derivative
    def f(*x):
        return c * np.prod(np.array(x) ** exponents)

    def df(i, *x):
        if exponents[i] == 0:
            return 0.0
        else:
            return (
                c
                * exponents[i]
                * np.prod(
                    np.array(x)
                    ** [exp if j != i else exp - 1 for j, exp in enumerate(exponents)]
                )
            )

    return f, df


# Tests


def test_n2l_and_l2n():
    for n in ns(1):
        # Generate unisolvent nodes 1d
        nodes = lp.utils.leja_nodes(n, lp.utils.cheb)

        # Forward and backward transformations
        l2n = lp.utils.l2n(nodes)
        n2l = lp.utils.n2l(nodes)

        # Check if the transformations are inverse to each other
        identity = n2l @ l2n
        assert np.allclose(identity, np.eye(n), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_tube_absolute_degree(m: int):
    for n in ns(m):
        # Check if the tube is valid for p = 1.0
        tube = lp.utils.tube(m, n, 1.0)
        tube_sum = np.sum(tube)
        binom = lp.utils._binomial(n + m, m)
        assert tube_sum == binom


@pytest.mark.parametrize("m, p", [(m, p) for m in ms for p in ps])
def test_newton_fnt_and_ifnt(m: int, p: float):
    for n in ns(m):
        # Transform object
        t = lp.Transform(m, n, p)

        # Generate random function values
        function_values = np.random.rand(len(t))

        # Apply forward and backward transformations
        reconstruction = t.ifnt(t.fnt(function_values))

        # Compare the reconstruction with the exact
        assert np.allclose(reconstruction, function_values)


@pytest.mark.parametrize("m, p", [(m, p) for m in ms for p in ps])
def test_newton_dx(m: int, p: float):
    if m > 3:
        return
    for n in range(3, 8):
        # Generate a monomial
        f, df = monomial(m, n)

        # We need to increase the number of nodes for p != infinity
        n_prime = int(1 + 2 * m / p) * (n + 1)

        # Transform object
        t = lp.Transform(m, n_prime, p)

        # Calculate the exact function values
        function_values = np.array([f(*x) for x in t.grid])

        # Perform the fast Newton transformation
        coeffs = t.fnt(function_values)

        for i in range(m):
            # Calculate the exact derivative
            dx_function_values = np.array([df(i, *x) for x in t.grid])

            # Apply forward, derivative and backward transformations
            dx_reconstruction = t.ifnt(t.dx(coeffs, i))

            # Compare the reconstruction with the exact
            assert np.allclose(dx_reconstruction, dx_function_values)
