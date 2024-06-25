import pytest
import numpy as np
import lpfun as nf

# Parameters
ms = [1, 2, 3, 4, 5]
ps = [1.0, 2.0, np.infty]


def ns(m: int) -> nf.NP_ARRAY:
    # Generate random ns
    n_max_m = {1: 100, 2: 60, 3: 50, 4: 40, 5: 20, 6: 10}
    n_max = n_max_m.get(m, 5)
    num = max(int(n_max * 0.2), 1)
    ns = np.random.randint(2, n_max + 1, num)
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


def test_n2l_l2n():
    for n in ns(1):
        # Generate unisolvent nodes 1d
        nodes = nf.utils.unisolvent_nodes_1d(n, nf.utils.cheb)

        # Forward and backward transformations
        l2n = nf.utils.l2n(nodes)
        n2l = nf.utils.n2l(nodes)

        # Check if the transformations are inverse to each other
        identity = n2l @ l2n
        assert np.allclose(identity, np.eye(n))  # , rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_tiling_md_absolute(m: int):
    for n in ns(m):
        # Check if the tiling is valid for p = 1.0
        tiling = nf.utils.tiling(m, n, 1.0)
        tiling_sum = np.sum(tiling)
        binom = nf.utils._binomial(n + m, m)
        assert tiling_sum == binom


@pytest.mark.parametrize("m, p", [(m, p) for m in ms for p in ps])
def test_fnt_ifnt_md(m: int, p: float):
    for n in ns(m):
        # Transform object
        t = nf.Transform(m, n, p)

        # Generate random function values
        function_values = np.random.rand(len(t))

        # Apply forward and backward transformations
        reconstruction = t.pull(t.push(function_values))

        # Compare the reconstruction with the exact
        assert np.allclose(reconstruction, function_values)


@pytest.mark.parametrize("m, p", [(m, p) for m in ms for p in ps])
def test_fnt_ifnt_md_derivative(m: int, p: float):
    if m > 3:
        return
    for n in range(3, 8):
        # Generate a monomial
        f, df = monomial(m, n)

        n_prime = int(1 + 2 * m / p) * (n + 1)

        # Transform object
        t = nf.Transform(m, n_prime, p)

        # Calculate the exact function values
        function_values = np.array([f(*x) for x in t.unisolvent_nodes])

        # Perform the fast Newton transformation
        coeffs = t.push(function_values)

        for i in range(m):
            # Calculate the exact derivative
            dx_function_values = np.array([df(i, *x) for x in t.unisolvent_nodes])

            # Apply forward, derivative and backward transformations
            reconstruction = t.pull(t.dx(coeffs, i))

            # Compare the reconstruction with the exact
            assert np.allclose(reconstruction, dx_function_values)
