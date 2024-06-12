import pytest
import numpy as np
import newfun as nf


def gen_ns(m: int) -> nf.NP_ARRAY:
    # Generate random ns
    n_max_m = {1: 100, 2: 60, 3: 50, 4: 40, 5: 20, 6:10}
    n_max = n_max_m.get(m, 5)
    num = max(int(n_max * 0.2), 1)
    ns = np.random.randint(2, n_max + 1, num)
    return ns


def test_n2l_l2n():
    ns = gen_ns(1)
    for n in ns:
        # Generate unisolvent nodes 1d
        nodes = nf.utils.unisolvent_nodes_1d(n)

        # Forward and backward transformations
        l2n = nf.utils.l2n(nodes)
        n2l = nf.utils.n2l(nodes)

        # Check if the transformations are inverse to each other
        identity = n2l @ l2n
        assert np.allclose(identity, np.eye(n))  # , rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_tiling_md_absolute(m: int):
    ns = gen_ns(m)
    for n in ns:
        # Check if the tiling is valid for p = 1.0
        tiling = nf.utils.tiling(m, n, 1.0)
        tiling_sum = np.sum(tiling)
        binom = nf.utils_njit._binomial(n + m, m)
        assert tiling_sum == binom


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_fnt_ifnt_md_absolute(m: int):
    ns = gen_ns(m)
    for n in ns:
        # Transform object for p = 1.0
        t = nf.Transform(m, n, 1.0)

        # Generate random function values
        function_values = np.random.rand(len(t))

        # Apply forward and backward transformations
        reconstruction = t.ifnt(t.fnt(function_values))

        # Forward and backward transformations
        assert np.allclose(reconstruction, function_values)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_fnt_ifnt_md_euclidean(m: int):
    ns = gen_ns(m)
    for n in ns:
        # Transform object for p = 2.0
        t = nf.Transform(m, n)

        # Generate random function values
        function_values = np.random.rand(len(t))

        # Apply forward and backward transformations
        reconstruction = t.ifnt(t.fnt(function_values))

        # Forward and backward transformations
        assert np.allclose(reconstruction, function_values)

@pytest.mark.parametrize("m", [1, 2, 3])
def test_fnt_ifnt_md_maximal(m: int):
    ns = gen_ns(m)
    for n in ns:
        # Transform object for p = infinity
        t = nf.Transform(m, n, p=np.infty)

        # Generate random function values
        function_values = np.random.rand(len(t))

        # Apply forward and backward transformations
        reconstruction = t.ifnt(t.fnt(function_values))

        # Forward and backward transformations
        assert np.allclose(reconstruction, function_values)

