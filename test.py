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


# BLAS backend of the LU transforms

BLAS_NS = [1, 4, 7]


@pytest.mark.parametrize("m, n, p, ba, pr", list(product([2, 3, 4, 5], BLAS_NS, ps, bases, precomputation)))
def test_function_blas(m: int, n: int, p: float, ba: str, pr: bool):
    rng = np.random.default_rng(0)
    kwargs = dict(basis=ba, precomputation=pr, precompilation=False, report=False)
    fun_blas = lpfun.Function(m, n, p, backend="blas", **kwargs)
    fun_tube = lpfun.Function(m, n, p, backend="tube", **kwargs)
    assert fun_blas._backend == "blas"
    values = rng.random(len(fun_blas))
    coeffs = rng.random(len(fun_blas))
    # the GEMMs sum in a different order than the tube kernels; entries can cancel to
    # well below the vector's scale (|eval| ~ 1e3 at m = 5), so atol scales with it
    for got, ref in ((fun_blas.interp(values), fun_tube.interp(values)), (fun_blas.eval(coeffs), fun_tube.eval(coeffs))):
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * max(1.0, np.abs(ref).max()))
    # round trip
    np.testing.assert_allclose(fun_blas.eval(fun_blas.interp(values)), values, rtol=0, atol=1e-10)


def test_function_blas_precompilation():
    fun = lpfun.Function(3, 6, 2.0, backend="blas", report=False)
    values = np.random.default_rng(1).random(len(fun))
    np.testing.assert_allclose(fun.eval(fun.interp(values)), values, rtol=0, atol=1e-10)


def test_function_blas_invalid_backend():
    with pytest.raises(ValueError):
        lpfun.Function(2, 3, backend="lapack", precompilation=False, report=False)


@pytest.mark.parametrize("m, p, ba", list(product([2, 3], ps, bases)))
def test_diff_blas(m: int, p: float, ba: str):
    # for m = 2 and m = 3 the BLAS backend also differentiates: same results as tube-wise
    kw = dict(basis=ba, precompilation=False, report=False)
    fun_blas = lpfun.Function(m, 7, p, backend="blas", **kw)
    fun_tube = lpfun.Function(m, 7, p, backend="tube", **kw)
    x = np.random.default_rng(m).uniform(-1, 1, len(fun_blas))
    for dim, order in product(range(m), [1, 2, 3]):
        for T in ("diff", "diffT"):
            got, ref = getattr(fun_blas, T)(x, dim, order), getattr(fun_tube, T)(x, dim, order)
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10 * max(1.0, np.abs(ref).max()))


@pytest.mark.parametrize("pr", precomputation)
def test_function_blas_state(pr: bool):
    # the plan and the factors exist for both precomputation values; "tube" keeps no BLAS
    # state and "blas" no tube-wise inverses
    g = lpfun.Function(3, 5, basis="chebyshev", backend="blas", precomputation=pr, precompilation=False, report=False)
    assert set(g._blas_args) == {"m", "buckets", "padpos", "between", "final"}
    assert len(g._blas_fwd) == 2 and g._blas_trsm == (not pr)
    assert hasattr(g, "_blas_inv") == pr and hasattr(g, "_blas_solve") == (not pr)
    assert not hasattr(g, "_inv_Vx") and not hasattr(g, "_inv_Vx_L")
    newton = lpfun.Function(3, 5, backend="blas", precomputation=pr, precompilation=False, report=False)
    assert len(newton._blas_fwd) == 1 and (pr or len(newton._blas_solve) == 1)
    u = lpfun.Function(3, 5, backend="tube", precomputation=pr, precompilation=False, report=False)
    assert not any(hasattr(u, a) for a in ("_blas_args", "_blas_fwd", "_blas_inv", "_blas_solve"))
    assert "blas" in repr(g) and "tube" in repr(u)


@pytest.mark.parametrize(
    "m, n, p, ba, pr",
    [(3, 20, 2.0, "chebyshev", True), (3, 16, 1.0, "newton", True), (3, 9, np.inf, "legendre", False),
     (4, 12, 2.0, "chebyshev", False), (4, 9, 1.0, "legendre", True), (5, 8, 2.0, "newton", False),
     (6, 6, 2.0, "chebyshev", True), (2, 60, 0.5, "chebyshev", True), (2, 40, 1.0, "newton", False),
     (2, 48, 2.0, "legendre", True), (2, 30, np.inf, "chebyshev", False)],
)
def test_blas_layout(m: int, n: int, p: float, ba: str, pr: bool):
    """The padded layout: buckets by the DP, aligned blocks, valid index maps; products and
    triangular solves agree with the tube-wise kernels at degrees where the DP takes
    several buckets."""
    from lpfun.core.blas import optimal_bucket_lengths, transform_blas, ALIGN
    kw = dict(basis=ba, precomputation=pr, precompilation=False, report=False)
    fun = lpfun.Function(m, n, p, backend="blas", **kw)
    plan = fun._blas_args
    N, S, buckets = len(fun), len(plan["between"]), plan["buckets"]
    # the DP's tops cover every tube and are a partition into ranges (padded to the range's
    # longest length); blocks are aligned and tile the buffer
    tops = optimal_bucket_lengths(fun._T)
    assert np.all(tops >= fun._T) and np.all(np.isin(np.unique(tops), fun._T))
    assert plan["m"] == m and len(buckets) == len(np.unique(tops)) <= len(np.unique(fun._T))
    assert np.all(buckets[:, 1:] % ALIGN == 0)
    assert np.array_equal(buckets[:, 1], np.r_[0, np.cumsum(buckets[:, 0] * buckets[:, 2])][:-1])
    assert S == int(np.sum(buckets[:, 0] * buckets[:, 2]))
    assert N <= S <= len(fun._T) * (n + 1) + ALIGN * (n + 1) * len(buckets)
    # every entry has its own slot; the pads read the zero slot; both gathers permute the slots
    padpos = plan["padpos"]
    assert padpos.dtype == np.int32 and plan["between"].dtype == np.int32
    assert len(np.unique(padpos)) == N and padpos.min() >= 0 and padpos.max() < S
    free = np.setdiff1d(np.arange(S), padpos)
    assert np.all(plan["between"][free] == S)
    assert np.array_equal(np.sort(plan["between"][padpos]), np.sort(padpos))
    assert np.array_equal(np.sort(plan["final"]), np.sort(padpos))
    rng = np.random.default_rng(3)
    f = rng.uniform(-1, 1, N)
    ref = lpfun.Function(m, n, p, backend="tube", **kw)
    for got, want in ((fun.interp(f), ref.interp(f)), (fun.eval(f), ref.eval(f))):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12 * max(1.0, np.abs(want).max()))
    with pytest.raises(ValueError):  # a short input would be read past its end
        transform_blas(fun._blas_fwd, f[:-1], **plan)


@pytest.mark.parametrize("size", [1, 1000, 70_000])
def test_blas_gather_scatter(size: int):
    """The gather and scatter kernels against numpy, on their serial and parallel paths."""
    from lpfun.core import blas
    assert 1000 < blas.SERIAL < 70_000
    rng = np.random.default_rng(size)
    src, idx = rng.random(size + 5), rng.permutation(size + 5)[:size].astype(np.int32)
    dst = np.full(size + 5, -1.0)
    blas._gather(src, idx, dst)
    assert np.array_equal(dst[:size], src[idx]) and np.all(dst[size:] == -1.0)
    dst = np.full(size + 5, -1.0)
    blas._scatter(src[:size], idx, dst)
    expected = np.full(size + 5, -1.0)
    expected[idx] = src[:size]
    assert np.array_equal(dst, expected)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_blas_transforms(m: int):
    """`transform_blas` and `itransform_blas` on their own: the identity round-trips bit
    for bit through the gather chain; a lower factor's garbage in the pads never reaches
    the upper factor that follows it."""
    from lpfun.core.blas import blas_plan, transform_blas, itransform_blas
    n = 7
    kw = dict(basis="chebyshev", precompilation=False, report=False)
    fun = lpfun.Function(m, n, 1.0, backend="blas", **kw)  # p = 1: many distinct lengths
    tube = lpfun.Function(m, n, 1.0, backend="tube", **kw)
    plan = blas_plan(fun._T, fun._cs_T, fun._pi, m)
    x = np.random.default_rng(m).uniform(-1, 1, len(fun))
    I = np.eye(n + 1)
    np.testing.assert_array_equal(transform_blas((I,), x, **plan), x)
    np.testing.assert_array_equal(transform_blas((I, I), x, **plan), x)
    np.testing.assert_array_equal(itransform_blas((np.asfortranarray(I),), x, **plan), x)
    np.testing.assert_array_equal(itransform_blas((np.asfortranarray(I), np.asfortranarray(I)), x, **plan), x)
    # the factors one at a time and in one call; products with the inverses and solves
    U, L = fun._blas_fwd
    inv_L, inv_U = fun._blas_inv
    np.testing.assert_allclose(transform_blas((L,), transform_blas((U,), x, **plan), **plan), tube.eval(x), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(transform_blas((U, L), x, **plan), tube.eval(x), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(transform_blas((inv_L, inv_U), x, **plan), tube.interp(x), rtol=1e-10, atol=1e-12)
    solved = itransform_blas((np.asfortranarray(L), np.asfortranarray(U)), x, **plan)
    np.testing.assert_allclose(solved, tube.interp(x), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(transform_blas((U, L), solved, **plan), x, rtol=1e-10, atol=1e-12)
    # a single lower factor (the Newton case): the solve against the product with the inverse
    solved = itransform_blas((np.asfortranarray(L),), x, **plan)
    np.testing.assert_allclose(solved, transform_blas((inv_L,), x, **plan), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("m, p", list(product([2, 3, 4, 5], [1.0, 2.0])))
def test_dtransform_blas(m: int, p: float):
    """`dtransform_blas` along every direction against the tube-wise kernels, for an upper
    and a lower triangular matrix; `Function` uses it for m <= 3 only."""
    from lpfun.core.blas import blas_plan, dtransform_blas
    n = 7
    kw = dict(basis="chebyshev", precompilation=False, report=False)
    fun = lpfun.Function(m, n, p, backend="blas", **kw)
    tube = lpfun.Function(m, n, p, backend="tube", **kw)
    assert hasattr(fun, "_blas_Dx") == (m <= 3) and not hasattr(tube, "_blas_Dx")
    plan = blas_plan(fun._T, fun._cs_T, fun._pi, m)
    x = np.random.default_rng(m).uniform(-1, 1, len(fun))
    D = np.triu(lpfun.basis.diff.chebyshev2derivative(fun.nodes))
    for dim in range(m):
        ref = tube.diff(x, dim)
        got = dtransform_blas(D, x, dim, **plan)
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-11 * np.abs(ref).max())
        ref = tube.diffT(x, dim)
        got = dtransform_blas(np.ascontiguousarray(D.T), x, dim, **plan)
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-11 * np.abs(ref).max())


@pytest.mark.parametrize("call, w0", [(50_000, 32), (50_000, 0), (0, 32), (300, 3), (2_000, 100), (0, 0)])
def test_optimal_bucket_lengths_is_the_exact_minimum(call: int, w0: int):
    """`optimal_bucket_lengths` returns a partition of the sorted lengths into ranges whose modelled
    time, sum of ``call + L^2 (w + w0)``, is the least over all 2^(K-1) such partitions."""
    from itertools import combinations
    from lpfun.core import blas

    def cost(tops, T):
        L, cnt = np.unique(tops, return_counts=True)
        w = -(-cnt // blas.ALIGN) * blas.ALIGN
        return float(np.sum(call + L.astype(float) ** 2 * (w + w0)))

    rng = np.random.default_rng(call + w0)
    for trial in range(6):
        u = np.sort(rng.choice(np.arange(1, 40), size=rng.integers(1, 10), replace=False))
        T = np.repeat(u, rng.integers(1, 60, size=len(u)))[::-1].copy()  # nonincreasing, as in a plan
        tops = blas.optimal_bucket_lengths(T, call=call, w0=w0)
        assert np.all(tops >= T) and np.all(np.isin(tops, u))
        assert np.all(np.diff(tops) <= 0)  # ranges of consecutive lengths
        best = np.inf
        for r in range(len(u)):
            for cuts in combinations(range(1, len(u)), r):
                ends = np.r_[np.array(cuts, dtype=np.int64), len(u)]  # range j ends before u[ends[j]]
                tops_c = u[ends - 1][np.searchsorted(ends, np.searchsorted(u, T), side="right")]
                best = min(best, cost(tops_c, T))
        assert cost(tops, T) == pytest.approx(best, rel=1e-12)


def test_optimal_bucket_lengths_width_term_merges_narrow_buckets(monkeypatch):
    """The pass over the operand, ``L^2 w0`` per bucket, merges buckets of a few columns:
    for m = 2 every length is shared by one or two tubes. The constants come from the
    arguments, else from the module constants at the time of the call."""
    from lpfun.core import blas

    fun = lpfun.Function(2, 300, 2.0, backend="tube", precompilation=False, report=False)
    T = fun._T
    counts = [len(np.unique(blas.optimal_bucket_lengths(T, w0=w0))) for w0 in (0, 32, 10**9)]
    assert counts[0] > counts[1] > counts[2] == 1
    assert blas.W0 == 32 and len(np.unique(blas.optimal_bucket_lengths(T))) == counts[1]
    plans = [blas.blas_plan(T, fun._cs_T, fun._pi, 2, w0=w0) for w0 in (0, None, 10**9)]
    assert [len(q["buckets"]) for q in plans] == counts
    assert len(blas.blas_plan(T, fun._cs_T, fun._pi, 2, call=10**12, w0=0)["buckets"]) == 1
    monkeypatch.setattr(blas, "W0", 0)
    assert len(np.unique(blas.optimal_bucket_lengths(T))) == counts[0]
    monkeypatch.setattr(blas, "CALL", 10**12)
    assert len(blas.blas_plan(T, fun._cs_T, fun._pi, 2)["buckets"]) == 1
    for bad in (dict(call=-1.0), dict(w0=-1.0), dict(w0=float("nan"))):
        with pytest.raises(ValueError):
            blas.optimal_bucket_lengths(T, **bad)


# defaults and input checks

KW = dict(precompilation=False, report=False)


@pytest.mark.parametrize("ba", bases)
def test_auto_rule(ba: str):
    for m in range(1, 7):
        assert lpfun.Function(m, 4, basis=ba, **KW).backend == ("tube" if m == 1 else "blas")
    for p in (0.5, 1.0, 2.0, np.inf):  # thin sets included
        assert lpfun.Function(2, 20, p, basis=ba, **KW).backend == "blas"
    assert lpfun.Function(3, 6, 0.5, basis=ba, **KW).backend == "blas"
    # explicit choices are kept
    assert lpfun.Function(4, 5, basis=ba, backend="tube", **KW)._backend == "tube"
    assert lpfun.Function(1, 5, basis=ba, backend="blas", **KW)._backend == "tube"  # a single tube


def test_repr_and_eq():
    g = lpfun.Function(2, 5, basis="chebyshev", **KW)
    u = lpfun.Function(2, 5, basis="chebyshev", backend="tube", **KW)
    assert "blas" in repr(g)
    # the transform changes neither the space nor the meaning of the coefficients
    assert g == u



@pytest.mark.parametrize("T", ["tube", "blas"])
def test_input_length_checked(T: str):
    # a short or long input raises instead of reading past the array on the BLAS path
    fun = lpfun.Function(3, 6, backend=T, **KW)
    for bad in (np.ones(len(fun) - 1), np.ones(len(fun) + 7), np.ones((len(fun), 1))):
        with pytest.raises(ValueError):
            fun.interp(bad)
        with pytest.raises(ValueError):
            fun.eval(bad)
        with pytest.raises(ValueError):
            fun.diff(bad, 0)
        with pytest.raises(ValueError):
            fun.diffT(bad, 0)
    assert fun.interp(list(np.ones(len(fun)))).shape == (len(fun),)
