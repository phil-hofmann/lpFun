"""
The LU transforms behind `interp`, `eval`, `diff` and `diffT`.

A transform applies a triangular (n+1) x (n+1) matrix M, a factor of the 1D Vandermonde
matrix, of its inverse or of a differentiation matrix, to every tube of the index set: a
tube of length t is multiplied by the leading block M[:t, :t], or solved with it.
`transform` (products) and `itransform` (solves) do so along every direction, one after
another; `dtransform` (products) along a single direction.

Both backends work on buckets of tubes. A bucket stores its tubes, padded with zeros to
its length L, as the columns of one contiguous block of shape (L, width). After each
direction the entries are rotated: the next direction comes to the columns.

* ``backend="tube"``: every tube is its own bucket of width 1. Nothing is padded, the
  layout is the vector itself, and a numba kernel applies M[:t, :t] tube by tube. It
  writes every result straight to its place in the next direction.
* ``backend="blas"``: the tubes are grouped into buckets of similar length and every block
  is applied by one BLAS call: GEMM, the GEneral Matrix-Matrix product C = A @ B, or TRSM,
  the TRiangular Solve with a Matrix of right-hand sides. A gather then rotates the
  entries.

The padding is exact: a lower triangular matrix never reads the pads, and an upper
triangular one reads them as the zeros they hold. The gathers refill every pad from a zero
slot.

The buckets of ``backend="blas"`` are chosen by `optimal_bucket_lengths`, which minimises
a model of the time of the products; the solves use the same buckets. Its two constants
`CALL` and `W0` belong to the machine and its BLAS library; the results do not depend on
them. To tune them, assign the module constants before constructing a `Function`, or pass
``call`` and ``w0`` to `transform_plan`.
"""

import numpy as np
import numba as nb
from typing import Optional, Sequence, Tuple
from scipy.linalg.blas import dtrsm as _dtrsm
from lpfun import CACHE, PARALLEL


def njit(*args, **kwargs):
    kwargs.setdefault("cache", CACHE)
    return nb.njit(*args, **kwargs)


prange = nb.prange


ALIGN = 8  # bucket widths are multiples of 8 columns: 8 doubles are one 64-byte cache line
CALL = 50_000  # latency of one BLAS call, in multiply-adds
W0 = 32  # one pass over the matrix operand of a product, in columns of arithmetic
SERIAL = 65_536  # a gather of fewer entries is faster on one thread than on the thread pool
SERIAL_TUBES = 65_536  # so are the tube-wise kernels below this many multiply-adds per direction


# Public functions


def optimal_bucket_lengths(
    T: np.ndarray,
    call: Optional[float] = None,
    w0: Optional[float] = None,
) -> np.ndarray:
    """O(K^2), K = number of distinct tube lengths

    Padded length of every tube. The sorted distinct lengths are cut into ranges, each
    padded to its longest length L and holding w tubes (rounded up to a multiple of
    `ALIGN`), such that the sum of ``call + L^2 (w + w0)`` over the ranges is the least
    possible: the arithmetic ``L^2 w``, one pass over the L x L operand, and the latency of
    the call. None takes the module constant `CALL` or `W0` at the time of the call.
    """
    call = float(CALL if call is None else call)
    w0 = float(W0 if w0 is None else w0)
    if not (call >= 0.0 and w0 >= 0.0):
        raise ValueError(f"call and w0 must be nonnegative, but got {call} and {w0}.")
    T = np.asarray(T, dtype=np.int64)
    u, count = np.unique(T, return_counts=True)  # ascending
    K = len(u)
    cs = np.r_[0, np.cumsum(count)]
    ### best[j]: least cost of the first j lengths, its last range is u[arg[j]:j]
    best = np.zeros(K + 1)
    arg = np.zeros(K + 1, dtype=np.int64)
    for j in range(1, K + 1):
        L = float(u[j - 1])
        w = np.ceil((cs[j] - cs[:j]) / ALIGN) * ALIGN
        v = best[:j] + call + L * L * (w + w0)
        i = int(np.argmin(v))
        best[j], arg[j] = v[i], i
    ###
    tops = np.empty(K, dtype=np.int64)
    j = K
    while j > 0:
        i = arg[j]
        tops[i:j] = u[j - 1]
        j = i
    return tops[np.searchsorted(u, T)]


def transform_plan(
    T: np.ndarray,
    cs_T: np.ndarray,
    pi: np.ndarray,
    m: int,
    backend: str,
    call: Optional[float] = None,
    w0: Optional[float] = None,
) -> dict:
    """O(|A|)

    Index maps of the layout, from the tube lengths `T`, their running totals `cs_T` and
    the transposition `pi` between two directions. With ``backend="tube"`` every tube is
    its own bucket; with ``backend="blas"`` the buckets come from `optimal_bucket_lengths`,
    which takes `call` and `w0`.

    Returns a dict, the keyword arguments of `transform`, `itransform` and `dtransform`.
    Its keys, with S the size of the layout (S = |A| for ``backend="tube"``):

    m       : number of directions, as given.
    backend : "tube" or "blas", as given.
    buckets : rows (length, base, width); the tubes of a bucket are the columns of the
              block (length, width) at `base`.

    ``backend="tube"`` rotates inside its kernels and needs only

    pi      : the transposition, as given: entry i of the current direction is entry pi[i]
              of the next.
    serial  : whether the kernels run on one thread: for a single tube, and below
              `SERIAL_TUBES` multiply-adds per direction, where waking the thread pool
              costs more than it saves.

    ``backend="blas"`` rotates by gathers and needs

    padpos  : slot of every entry, base + row * width + column: row is the position of the
              entry in its tube, column the rank of the tube in its bucket.
    between : gather between two directions, of length S: the transposition `pi` composed
              with `padpos`; every pad reads the zero slot S.
    final   : gather after the last direction.

    The keys of the other backend are None (`serial` False). The gathers are stored rather
    than composed from `pi` and `padpos` on every call: the composition costs a second pass
    and a zero fill, 2.5 to 8 times one gather.
    """
    if not backend in ["tube", "blas"]:
        raise ValueError("Invalid choice for backend.")
    T = np.asarray(T, dtype=np.int64)
    cs_T = np.asarray(cs_T, dtype=np.int64)
    pi = np.asarray(pi, dtype=np.int64)
    N = int(cs_T[-1])
    if backend == "tube":
        ### every tube is a bucket of width 1, at its place in the vector
        width = np.ones(len(T), dtype=np.int64)
        work = int(np.sum(T * T)) // 2  # multiply-adds of one triangular product along a direction
        return {
            "m": int(m),
            "backend": "tube",
            "buckets": np.column_stack((T, cs_T[:-1], width)),
            "pi": pi.astype(np.int32 if N < 2**31 - 1 else np.int64),
            "serial": bool(len(T) == 1 or work < SERIAL_TUBES),
            "padpos": None,
            "between": None,
            "final": None,
        }
    ### buckets: one per padded length, its block (length, width) starts at base
    tops = optimal_bucket_lengths(T, call, w0)
    length, bucket, count = np.unique(tops, return_inverse=True, return_counts=True)
    bucket = bucket.ravel().astype(np.int64)  # bucket of every tube
    width = -(-count.astype(np.int64) // ALIGN) * ALIGN  # rounded up to a multiple of ALIGN
    size = length * width
    base = np.cumsum(size) - size
    S = int(np.sum(size))
    ### column of every tube in its block: its rank among the tubes of its bucket
    order = np.argsort(bucket, kind="stable")  # the tubes, bucket by bucket
    first = np.cumsum(count) - count  # number of tubes in the earlier buckets
    column = np.empty(len(T), dtype=np.int64)
    column[order] = np.arange(len(T), dtype=np.int64) - first[bucket[order]]
    ### slot of every entry: (row, column) of the block of its bucket, in C order
    tube = np.repeat(np.arange(len(T), dtype=np.int64), T)  # tube of every entry
    row = np.arange(N, dtype=np.int64) - cs_T[tube]
    b = bucket[tube]
    padpos = base[b] + row * width[b] + column[tube]
    ### gathers: entry i of the next direction is entry ipi[i] of the current one
    ipi = np.empty(N, dtype=np.int64)
    ipi[pi] = np.arange(N, dtype=np.int64)
    final = padpos[ipi]  # into the plain vector
    between = np.full(S, S, dtype=np.int64)  # into the layout: the pads read slot S
    between[padpos] = final
    ###
    dtype = np.int32 if S + 1 < 2**31 - 1 else np.int64
    return {
        "m": int(m),
        "backend": "blas",
        "buckets": np.column_stack((length, base, width)).astype(np.int64),
        "pi": None,
        "serial": False,
        "padpos": padpos.astype(dtype),
        "between": between.astype(dtype),
        "final": final.astype(dtype),
    }


def transform(
    Ms: Sequence[np.ndarray],
    lower: Sequence[bool],
    x: np.ndarray,
    m: int,
    backend: str,
    buckets: np.ndarray,
    pi: Optional[np.ndarray],
    serial: bool,
    padpos: Optional[np.ndarray],
    between: Optional[np.ndarray],
    final: Optional[np.ndarray],
) -> np.ndarray:
    """O(m S n) per matrix, S = size of the layout

    Products with the triangular matrices `Ms` (dense, C order) one after another, each
    along every direction. `lower` tells for every matrix whether it is lower triangular.
    """
    if backend == "tube":
        return _transform_tubes(Ms, lower, _vector(x, len(pi)), m, buckets, pi, serial)
    return _transform_blocks(Ms, _vector(x, len(padpos)), m, buckets, padpos, between, final)


def itransform(
    Ms: Sequence[np.ndarray],
    lower: Sequence[bool],
    x: np.ndarray,
    m: int,
    backend: str,
    buckets: np.ndarray,
    pi: Optional[np.ndarray],
    serial: bool,
    padpos: Optional[np.ndarray],
    between: Optional[np.ndarray],
    final: Optional[np.ndarray],
) -> np.ndarray:
    """O(m S n) per matrix, S = size of the layout

    Solves with the triangular matrices `Ms` (dense, C order) one after another, each
    along every direction. `lower` tells for every matrix whether it is lower triangular.
    """
    if backend == "tube":
        return _itransform_tubes(Ms, lower, _vector(x, len(pi)), m, buckets, pi, serial)
    x = _vector(x, len(padpos))
    return _itransform_blocks(Ms, lower, x, m, buckets, padpos, between, final)


def dtransform(
    D: np.ndarray,
    lower: bool,
    x: np.ndarray,
    perm: np.ndarray,
    m: int,
    backend: str,
    buckets: np.ndarray,
    pi: Optional[np.ndarray],
    serial: bool,
    padpos: Optional[np.ndarray],
    between: Optional[np.ndarray],
    final: Optional[np.ndarray],
) -> np.ndarray:
    """O(S n), S = size of the layout

    Product with the triangular matrix `D` (dense, C order) along a single direction: the
    permutation `perm` brings that direction to the tubes, ``x[perm]``, and back.
    """
    if backend == "tube":
        return _dtransform_tubes(D, lower, _vector(x, len(perm)), perm, buckets, serial)
    return _dtransform_blocks(D, _vector(x, len(padpos)), perm, buckets, padpos, between)


# tube-wise backend: every bucket is one tube, at its place in the vector
#
# The kernels multiply (or solve) and rotate in one pass: every result goes straight to
# its place in the next direction, B[rot[i]] for the entry i. So there are no gathers, and
# two work vectors are enough.
#
# The serial kernels are separate functions on purpose. Compiling one function twice, with
# and without parallel=True, does not work: numba's cache does not tell the two apart and
# hands both the code of whichever was compiled first.


def _transform_tubes(
    Ms: Sequence[np.ndarray],
    lower: Sequence[bool],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    pi: np.ndarray,
    serial: bool,
) -> np.ndarray:
    """O(m n|A|) per matrix

    `transform` for ``backend="tube"``.
    """
    multiply = _multiply_tubes_serial if serial else _multiply_tubes
    X, B = np.empty(len(x), dtype=np.float64), np.empty(len(x), dtype=np.float64)
    src = x  # the first product reads the input itself, the later ones the work vectors
    for M, low in zip(Ms, lower):
        for _ in range(m):  # along every direction
            multiply(M, low, buckets, pi, src, B)  # B[pi] = M @ src, tube by tube
            X, B = B, X
            src = X
    return X


def _itransform_tubes(
    Ms: Sequence[np.ndarray],
    lower: Sequence[bool],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    pi: np.ndarray,
    serial: bool,
) -> np.ndarray:
    """O(m n|A|) per matrix

    `itransform` for ``backend="tube"``.
    """
    solve = _solve_tubes_serial if serial else _solve_tubes
    X = x.copy()  # the solves are in place
    B = np.empty(len(x), dtype=np.float64)
    for M, low in zip(Ms, lower):
        for _ in range(m):  # along every direction
            solve(M, low, buckets, pi, X, B)  # B[pi] = inv(M) @ X, tube by tube
            X, B = B, X
    return X


def _dtransform_tubes(
    D: np.ndarray,
    lower: bool,
    x: np.ndarray,
    perm: np.ndarray,
    buckets: np.ndarray,
    serial: bool,
) -> np.ndarray:
    """O(n|A|)

    `dtransform` for ``backend="tube"``.
    """
    multiply = _multiply_tubes_serial if serial else _multiply_tubes
    X = np.empty(len(x), dtype=np.float64)
    _gather(x, perm, X)  # X = x[perm]: the direction to the tubes
    dot = np.empty(len(x), dtype=np.float64)
    multiply(D, lower, buckets, perm, X, dot)  # dot[perm] = D @ X, tube by tube: and back
    return dot


@njit(inline="always")
def _multiply_tube(
    M: np.ndarray,
    lower: bool,
    t: int,
    pos: int,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(t^2)

    B[rot] <- M[:t, :t] @ X for the tube of length t at `pos`.
    """
    for k in range(t):
        s = 0.0
        if lower:
            for l in range(k + 1):
                s += M[k, l] * X[pos + l]
        else:
            for l in range(k, t):
                s += M[k, l] * X[pos + l]
        B[rot[pos + k]] = s


@njit(inline="always")
def _solve_tube(
    M: np.ndarray,
    lower: bool,
    t: int,
    pos: int,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(t^2)

    In place X <- inv(M[:t, :t]) @ X for the tube of length t at `pos`, by forward (lower)
    or back (upper) substitution, and B[rot] <- X.
    """
    if lower:
        for k in range(t):
            s = 0.0
            for l in range(k):
                s += M[k, l] * X[pos + l]
            X[pos + k] = (X[pos + k] - s) / M[k, k]
            B[rot[pos + k]] = X[pos + k]
    else:
        for k in range(t - 1, -1, -1):
            s = 0.0
            for l in range(k + 1, t):
                s += M[k, l] * X[pos + l]
            X[pos + k] = (X[pos + k] - s) / M[k, k]
            B[rot[pos + k]] = X[pos + k]


@njit(parallel=PARALLEL)
def _multiply_tubes(
    M: np.ndarray,
    lower: bool,
    buckets: np.ndarray,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(n|A|)

    B[rot] <- M[:t, :t] @ X for every tube of length t: the product along the current
    direction, written to the places `rot` of the entries.
    """
    for i in prange(len(buckets)):
        _multiply_tube(M, lower, buckets[i, 0], buckets[i, 1], rot, X, B)


@njit
def _multiply_tubes_serial(
    M: np.ndarray,
    lower: bool,
    buckets: np.ndarray,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(n|A|)

    `_multiply_tubes` on one thread.
    """
    for i in range(len(buckets)):
        _multiply_tube(M, lower, buckets[i, 0], buckets[i, 1], rot, X, B)


@njit(parallel=PARALLEL)
def _solve_tubes(
    M: np.ndarray,
    lower: bool,
    buckets: np.ndarray,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(n|A|)

    In place X <- inv(M[:t, :t]) @ X for every tube of length t, and B[rot] <- X: the
    solve along the current direction, written to the places `rot` of the entries.
    """
    for i in prange(len(buckets)):
        _solve_tube(M, lower, buckets[i, 0], buckets[i, 1], rot, X, B)


@njit
def _solve_tubes_serial(
    M: np.ndarray,
    lower: bool,
    buckets: np.ndarray,
    rot: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
) -> None:
    """O(n|A|)

    `_solve_tubes` on one thread.
    """
    for i in range(len(buckets)):
        _solve_tube(M, lower, buckets[i, 0], buckets[i, 1], rot, X, B)


# BLAS backend: one call per bucket, then a gather to the next direction
#
# A BLAS call writes a whole block in one piece, so it cannot send every result to its
# place in the next direction as the tube-wise kernels do: the rotation is a gather of its
# own, `between`, and `final` after the last direction.
#
# The loops over the buckets are plain Python on purpose: they run once per bucket, a
# handful of times, and all their time is spent inside BLAS. Inside numba the product
# would need contiguous operands (a copy of M[:t, :t] per call) and scipy's dtrsm is not
# available at all. BLAS threads on its own, so there is nothing to gain from a parallel
# loop around it.


def _transform_blocks(
    Ms: Sequence[np.ndarray],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    padpos: np.ndarray,
    between: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    """O(m S n) per matrix

    `transform` for ``backend="blas"``.
    """
    products = []
    for M in Ms:
        products += [M] * m  # each matrix along every direction
    last_M = products.pop()  # the last product is followed by the final gather
    X, B = _buffers(x, padpos, len(between))
    bx, bb = _blocks(buckets, X), _blocks(buckets, B)
    for M in products:
        _multiply_blocks(M, bx, bb)
        _gather(B, between, X)  # X[:S] = B[between]: rotate to the next direction
    _multiply_blocks(last_M, bx, bb)
    dot = np.empty(len(x), dtype=np.float64)
    _gather(B, final, dot)  # dot = B[final]: the last rotation, straight into the output
    return dot


def _itransform_blocks(
    Ms: Sequence[np.ndarray],
    lower: Sequence[bool],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    padpos: np.ndarray,
    between: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    """O(m S n) per matrix

    `itransform` for ``backend="blas"``.
    """
    solves = []
    for M, low in zip(Ms, lower):
        solves += [(M, low)] * m  # each matrix along every direction
    last_M, last_low = solves.pop()  # the last solve is followed by the final gather
    X, B = _buffers(x, padpos, len(between))
    bx, bb = _blocks(buckets, X), _blocks(buckets, B)
    for M, low in solves:
        _solve_blocks(M, low, bx)
        _gather(X, between, B)  # B[:S] = X[between]: rotate to the next direction
        X, B = B, X
        bx, bb = bb, bx
    _solve_blocks(last_M, last_low, bx)
    dot = np.empty(len(x), dtype=np.float64)
    _gather(X, final, dot)  # dot = X[final]: the last rotation, straight into the output
    return dot


def _dtransform_blocks(
    D: np.ndarray,
    x: np.ndarray,
    perm: np.ndarray,
    buckets: np.ndarray,
    padpos: np.ndarray,
    between: np.ndarray,
) -> np.ndarray:
    """O(S n)

    `dtransform` for ``backend="blas"``.
    """
    S = len(between)
    X, B = _aligned_zeros(S + 1), _aligned_zeros(S + 1)
    _move(x, perm, padpos, X)  # X[padpos] = x[perm]: the direction to the tubes, into the layout
    _multiply_blocks(D, _blocks(buckets, X), _blocks(buckets, B))
    dot = np.empty(len(x), dtype=np.float64)
    _move(B, padpos, perm, dot)  # dot[perm] = B[padpos]: and back
    return dot


def _multiply_blocks(
    M: np.ndarray,
    bx: list,
    bb: list,
) -> None:
    """O(S n)

    Bb <- M[:t, :t] @ Xb for every pair of blocks: the product along the current
    direction. The zeros of the triangular M are multiplied like any entry.
    """
    for (t, Xb), (_, Bb) in zip(bx, bb):
        np.matmul(M[:t, :t], Xb, out=Bb)


def _solve_blocks(
    M: np.ndarray,
    lower: bool,
    bx: list,
) -> None:
    """O(S n)

    In place Xb <- inv(M[:t, :t]) @ Xb for every block: the solve along the current
    direction. dtrsm wants Fortran order, and the C-order Xb is the Fortran-order Xb.T, so
    it solves Y @ M.T = Xb.T from the right: no copy of Xb.
    """
    for t, Xb in bx:
        Mf = np.asfortranarray(M[:t, :t])
        _dtrsm(1.0, Mf, Xb.T, side=1, lower=lower, trans_a=1, diag=0, overwrite_b=1)


def _blocks(
    buckets: np.ndarray,
    X: np.ndarray,
) -> list:
    """O(len(buckets))

    Every bucket's stretch of the flat buffer X as a (length, width) matrix, paired with
    its length: views, no copies, built once per call.
    """
    return [(t, X[b : b + t * w].reshape(t, w)) for t, b, w in buckets]


def _buffers(
    x: np.ndarray,
    padpos: np.ndarray,
    S: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """O(S)

    Two work buffers with the zero slot S, the first holding x in the layout.
    """
    X, B = _aligned_zeros(S + 1), _aligned_zeros(S + 1)
    _scatter(x, padpos, X)  # X[padpos] = x
    return X, B


# gathers and helpers
#
# The gathers are numba loops rather than numpy's `take` or fancy indexing: numpy runs on
# one thread and fancy indexing allocates a temporary, 3 to 11 times slower.


@njit(parallel=PARALLEL)
def _gather(
    src: np.ndarray,
    idx: np.ndarray,
    dst: np.ndarray,
) -> None:
    """O(len(idx))

    numpy's dst[: len(idx)] = src[idx], but into an existing buffer and, from `SERIAL`
    entries on, in parallel.
    """
    if idx.shape[0] < SERIAL:
        for i in range(idx.shape[0]):
            dst[i] = src[idx[i]]
    else:
        for i in prange(idx.shape[0]):
            dst[i] = src[idx[i]]


@njit(parallel=PARALLEL)
def _scatter(
    src: np.ndarray,
    idx: np.ndarray,
    dst: np.ndarray,
) -> None:
    """O(len(idx))

    numpy's dst[idx] = src, from `SERIAL` entries on in parallel: idx has no repeats, so
    the parallel loop is safe.
    """
    if idx.shape[0] < SERIAL:
        for i in range(idx.shape[0]):
            dst[idx[i]] = src[i]
    else:
        for i in prange(idx.shape[0]):
            dst[idx[i]] = src[i]


@njit(parallel=PARALLEL)
def _move(
    src: np.ndarray,
    isrc: np.ndarray,
    idst: np.ndarray,
    dst: np.ndarray,
) -> None:
    """O(len(isrc))

    numpy's dst[idst] = src[isrc], from `SERIAL` entries on in parallel: idst has no
    repeats, so the parallel loop is safe.
    """
    if isrc.shape[0] < SERIAL:
        for i in range(isrc.shape[0]):
            dst[idst[i]] = src[isrc[i]]
    else:
        for i in prange(isrc.shape[0]):
            dst[idst[i]] = src[isrc[i]]


def _aligned_zeros(n: int) -> np.ndarray:
    """O(n)

    n zeros starting on a cache line (64 bytes).
    """
    buf = np.zeros(n + 8, dtype=np.float64)
    off = (-buf.ctypes.data % 64) // 8
    return buf[off : off + n]


def _vector(
    x: np.ndarray,
    N: int,
) -> np.ndarray:
    """O(N)

    x as a contiguous vector of N doubles; no copy if it is one already.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.shape != (N,):
        raise ValueError(f"x must have shape {(N,)}, but got {x.shape}.")
    return x
