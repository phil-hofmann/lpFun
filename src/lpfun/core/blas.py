"""
BLAS backend for the LU transforms (``backend="blas"``).

GEMM is the BLAS routine for the GEneral Matrix-Matrix product, C = A @ B with dense A and
B. TRSM is the BLAS routine for the TRiangular Solve with a Matrix of right-hand sides.

The LU path applies a triangular (n+1) x (n+1) matrix M, a factor of the 1D Vandermonde
matrix or of its inverse, to every tube of the index set along every direction: a tube of
length t is multiplied by the leading block M[:t, :t]. Here the tubes are grouped into
buckets of similar length, padded with zeros to the longest length L of their bucket and
applied bucket by bucket, as dense matrix products (BLAS GEMM) or triangular solves (BLAS
TRSM) with M[:L, :L].

The padding is exact: a lower triangular factor never reads the pads, and an upper
triangular one reads them as the zeros they hold. A bucket stores its tubes as the columns
of one contiguous block. After each direction a gather brings the next direction to the
columns and refills every pad from a zero slot.

The buckets are chosen by `optimal_bucket_lengths`, which minimises a model of the time
of the products; the solves use the same buckets. Its two constants `CALL` and `W0`
belong to the machine and its BLAS library; the results do not depend on them. To tune
them, assign the module constants before constructing a `Function`, or pass ``call`` and
``w0`` to `blas_plan`.
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


ALIGN = 8  # bucket widths are multiples of this many columns: every block starts on a cache line
CALL = 50_000  # latency of one BLAS call, in multiply-adds
W0 = 32  # one pass over the matrix operand of a product, in columns of arithmetic
SERIAL = 65_536  # a gather of fewer entries is faster on one thread than on the thread pool


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


def blas_plan(
    T: np.ndarray,
    cs_T: np.ndarray,
    pi: np.ndarray,
    m: int,
    call: Optional[float] = None,
    w0: Optional[float] = None,
) -> dict:
    """O(|A|)

    Index maps of the padded layout, from the tube lengths `T`, their running totals
    `cs_T` and the transposition `pi` between two directions; `call` and `w0` go to
    `optimal_bucket_lengths`.

    Returns a dict, the keyword arguments of `transform_blas` and `itransform_blas`. Its
    keys, with S the padded size:

    m       : number of directions, as given.
    buckets : rows (length, base, width); the tubes of a bucket are the columns of the
              block (length, width) at `base`.
    padpos  : slot of every entry, base + row * width + column: row is the position of the
              entry in its tube, column the rank of the tube in its bucket.
    between : gather between two directions, of length S: the transposition `pi` composed
              with `padpos`; every pad reads the zero slot S.
    final   : gather after the last direction.
    """
    T = np.asarray(T, dtype=np.int64)
    cs_T = np.asarray(cs_T, dtype=np.int64)
    N = int(cs_T[-1])
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
    ipi[np.asarray(pi, dtype=np.int64)] = np.arange(N, dtype=np.int64)
    final = padpos[ipi]  # into the plain vector
    between = np.full(S, S, dtype=np.int64)  # into the padded layout: the pads read slot S
    between[padpos] = final
    ###
    dtype = np.int32 if S + 1 < 2**31 - 1 else np.int64
    return {
        "m": int(m),
        "buckets": np.column_stack((length, base, width)).astype(np.int64),
        "padpos": padpos.astype(dtype),
        "between": between.astype(dtype),
        "final": final.astype(dtype),
    }


def transform_blas(
    Ms: Sequence[np.ndarray],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    padpos: np.ndarray,
    between: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    """O(m S n) per matrix, S = padded size

    Products with the triangular matrices `Ms` (dense, C order) one after another, each
    along every direction.
    """
    products = []
    for M in Ms:
        products += [M] * m  # each matrix along every direction
    last = products.pop()  # the last product is followed by the final gather
    X, B = _buffers(x, padpos, len(between))
    bx, bb = _blocks(buckets, X), _blocks(buckets, B)
    for M in products:
        _multiply_blocks(M, bx, bb)
        _gather(B, between, X)  # rotate to the next direction
    _multiply_blocks(last, bx, bb)
    dot = np.empty(len(padpos), dtype=np.float64)
    _gather(B, final, dot)  # the last rotation, straight into the output
    return dot


def itransform_blas(
    Ms: Sequence[np.ndarray],
    x: np.ndarray,
    m: int,
    buckets: np.ndarray,
    padpos: np.ndarray,
    between: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    """O(m S n) per matrix, S = padded size

    Solves with the triangular matrices `Ms` (dense, Fortran order) one after another,
    each along every direction. The first is lower triangular; a second one is upper.
    """
    solves = []
    for k, M in enumerate(Ms):
        solves += [(M, k == 0)] * m  # (matrix, lower) along every direction
    last = solves.pop()  # the last solve is followed by the final gather
    X, B = _buffers(x, padpos, len(between))
    bx, bb = _blocks(buckets, X), _blocks(buckets, B)
    for M, lower in solves:
        _solve_blocks(M, lower, bx)
        _gather(X, between, B)  # rotate to the next direction
        X, B = B, X
        bx, bb = bb, bx
    _solve_blocks(*last, bx)
    dot = np.empty(len(padpos), dtype=np.float64)
    _gather(X, final, dot)  # the last rotation, straight into the output
    return dot


# kernels


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


def _aligned_zeros(n: int) -> np.ndarray:
    """O(n)

    n zeros starting on a cache line (64 bytes).
    """
    buf = np.zeros(n + 8, dtype=np.float64)
    off = (-buf.ctypes.data % 64) // 8
    return buf[off : off + n]


def _buffers(
    x: np.ndarray,
    padpos: np.ndarray,
    S: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """O(S)

    Two work buffers with the zero slot S, the first holding x in the padded layout.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.shape != padpos.shape:
        raise ValueError(f"x must have shape {padpos.shape}, but got {x.shape}.")
    X, B = _aligned_zeros(S + 1), _aligned_zeros(S + 1)
    _scatter(x, padpos, X)
    return X, B


def _blocks(
    buckets: np.ndarray,
    X: np.ndarray,
) -> list:
    """O(len(buckets))

    Every bucket's stretch of the flat buffer X as a (length, width) matrix, paired with
    its length: views, no copies.
    """
    return [(t, X[b : b + t * w].reshape(t, w)) for t, b, w in buckets]


def _multiply_blocks(
    M: np.ndarray,
    bx: list,
    bb: list,
) -> None:
    """O(S n)

    Bb <- M @ Xb for every pair of blocks: the product along the current direction.
    """
    for (t, Xb), (_, Bb) in zip(bx, bb):
        np.matmul(M[:t, :t], Xb, out=Bb)


def _solve_blocks(
    M: np.ndarray,
    lower: bool,
    bx: list,
) -> None:
    """O(S n)

    In place Xb <- inv(M) @ Xb for every block: the solve along the current direction.
    dtrsm wants Fortran order, and the C-order Xb is the Fortran-order Xb.T, so it solves
    Y @ M.T = Xb.T from the right: no copy of Xb.
    """
    for t, Xb in bx:
        Mf = np.asfortranarray(M[:t, :t])
        _dtrsm(1.0, Mf, Xb.T, side=1, lower=lower, trans_a=1, diag=0, overwrite_b=1)
