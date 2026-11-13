"""
Measure the two constants of the bucket cost of the BLAS backend and how much they matter.

The BLAS backend (`lpfun.core.blas`) groups the tubes into buckets by minimising a model
of the time of its matrix products, ``CALL + L^2 (w + W0)`` per bucket in units of one
multiply-add. ``CALL`` and ``W0`` belong to the machine and its BLAS library. This script
has two commands:

``fit``
    Times the product ``M[:L, :L] @ X`` with ``X`` of shape (L, w) alone, over a grid of
    lengths L and widths w, and fits ``t(L, w) = c + b L^2 + a L^2 w`` by least squares
    on the relative error. The three terms are the latency of a call, one pass over the
    L x L operand, and the arithmetic, so ``CALL = c / a`` and ``W0 = b / a``.

``sweep``
    Times whole transforms (eval and interp) with the buckets chosen for a range of
    values of one constant, the other held at its default, and reports for every value
    the slowdown against the best value of each case. A flat row means the constant
    does not need tuning.

Usage::

    python scripts/tune_fnt_blas.py fit
    python scripts/tune_fnt_blas.py sweep
    python scripts/tune_fnt_blas.py sweep --constant call --values 0 1e4 5e4 2e5 1e6
    python scripts/tune_fnt_blas.py sweep --cases 2,1024,2 3,128,0.5 --csv sweep.csv

To use measured values, assign ``lpfun.core.blas.CALL`` and ``lpfun.core.blas.W0`` before
constructing a `Function`. Run on a quiet machine: the timings are minima over a few
repetitions, which removes most but not all of the disturbance from other processes.
"""

import argparse
import csv
import os
import platform
import time

import numpy as np

import lpfun
from lpfun.core import blas

FIT_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2000)
FIT_WIDTHS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)

SWEEP_W0 = (0, 8, 16, 32, 64, 128, 256)
SWEEP_CALL = (0, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5)
# (m, n, p): narrow buckets (m = 2, thin sets) and wide ones (m >= 3, full sets)
SWEEP_CASES = (
    (2, 256, 0.5), (2, 256, 2.0), (2, 1024, 0.5), (2, 1024, 1.0), (2, 1024, 2.0),
    (3, 64, 1.0), (3, 128, 0.5), (3, 128, 2.0), (4, 24, 1.0), (4, 48, 2.0), (5, 16, 2.0),
)  # fmt: skip


def header() -> None:
    print(f"machine : {platform.platform()}, {platform.machine()}, {os.cpu_count()} cores")
    print(f"software: python {platform.python_version()}, numpy {np.__version__}, lpfun defaults CALL = {blas.CALL:g}, W0 = {blas.W0:g}")
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        version = info.get("version")
        print(f"blas    : {info.get('name')}" + (f" {version}" if version and version != "unknown" else ""))
    except Exception:
        pass


def best_time(f, seconds: float, min_reps: int = 5, max_reps: int = 2000) -> float:
    """Least time of one call of f, repeating for about `seconds`."""
    f()
    best, spent, reps = np.inf, 0.0, 0
    while reps < min_reps or (spent < seconds and reps < max_reps):
        t = time.perf_counter()
        f()
        t = time.perf_counter() - t
        best, spent, reps = min(best, t), spent + t, reps + 1
    return best


def relative_fit(columns, t):
    """Coefficients x minimising sum_i (columns_i . x / t_i - 1)^2, and the relative errors."""
    A = np.column_stack(columns)
    x, *_ = np.linalg.lstsq(A / t[:, None], np.ones_like(t), rcond=None)
    return x, A @ x / t - 1.0


def fit(args) -> None:
    header()
    lengths = [L for L in FIT_LENGTHS if L <= args.max_length]
    rng = np.random.default_rng(0)
    M = np.ascontiguousarray(np.tril(rng.uniform(-1, 1, (max(lengths) + 1, max(lengths) + 1))))
    L, w, t = [], [], []
    print("\nmicroseconds per product M[:L, :L] @ X, X of shape (L, w)")
    print("      L \\ w " + " ".join(f"{x:>9}" for x in FIT_WIDTHS))
    for Li in lengths:
        row = []
        for wi in FIT_WIDTHS:
            # the same operands as the backend: a leading block of a larger matrix, aligned buffers
            X = blas._aligned_zeros(Li * wi).reshape(Li, wi)
            X[:] = rng.uniform(-1, 1, (Li, wi))
            B = blas._aligned_zeros(Li * wi).reshape(Li, wi)
            A = M[:Li, :Li]
            ti = best_time(lambda: np.matmul(A, X, out=B), args.seconds)
            L.append(Li), w.append(wi), t.append(ti), row.append(ti)
        print(f"{Li:>11} " + " ".join(f"{x * 1e6:>9.1f}" for x in row), flush=True)
    L, w, t = np.array(L, dtype=float), np.array(w, dtype=float), np.array(t)

    one = np.ones_like(t)
    (c, b, a), err = relative_fit([one, L**2, L**2 * w], t)
    (c2, a2), err2 = relative_fit([one, L**2 * w], t)
    rms = lambda e: float(np.sqrt(np.mean(e**2)))
    print(f"\nfit t = c + b L^2 + a L^2 w over {len(t)} products")
    print(f"  a = {a * 1e9:.4f} ns per multiply-add")
    print(f"  b = {b * 1e9:.4f} ns per entry of the operand")
    print(f"  c = {c * 1e6:.3f} us per call")
    print(f"  relative error: rms {rms(err):.2f}, largest {np.abs(err).max():.2f}")
    print(f"  without the term b L^2: rms {rms(err2):.2f}, largest {np.abs(err2).max():.2f}")
    if a <= 0:
        print("\nThe fitted arithmetic rate is not positive: the timings are too disturbed to use.")
        return
    print(f"\n  CALL = c / a = {c / a:,.0f}      (default {blas.CALL:,})")
    print(f"  W0   = b / a = {b / a:.1f}      (default {blas.W0})")
    if c < 0 or b < 0:
        print("  A negative value means the grid does not resolve that term; widen it or rerun.")
    print("\nper range of lengths (a single W0 is an approximation: the pass is cheaper while the")
    print("operand fits in cache):")
    for label, mask in (("L <= 128", L <= 128), ("L >= 256", L >= 256)):
        if mask.sum() >= 6:
            (ci, bi, ai), ei = relative_fit([one[mask], L[mask] ** 2, (L**2 * w)[mask]], t[mask])
            print(f"  {label}: CALL = {ci / ai:>10,.0f}, W0 = {bi / ai:>6.1f}, rms relative error {rms(ei):.2f}")


def parse_case(text: str):
    m, n, p = text.split(",")
    return int(m), int(n), float(p)


def sweep(args) -> None:
    header()
    values = args.values or (SWEEP_W0 if args.constant == "w0" else SWEEP_CALL)
    cases = [parse_case(c) for c in args.cases] if args.cases else list(SWEEP_CASES)
    print(f"\nsweep of {args.constant.upper()}, the other constant at its default; basis {args.basis}")
    print("times in ms, eval / interp; (buckets, padded size / |A|)\n")
    rows, ratio = [], {v: [] for v in values}
    for m, n, p in cases:
        fun = lpfun.Function(m, n, lp_degree=p, basis=args.basis, backend="blas", report=False, precompilation=False)
        N, x = len(fun), np.random.default_rng(0).uniform(-1, 1, len(fun))
        plans = [blas.blas_plan(fun._T, fun._cs_T, fun._pi, m, **{args.constant: v}) for v in values]
        runs = []
        for plan in plans:
            runs.append(lambda plan=plan: blas.transform_blas(fun._blas_fwd, x, **plan))
            runs.append(lambda plan=plan: blas.transform_blas(fun._blas_inv, x, **plan))
        # interleave the candidates, so that a disturbance hits all of them alike
        for f in runs:
            f()
        times = np.full(len(runs), np.inf)
        start, reps = time.perf_counter(), 0
        while reps < 5 or (time.perf_counter() - start < args.seconds and reps < 500):
            for k, f in enumerate(runs):
                t = time.perf_counter()
                f()
                times[k] = min(times[k], time.perf_counter() - t)
            reps += 1
        times = times.reshape(len(values), 2) * 1e3
        total = times.sum(axis=1)
        print(f"m = {m}, n = {n}, p = {p:g}, |A| = {N}")
        for v, plan, (te, ti), tot in zip(values, plans, times, total):
            ratio[v].append(tot / total.min())
            mark = "  <- best" if tot == total.min() else ""
            buckets, ratio_S = len(plan["buckets"]), len(plan["between"]) / N
            print(f"  {args.constant} = {v:>9g}: {te:>9.3f} / {ti:>9.3f}   ({buckets:>3}, {ratio_S:.2f})   x{tot / total.min():.2f}{mark}")
            rows.append(dict(constant=args.constant, value=v, m=m, n=n, p=p, size=N, buckets=buckets,
                             padded_ratio=ratio_S, eval_ms=te, interp_ms=ti, slowdown=tot / total.min()))  # fmt: skip
        print(flush=True)
    print(f"slowdown against the best {args.constant.upper()} of every case (eval + interp)")
    print(f"  {'value':>10} {'geometric mean':>15} {'worst case':>11}")
    default = blas.W0 if args.constant == "w0" else blas.CALL
    for v in values:
        r = np.array(ratio[v])
        note = "   (default)" if v == default else ""
        print(f"  {v:>10g} {np.exp(np.mean(np.log(r))):>15.2f} {r.max():>11.2f}{note}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="time the bare product and fit CALL and W0")
    f.add_argument("--max-length", type=int, default=max(FIT_LENGTHS), help="largest length L of the grid")
    f.add_argument("--seconds", type=float, default=0.15, help="time spent per product")
    f.set_defaults(run=fit)
    s = sub.add_parser("sweep", help="time whole transforms over a range of one constant")
    s.add_argument("--constant", choices=("w0", "call"), default="w0")
    s.add_argument("--values", type=float, nargs="+", help="values of the constant (default: a range around the default)")
    s.add_argument("--cases", nargs="+", metavar="M,N,P", help="cases as m,n,p (default: a set from m = 2 to 5)")
    s.add_argument("--basis", choices=("newton", "chebyshev", "legendre"), default="chebyshev")
    s.add_argument("--seconds", type=float, default=1.0, help="time spent per case")
    s.add_argument("--csv", help="write the timings to this file")
    s.set_defaults(run=sweep)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
