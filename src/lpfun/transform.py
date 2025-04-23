import sys
import time
import threading
import numpy as np
from typing import Literal
from lpfun import NP_FLOAT
from lpfun.core.molecules import (
    transform,
    itransform,
    dtransform,
)
from lpfun.utils import (
    cheb2nd,
    newton2lagrange,
    newton2derivative,
    newton2point,
    chebyshev2lagrange,
    chebyshev2derivative,
    # chebyshev2point, # TODO
    apply_permutation,
    inv,
    is_lower_triangular,
    leja_nodes,
    grid,
    lu,
    rmo,
    tube,
)


class Transform:
    """Transform class."""

    def __init__(
        self,
        spatial_dimension: int,  # ... m
        polynomial_degree: int,  # ... n
        lp_degree: float = 2.0,  # ... p
        nodes: callable = cheb2nd,  # ... x
        basis: Literal["newton", "chebyshev"] = "newton",
        parallel=True,
        precompilation: bool = True,
        threshold: int = 20_000_000,
        report: bool = True,
    ):
        """
        Initialize the Transform object.

        Args:
            spatial_dimension (int): The dimension of the spatial domain.
            polynomial_degree (int): The degree of the polynomial.
            lp_degree (float): The p-norm of the polynomial space.
            nodes (callable): A callable function that takes an integer and returns a one dimensional numpy array of nodes.
            basis (str): The basis to use for the Vandermonde and differentiation matrices. Either "newton" or "chebyshev".
            parallel (bool): Decide whether to use parallel execution on the CPU. Utilizes precomputations for the inverse of the Vandermonde matrix if activated.
            precompilation (bool): Precompile all the JIT functions with dummy inputs.
            threshold (int): The threshold for the dimension of the lower space. If the dimension is greater than the threshold, an error is raised.
            report (bool): Print a report after initialization.
        """

        self._start_spinner() if report else None
        construction_start = time.time()
        self._m = int(spatial_dimension)
        self._n = int(polynomial_degree)
        self._p = float(lp_degree)
        self._parallel = bool(parallel)

        if basis == "newton":
            vandermonde_matrix = newton2lagrange
            differentiation_matrix = newton2derivative
            self._eval_at_point = newton2point
        elif basis == "chebyshev":
            vandermonde_matrix = chebyshev2lagrange
            differentiation_matrix = chebyshev2derivative
            self._eval_at_point = None
        else:
            self._stop_spinner() if report else None
            raise ValueError("Invalid choice for basis.")

        # Compute tube only if p is not np.inf
        self._spinner_label = "Compute Tubes"
        self._T = tube(self._m, self._n, self._p)
        self._length = np.sum(self._T)

        # Check the threshold
        if threshold is None:
            warnings.warn("Threshold is set to None. This may lead to memory issues.")
        elif self._length > threshold:
            self._stop_spinner() if report else None
            raise ValueError(
                f"""
                    Dimension exceeds threshold: {format(self._length, "_")} > {format(threshold, "_")}.
                    If this operation should be executed anyways, please set threshold to None.
                """
            )

        # TODO REMOVE...
        # if not parallel and self._m > 3:
        #     self._stop_spinner() if report else None
        #     raise ValueError(
        #         "Precomputation must be enabled when the spatial dimension is greater than three."
        #     )

        # One dimensional (unisolvent, leja-ordered) nodes
        self._spinner_label = "Compute Nodes"
        x = nodes(self._n + 1)
        if len(np.unique(x)) != self._n + 1:
            self._stop_spinner() if report else None
            raise ValueError("The provided nodes are not pairwise distinct.")
        x = leja_nodes(x)
        self._x = x

        # Grid
        G = grid(x, self._m, self._n, self._p)

        # Lex order: User experience
        self._lex_order = np.lexsort(G.T)
        self._G = apply_permutation(self._lex_order, G, invert=False)

        # Compute matrices
        self._spinner_label = "Compute Matrices"
        V = vandermonde_matrix(x)
        D = differentiation_matrix(x)

        # Compute condition number
        self._cond_V = np.linalg.cond(V)

        # Row major ordering V
        self._spinner_label = "Row Major Ordering V"
        if not is_lower_triangular(V):
            self._V, self._invV = (None, None)
            V_L, V_U = lu(V)
            self._V_L, self._V_U = rmo(V_L), rmo(V_U[::-1, ::-1])[::-1]
            invV_L, invV_U = inv(V_L), inv(V_U[::-1, ::-1])[::-1, ::-1]
            self._invV_L, self._invV_U = (
                (rmo(invV_L), rmo(invV_U[::-1, ::-1])[::-1])
                if self._parallel
                else (None, None)
            )
        else:
            self._V, self._invV = rmo(V), rmo(inv(V)) if self._parallel else None
            self._V_L, self._V_U, self._invV_L, self._invV_U = (
                None,
                None,
                None,
                None,
            )

        # Row major ordering D
        self._spinner_label = "Row Major Ordering D"
        if not is_lower_triangular(D.T):
            self._D, self._DT = (None, None)
            D_L, D_U = lu(D)
            self._D_L, self._D_U = (rmo(D_L), rmo(D_U[::-1, ::-1])[::-1])
            self._DT_L, self._DT_U = (rmo(D_U.T), rmo(D_L.T[::-1, ::-1])[::-1])
        else:
            self._D, self._DT = rmo(D[::-1, ::-1])[::-1], rmo(D.T)
            self._D_L, self._D_U, self._DT_L, self._DT_U = (None, None, None, None)

        construction_end = time.time()
        self._construction_ms = (construction_end - construction_start) * 1000

        # Warmup the JIT compiler
        self._spinner_label = "Precompilation"
        precompilation_start = time.time()
        if precompilation:
            self.warmup()
        precompilation_end = time.time()
        self._precompilation_ms = (precompilation_end - precompilation_start) * 1000

        self._stop_spinner() if report else None

        # Print report
        print()
        print(self) if report else None

    def _start_spinner(self):
        self._loading = True
        self._spinner_label = "Intializations"
        self._spinner_thread = threading.Thread(target=self._show_spinner)
        self._spinner_thread.start()

    def _show_spinner(self):
        symbols = ["|", "/", "-", "\\"]
        idx = 0
        while self._loading:
            sys.stdout.write(
                f"\r>>>{' '*10}{self._spinner_label} ({symbols[idx]}){' '*10}<<<{' '*20}"
            )
            sys.stdout.flush()
            idx = (idx + 1) % len(symbols)
            time.sleep(0.1)

    def _stop_spinner(self):
        self._loading = False
        self._spinner_thread.join()
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    @property
    def spatial_dimension(self) -> int:
        return self._m

    @property
    def polynomial_degree(self) -> int:
        return self._n

    @property
    def lp_degree(self) -> int:
        return self._p

    @property
    def nodes(self) -> np.ndarray:
        return self._x

    @property
    def grid(self) -> np.ndarray:
        return self._G

    @property
    def tube(self) -> np.ndarray:
        return self._T if len(self._T) > 0 else None

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        zeros_N = np.zeros(len(self), dtype=NP_FLOAT)
        zeros_m = np.zeros(self._m, dtype=NP_FLOAT)
        self.fnt(zeros_N)
        self.ifnt(zeros_N)
        self.dx(zeros_N, 0)
        self.dxT(zeros_N, 0)
        self.eval(zeros_N, zeros_m)

    def fnt(self, function_values: np.ndarray) -> np.ndarray:
        """Fast Newton Transform"""
        function_values = np.asarray(function_values).astype(NP_FLOAT)
        function_values = apply_permutation(
            self._lex_order, function_values, invert=True
        )
        ###
        coefficients = np.zeros(len(self), dtype=NP_FLOAT)
        if self._invV is not None:
            coefficients = itransform(
                self._invV,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
                parallel=self._parallel,
            )
        elif self._V is not None:
            coefficients = transform(
                self._V,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
        elif self._invV_L is not None and self._invV_U is not None:
            coefficients = itransform(
                self._invV_L,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
                parallel=self._parallel,
            )
            coefficients = itransform(
                self._invV_U,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="upper",
                parallel=self._parallel,
            )
        elif self._V_L is not None and self._V_U is not None:
            coefficients = transform(
                self._V_L,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
            coefficients = transform(
                self._V_U,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="upper",
            )
        else:
            raise ValueError("Unexpected error.")
        ###
        return coefficients

    def ifnt(self, coefficients: np.ndarray) -> np.ndarray:
        """Inverse Fast Newton Transform"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        ###
        function_values = np.zeros(len(self), dtype=NP_FLOAT)
        if self._V is not None:
            function_values = itransform(
                self._V,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="lower",
                parallel=self._parallel,
            )
        elif self._V_L is not None and self._V_U is not None:
            function_values = itransform(
                self._V_U,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="upper",
                parallel=self._parallel,
            )
            function_values = itransform(
                self._V_L,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
                parallel=self._parallel,
            )
        else:
            raise ValueError("Unexpected error.")
        ###
        function_values = apply_permutation(
            self._lex_order, function_values, invert=False
        )
        return function_values

    def dx(self, coefficients: np.ndarray, i: int) -> np.ndarray:
        """Fast Differentiation"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        ###
        if self._D is not None:
            coefficients = dtransform(
                self._D,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
                parallel=self._parallel,
            )
        elif self._D_L is not None and self._D_U is not None:
            coefficients = dtransform(
                self._D_U,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
                parallel=self._parallel,
            )
            coefficients = dtransform(
                self._D_L,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
                parallel=self._parallel,
            )
        else:
            raise ValueError("Unexpected error.")
        ###
        return coefficients

    def dxT(self, coefficients: np.ndarray, i: int) -> np.ndarray:
        """Fast Differentiation (Transpose)"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        ###
        if self._DT is not None:
            coefficients = dtransform(
                self._DT,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
                parallel=self._parallel,
            )
        elif self._DT_L is not None and self._DT_U is not None:
            coefficients = dtransform(
                self._DT_U,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
                parallel=self._parallel,
            )
            coefficients = dtransform(
                self._DT_L,
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
                parallel=self._parallel,
            )
        else:
            raise ValueError("Unexpected error.")
        ###
        return coefficients

    def eval(self, coefficients: np.ndarray, x: np.ndarray) -> NP_FLOAT:
        """Point Evaluation"""
        if self._eval_at_point is None:  # TODO Remove this
            return 0.0
        return self._eval_at_point(coefficients, self._x, x, self._m, self._p)

    def __len__(self) -> int:
        return self._length

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transform):
            return False
        if not value.dimension == self.dimension:
            return False
        if not value.polynomial_degree == self.polynomial_degree:
            return False
        if not value.p == self.p:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"{'-'*20}-+-{'-'*20}\n"
            f"{' '*19}Report{' '*18}\n"
            f"{'-'*20}-+-{'-'*20}\n"
            f"{'Spatial Dimension':<20} | {self._m}\n"
            f"{'Polynomial Degree':<20} | {self._n:_}\n"
            f"{'lp Degree':<20} | {self._p}\n"
            f"{'Condition V':<20} | {self._cond_V:.2e}\n"
            f"{'Amount of Coeffs':<20} | {len(self):_}\n"
            f"{'Construction':<20} | {self._construction_ms:_.2f} ms\n"
            f"{'Precompilation':<20} | {self._precompilation_ms:_.2f} ms\n"
            f"{'-'*20}-+-{'-'*20}\n"
        )
