import sys
import time
import threading
import numpy as np
from typing import Literal
from lpfun import NP_FLOAT
from abc import ABC, abstractmethod
from lpfun.core.set import lp_set, lp_tube, ordinal_embedding
from lpfun.core.molecules import (
    transform,
    itransform,
    dtransform,
)
from lpfun.core.utils import apply_permutation
from lpfun.utils import (
    classify,
    cheb2nd,
    ###
    newton2lagrange,
    newton2derivative,
    newton2point,
    ###
    chebyshev2lagrange,
    chebyshev2derivative,
    chebyshev2point,
    ###
    # inv,
    is_lower_triangular,
    leja_nodes,
    gen_grid,
    lu,
    rmo,
)

import numpy as np
from abc import ABC, abstractmethod


class AbstractTransform(ABC):
    @property
    @abstractmethod
    def spatial_dimension(self) -> int:
        """Returns the spatial dimension"""
        pass

    @property
    @abstractmethod
    def polynomial_degree(self) -> int:
        """Returns the polynomial degree"""
        pass

    @property
    @abstractmethod
    def lp_degree(self) -> int:
        """Returns the lp degree"""
        pass

    @abstractmethod
    def tube(self) -> np.ndarray:
        """Returns the tube array"""
        pass

    @property
    @abstractmethod
    def multi_index_set(self) -> np.ndarray:
        """Returns the multi-index set array"""
        pass

    @property
    @abstractmethod
    def nodes(self) -> np.ndarray:
        """Returns the nodes array"""
        pass

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        """Returns the grid array"""
        pass


class Transform(AbstractTransform):
    """Transform class."""

    def __init__(
        self,
        spatial_dimension: int,  # ... m
        polynomial_degree: int,  # ... n
        lp_degree: float = 2.0,  # ... p
        nodes: callable = cheb2nd,  # ... x
        basis: Literal["newton", "chebyshev"] = "newton",
        precompilation: bool = True,
        lex_order: bool = True,
        threshold: int = 150_000_000,
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
            precompilation (bool): Precompile all the JIT functions with dummy inputs.
            threshold (int): The threshold for the dimension of the lower space. If the dimension is greater than the threshold, an error is raised.
            report (bool): Print a report after initialization.
        """

        self._start_spinner() if report else None
        construction_start = time.time()
        self._m = int(spatial_dimension)
        self._n = int(polynomial_degree)
        self._p = float(lp_degree)
        self._basis = str(basis)
        classify(self._m, self._n, self._p)

        if not basis in ["newton", "chebyshev"]:
            self._stop_spinner() if report else None
            raise ValueError("Invalid choice for basis.")

        # multi index set
        self._spinner_label = "Construct multi index set"
        self._A = lp_set(self._m, self._n, self._p)
        self._length = (self._n + 1) ** self._m if self._p is np.inf else len(self._A)

        # tube projection
        self._spinner_label = "Construct tube projections"
        self._T = lp_tube(self._A, self._m, self._n, self._p)

        # check threshold
        self._spinner_label = "Check threshold"
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

        # one dimensional (unisolvent, leja-ordered) nodes
        self._spinner_label = "Construct nodes"
        x = nodes(self._n + 1)
        if len(np.unique(x)) != self._n + 1:
            self._stop_spinner() if report else None
            raise ValueError("The provided nodes are not pairwise distinct.")
        x = leja_nodes(x)
        self._x = x

        # grid
        self._spinner_label = "Construct grid"
        grid = gen_grid(x, self._A, self._m, self._n, self._p)
        self._lex_order = np.lexsort(grid.T) if lex_order else None  # user experience
        self._grid = (
            apply_permutation(self._lex_order, grid, invert=False)
            if lex_order
            else grid
        )

        # compute matrices
        self._spinner_label = "Construct matrices"
        if basis == "newton":
            self._Vx = newton2lagrange(x)
            self._Dx = newton2derivative(x)
        elif basis == "chebyshev":
            self._Vx = chebyshev2lagrange(x)
            self._Dx = chebyshev2derivative(x)
        self._Dx2 = self._Dx @ self._Dx
        self._Dx3 = self._Dx @ self._Dx2

        # compute condition number
        self._cond_Vx = np.linalg.cond(self._Vx)

        # row major ordering V
        self._spinner_label = "Row major ordering V"
        if not is_lower_triangular(self._Vx):
            Vx_lt, Vx_ut = lu(self._Vx)
            self._Vx_lt, self._inv_Vx_lt = rmo(Vx_lt), rmo(np.linalg.inv(Vx_lt))
            self._Vx_ut, self._inv_Vx_ut = (
                rmo(Vx_ut[::-1, ::-1])[::-1],
                rmo(np.linalg.inv(Vx_ut[::-1, ::-1]))[::-1],
            )
        else:
            self._Vx_lt, self._inv_Vx_lt = rmo(self._Vx), rmo(np.linalg.inv(self._Vx))
            self._Vx_ut, self._inv_Vx_ut = None, None

        # row major ordering D
        self._spinner_label = "Row major ordering D"
        if not is_lower_triangular(self._Dx.T):
            Dx_lt, Dx_ut = lu(self._Dx)
            Dx2_lt, Dx2_ut = lu(self._Dx2)
            Dx3_lt, Dx3_ut = lu(self._Dx3)
            #
            self._Dx_lt = [rmo(Dx_lt), rmo(Dx2_lt), rmo(Dx3_lt)]
            self._Dx_ut = [
                rmo(Dx_ut[::-1, ::-1])[::-1],
                rmo(Dx2_ut[::-1, ::-1])[::-1],
                rmo(Dx3_ut[::-1, ::-1])[::-1],
            ]
            #
            self._DxT_lt = [rmo(Dx_ut.T), rmo(Dx2_ut.T), rmo(Dx3_ut.T)]
            self._DxT_ut = [
                rmo(Dx_lt.T[::-1, ::-1])[::-1],
                rmo(Dx2_lt.T[::-1, ::-1])[::-1],
                rmo(Dx3_lt.T[::-1, ::-1])[::-1],
            ]
        else:
            self._Dx_lt = [
                rmo(self._Dx[::-1, ::-1])[::-1],
                rmo(self._Dx2[::-1, ::-1])[::-1],
                rmo(self._Dx3[::-1, ::-1])[::-1],
            ]
            self._DxT_lt = [rmo(self._Dx.T), rmo(self._Dx2.T), rmo(self._Dx3.T)]
            #
            self._Dx_ut, self._DxT_ut = None, None

        construction_end = time.time()
        self._construction_ms = (construction_end - construction_start) * 1000

        # warmup JIT compiler
        self._spinner_label = "Precompile jit functions"
        precompilation_start = time.time()
        if precompilation:
            self.warmup()
        precompilation_end = time.time()
        self._precompilation_ms = (precompilation_end - precompilation_start) * 1000

        # stop spinner
        self._stop_spinner() if report else None

        # print report
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
    def tube(self) -> np.ndarray:
        return self._T

    @property
    def multi_index_set(self) -> np.ndarray:
        return self._A

    @property
    def nodes(self) -> np.ndarray:
        return self._x

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        zeros_N = np.zeros(len(self), dtype=NP_FLOAT)
        one_zero = np.zeros((1, self._m), dtype=NP_FLOAT)
        self._spinner_label = "Precompile fast Newton transform"
        self.fnt(zeros_N)
        self._spinner_label = "Precompile inverse fast Newton transform"
        self.ifnt(zeros_N)
        self._spinner_label = "Precompile derivative"
        self.dx(zeros_N, 0)
        self._spinner_label = "Precompile transposed derivative"
        self.dxT(zeros_N, 0)
        self._spinner_label = "Precompile point evaluation"
        self.eval(zeros_N, one_zero)

    def fnt(self, function_values: np.ndarray) -> np.ndarray:
        """Fast Newton Transform"""
        function_values = np.asarray(function_values).astype(NP_FLOAT)
        if self._lex_order is not None:
            function_values = apply_permutation(
                self._lex_order, function_values, invert=True
            )
        ###
        if self._inv_Vx_lt is not None and self._inv_Vx_ut is None:
            return transform(
                self._Vx_lt,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
        elif self._inv_Vx_ut is not None and self._inv_Vx_ut is not None:
            coefficients = transform(
                self._Vx_lt,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
            return transform(
                self._Vx_ut,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="upper",
            )
        else:
            raise ValueError("Unexpected error: _Vx_lt must exist, _Vx_ut is optional.")
        ###

    def ifnt(self, coefficients: np.ndarray) -> np.ndarray:
        """Inverse Fast Newton Transform"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        ###
        function_values = np.zeros(len(self), dtype=NP_FLOAT)
        if self._Vx_lt is not None and self._Vx_ut is None:
            function_values = itransform(
                self._Vx_lt,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
        elif self._Vx_lt is not None and self._Vx_ut is not None:
            function_values = itransform(
                self._Vx_ut,
                coefficients,
                self._T,
                self._m,
                self._p,
                mode="upper",
            )
            function_values = itransform(
                self._Vx_lt,
                function_values,
                self._T,
                self._m,
                self._p,
                mode="lower",
            )
        else:
            raise ValueError("Unexpected error: _Vx_lt must exist, _Vx_ut is optional.")
        ###
        if self._lex_order is not None:
            function_values = apply_permutation(
                self._lex_order, function_values, invert=False
            )
        return function_values

    def dx(
        self, coefficients: np.ndarray, i: int, k: Literal[1, 2, 3] = 1
    ) -> np.ndarray:
        """Fast Differentiation"""
        coefficients, i, k = (
            np.asarray(coefficients).astype(NP_FLOAT),
            int(i),
            int(k),
        )
        if (i < 0) or (i > self._m):
            raise ValueError(
                f"Invalid value for i. Please choose i in between 1 and {self._m}."
            )
        if not k in [1, 2, 3]:
            raise ValueError("Invalid value for k. Please choose 1, 2 or 3.")
        ###
        if self._Dx_lt is not None and self._Dx_ut is None:
            return dtransform(
                self._Dx_lt[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
            )
        elif self._Dx_lt is not None and self._Dx_ut is not None:
            coefficients = dtransform(
                self._Dx_ut[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
            )
            return dtransform(
                self._Dx_lt[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
            )
        else:
            raise ValueError("Unexpected error: _Dx_lt must exist, _Dx_ut is optional.")
        ###

    def dxT(
        self, coefficients: np.ndarray, i: int, k: Literal[1, 2, 3] = 1
    ) -> np.ndarray:
        """Fast Differentiation (Transpose)"""
        coefficients, i, k = (
            np.asarray(coefficients).astype(NP_FLOAT),
            int(i),
            int(k),
        )
        if (i < 0) or (i > self._m):
            raise ValueError(
                f"Invalid value for i. Please choose i in between 1 and {self._m}."
            )
        if not k in [1, 2, 3]:
            raise ValueError("Invalid value for k. Please choose 1, 2 or 3.")
        ###
        if self._DxT_lt is not None and self._DxT_ut is None:
            return dtransform(
                self._DxT_lt[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
            )
        elif self._DxT_lt is not None and self._DxT_ut is not None:
            coefficients = dtransform(
                self._DxT_ut[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="upper",
            )
            return dtransform(
                self._DxT_lt[k - 1],
                coefficients,
                self._T,
                self._m,
                self._p,
                i,
                mode="lower",
            )
        else:
            raise ValueError(
                "Unexpected error: _DxT_lt must exist, _DxT_ut is optional."
            )
        ###

    def eval(self, coefficients: np.ndarray, points: np.ndarray) -> NP_FLOAT:
        """Point Evaluation"""
        coefficients, points = (
            np.asarray(coefficients).astype(NP_FLOAT),
            np.asarray(points).astype(NP_FLOAT),
        )

        if self._basis == "newton":
            return newton2point(
                coefficients, self._x, points, self._A, self._m, self._n
            )
        elif self._basis == "chebyshev":
            return chebyshev2point(coefficients, points, self._A, self._m, self._n)

    def embed(self, t: AbstractTransform) -> np.ndarray:
        return ordinal_embedding(self._m, t.tube, self._T)

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
            f"{'Condition V':<20} | {self._cond_Vx:.2e}\n"
            f"{'Amount of Coeffs':<20} | {len(self):_}\n"
            f"{'Construction':<20} | {self._construction_ms:_.2f} ms\n"
            f"{'Precompilation':<20} | {self._precompilation_ms:_.2f} ms\n"
            f"{'-'*20}-+-{'-'*20}\n"
        )
