import sys
import time
import threading
import numpy as np
from typing import Literal, Callable
from abc import ABC, abstractmethod
from lpfun.core.set import lp_set, lp_tube, ordinal_embedding, entropy
from lpfun.core.molecules import (
    transform,
    itransform,
    dtransform,
)
from lpfun.core.utils import apply_permutation
from lpfun.utils import (
    classify,
    ###
    cheb2nd_nodes,
    leja_nodes,  # NOTE alternative for adaptivity
    ###
    newton2lagrange,
    newton2derivative,
    newton2point,
    ###
    chebyshev2lagrange,
    chebyshev2derivative,
    chebyshev2point,
    ###
    is_lower_triangular,
    get_leja_order,
    get_grid,
    get_lu,
    get_rmo,
    # get_lu_pivot, # TODO
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

    @property
    @abstractmethod
    def colex_order(self) -> np.ndarray:
        """Returns the co-lexicographic order array"""
        pass

    @property
    @abstractmethod
    def leja_order(self) -> np.ndarray:
        """Returns the leja order array"""
        pass


class Transform(AbstractTransform):
    """
    Transform class for polynomial multivariate interpolation and differentiation.

    Attributes
    ----------
    spatial_dimension : int
        The dimension `m` of the spatial domain, representing the number of input variables.
    polynomial_degree : int
        The maximum total degree `n` of the polynomial basis used for approximation.
    lp_degree : float
        The degree `p` of the ℓ^p norm that defines the polynomial space (default is 2.0, Euclidean degree).
    tube : numpy.ndarray
        Array representing directional polynomial degree constraints ("tube").
    multi_index_set : numpy.ndarray
        The multi-index set defining the exponents.
    nodes : np.ndarray
        The one-dimensional interpolation nodes.
    grid : numpy.ndarray
        The multidimensional grid points (shape: [num_points, spatial_dimension]).
    colex_order: TODO
    leja_order: TODO
    """

    def __init__(
        self,
        spatial_dimension: int,
        polynomial_degree: int,
        lp_degree: float = 2.0,
        nodes: Callable[[int], np.ndarray] = cheb2nd_nodes,
        basis: Literal["newton", "chebyshev"] = "newton",
        precomputation: bool = True,
        precompilation: bool = True,
        colex_order: bool = True,
        threshold: int = 150_000_000,
        report: bool = True,
    ):
        """
        Initialize the Transform object, which constructs and manages the polynomial transform
        used for multivariate function interpolation and differentiation on non-tensorial grids.

        Parameters
        ----------
        spatial_dimension : int
            The dimension `m` of the spatial domain, representing the number of input variables.
        polynomial_degree : int
            The maximum total degree `n` of the polynomial basis used for approximation.
        lp_degree : float, optional
            The degree `p` of the ℓ^p norm that defines the polynomial space (default is 2.0, Euclidean degree).
        nodes : callable, optional
            A callable that, given an integer `n`, returns an array of `n` nodes in one dimension.
            Typically, this is a function returning Chebyshev nodes `cheb2nd_nodes` or Leja nodes `leja_nodes`.
        basis : {"newton", "chebyshev"}, optional
            The polynomial basis to use for constructing Vandermonde and differentiation matrices.
            Options are:
            - "newton": Newton basis polynomials
            - "chebyshev": Chebyshev basis polynomials
            Default is "newton".
        precomputation : bool, optional
            If True, precompute the inverse Vandermonde matrix to speed up transforms.
            Precomputation can be less stable for large problems (default is True).
        precompilation : bool, optional
            If True, precompile all just-in-time (JIT) compiled functions with dummy inputs
            during initialization to reduce runtime overhead during actual calls (default is True).
        colex_order : bool, optional
            If True, reassigns co-lexicographic ordering for nodes.
            otherwise, returns
        threshold : int, optional
            A safety threshold for the dimension of the polynomial space.
            If the dimension exceeds this number, initialization will raise an error to prevent
            excessive memory usage or computational cost (default is 150,000,000).
        report : bool, optional
            If True, print detailed initialization information and statistics after setup
            (default is True).

        Raises
        ------
        ValueError
            If the polynomial space dimension exceeds the specified `threshold`.

        Notes
        -----
        The Transform class is designed for efficient multivariate polynomial interpolation,
        evaluation, and differentiation. The choice of basis and nodes directly affects
        numerical stability and accuracy. Precomputation and precompilation optimize performance
        at the cost of initial setup time and memory usage.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        >>> coeffs_dx_f = t.dx(coeffs_f, i=0, k=1)
        >>> rec_dx_f = t.ifnt(coeffs_dx_f)
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
        self._N_0 = (self._n + 1) ** self._m if self._p is np.inf else len(self._A)

        # tube projection
        self._spinner_label = "Construct tube projections"
        self._T = lp_tube(self._A, self._m, self._n, self._p)
        zero = np.array([0], dtype=np.int64)
        self._cs_T = np.concatenate((zero, np.cumsum(self._T)))
        self._N_1 = len(self._T)
        self._V_2 = (
            np.array(
                [
                    np.sum(self._T[self._cs_T[i] : self._cs_T[i + 1]])
                    for i in range(self._T[0])
                ],
                dtype=np.int64,
            )
            if self._m == 3
            else zero
        )
        self._cs_V_2 = (
            np.concatenate((zero, np.cumsum(self._V_2))) if self._m == 3 else zero
        )
        self._e_T = entropy(self._T) if self._m > 3 else zero

        # check threshold
        self._spinner_label = "Check threshold"
        if threshold is None:
            warnings.warn("Threshold is set to None. This may lead to memory issues.")
        elif self._N_0 > threshold:
            self._stop_spinner() if report else None
            raise ValueError(
                f"""
                    Dimension exceeds threshold: {format(self._N_0, "_")} > {format(threshold, "_")}.
                    If this operation should be executed anyways, please set threshold to None.
                """
            )

        # one dimensional (unisolvent, leja-ordered) nodes
        self._spinner_label = "Construct nodes"
        x = nodes(self._n + 1)
        if len(np.unique(x)) != self._n + 1:
            self._stop_spinner() if report else None
            raise ValueError("The provided nodes are not pairwise distinct.")
        self._leja_order = get_leja_order(x)
        self._x = x[self._leja_order]

        # grid
        self._spinner_label = "Construct grid"
        grid = get_grid(self._x, self._A, self._m, self._n, self._p)
        self._colex_order = (
            np.lexsort(grid.T) if colex_order else None
        )  # user experience
        self._grid = (
            apply_permutation(self._colex_order, grid, invert=False)
            if colex_order
            else grid
        )

        # compute matrices
        self._spinner_label = "Construct matrices"
        if basis == "newton":
            self._Vx = newton2lagrange(self._x)
            self._Dx = newton2derivative(self._x)
        elif basis == "chebyshev":
            self._Vx = chebyshev2lagrange(self._x)
            self._Dx = chebyshev2derivative(self._x)
        self._Dx2 = self._Dx @ self._Dx
        self._Dx3 = self._Dx @ self._Dx2

        # compute condition number
        self._cond_Vx = np.linalg.cond(self._Vx)

        # row major ordering V
        self._spinner_label = "Row major ordering V"
        lt = is_lower_triangular(self._Vx)
        if lt:
            self._Vx_lt, self._Vx_ut = get_rmo(self._Vx), None
            self._inv_Vx_lt, self._inv_Vx_ut = (
                (get_rmo(np.linalg.inv(self._Vx)), None)
                if precomputation
                else (None, None)
            )
        else:
            Vx_lt, Vx_ut = get_lu(self._Vx)
            self._Vx_lt, self._Vx_ut = get_rmo(Vx_lt), get_rmo(Vx_ut[::-1, ::-1])[::-1]
            self._inv_Vx_lt, self._inv_Vx_ut = (
                (
                    get_rmo(np.linalg.inv(Vx_lt)),
                    get_rmo(np.linalg.inv(Vx_ut)[::-1, ::-1])[::-1],
                )
                if precomputation
                else (None, None)
            )

        # row major ordering D
        self._spinner_label = "Row major ordering D"
        if not is_lower_triangular(self._Dx.T):
            Dx_lt, Dx_ut = get_lu(self._Dx)
            Dx2_lt, Dx2_ut = get_lu(self._Dx2)
            Dx3_lt, Dx3_ut = get_lu(self._Dx3)
            #
            self._Dx_lt = [get_rmo(Dx_lt), get_rmo(Dx2_lt), get_rmo(Dx3_lt)]
            self._Dx_ut = [
                get_rmo(Dx_ut[::-1, ::-1])[::-1],
                get_rmo(Dx2_ut[::-1, ::-1])[::-1],
                get_rmo(Dx3_ut[::-1, ::-1])[::-1],
            ]
            #
            self._DxT_lt = [get_rmo(Dx_ut.T), get_rmo(Dx2_ut.T), get_rmo(Dx3_ut.T)]
            self._DxT_ut = [
                get_rmo(Dx_lt.T[::-1, ::-1])[::-1],
                get_rmo(Dx2_lt.T[::-1, ::-1])[::-1],
                get_rmo(Dx3_lt.T[::-1, ::-1])[::-1],
            ]
        else:
            self._Dx_lt = [
                get_rmo(self._Dx[::-1, ::-1])[::-1],
                get_rmo(self._Dx2[::-1, ::-1])[::-1],
                get_rmo(self._Dx3[::-1, ::-1])[::-1],
            ]
            self._DxT_lt = [
                get_rmo(self._Dx.T),
                get_rmo(self._Dx2.T),
                get_rmo(self._Dx3.T),
            ]
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

    @property
    def colex_order(self) -> np.ndarray:
        return self._colex_order

    @property
    def leja_order(self) -> np.ndarray:
        return self._leja_order

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        zeros_N = np.zeros(len(self), dtype=np.float64)
        one_zero = np.zeros((1, self._m), dtype=np.float64)
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
        """
        Compute the Fast Newton Transform (FNT) of given function values.

        Parameters
        ----------
        function_values : np.ndarray
            Function values sampled at the interpolation grid points.
            Shape must match the number of grid points.

        Returns
        -------
        np.ndarray
            Coefficients of the function in the predefined polynomial basis.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        """
        function_values = np.asarray(function_values).astype(np.float64)
        if self._colex_order is not None:
            function_values = apply_permutation(
                self._colex_order, function_values, invert=True
            )
        ###
        if self._inv_Vx_lt is not None and self._inv_Vx_ut is None:
            return itransform(
                self._inv_Vx_lt,
                function_values,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
        elif self._Vx_lt is not None and self._Vx_ut is None:
            return transform(
                self._Vx_lt,
                function_values,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
        elif self._inv_Vx_lt is not None and self._inv_Vx_ut is not None:
            coefficients = itransform(
                self._inv_Vx_lt,
                function_values,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
            return itransform(
                self._inv_Vx_ut,
                coefficients,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="upper",
            )
        elif self._Vx_ut is not None and self._Vx_ut is not None:
            coefficients = transform(
                self._Vx_lt,
                function_values,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
            return transform(
                self._Vx_ut,
                coefficients,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="upper",
            )
        else:
            raise ValueError("Unexpected error: _Vx_lt must exist, _Vx_ut is optional.")
        ###

    def ifnt(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Compute the Inverse Fast Newton Transform (IFNT) to reconstruct function values from coefficients.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients of the function in the predefined polynomial basis.

        Returns
        -------
        np.ndarray
            Reconstructed function values at the interpolation grid points.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        >>> rec_f = t.ifnt(coeffs_f)
        """
        coefficients = np.asarray(coefficients).astype(np.float64)
        ###
        function_values = np.zeros(len(self), dtype=np.float64)
        if self._Vx_lt is not None and self._Vx_ut is None:
            function_values = itransform(
                self._Vx_lt,
                coefficients,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
        elif self._Vx_lt is not None and self._Vx_ut is not None:
            function_values = itransform(
                self._Vx_ut,
                coefficients,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="upper",
            )
            function_values = itransform(
                self._Vx_lt,
                function_values,
                self._T,
                self._cs_T,
                self._V_2,
                self._cs_V_2,
                self._e_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                mode="lower",
            )
        else:
            raise ValueError("Unexpected error: _Vx_lt must exist, _Vx_ut is optional.")
        ###
        if self._colex_order is not None:
            function_values = apply_permutation(
                self._colex_order, function_values, invert=False
            )
        return function_values

    def dx(
        self, coefficients: np.ndarray, i: int, k: Literal[1, 2, 3] = 1
    ) -> np.ndarray:
        """
        Apply the k-th partial derivative along the i-th spatial direction using a fast Differentiation algorithm.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients of the function in the predefined polynomial basis.
        i : int
            Index of the spatial direction along which to differentiate (0-based).
        k : {1, 2, 3}, optional
            Order of the derivative, default is 1 (first derivative).

        Returns
        -------
        np.ndarray
            Coefficients of the differentiated function (in the same basis).

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        >>> dfdx = t.dx(coeffs_f, i=0)
        >>> dfdy2 = t.dx(coeffs_f, i=1, k=2)
        """
        coefficients, i, k = (
            np.asarray(coefficients).astype(np.float64),
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
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                i,
                mode="upper",
            )
        elif self._Dx_lt is not None and self._Dx_ut is not None:
            coefficients = dtransform(
                self._Dx_ut[k - 1],
                coefficients,
                self._T,
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                i,
                mode="upper",
            )
            return dtransform(
                self._Dx_lt[k - 1],
                coefficients,
                self._T,
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
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
        """
        Apply the transpose of the k-th partial derivative operator along the i-th spatial direction using a fast Differentiation algorithm.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients in the predefined polynomial basis to which the transposed differentiation operator is applied.
        i : int
            Index of the spatial direction along which the transposed derivative is taken (0-based).
        k : {1, 2, 3}, optional
            Order of the derivative, default is 1 (first derivative).

        Returns
        -------
        np.ndarray
            Transformed coefficients after applying the transposed derivative operator.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        >>> adj_dfdx = t.dxT(coeffs_f, i=0)
        >>> adj_dfdy3 = t.dxT(coeffs_f, i=1, k=3)
        """
        coefficients, i, k = (
            np.asarray(coefficients).astype(np.float64),
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
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                i,
                mode="lower",
            )
        elif self._DxT_lt is not None and self._DxT_ut is not None:
            coefficients = dtransform(
                self._DxT_ut[k - 1],
                coefficients,
                self._T,
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                i,
                mode="upper",
            )
            return dtransform(
                self._DxT_lt[k - 1],
                coefficients,
                self._T,
                self._cs_T,
                self._N_1,
                self._m,
                self._n,
                self._p,
                i,
                mode="lower",
            )
        else:
            raise ValueError("Unexpected error.")
        ###

    def eval(self, coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the function represented by given coefficients at specified points.

        This method computes the values of the interpolated function at one or more points 
        by evaluating the polynomial expansion defined by the coefficients.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients of the function in the predefined polynomial basis.
        points : np.ndarray
            A 2D array of shape (num_points, spatial_dimension) representing the coordinates 
            of the evaluation points.

        Returns
        -------
        np.ndarray
            Function values at the specified points.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> t = Transform(spatial_dimension=2, polynomial_degree=10)
        >>> values_f = f(t.grid[:, 0], t.grid[:, 1])
        >>> coeffs_f = t.fnt(values_f)
        >>> pts = np.array([[0.0, 0.0], [0.1, 0.2]])
        >>> values_at_pts = t.eval(coeffs_f, pts)
        """
        coefficients, points = (
            np.asarray(coefficients).astype(np.float64),
            np.asarray(points).astype(np.float64),
        )

        if self._basis == "newton":
            return newton2point(
                coefficients, self._x, points, self._A, self._m, self._n
            )
        elif self._basis == "chebyshev":
            return chebyshev2point(coefficients, points, self._A, self._m, self._n)

    def embed(self, t: AbstractTransform) -> np.ndarray:
        """
        Embed a function from a lower-resolution transform `self` into a higher-resolution transform `t`.

        This method returns the index array needed to embed coefficients from the polynomial basis of `self`
        into that of `t`. The embedding is only valid if both transforms use the same nodes up to the 
        polynomial degree of `self`, and if they share the same spatial dimension.

        Parameters
        ----------
        t : AbstractTransform
            The target transform whose multi index set contains the one of `self`.

        Returns
        -------
        np.ndarray
            An array of shape `(self.size,)` containing the embedding indices into `t`.

        Raises
        ------
        ValueError
            If spatial dimensions differ.
            If the nodes do not match up to the polynomial degree.
            If not both, the lp degree and the polynomial degree are greater or equal.

        Examples
        --------
        >>> import numpy as np
        >>> from lpfun import Transform
        >>> t_coarse = Transform(spatial_dimension=2, polynomial_degree=4)
        >>> t_fine = Transform(spatial_dimension=2, polynomial_degree=8)
        >>> embed_idx = t_coarse.embed(t_fine)
        >>> coeffs_coarse = t_coarse.fnt(np.sin(t_coarse.grid[:, 0]))
        >>> coeffs_fine = np.zeros(len(t_fine))
        >>> coeffs_fine[embed_idx] = coeffs_coarse
        """
        if t.spatial_dimension != self.spatial_dimension:
            raise ValueError("Spatial dimensions do not match.")
        if not np.allclose(t.nodes[: self.polynomial_degree + 1], self.nodes):
            print(self.nodes)
            print(t.nodes[: self.polynomial_degree + 1])
            raise ValueError("Nodes mismatch: The nodes of `self` must be the starting nodes of `t`.")
        if not (
            (t.lp_degree >= self.lp_degree)
            and (t.polynomial_degree >= self.polynomial_degree)
        ):
            raise ValueError(
                "The index set of the transform `t` must already contain the index set of `self` for embedding."
            )
        return ordinal_embedding(self._m, self._T, t.tube)

    def __len__(self) -> int:
        return self._N_0

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
