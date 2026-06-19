import sys
import time
import warnings
import threading
import numpy as np
from typing import Literal, Callable
from abc import ABC, abstractmethod

from lpfun.utils import (
    classify,
    is_lower_triangular,
    get_lu,
    get_rmo,
)

# core
from lpfun.core.grid import (
    get_leja_order,
    get_grid,
)
from lpfun.core.set import (
    lp_set,
    lp_tube,
    rank_embedding,
    transposition,
    permutations,
    permutations_max,
)
from lpfun.core.transform import (
    transform_lt,
    transform_ut,
    itransform_lt,
    itransform_ut,
    dtransform_lt,
    dtransform_ut,
)

# basis
from lpfun.basis.diff import (
    newton2derivative,
    chebyshev2derivative,
    legendre2derivative,
)
from lpfun.basis.eval import (
    newton2point,
    chebyshev2point,
    legendre2point,
)
from lpfun.basis.nodes import (
    cheb2nd_nodes,
    leja_nodes,  # NOTE alternative for adaptivity
)
from lpfun.basis.vander import (
    newton2lagrange,
    chebyshev2lagrange,
    legendre2lagrange,
)


class AbstractFunction(ABC):
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
        """Returns the l^p degree"""
        pass

    @abstractmethod
    def tube(self) -> np.ndarray:
        """Returns the tube array"""
        pass

    @property
    @abstractmethod
    def index_set(self) -> np.ndarray:
        """Returns the index set array"""
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
    def leja_order(self) -> np.ndarray:
        """Returns the leja order array"""
        pass


class Function(AbstractFunction):
    """
    Function class for polynomial multivariate interpolation on quasi-tensorial grids associated with l^p-type polynomial spaces.

    Attributes
    ----------
    spatial_dimension : int
        The dimension `m` of the spatial domain, i.e. the number of input variables.
    polynomial_degree : int
        The maximum total degree `n` used for the approximation.
    lp_degree : float
        The parameter `p` of the l^p-type degree used to define the index set and polynomial space.
        The default is 2.0, coressponding to the Euclidean degree
    tube : numpy.ndarray
        Array encoding the directional polynomial degree constraints.
    index_set : numpy.ndarray
        The index set defining the polynomial exponents.
    nodes : np.ndarray
        The one-dimensional interpolation nodes.
    grid : numpy.ndarray
        The multidimensional grid points, with shape `(num_points, spatial_dimension)`.
    """

    def __init__(
        self,
        spatial_dimension: int,
        polynomial_degree: int,
        lp_degree: float = 2.0,
        nodes: Callable[[int], np.ndarray] = cheb2nd_nodes,
        basis: Literal["newton", "chebyshev", "legendre"] = "newton",
        precomputation: bool = True,
        precompilation: bool = True,
        threshold: int = 150_000_000,
        report: bool = True,
    ):
        """
        Initialize the `Function` object for multivariate polynomial interpolation on quasi-tensorial grids.

        Parameters
        ----------
        spatial_dimension : int
            The dimension `m` of the spatial domain, i.e. the number of input variables.
        polynomial_degree : int
            The maximum total degree `n` used for the approximation.
        lp_degree : float
            The `p` degree  of the l^p norm that defining the index set and polynomial space.
            The default is 2.0, coressponding to the Euclidean degree
        nodes : callable, optional
            A callable that takes an integer `n` and returns an array of `n` one-dimensional distinct interpolation nodes.
            Typical choices are Chebyshev nodes `cheb2nd_nodes` or Leja nodes `leja_nodes`.
        basis: str, optional
            The polynomial basis used for constructing Vandermonde and differentiation matrices.
            The default is "newton".
        precomputation : bool, optional
            If True, precompute the inverse of the Vandermonde matrix.
            This improves runtime performance at the cost of additional setup time and memory usage.
            The default is True.
        precompilation : bool, optional
            If True, precompile just-in-time compiled functions during initialization to reduce overhead in subsequent calls.
            The default is True.
        threshold:
            Safety threshold for the dimension of the polynomial space.
            If the dimension exceeds this value, initialization raises a `ValueError` to avoid excessive memory usage or computational cost.
            The default is 150,000,000.
        report:
            If True, print initialization information and setup statistics.
            The default is True.

        Raises
        ------
        ValueError
            If the dimension of the polynomial space exceeds `threshold`.

        Notes
        -----
        The choice of basis, nodes, and index set affects numerical stability, accuracy, and computational efficiency.
        Precomputation and precompilation can improve runtime performance, but may increase initialization time and memory usage.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        >>> coeffs_dx = fun.diff(coeffs, dim=0, order=1)
        >>> values_dx = fun.eval(coeffs_dx)
        """

        self._start_spinner() if report else None
        construction_start = time.time()
        self._m = int(spatial_dimension)
        self._n = int(polynomial_degree)
        self._p = float(lp_degree)
        self._basis = str(basis)
        classify(self._m, self._n, self._p)

        if not basis in ["newton", "chebyshev", "legendre"]:
            self._stop_spinner() if report else None
            raise ValueError("Invalid choice for basis.")

        # index set
        self._spinner_label = "Constructing index set..."
        self._A = lp_set(self._m, self._n, self._p)

        # tube size projection
        self._spinner_label = "Constructing tube size projection..."
        self._T = lp_tube(self._A, self._m, self._n, self._p)
        self._cs_T = np.r_[0, np.cumsum(self._T)]

        if self._m == 3:
            self._V_2 = np.array(
                [
                    np.sum(self._T[self._cs_T[i] : self._cs_T[i + 1]])
                    for i in range(self._T[0])
                ],
                dtype=np.int64,
            )
            self._cs_V_2 = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(self._V_2))
            )

        # threshold
        self._spinner_label = "Checking threshold..."
        if threshold is None:
            warnings.warn("Threshold is set to None. This may lead to memory issues.")
        elif len(self) > threshold:
            self._stop_spinner() if report else None
            raise ValueError(f"""
                    Dimension exceeds threshold: {format(len(self), "_")} > {format(threshold, "_")}.
                    If this operation should be executed anyways, please set threshold to None.
                """)

        # nodes
        self._spinner_label = "Constructing nodes..."
        x = nodes(self._n + 1)
        if len(np.unique(x)) != self._n + 1:
            self._stop_spinner() if report else None
            raise ValueError("The provided nodes are not pairwise distinct.")
        self._leja_order = get_leja_order(x)
        self._x = x[self._leja_order]

        # grid
        self._spinner_label = "Constructing grid..."
        self._grid = get_grid(self._x, self._A, self._m, self._n, self._p)

        # matrices
        self._spinner_label = "Constructing matrices..."
        Vx = None
        Dx = None
        if basis == "newton":
            Vx = newton2lagrange(self._x)
            Dx = newton2derivative(self._x)
        elif basis == "chebyshev":
            Vx = chebyshev2lagrange(self._x)
            Dx = chebyshev2derivative(self._x)
        elif basis == "legendre":
            Vx = legendre2lagrange(self._x)
            Dx = legendre2derivative(self._x)
        Dx2 = Dx @ Dx
        Dx3 = Dx @ Dx2

        # condition number
        self._condition_number = np.linalg.cond(Vx)

        # LU decompositions
        self._spinner_label = "Computing LU decompositions..."
        lt_Vx = is_lower_triangular(Vx)
        Vx_L, Vx_U = None, None

        if lt_Vx:
            self._Vx = get_rmo(Vx)
        else:
            Vx_L, Vx_U = get_lu(Vx)
            self._Vx_L, self._Vx_U = get_rmo(Vx_L), get_rmo(Vx_U[::-1, ::-1])[::-1]

        if is_lower_triangular(Dx.T):
            self._Dx = tuple(get_rmo(D[::-1, ::-1])[::-1] for D in (Dx, Dx2, Dx3))
            self._Dx_T = tuple(get_rmo(D.T) for D in (Dx, Dx2, Dx3))
        else:
            Dx_LU = tuple(get_lu(Dx) for Dx in [Dx, Dx2, Dx3])

            self._Dx_L = tuple(get_rmo(lt) for lt, _ in Dx_LU)
            self._Dx_U = tuple(get_rmo(ut[::-1, ::-1])[::-1] for _, ut in Dx_LU)

            self._Dx_T_L = tuple(get_rmo(ut.T) for _, ut in Dx_LU)
            self._Dx_T_U = tuple(get_rmo(lt.T[::-1, ::-1])[::-1] for lt, _ in Dx_LU)

        # Precompute inverses
        self._spinner_label = "Precomputing inverses..."
        if precomputation and lt_Vx:
            self._inv_Vx = get_rmo(np.linalg.inv(Vx))

        if precomputation and not lt_Vx:
            self._inv_Vx_L = get_rmo(np.linalg.inv(Vx_L))
            self._inv_Vx_U = get_rmo(np.linalg.inv(Vx_U)[::-1, ::-1])[::-1]

        construction_end = time.time()
        self._construction_ms = (construction_end - construction_start) * 1000

        # Construct permutations
        self._spinner_label = "Constructing permutations..."
        self._pi = transposition(self._T)
        if self._p == np.inf:
            self._permutations = permutations_max(self._m, self._n)
        else:
            self._permutations = permutations(len(self), self._m, self._pi)

        # warmup JIT compiler
        self._spinner_label = "Precompiling jit functions..."
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
        self._spinner_label = "Initializing..."
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
    def lp_degree(self) -> float:
        return self._p

    @property
    def tube(self) -> np.ndarray:
        return self._T

    @property
    def index_set(self) -> np.ndarray:
        return self._A

    @property
    def nodes(self) -> np.ndarray:
        return self._x

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def leja_order(self) -> np.ndarray:
        return self._leja_order

    @property
    def _transform_args(self) -> dict:
        return {
            "m": self._m,
            "n": self._n + 1,
            "pi": self._pi,
            "T": self._T,
            "cs_T": self._cs_T,
            "V_2": self._V_2 if hasattr(self, "_V_2") else None,
            "cs_V_2": self._cs_V_2 if hasattr(self, "_cs_V_2") else None,
        }

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        zeros_N = np.zeros(len(self), dtype=np.float64)
        one_zero = np.zeros((1, self._m), dtype=np.float64)
        self._spinner_label = "Precompiling interpolation..."
        self.interp(zeros_N)
        self._spinner_label = "Precompiling evaluation..."
        self.eval(zeros_N)
        self._spinner_label = "Precompiling differentiation..."
        self.diff(zeros_N, 0)
        self._spinner_label = "Precompiling transposed differentiation..."
        self.diffT(zeros_N, 0)
        self._spinner_label = "Precompiling point evaluation..."
        self(zeros_N, one_zero)

    def interp(self, function_values: np.ndarray) -> np.ndarray:
        """
        Interpolate function values by computing their polynomial coefficients.

        This method applies the Fast Newton Transform (FNT) to function values
        sampled on the interpolation grid and returns the corresponding coefficients
        in the chosen polynomial basis.

        Parameters
        ----------
        function_values : np.ndarray
            Function values sampled on the interpolation grid points.
            The array must have length equal to the number of grid points.

        Returns
        -------
        np.ndarray
            Polynomial coefficients representing the interpolant in the chosen basis.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        """
        function_values = np.asarray(function_values).astype(np.float64)
        if hasattr(self, "_inv_Vx"):
            return transform_lt(
                L=self._inv_Vx, x=function_values, **self._transform_args
            )
        elif hasattr(self, "_inv_Vx_L") and hasattr(self, "_inv_Vx_U"):
            coefficients = transform_lt(
                L=self._inv_Vx_L, x=function_values, **self._transform_args
            )
            return transform_ut(
                U=self._inv_Vx_U, x=coefficients, **self._transform_args
            )
        elif hasattr(self, "_Vx"):
            return itransform_lt(L=self._Vx, x=function_values, **self._transform_args)
        elif hasattr(self, "_Vx_L") and hasattr(self, "_Vx_U"):
            coefficients = itransform_lt(
                L=self._Vx_L, x=function_values, **self._transform_args
            )
            return itransform_ut(U=self._Vx_U, x=coefficients, **self._transform_args)
        else:
            raise ValueError(
                "Unexpected error: _Vx_L and _Vx_U must exist for non-lower-triangular case."
            )

    def eval(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial interpolant on the interpolation grid points.

        This method applies the inverse Fast Newton Transform (IFNT) to
        polynomial coefficients and returns the corresponding function values
        on the interpolation grid.

        Parameters
        ----------
        coefficients : np.ndarray
            Polynomial coefficients of the function in the chosen basis.

        Returns
        -------
        np.ndarray
            Function values of the interpolant at the interpolation grid points.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        >>> values_rec = fun.eval(coeffs)
        """
        coefficients = np.asarray(coefficients).astype(np.float64)
        function_values = np.zeros(len(self), dtype=np.float64)
        if hasattr(self, "_Vx"):
            function_values = transform_lt(
                self._Vx, coefficients, **self._transform_args
            )
        elif hasattr(self, "_Vx_U") and hasattr(self, "_Vx_L"):
            function_values = transform_ut(
                U=self._Vx_U, x=coefficients, **self._transform_args
            )
            function_values = transform_lt(
                self._Vx_L, function_values, **self._transform_args
            )
        else:
            raise ValueError("Unexpected error.")
        return function_values

    def diff(
        self, coefficients: np.ndarray, dim: int, order: Literal[1, 2, 3] = 1
    ) -> np.ndarray:
        """
        Differentiate a polynomial interpolant along a spatial direction.

        This method applies a fast differentiation transform to the given
        polynomial coefficients and returns the coefficients of the corresponding
        partial derivative.

        Parameters
        ----------
        coefficients : np.ndarray
            Polynomial coefficients representing the interpolant in the chosen basis.
        dim : int
            Spatial direction along which to differentiate, using zero-based indexing.
        order : {1, 2, 3}, optional
            Derivative order. The default is 1.

        Returns
        -------
        np.ndarray
            Polynomial coefficients of the differentiated interpolant in the same basis.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        >>> coeffs_dx = fun.diff(coeffs, dim=0)
        >>> coeffs_dyy = fun.diff(coeffs, dim=1, order=2)
        """
        coefficients, dim, order = (
            np.asarray(coefficients).astype(np.float64),
            int(dim),
            int(order),
        )
        if (dim < 0) or (dim >= self._m):
            raise ValueError(
                f"Invalid value for dim. Please choose dim in between 0 and {self._m - 1}."
            )
        if not order in [1, 2, 3]:
            raise ValueError("Invalid value for k. Please choose 1, 2 or 3.")
        if hasattr(self, "_Dx"):
            return dtransform_ut(
                D=self._Dx[order - 1],
                x=coefficients,
                n=self._n + 1,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )
        elif hasattr(self, "_Dx_U") and hasattr(self, "_Dx_L"):
            coefficients = dtransform_lt(
                D=self._Dx_L[order - 1],
                x=coefficients,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )
            return dtransform_ut(
                D=self._Dx_U[order - 1],
                x=coefficients,
                n=self._n + 1,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )
        else:
            raise ValueError("Unexpected error.")

    def diffT(
        self, coefficients: np.ndarray, dim: int, order: Literal[1, 2, 3] = 1
    ) -> np.ndarray:
        """
        Apply the transpose of a partial differentiation operator.

        This method applies the transpose of the fast differentiation transform
        along a given spatial direction to polynomial coefficients.

        Parameters
        ----------
        coefficients : numpy.ndarray
            Polynomial coefficients to which the transposed differentiation
            operator is applied.
        dim : int
            Spatial direction of the derivative, using zero-based indexing.
        order : {1, 2, 3}, optional
            Derivative order. The default is 1.

        Returns
        -------
        numpy.ndarray
            Polynomial coefficients after applying the transposed differentiation
            operator.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        >>> coeffs_dx_T = fun.diffT(coeffs, dim=0)
        >>> coeffs_dyyy_T = fun.diffT(coeffs, dim=1, order=3)
        """
        coefficients, dim, order = (
            np.asarray(coefficients).astype(np.float64),
            int(dim),
            int(order),
        )

        if (dim < 0) or (dim >= self._m):
            raise ValueError(
                f"Invalid value for dim. Please choose dim between 0 and {self._m - 1}."
            )

        if order not in [1, 2, 3]:
            raise ValueError("Invalid value for order. Please choose 1, 2, or 3.")

        if hasattr(self, "_Dx"):
            return dtransform_lt(
                D=self._Dx_T[order - 1],
                x=coefficients,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )

        elif hasattr(self, "_Dx_U") and hasattr(self, "_Dx_L"):
            coefficients = dtransform_lt(
                D=self._Dx_T_L[order - 1],
                x=coefficients,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )
            return dtransform_ut(
                D=self._Dx_T_U[order - 1],
                x=coefficients,
                n=self._n + 1,
                perm=self._permutations[dim],
                T=self._T,
                cs_T=self._cs_T,
            )

        else:
            raise ValueError("Unexpected error.")

    def __call__(self, coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Evaluate a polynomial interpolant at specified points.

        This method evaluates the polynomial expansion defined by the given
        coefficients at one or more points in the spatial domain.

        Parameters
        ----------
        coefficients : np.ndarray
            Polynomial coefficients representing the interpolant in the chosen basis.
        points : np.ndarray
            Evaluation points with shape `(num_points, spatial_dimension)`.

        Returns
        -------
        np.ndarray
            Function values of the interpolant at the specified points.

        Example
        -------
        >>> import numpy as np
        >>> from lpfun import Function
        >>> def f(x, y):
        ...     return np.sin(x) * np.cos(y)
        >>> fun = Function(spatial_dimension=2, polynomial_degree=10)
        >>> values = f(fun.grid[:, 0], fun.grid[:, 1])
        >>> coeffs = fun.interp(values)
        >>> pts = np.array([[0.0, 0.0], [0.1, 0.2]])
        >>> values_at_pts = fun(coeffs, pts)
        """

        coefficients, points = (
            np.asarray(coefficients).astype(np.float64),
            np.asarray(points).astype(np.float64),
        )

        if points.ndim == 1:
            points = points.reshape(1, -1)

        if points.shape[1] != self._m:
            raise ValueError(
                f"Points must have shape (num_points, {self._m}), "
                f"but got {points.shape}."
            )

        if coefficients.shape[0] != len(self._A):
            raise ValueError(
                f"Coefficients must have length {len(self._A)}, "
                f"but got {coefficients.shape[0]}."
            )

        if self._basis == "newton":
            return newton2point(
                coefficients, self._x, points, self._A, self._m, self._n
            )
        elif self._basis == "chebyshev":
            return chebyshev2point(coefficients, points, self._A, self._m, self._n)
        elif self._basis == "legendre":
            return legendre2point(coefficients, points, self._A, self._m, self._n)
        else:
            raise ValueError(f"Unknown basis: {self._basis}")

    def embed(self, larger_fun: AbstractFunction) -> np.ndarray:
        """
        Return embedding indices into a larger polynomial function space.

        This method computes the index array needed to embed coefficients from
        the polynomial space of `self` into the polynomial space of `larger_fun`. The
        embedding is valid only if both function spaces have the same spatial
        dimension, use compatible interpolation nodes, and the index set of
        `self` is contained in the index set of `larger_fun`.

        Parameters
        ----------
        larger_fun : AbstractFunction
            Target function space into which the coefficients of `self` are embedded

        Returns
        -------
        np.ndarray
            Embedding indices into `larger_fun`.

        Raises
        ------
        ValueError
            If spatial dimensions differ.
        ValueError
            If the interpolation nodes of `self` do not match the initial nodes
            of `larger_fun` up to degree `self.polynomial_degree`.
        ValueError
            If the index set of `self` is not contained in the index set of `larger_fun`.

         Example
         -------
         >>> import numpy as np
         >>> from lpfun import Function
         >>> fun = Function(spatial_dimension=2, polynomial_degree=4)
         >>> larger_fun = Function(spatial_dimension=2, polynomial_degree=8)
         >>> embed_idx = fun.embed(larger_fun)
         >>> coeffs = fun.interp(np.sin(fun.grid[:, 0]))
         >>> coeffs_larger = np.zeros(len(larger_fun))
         >>> coeffs_larger[embed_idx] = coeffs
        """
        if larger_fun.spatial_dimension != self.spatial_dimension:
            raise ValueError("Spatial dimensions do not match.")
        if not np.allclose(larger_fun.nodes[: self.polynomial_degree + 1], self.nodes):
            print(self.nodes)
            print(larger_fun.nodes[: self.polynomial_degree + 1])
            raise ValueError(
                "Nodes mismatch: The nodes of `self` must be the starting nodes of `larger_fun`."
            )
        if not (
            (larger_fun.lp_degree >= self.lp_degree)
            and (larger_fun.polynomial_degree >= self.polynomial_degree)
        ):
            raise ValueError(
                "The index set of the Function object `larger_fun` must already contain the index set of `self` for embedding."
            )
        return rank_embedding(self._m, self._T, larger_fun.tube)

    def __len__(self) -> int:
        return len(self._A)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Function):
            return False
        return (
            value.spatial_dimension == self.spatial_dimension
            and value.polynomial_degree == self.polynomial_degree
            and value.lp_degree == self.lp_degree
            and value._basis == self._basis
        )

    def __repr__(self) -> str:
        return (
            f"{'-'*20}-+-{'-'*20}\n"
            f"{' '*19}Report{' '*18}\n"
            f"{'-'*20}-+-{'-'*20}\n"
            f"{'Spatial Dimension':<20} | {self._m}\n"
            f"{'Polynomial Degree':<20} | {self._n:_}\n"
            f"{'l^p Degree':<20} | {self._p}\n"
            f"{'Condition V':<20} | {self._condition_number:.2e}\n"
            f"{'Amount of Coeffs':<20} | {len(self):_}\n"
            f"{'Construction':<20} | {self._construction_ms:_.2f} ms\n"
            f"{'Precompilation':<20} | {self._precompilation_ms:_.2f} ms\n"
            f"{'-'*20}-+-{'-'*20}\n"
        )
