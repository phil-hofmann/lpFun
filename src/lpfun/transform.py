import numpy as np
from lpfun import NP_FLOAT, WARMUP, MESSAGES, PARALLEL
from lpfun.utils import (
    cheb2nd,
    newton2lagrange,
    newton2derivative,
    newton2point,
    ###
    tube,
    test_threshold,
    leja_nodes,
    lower_grid,
    apply_permutation,
    is_lower_triangular,
    lu,
    rmo,
    inv,
)
from lpfun.core.molecules import transform, itransform, dtransform


class Transform:
    """Transform class."""

    def __init__(
        self,
        spatial_dimension: int,  # ... m
        polynomial_degree: int,  # ... n
        # Optional parameters for customisation
        lp_degree: float = 2.0,  # ... p
        nodes: callable = cheb2nd,  # ... x
        interpolation_matrix: callable = newton2lagrange,  # ... Q
        differentiation_matrix: callable = newton2derivative,  # ... D
        eval_at_point: callable = newton2point,
        precomp: bool = PARALLEL,
    ):
        """
        Initialize the Transform object.

        Args:
            spatial_dimension (int): The dimension of the spatial domain.
            polynomial_degree (int): The degree of the polynomial.
            lp_degree (float): The p-norm of the polynomial space.
            nodes (callable): A callable function that takes an integer and returns a one dimensional numpy array of nodes.
            interpolation_matrix (callable): A callable function that takes a one dimensional numpy array of nodes and returns the interpolation matrix.
            differentiation_matrix (bool): A callable function that takes a one dimensional numpy array of nodes and returns the differentiation matrix.
            eval_at_point (callable): TODO
            precomp (bool): Do you want to precompute the inverse of the interpolation matrix? This enables faster parallel execution. Default is set to PARALLEL.
        """

        self._m = spatial_dimension
        self._n = polynomial_degree
        self._p = lp_degree
        self._eval_at_point = eval_at_point

        # Compute tube only if p is not np.inf
        self._T = tube(self._m, self._n, self._p)

        # Check the threshold
        test_threshold(self._T)

        # One dimensional (unisolvent, leja-ordered) nodes
        x = nodes(self._n + 1)
        if len(np.unique(x)) != len(x):
            raise ValueError("The provided nodes are not pairwise distinct.")
        self._x = leja_nodes(x)

        # Lower grid
        G = lower_grid(self._x, self._m, self._n, self._p)

        # Lex order: User experience
        self._lex_order = np.lexsort(G.T)
        self._G = apply_permutation(self._lex_order, G, invert=False)

        # Compute matrices
        Q = interpolation_matrix(self._x)
        D = differentiation_matrix(self._x)

        # Compute condition number
        self._cond_Q = np.linalg.cond(Q)

        # Row major ordering Q
        if not is_lower_triangular(Q):
            # TODO
            print(f"{'-'*12}Perform LU for (Q)-{'-'*12}\n") if MESSAGES else None
            QL, QU = lu(Q)
            QU = QU[::-1, ::-1]
            self._QL, self._QU = (rmo(QL), rmo(QU))
            self._invQL, self._invQU = (
                (rmo(inv(QL)), rmo(inv(QU))) if precomp else (None, None)
            )
            self._Q, self._invQ = (None, None)
        else:
            self._QL, self._QU, self._invQL, self._invQU = (None, None, None, None)
            self._Q, self._invQ = rmo(Q), rmo(inv(Q)) if precomp else None

        # Row major ordering D
        if not is_lower_triangular(D.T):
            print(f"{'-'*12}Perform LU for (D)-{'-'*12}\n") if MESSAGES else None
            DL, DU = lu(D)
            self._DL, self._DU = (rmo(DL), rmo(DU[::-1, ::-1]))
            self._DTL, self._DTU = (rmo(DU.T), rmo(DL.T[::-1, ::-1]))
        else:
            self._D, self._DL, self._DU = (rmo(D[::-1, ::-1]), None, None)
            self._DT, self._DTL, self._DTU = (rmo(D.T), None, None)

        # Warmup the JIT compiler
        if WARMUP:
            print(f"{'-'*12}Warmup JIT compiler{'-'*12}\n") if MESSAGES else None
            self.warmup()

        print(self) if MESSAGES else None

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
        if self._invQ is not None:
            coefficients = itransform(
                self._invQ, function_values, self._T, self._m, self._p
            )
        elif self._Q is not None:
            coefficients = transform(
                self._Q, function_values, self._T, self._m, self._p
            )
        elif self._invQL is not None and self._invQU is not None:
            # TODO
            pass
        elif self._QL is not None and self._QU is not None:
            # TODO
            coefficients = transform(
                self._QL, function_values, self._T, self._m, self._p
            )
            coefficients = transform(
                self._QU, coefficients[::-1], self._T[::-1], self._m, self._p
            )[::-1]
        else:
            raise ValueError("Unexpected error.")
        ###
        return coefficients

    def ifnt(self, coefficients: np.ndarray) -> np.ndarray:
        """Inverse Fast Newton Transform"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        ###
        function_values = np.zeros(len(self), dtype=NP_FLOAT)
        if self._Q is not None:
            function_values = itransform(
                self._Q, coefficients, self._T, self._m, self._p
            )
        elif self._QL is not None and self._QU is not None:
            # TODO
            function_values = itransform(
                self._QU, coefficients[::-1], self._T[::-1], self._m, self._p
            )[::-1]
            function_values = itransform(
                self._QL, function_values, self._T, self._m, self._p
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
                self._D, coefficients[::-1], self._T[::-1], self._m, self._n, self._p, i
            )[::-1]
        elif self._DL is not None and self._DU is not None:
            # TODO test
            coefficients = dtransform(
                self._DL, coefficients, self._T, self._m, self._n, self._p, i
            )
            coefficients = dtransform(
                self._DU,
                coefficients[::-1],
                self._T[::-1],
                self._m,
                self._n,
                self._p,
                i,
            )[::-1]
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
                self._DT, coefficients, self._T, self._m, self._n, self._p, i
            )
        elif self._DTL is not None and self._DU is not None:
            # TODO test
            coefficients = dtransform(
                self._DTL, coefficients, self._T, self._m, self._n, self._p, i
            )
            coefficients = dtransform(
                self._DTU,
                coefficients[::-1],
                self._T[::-1],
                self._m,
                self._n,
                self._p,
                i,
            )[::-1]
        else:
            raise ValueError("Unexpected error.")
        ###
        return coefficients

    def eval(self, coefficients: np.ndarray, x: np.ndarray) -> NP_FLOAT:
        """Point Evaluation"""
        return self._eval_at_point(coefficients, self._x, x, self._m, self._p)

    def __len__(self) -> int:
        return len(self._G)

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
            f"{'Spatial Dimension':<20} | {self._m:<20}\n"
            f"{'Polynomial Degree':<20} | {self._n:<20}\n"
            f"{'lp Degree':<20} | {self._p:<20}\n"
            f"{'Condition Q':<20} | {self._cond_Q:<20.2e}\n"
            f"{'Length':<20} | {len(self):<20}\n"
            f"{'-'*20}-+-{'-'*20}\n"
        )
