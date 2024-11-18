import numpy as np
from typing import Literal
from lpfun import NP_FLOAT, EXPENSIVE
from lpfun.utils import (
    leja_nodes,
    lower_grid,
    apply_permutation,
    cheb,
    tube,
    l2n,
    n2l,
    n_dx,
    rmo,
    n_eval_at_point,
)
from lpfun.core.molecules import transform, dx_transform


class Transform:
    """Transform class."""

    def __init__(
        self,
        spatial_dimension: int,
        polynomial_degree: int,
        lp_degree: float = 2.0,
        nodes: callable = cheb,
        expensive=EXPENSIVE,
        warmup: bool = True,
    ):
        """
        Initialize the Transform object.

        Args:
            spatial_dimension (int): Dimension of the spatial domain.
            polynomial_degree (int): Degree of the polynomial.
            lp_degree (float): p-norm of the polynomial space.
            nodes (callable): One dimensional nodes.
            expensive (int): Expensive operation threshold.
            warmup (bool): Warmup the JIT compiler.
        """

        self._m = spatial_dimension
        self._n = polynomial_degree
        self._p = lp_degree

        # Check supported modes
        if self._m >= 5 and self._p != 1.0 and self._p != np.inf:
            # TODO: Add support for p != 1.0 and p != np.inf
            print(
                "Currently, only p = 1.0 or p = infinity is supported for spatial dimension >= 5."
            )
            self._p = 1.0

        # Compute tube only if p is not np.inf
        self._T = (
            np.array([])
            if self._p == np.inf or self._m == 1
            else tube(self._m, self._n, self._p)
        )

        # Check if the operation is too expensive
        if expensive is not None:
            if self._T is None:
                length = (self._n + 1) ** self._m
                if length > expensive:
                    raise ValueError(
                        f"""
                            Operation too expensive: {length} > {expensive}.
                            If this operation should be executed anyways, please set expensive to None.
                        """
                    )
            else:
                length = np.sum(self._T)
                if length > expensive:
                    raise ValueError(
                        f"""
                            Operation too expensive: {length} > {expensive}.
                            If this operation should be executed anyways, please set expensive to None.
                        """
                    )

        # One dimensional (unisolvent, leja-ordered) nodes
        self._nodes = leja_nodes(self._n + 1, nodes)

        # Lower grid
        grid = lower_grid(self._nodes, self._m, self._n, self._p)

        # Lex order: User experience
        self._lex_order = np.lexsort(grid.T)
        self._grid = apply_permutation(self._lex_order, grid, invert=False)

        # Lagrange to Newton transformation 1D
        self._l2n = rmo(l2n(self._nodes))

        # Newton to Lagrange transformation 1D
        self._n2l = rmo(n2l(self._nodes))

        # Newton differentiation matrix 1D
        self._dx = rmo(n_dx(self._nodes), mode="upper")

        # Warmup the JIT compiler
        if warmup:
            self.warmup()

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
        return self._nodes

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        length = len(self._grid)
        zeros = np.zeros(length, dtype=NP_FLOAT)
        self.dx(zeros, 0)
        self.dx(zeros, 0, True)
        if self._m > 1:
            self.dx(zeros, 1)
        self.eval(zeros, np.zeros(self._m, dtype=NP_FLOAT))
        self.fnt(zeros)
        self.ifnt(zeros)

    def fnt(self, function_values: np.ndarray) -> np.ndarray:
        """Fast Newton Transform"""
        function_values = np.asarray(function_values).astype(NP_FLOAT)
        function_values = apply_permutation(
            self._lex_order, function_values, invert=True
        )
        coefficients = transform(
            self._l2n, function_values, self._T, self._m, self._p
        )
        return coefficients

    def ifnt(self, coefficients: np.ndarray) -> np.ndarray:
        """Inverse Fast Newton Transform"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        function_values = transform(
            self._n2l, coefficients, self._T, self._m, self._p
        )
        function_values = apply_permutation(
            self._lex_order, function_values, invert=False
        )
        return function_values

    def dx(
        self, coefficients: np.ndarray, i: int, transpose: bool = False
    ) -> np.ndarray:
        """Fast Spectral Differentiation"""
        coefficients = np.asarray(coefficients).astype(NP_FLOAT)
        coefficients = dx_transform(
            self._dx, coefficients, self._T, self._m, self._n, self._p, i, transpose
        )
        return coefficients

    def eval(self, coefficients: np.ndarray, x: np.ndarray) -> NP_FLOAT:
        """Point Evaluation"""
        # TODO Add tests for this method
        return n_eval_at_point(coefficients, self._nodes, x, self._m, self._p)

    def __len__(self) -> int:
        return len(self._grid)

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
