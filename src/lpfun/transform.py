import numpy as np
from lpfun import NP_ARRAY, NP_FLOAT, EXPENSIVE
from lpfun.utils import (
    unisolvent_nodes_1d,
    unisolvent_nodes,
    cheb,
    tiling,
    l2n,
    n2l,
    n_dx,
    l_dx,
    rmo,
    n_eval_at_point,
)
from lpfun.core.molecules import n_transform, n_dx_transform, l_dx_transform


class Transform:
    """Transform class."""

    def __init__(
        self,
        spatial_dimension: int,
        polynomial_degree: int,
        p: float = 2.0,
        nodes: callable = cheb,
        mode: str = "newton",
        expensive=EXPENSIVE,
    ) -> None:
        """
        Initialize the Transform object.

        Args:
            spatial_dimension (int): Dimension of the spatial domain.
            polynomial_degree (int): Degree of the polynomial.
            p (float): p-norm of the polynomial space.
            nodes (callable): One dimensional nodes.
            mode (str): Transformation mode. Default is 'newton'. The other option is 'lagrange' and is only available for m=1 or p=np.inf.
            expensive (int): Expensive operation threshold.
        """

        self._m = spatial_dimension
        self._n = polynomial_degree
        self._p = p
        self._mode = mode

        # Check supported modes
        if mode not in ["newton", "lagrange"]:
            raise ValueError("Invalid mode. Choose 'newton' or 'lagrange'.")
        if (
            mode == "newton"
            and self._m >= 5
            and self._p != 1.0
            and self._p != np.inf
        ):
            # TODO: Add support for p != 1.0 and p != np.inf
            print(
                "For Newton mode, only p = 1.0 or p = infinity is supported for spatial dimension >= 5."
            )
            self._p = 1.0

        if mode == "lagrange" and self._m > 1 and self._p != np.inf:
            # Problem: Lagrange differentiation matrices
            # One would need to compute the differentiation matrices with respect to every occuring number in self._T.
            # Then, one would have in some areas differentiation matrices with shape (1, 1).
            # Hence, in this mode it is impossible to approximate the derivative of a function.
            raise ValueError(
                "The Lagrange mode is not supported for spatial dimension > 1 and p != infinity."
            )

        # Tiling of the Newton transformations only if p is not np.inf
        self._T = (
            np.array([])
            if p == np.inf or self._m == 1
            else tiling(self._m, self._n, self._p)
        )

        # Check if the operation is too expensive
        if expensive is not None:
            if self._T is None:
                length = (self._n + 1) ** self._m
                if length > expensive:
                    raise ValueError(
                        f"Operation too expensive: {length} > {expensive}. If this operation should be executed anyways, please set expensive to None."
                    )
            else:
                length = np.sum(self._T)
                if length > expensive:
                    raise ValueError(
                        f"Operation too expensive: {length} > {expensive}. If this operation should be executed anyways, please set expensive to None."
                    )

        # One dimensional unisolvent nodes
        self._nodes = unisolvent_nodes_1d(self._n + 1, nodes)

        # Multi-dimensional unisolvent nodes
        self._unisolvent_nodes = unisolvent_nodes(
            self._nodes, self._m, self._n, self._p
        )

        if mode == "newton":
            # Lagrange to Newton transformation 1D
            self._l2n = rmo(l2n(self._nodes))

            # Newton to Lagrange transformation 1D
            self._n2l = rmo(n2l(self._nodes))

            # Newton differentiation matrix 1D
            self._dx = rmo(n_dx(self._nodes), mode="upper")

        elif mode == "lagrange":
            # Lagrange differentiation matrix 1D
            self._dx = l_dx(self._nodes)

    @property
    def dimension(self) -> int:
        return self._m

    @property
    def polynomial_degree(self) -> int:
        return self._n

    @property
    def p(self) -> int:
        return self._p

    @property
    def nodes(self) -> NP_ARRAY:
        return self._nodes

    @property
    def unisolvent_nodes(self) -> NP_ARRAY:
        return self._unisolvent_nodes

    def warmup(self) -> None:
        """Warmup the JIT compiler."""
        length = len(self._unisolvent_nodes)
        zeros = np.zeros(length, dtype=NP_FLOAT)
        self.dx(zeros, 0)
        self.dx(zeros, 0, True)
        if self._m > 1:
            self.dx(zeros, 1)
        if self._mode == "lagrange":
            return
        self.eval(zeros, np.zeros(self._m, dtype=NP_FLOAT))
        self.push(zeros)
        self.pull(zeros)

    def push(self, function_values: NP_ARRAY) -> NP_ARRAY:
        """Fast l^p Transformation"""
        function_values = np.asarray(function_values).astype(np.float64)
        if self._mode == "newton":
            return n_transform(self._l2n, function_values, self._T, self._m, self._p)
        elif self._mode == "lagrange":
            raise ValueError("The push method is not needed for the Lagrange mode.")

    def pull(self, coefficients: NP_ARRAY) -> NP_ARRAY:
        """Inverse Fast l^p Transformation"""
        if self._mode == "lagrange":
            raise ValueError("The pull method is not needed for the Lagrange mode.")
        coefficients = np.asarray(coefficients).astype(np.float64)
        return n_transform(self._n2l, coefficients, self._T, self._m, self._p)

    def dx(self, coefficients: NP_ARRAY, i: int, transpose: bool = False) -> NP_ARRAY:
        """Spectral l^p Differentiation"""
        coefficients = np.asarray(coefficients).astype(np.float64)
        if self._mode == "newton":
            return n_dx_transform(
                self._dx, coefficients, self._T, self._m, self._n, self._p, i, transpose
            )
        elif self._mode == "lagrange":
            return l_dx_transform(
                self._dx,
                coefficients,
                self._m,
                self._n,
                i,
                transpose,
            )

    def eval(self, coefficients: NP_ARRAY, x: NP_ARRAY) -> NP_FLOAT:
        """Point Evaluation"""
        # TODO Add tests for this method
        if self._mode == "newton":
            return n_eval_at_point(coefficients, self._nodes, x, self._m, self._p)
        elif self._mode == "lagrange":
            raise NotImplementedError(
                "The eval method is not implemented for the Lagrange mode yet."
            )

    def __len__(self) -> int:
        return len(self._unisolvent_nodes)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transform):
            return False
        if not value.dimension == self.dimension:
            return False
        if not value.polynomial_degree == self.polynomial_degree:
            return False
        if not value.p == self.p:
            return False