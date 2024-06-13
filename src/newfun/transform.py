import numpy as np
from newfun.utils_njit import eval_at_point
from newfun.utils import unisolvent_nodes_1d, unisolvent_nodes, cheb, tiling, l2n, n2l, rmo
from newfun.transform_utils import transform, transform_diag
from newfun import NP_ARRAY, EXPENSIVE

class Transform:

    """Transform class."""

    def __init__(self, dimension: int, polynomial_degree: int, p: float = 2.0, nodes:callable=cheb, expensive=EXPENSIVE) -> None:
        
        self._m = dimension
        self._n = polynomial_degree
        self._p = p

        # TODO If the dimension is five or higher only p = 1.0 or p = infinity is supported
        if self._m >= 5 and not (self._p == 1.0 or self._p == np.infty):
            print("Only p = 1.0 or p = infinity is supported for m >= 5.")
            self._p = 1.0

        # Tiling of the Newton transformations only if p is not np.infty
        self._T = None if p == np.infty or self._m == 1 else tiling(self._m, self._n, self._p)

        # Check if the operation is too expensive
        if expensive is not None:
            if self._T is None:
                length = (self._n + 1)**self._m
                if length > expensive:
                    raise ValueError(f"Operation too expensive: {length} > {expensive}. If this operation should be executed anyways, please set expensive to None.")
            else:
                length = np.sum(self._T)
                if length > expensive:
                    raise ValueError(f"Operation too expensive: {length} > {expensive}. If this operation should be executed anyways, please set expensive to None.")
        
        # One dimensional unisolvent nodes
        self._nodes = unisolvent_nodes_1d(self._n + 1, nodes)

        # Multi-dimensional unisolvent nodes
        self._unisolvent_nodes = unisolvent_nodes(self._nodes, self._m, self._n, self._p)

        # Lagrange to Newton transformation 1D
        self._l2n = rmo(l2n(self._nodes))

        # Newton to Lagrange transformation 1D
        self._n2l = rmo(n2l(self._nodes))

        # Newton differentiation matrix
        # self._dx = rmo(dx(self._nodes)) # TODO
    
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
    
    def fnt(self, function_values: NP_ARRAY) -> NP_ARRAY:
        """Forward Newton Transformation (FNT)"""
        function_values = np.asarray(function_values).astype(np.float64)
        return transform(self._l2n, function_values, self._T, self._m, self._p)

    def ifnt(self, coefficients: NP_ARRAY) -> NP_ARRAY:
        """Inverse Newton Transformation (IFNT)"""
        coefficients = np.asarray(coefficients).astype(np.float64)
        return transform(self._n2l, coefficients, self._T, self._m, self._p)
    
    def dfnt(self, coefficients: NP_ARRAY, i: int) -> NP_ARRAY:
        """Differentiation using Forward Newton Transformation (DFNT)"""
        coefficients = np.asarray(coefficients).astype(np.float64)
        # coefficients_prime = coordinate-transform(coeffs, i)
        # d_coefficients_prime = transform_diag(self._dx, coefficients, self._T, self._p)
        # d_coefficients = backward-coordinate-transform(d_coefficients_prime, i)
        # return d_coefficients
        raise NotImplementedError("Not implemented yet.")
    
    def eval(self, coefficients: NP_ARRAY, x: NP_ARRAY): # TODO ADD TESTS!
        """Point Evaluation"""
        return eval_at_point(coefficients, self._nodes, x, self._m, self._p)
    
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
        
    def __call__(self, function_values: NP_ARRAY) -> NP_ARRAY:
        return self.fnt(function_values)

if __name__ == "__main__":

    # TODO

    # def d_x_f(x, y, z):
    #     return 2*x + 6*x**2
    
    # dx_function_values = [d_x_f(*x) for x in t.unisolvent_nodes]
    
    # dx_coeffs = t.dfnt(coeffs, (1, 0, 0))

    # dx_reconstruction = t.ifnt(dx_coeffs)

    # print(f"||dx_reconstruction-dx_function_values||_1: {np.linalg.norm(reconstruction-function_values)}")

    pass