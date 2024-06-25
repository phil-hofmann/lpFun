import numpy as np
from numba import boolean
from numba.experimental import jitclass
from lpfun import NP_INT, NP_FLOAT, NP_ARRAY, NB_INT, NB_FLOAT

@jitclass(
    [
        ("_m", NB_INT),
        ("_n", NB_INT),
        ("_p", NB_FLOAT),
        ("_n_p", NB_FLOAT),
        ("_num_p", NB_FLOAT[:]),
        ("_next_called", boolean),
        ("_multi_index", NB_INT[:]),
        ("_multi_index_p", NB_FLOAT[:]),
        ("_sum_multi_index_p", NB_FLOAT),
        ("_i", NB_INT),
        ("_k", NB_INT),
        ("_l", NB_INT),
        ("_j", NB_INT),
    ]
)
class MultiIndexSet(object):

    def __init__(self, m: int, n: int, p: float):
        self._m = m
        self._n = n
        self._p = p
        self._n_p = n**p
        self._num_p = np.arange(0, n + 1).astype(NP_FLOAT) ** p
        self.clear()

    @property
    def multi_index(self) -> NP_ARRAY:
        return self._multi_index

    @property
    def i(self) -> int:
        return self._i

    @property
    def k(self) -> int:
        return self._k

    @property
    def l(self) -> int:
        return self._l

    def clear(self):
        self._next_called = False
        self._multi_index = np.zeros(self._m, dtype=NP_INT)
        self._multi_index_p = np.zeros(self._m, dtype=NP_FLOAT)
        self._sum_multi_index_p = 0.0
        self._i = 0  # absolute count
        self._k = 1  # inner tiling count
        self._l = 0  # outer tiling count
        self._j = 0

    def next(self):
        """."""
        if not self._next_called:
            self._next_called = True
            return True

        while True:
            while True:
                if self._j >= self._m:
                    return False
                elif self._multi_index[self._j] < self._n:
                    self._sum_multi_index_p = (
                        self._sum_multi_index_p - self._multi_index_p[self._j]
                    )
                    self._multi_index[self._j] += 1
                    self._multi_index_p[self._j] = self._num_p[
                        self._multi_index[self._j]
                    ]
                    self._sum_multi_index_p = (
                        self._sum_multi_index_p + self._multi_index_p[self._j]
                    )
                    break
                else:
                    self._sum_multi_index_p = (
                        self._sum_multi_index_p - self._multi_index_p[self._j]
                    )
                    self._multi_index[self._j] = 0
                    self._multi_index_p[self._j] = 0
                    self._j += 1
                    if self._k > 0:
                        self._l += 1
                    self._k = 0
            if self._sum_multi_index_p <= self._n_p:
                self._i += 1
                self._k += 1
                self._j = 0
                return True
            else:
                self._sum_multi_index_p = np.sum(self._multi_index**self._p)
                if self._sum_multi_index_p <= self._n_p:
                    self._i += 1
                    self._k += 1
                    self._j = 0
                    return True
                else:
                    self._sum_multi_index_p = (
                        self._sum_multi_index_p - self._multi_index_p[self._j]
                    )
                    self._multi_index[self._j] = 0
                    self._multi_index_p[self._j] = 0
                    self._j += 1
                if self._k > 0:
                    self._l += 1
                self._k = 0
