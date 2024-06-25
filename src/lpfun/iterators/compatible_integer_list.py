import numpy as np
from numba.experimental import jitclass
from lpfun import NP_INT, NP_ARRAY, NB_INT

spec = [
    ("_cs", NB_INT[:]),
    ("_cs_2d_mask", NB_INT[:, :]),
    ("_len", NB_INT),
    ("_depth", NB_INT),
    ("_indices", NB_INT[:]),
    ("_element", NB_INT),
    ("_element_history", NB_INT[:, :]),
    ("_index", NB_INT),
]


@jitclass(spec)
class CompatibleIntegerList(object):

    def __init__(self):
        pass

    def init(self, tiling: NP_ARRAY, depth: int):
        tiling = np.asarray(tiling, dtype=NP_INT)
        self._len = len(tiling)
        self._cs = np.cumsum(tiling)
        cs_enum = np.append(np.array([0], dtype=NP_INT), self._cs[:-1])
        self._cs_2d_mask = np.array(
            [[itr_p, self._cs[idx_p]] for idx_p, itr_p in enumerate(cs_enum)],
            dtype=NP_INT,
        )
        self._depth = depth
        self.clear()

    def clear(self):
        depth = self._depth
        cs = self._cs
        length = self._len
        self._index = 0
        self._indices = np.zeros((max([depth - 1, 1])), dtype=NP_INT)
        self._element = cs[0]
        self._element_history = (
            np.concatenate(
                (
                    np.array([[0, length]], dtype=NP_INT),
                    np.array([[0, self._element]] * (depth - 1), dtype=NP_INT),
                ),
                axis=0,
            )
            if depth > 1
            else np.array([[0, length]], dtype=NP_INT)
        )

    @property
    def indices(self) -> NP_ARRAY:
        return self._indices

    @property
    def element(self) -> int:
        return self._element

    def eval(self, indices: NP_ARRAY) -> bool:
        depth = self._depth
        cs_2d_mask = self._cs_2d_mask
        length = self._len
        element_history = self._element_history
        last_indices = np.copy(self._indices)
        first_no_match = -1
        for i in range(depth - 1):
            if last_indices[i] != indices[i]:
                first_no_match = i
                break
        if first_no_match == -1:
            return True
        j = first_no_match
        kitkat = False
        if j >= depth:
            return False
        element = element_history[j]
        for i in range(j, depth - 1):
            idx = indices[i]
            if element[1] <= length:
                if idx < element[1] - element[0]:
                    element = cs_2d_mask[element[0] : element[1]][idx]
                    if not i == depth - 1:
                        element_history[i + 1] = np.copy(element)
                else:
                    kitkat = True
                    break
            else:
                return False
        self._indices = np.copy(indices)
        self._element_history = element_history
        if not kitkat:
            self._element = element[1] - element[0]
            return True
        else:
            return False

    def next(self) -> bool:
        if self._len == 1:
            return False
        else:
            return self._comp_next()

    def _comp_next(self) -> bool:
        depth = self._depth
        indices = np.copy(self._indices)
        cs_2d_mask = self._cs_2d_mask
        length = self._len
        element_history = np.copy(self._element_history)
        index = self._index
        j = depth - 2
        while True:
            indices[j] += 1
            kitkat = False
            element = element_history[j]
            for i in range(j, depth - 1):
                idx = indices[i]
                if element[1] <= length:
                    if idx < element[1] - element[0]:
                        element = cs_2d_mask[element[0] : element[1]][idx]
                        if not i == depth - 1:
                            element_history[i + 1] = np.copy(element)
                    else:
                        indices[j] = 0
                        j -= 1
                        kitkat = True
                        break
                else:
                    return False
            if kitkat:
                continue
            else:
                self._indices = np.copy(indices)
                self._element = element[1] - element[0]
                self._element_history = np.copy(element_history)
                self._index = index + 1
                return True
